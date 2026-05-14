# 内部状态诊断方案

这份文档承接 `02_faithfulness_problem_analysis.md` 的判断：当前项目主要在行为层压制某个显式 CoT step，但这不足以区分：

- 真实遗忘；
- 只是不再把原句说出来；
- 内部仍沿用同一路径，只是外显 CoT 改写；
- 内部改走另一条等价路径。

因此，这一阶段的目标不是继续优化 loss，而是补一层**机制诊断**，直接比较 unlearning 前后模型“怎么想”的变化。

---

## 1. 核心问题

我们要回答的不是单一问题，而是三个彼此相关的问题：

### 1.1 表征是否还在

当目标 step 被 unlearn 后：

- 与该 step 对应的语义信息，是否仍然能从隐藏状态中读出来？
- 最终答案 token 生成时，内部 residual state 是否仍然携带该 step 的信息？

### 1.2 答案是否仍然依赖该 step

即使模型不再输出原始 step 文本，它在计算最终答案时是否仍然依赖同一段信息？

### 1.3 是删除还是绕路

如果最终答案不变，原因到底是：

- 该 step 本来就不关键；
- 该 step 的表面文本被压掉了，但内部依赖没变；
- 内部依赖减少了，但模型换了一条等价路径；
- 相关知识真的被破坏了。

---

## 2. 总体原则

### 2.1 不把 attention 当成唯一证据

单看 attention score 不够，因为：

- attention 高不代表因果贡献高；
- attention 低不代表信息没被用到；
- 信息可能已经写进 residual stream；
- softmax 后的权重本身可解释性有限。

因此 attention 只能作为**辅助观测量**，不能单独当结论。

### 2.2 优先比较“答案生成时的内部状态”

我们真正关心的是：

> 模型在生成最终答案 token 时，是否还在依赖目标 step 对应的信息。

所以最重要的比较对象不是整段生成过程的所有状态，而是：

- 关键 step token 的表征；
- 最终答案 token 位置的表征；
- 答案位置对关键 step 的依赖。

### 2.3 行为层和机制层必须联动

这套诊断不能脱离当前项目已有指标。最终分析必须联合使用：

- step probability 变化；
- final answer 变化；
- new CoT 变化；
- 内部状态变化。

否则机制指标会失去语境。

---

## 3. 观测对象

这一阶段建议把内部对象分成四层。

### 3.1 残差流 / 隐藏状态

这是最优先的主指标。

建议采集：

- 目标 step token 在每层的 hidden states；
- 最终答案 token 在每层的 hidden states；
- 答案位置与 step 位置之间的层间相似性变化。

建议指标：

- cosine similarity
- L2 distance
- CKA / SVCCA / PWCCA（如后续需要更稳）

这层主要回答：

> 目标 step 的语义表征是否仍保留在模型内部。

### 3.2 Attention 结构

这里不只看 softmax attention matrix，而是分开看：

- answer token 对关键 step token 的 attention mass；
- answer token 对整个 prefix / CoT 的 attention 分布；
- 不同层、不同 head 的变化模式。

可进一步记录：

- Q/K 相似度变化；
- attention output 向量变化。

这层主要回答：

> 最终答案在注意力路由上是否仍然把关键 step 当作重要信息源。

### 3.3 投影向量与模块输出

如果只看 attention weight，容易错过更深层变化，因此还要看：

- Q / K / V 投影后的表示；
- attention output；
- MLP output；
- layer output / residual update。

这层的意义是：

- 区分“只是不再 attend”与“仍然用别的方式保留同样语义”；
- 找到真正变化的是哪个模块。

### 3.4 因果干预对象

最终要进入因果诊断，就要把以下对象当成可 patch 的单位：

- 某层某位置 residual stream；
- 某层某 head 的 attention output；
- 某层某 token 的 MLP output。

这些对象后面会用于 activation patching。

---

## 4. 建议的诊断指标

### 4.1 表征保留指标

对每个目标 `(instance, step)`，记录：

- `repr_sim_step[l]`
  - 原模型 vs unlearn 后模型，在目标 step token 位置、第 `l` 层 hidden state 的相似度
- `repr_sim_answer[l]`
  - 原模型 vs unlearn 后模型，在答案 token 位置、第 `l` 层 hidden state 的相似度
- `cross_sim[l]`
  - 答案位置 hidden state 与目标 step hidden state 的相似度变化

解释：

- 如果 `repr_sim_step` 很高，说明 step 表征基本还在；
- 如果 `repr_sim_answer` 很高而行为变了，说明更可能只是输出策略变化；
- 如果 `cross_sim` 明显下降，说明答案位置与该 step 的内部耦合减弱。

### 4.2 注意力依赖指标

定义：

- `attn_to_step[l,h]`
  - 答案 token 在层 `l`、head `h` 上对目标 step token 的注意力总质量
- `attn_to_prefix[l,h]`
  - 答案 token 对 step 前缀区域的注意力总质量
- `attn_redistribution[l,h]`
  - unlearning 前后 attention mass 在不同区段上的重分配

解释：

- 如果 `attn_to_step` 降低而答案不变，可能发生了 rerouting；
- 如果 `attn_to_step` 不变但 CoT 文本消失，可能更接近“学会不说”。

### 4.3 模块响应指标

定义：

- `q_sim[l,h]`, `k_sim[l,h]`, `v_sim[l,h]`
- `attn_out_sim[l,h]`
- `mlp_out_sim[l]`

这些指标用来回答：

- 是 query-key 匹配关系变了；
- 还是 value 携带的信息变了；
- 还是 residual/MLP 通路在重写答案计算。

### 4.4 因果恢复指标

通过 patching 定义：

- `patch_step_to_answer_gain`
  - 把原模型的关键激活 patch 回 unlearn 后模型后，目标答案 logit 恢复多少
- `patch_answer_flip_rate`
  - patch 后答案是否恢复为原答案
- `reverse_patch_damage`
  - 把 unlearn 后激活 patch 到原模型后，原模型答案是否被破坏

这层是最强证据，因为它不只是比较相关性，而是在做因果干预。

---

## 5. 实施步骤

建议按三阶段推进。

### 5.1 第一阶段：先做只读诊断

目标：

- 不改训练；
- 不改 loss；
- 先能稳定地抽取内部状态并离线分析。

需要实现：

1. 在 `v2` 路径里增加 hook / trace 收集器
2. 给一次前向生成保存：
   - hidden states
   - attentions
   - 可选的 Q/K/V 与模块输出
3. 定义统一的 trace 存储格式

建议 trace 单位：

- `instance_id`
- `step_idx`
- `model_version` (`base`, `unlearned`)
- `prompt_type` (`original`, `step_removed`, `fixed_cot`, `answer_only`)
- `token_spans`
  - target step span
  - answer token span
- per-layer tensors / summary stats

### 5.2 第二阶段：做比较分析

目标：

- 给定同一个 `(instance, step)`，比较 base model 与 unlearned adapter 的 trace。

最小分析矩阵：

1. 原模型 + 原 prompt
2. unlearned model + 原 prompt
3. 原模型 + 去掉关键 step 的 prompt
4. unlearned model + 去掉关键 step 的 prompt

比较输出：

- 表征相似度曲线
- attention redistribution heatmap
- Q/K/V 与 MLP 响应变化
- final answer logits 变化

### 5.3 第三阶段：做 patching

目标：

- 验证“关键 step 的内部信息”是否仍然对答案有因果作用。

建议 patching 顺序：

1. 先 patch residual stream
2. 再 patch attention output
3. 最后 patch MLP output

原因：

- residual stream 最稳定，也最容易解释；
- attention/MLP patching 更细，但也更贵、更难解释。

---

## 6. 建议的代码结构

为了不把现有训练脚本搅乱，建议在 `v2` 下单独加机制诊断模块。

建议新增：

- `v2/trace_utils.py`
  - hook 注册
  - token span 对齐
  - trace 存储/加载
- `v2/trace_compare.py`
  - 相似度计算
  - 指标汇总
  - 层/头级别比较
- `v2/patching.py`
  - residual patching
  - attention output patching
  - MLP output patching
- `v2/run_trace_eval.py`
  - 统一入口：给定 adapter record 或 run manifest，批量生成 trace 和分析结果

建议不要把这些逻辑直接塞进：

- `unlearn.py`
- `evaluate.py`

否则训练主线会迅速变脆。

---

## 7. 数据与对齐细节

### 7.1 必须记录 token span

如果不记录目标 step 在 token 序列中的准确 span，后续所有内部比较都会变得模糊。

因此每次 trace 必须保存：

- prompt token span
- prefix token span
- target step token span
- answer token span

### 7.2 比较时优先对齐“语义位置”

同一句 step 在不同设置下可能 tokenization 略有不同，因此分析时优先按：

- 问题部分
- prefix 部分
- target step 部分
- answer 部分

做分段对齐，而不是盲目按绝对 token index 对齐。

### 7.3 先限制问题规模

第一阶段不建议对所有样本全量存整层全头全 token 的 trace，因为代价很高。

更现实的范围：

- 只对一小批 representative instances
- 只对若干关键层
- 只对答案 token 与目标 step 区域

等诊断跑通后再扩。

---

## 8. 结果解释框架

这一阶段最终想支持三种判断。

### 8.1 更像表面压制

典型特征：

- 目标 step 文本概率下降；
- 最终答案不变；
- hidden states 和答案依赖模式变化很小；
- patching 基本没什么恢复空间，因为原依赖本来就没消失。

解释：

- 模型更像学会了不这样说，而不是忘了怎么想。

### 8.2 更像内部重路由

典型特征：

- 目标 step 文本概率下降；
- 最终答案不变；
- 对关键 step 的依赖下降；
- 但对其他 prefix / 其他 token 的依赖上升；
- patching 表明答案通路改了，但没有真正失去能力。

解释：

- 模型仍能做对，只是换了一条内部计算路径。

### 8.3 更像真实破坏

典型特征：

- step probability 下降；
- answer logits / answer correctness 也明显受影响；
- 关键表征与依赖同时下降；
- patch 回原始关键激活能显著恢复答案。

解释：

- 这更接近“与该 step 相关的内部机制真的被打坏了”。

---

## 9. 优先级建议

建议先做：

1. hidden states + token span 对齐
2. answer-to-step attention mass
3. residual stream patching

原因：

- 这三者已经足够回答大部分“是不是只是学会不说”的问题；
- 同时实现成本还可控。

后做：

4. Q/K/V 级别分析
5. attention output / MLP output patching
6. probe / linear readout

这些更强，但也更重。

---

## 10. 当前阶段的结论

如果要解决“模型是不是在撒谎、是不是只是学会别说那一步”的问题，单靠行为层指标不够。

最合理的补强方向是：

- 比较内部表征；
- 比较答案位置对关键 step 的依赖；
- 用 patching 验证这种依赖是否仍有因果作用。

因此这一阶段的目标不是再换一个 loss，而是建立一套：

> 行为评估 + 内部状态比较 + 因果 patching

的联合诊断框架。

---

## 11. 当前接入方案

为了先把机制诊断挂到现有实验流里，当前实现采用的是：

- 新增一个共享模块：
  - `mechanistic_diagnostics.py`
- 模块内统一封装三部分：
  1. hidden-state / residual 表征比较
  2. answer-to-step attention 依赖摘要
  3. projection / QKV 风格模块输出摘要
- 原版 `unlearn.py` 和 `v2/unlearn.py` 都只在 `evaluate()` 阶段按开关调用它

这样做的好处是：

- 不改当前训练逻辑；
- 不污染 `evaluate.py` 里的原有行为评估 API；
- 默认关闭，只有显式打开时才增加额外前向与 hook 开销；
- 原版和 `v2` 共享同一份诊断逻辑，便于横向比较。

### 11.1 当前开关

当前接入的两个参数是：

- `--mechanistic_diag`
  - 是否在评估阶段运行内部状态诊断
- `--mechanistic_diag_proj_limit`
  - 最多摘要多少个 projection / QKV 风格模块

### 11.2 当前返回字段

开启后，评估结果字典中会新增：

- `mechanistic_diagnostics`

其中当前包含：

- `prompt_lengths`
- `representation`
  - `step_answer_cosine_by_layer`
  - `answer_removed_cosine_by_layer`
  - `answer_norm_by_layer`
  - `removed_answer_norm_by_layer`
- `attention`
  - `answer_to_step_mass_by_layer`
  - `answer_to_prefix_mass_by_layer`
  - `answer_to_post_step_mass_by_layer`
- `projections`
  - 每个模块的 step/answer 相似度与 removed-step 对比
- `warnings`

### 11.3 当前版本刻意还没做的事

当前接入版还没有直接实现：

- activation patching
- probe 训练
- 多 adapter / 多 epoch trace 离线聚合器
- 大规模 trace 持久化

原因是这一版先解决：

> 如何以最小侵入方式，把机制诊断稳定挂到原版和 `v2` 的评估阶段

后续再在这个入口之上继续扩展 patching 和 trace 存储。
