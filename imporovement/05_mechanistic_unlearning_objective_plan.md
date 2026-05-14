# 机制约束的 Unlearning 目标设计

这份文档承接：

- `02_faithfulness_problem_analysis.md`
- `04_internal_state_diagnosis_plan.md`

前者讨论了当前方法更像行为层 suppression，后者给出了内部状态诊断方案。  
这一份文档继续往前走一步：把“行为层目标”和“机制层目标”结合起来，设计更合理的 unlearning loss。

核心问题是：

> 如果当前 NPO 只是在压低某条 step 文本的 token likelihood，那么模型完全可能只是学会“不这样说”，而不是“不会这样想”。  
> 因此，我们需要把内部状态或内部语义的约束，也纳入训练目标。

---

## 1. 我们目前讨论过的几个方向

### 1.1 只遗忘第一个 token 或前 `k` 个 token

用户观点：

- 如果模型在当前 prompt 下连目标 step 的第一个 token 都想不起来，那么它很难原样继续说出这一步。
- 因此可以只对第一个 token，或者前 `k` 个 token 做 forget objective。

这个方向的优点：

- 比整段 step 的 token-sum loss 更稳定；
- 梯度更集中；
- 长 step 与短 step 的尺度差异更小；
- 更像“阻断这一步的起始触发”。

这个方向的缺点：

- 很容易退化成字符串级 suppression；
- 模型可能只是换一个开头继续表达同样内容；
- 对 paraphrase 和近义说法过于脆弱。

结论：

- 适合作为一个很强的稳定 baseline；
- 不适合作为最终唯一目标。

---

### 1.2 直接对目标 step 的内部向量施加约束

用户观点：

- 对要遗忘的 step，在每一层都可以取一个语义向量；
- 与原始模型对应层的语义向量做余弦相似度比较；
- 对层做从后往前的折扣加权，类似多层 credit assignment；
- 再和原有损失做尺度对齐后相加。

这个方向的优点：

- 比 attention 更稳定；
- 比 token-level suppression 更接近机制层；
- 易于和已有 hidden-state hook 兼容。

这个方向的风险：

- 仅仅“远离原向量”不等于“进入合理的无该知识状态”；
- 可能只是把表示推向某个奇怪方向；
- 如果所有层都强推，容易误伤整体能力。

结论：

- 方向成立；
- 但不建议把“远离原始向量”作为最终形式；
- 更适合改造成“层加权的 counterfactual / prototype loss”。

---

### 1.3 对目标 step 建立语义原型（prototype）

用户观点：

- 同一个 step 可以有很多不同表述；
- 如果只压原句，模型可以换一种说法；
- 因此可以对该 step 的语义向量做平均，形成一个语义中心；
- 然后压制与这个语义中心接近的表征。

这个方向的优点：

- 从字符串级 suppression 升级到语义簇 suppression；
- 更能覆盖 paraphrase；
- 比“只压前几个 token”更贴近真正的知识区域。

这个方向的风险：

- 只有“远离 prototype”，仍然缺少“靠近什么”的目标；
- 单个平均向量可能太粗；
- 如果 prototype 太宽，容易误伤语义邻近但合理的推理。

结论：

- 这个方向很有价值；
- 但最好做成**正负 prototype 对比**，而不是只有单一负 prototype。

---

## 2. 从这些想法里抽出最有价值的部分

综合起来，最值得保留的部分是：

### 2.1 保留 first-`k` token 作为稳定行为层基线

这个想法很务实。

它不能解决“模型是不是还会这样想”的问题，但可以解决：

- 当前 full-step NPO 不稳定；
- 不同步长长度导致的尺度问题；
- 训练过重的问题。

因此：

- `first-k forget objective` 很适合保留成一条 baseline；
- 后续可以与机制项结合。

### 2.2 保留“层加权”的思想

用户提出的多层折扣非常合理。

原因：

- 越靠后的层越接近最终决策；
- 越靠前的层越通用；
- 因此后层应该权重大，前层权重小。

这条思想应当保留，并作为机制项的统一加权方式。

### 2.3 把“远离原向量”升级成“靠近反事实语义区域”

这是我认为最重要的改造。

与其只要求：

- 不像原始 step 的内部表征

不如要求：

- 更像“去掉该 step 后”的反事实表征
- 或更像“非该 step 语义簇”的正 prototype

这样训练目标更有语义方向。

### 2.4 把单一 prototype 升级成对比式 prototype

这也是关键。

不只建：

- `forget prototype`

还应当建：

- `retain / counterfactual prototype`

然后训练时：

- 远离 `forget prototype`
- 靠近 `retain / counterfactual prototype`

这比单独压负 prototype 更稳。

---

## 3. 推荐的目标结构

不建议完全抛弃当前 NPO。  
更合理的是：

> 保留 NPO 负责行为层压制，再叠加机制层 regularizer，防止模型只是学会不说。

统一形式可以写成：

\[
L
=
\lambda_{\text{forget}} L_{\text{forget}}
+
\lambda_{\text{mech}} L_{\text{mechanistic}}
+
\lambda_{\text{retain}} L_{\text{retain}}
\]

其中：

- `L_forget`
  - 行为层 forget objective
- `L_mechanistic`
  - 内部语义/表征/依赖约束
- `L_retain`
  - 原有 retain KL / CE 项

---

## 4. 行为层 forget objective：推荐保留两条版本

### 4.1 `full-step NPO`

这是当前项目的主线：

- 对整段 step continuation 做 NPO
- 语义最完整
- 但不够稳

### 4.2 `first-k NPO`

把 forget 目标限制在目标 step 的前 `k` 个 token：

\[
L_{\text{forget-first-k}}
=
L_{\text{NPO}}(\text{first } k \text{ target tokens})
\]

建议：

- `k ∈ {1, 2, 4, 8}`
- 作为稳定 baseline
- 后续与机制项叠加

推荐定位：

- `full-step NPO`：强基线
- `first-k NPO`：稳定基线

---

## 5. 机制项 1：层加权的 counterfactual representation loss

这是我最推荐的第一机制项。

### 5.1 直觉

与其让当前表征“远离原始表征”，不如让它“靠近没有该 step 时的反事实表征”。

设：

- `h^{cur}_{l,t}`：当前模型在层 `l`、位置 `t` 的表示
- `h^{cf}_{l,t}`：原模型在去掉该 step 的 prompt 下，对应位置的反事实表示

则可定义：

\[
L_{\text{repr-cf}}
=
\frac{1}{|T|}\sum_{t \in T}\sum_{l \in \mathcal{L}}
w_l \cdot
\bigl(1-\cos(h^{cur}_{l,t}, h^{cf}_{l,t})\bigr)
\]

这里：

- `T` 可以是目标 step token 集合
- 也可以是答案位置 token 集合
- `w_l` 是层权重

### 5.2 为什么它比“远离原始向量”更好

因为它不是无目标地推开，而是把模型拉向一个有语义解释的状态：

- “如果没有这个 step，模型应该是什么样”

### 5.3 推荐优先比较的位置

建议优先做两种版本：

1. `step-position repr loss`
   - 约束目标 step token 本身的内部表征
2. `answer-position repr loss`
   - 约束最终答案 token 位置的内部表征

其中更重要的是第二种：

- 答案位置的状态，才更直接反映模型是否仍在依赖该 step。

---

## 6. 机制项 2：prototype contrastive loss

这是把你的“平均语义向量”想法系统化后的版本。

### 6.1 负 prototype

对要遗忘 step 及其 paraphrases，抽 hidden states，构造：

\[
v^{(l)}_{\text{forget}}
=
\text{norm}\left(
\frac{1}{n}\sum_i \text{norm}(v^{(l)}_i)
\right)
\]

### 6.2 正 prototype

可选来源：

- 去掉该 step 后的 counterfactual continuation
- retain steps
- 中性或不依赖该知识的合理 continuation

构造：

\[
v^{(l)}_{\text{retain}}
\]

### 6.3 对比损失

令当前表示为 `h_l`，则：

\[
L_{\text{proto}}
=
\sum_{l \in \mathcal{L}} w_l \cdot
\max\Bigl(
0,\ 
\cos(h_l, v^{(l)}_{\text{forget}})
- \cos(h_l, v^{(l)}_{\text{retain}})
+ m
\Bigr)
\]

直觉：

- 离 forget prototype 远一些；
- 离 retain/counterfactual prototype 近一些。

### 6.4 为什么这比单一平均向量更好

因为它不只是说：

- “别像那个”

而是说：

- “别像那个，而且要往合理区域靠”

---

## 7. 机制项 3：层加权的相似度折扣

保留你提出的分层折扣思想。

如果总层数为 `L`，层编号为 `0 ... L-1`，推荐：

\[
w_l = \gamma^{(L-1-l)}
\]

其中：

- 最后一层权重最大
- 越往前层权重越小

建议：

- `\gamma ∈ [0.85, 0.97]`
- 若层很多，用 `0.95`
- 若只取最后几层，用 `0.9`

同时建议：

- 不必全层都用；
- 第一版只取最后 `1/3` 或最后 `1/4` 的层。

---

## 8. 不推荐直接采用的形式

### 8.1 不推荐：只用“与原始向量的余弦相似度”作为机制项

即：

\[
L = \sum_l w_l \cos(h_l, h^0_l)
\]

然后最小化。

问题：

- 它只是无目标地推离原始状态；
- 容易把模型推向某个奇怪方向；
- 不够语义化。

### 8.2 不推荐：只看第一个 token

即使 first-token 很稳，也太窄了。

更建议：

- `first-k`
- 而不是只 `first-1`

### 8.3 不推荐：只加 attention 抑制项

attention 可以作为辅助项，但不应作为机制训练主项。

---

## 9. 推荐的三档实验路线

### 9.1 档位 A：稳定基线

\[
L = L_{\text{first-k NPO}} + \lambda_{\text{retain}} L_{\text{retain}}
\]

用途：

- 测试更稳定的 forget objective 是否已足够。

### 9.2 档位 B：推荐主线

\[
L
=
L_{\text{first-k NPO}}
+
\lambda_{\text{repr}} L_{\text{repr-cf}}
+
\lambda_{\text{retain}} L_{\text{retain}}
\]

用途：

- 行为层阻断原 continuation
- 机制层推动答案内部状态向反事实靠近

这是我最推荐的第一正式版本。

### 9.3 档位 C：语义簇版本

\[
L
=
L_{\text{first-k NPO}}
+
\lambda_{\text{proto}} L_{\text{proto}}
+
\lambda_{\text{retain}} L_{\text{retain}}
\]

用途：

- 从“反事实单点目标”扩展到“语义簇目标”
- 更适合覆盖 paraphrase

这是更长期也更有潜力的一条线。

---

## 10. 关于尺度对齐

你提到“和现在的损失函数 scaling 一下确保尺度不变”，这是必须做的。

建议不要手调，而是先做统计归一化。

### 10.1 基本方法

在一个小批样本上估计：

- `E[L_forget]`
- `E[L_mech]`

然后设：

\[
\lambda_{\text{mech}}
=
\alpha \cdot \frac{E[L_{\text{forget}}]}{E[L_{\text{mech}}] + \epsilon}
\]

其中：

- `\alpha ∈ {0.05, 0.1, 0.2}`

### 10.2 为什么这样做

这样可以：

- 保证机制项不是完全淹没行为项；
- 也不会因为数值尺度太小而完全不起作用。

---

## 11. 实施优先级

建议顺序：

### 第一步

实现：

- `first-k NPO`

原因：

- 最稳定
- 改动最小
- 便于和现有 full-step NPO 直接对照

### 第二步

实现：

- `layer-weighted counterfactual repr loss`

原因：

- 它最直接回应“模型是不是只是学会不说”这个问题；
- 比 prototype 和 patching 都更容易先落地。

### 第三步

实现：

- `prototype contrastive loss`

原因：

- 它更像真正的语义簇 suppression；
- 但前置工作更多，需要 paraphrase / prototype 构建。

---

## 12. 当前推荐结论

综合用户提出的思路和目前最有价值的机制方向，当前最推荐的目标设计是：

### 短期推荐

\[
L
=
L_{\text{first-k NPO}}
+
\lambda_{\text{repr}} L_{\text{repr-cf}}
+
\lambda_{\text{retain}} L_{\text{retain}}
\]

其中：

- `first-k NPO`
  - 负责稳定阻断原 step continuation 的入口
- `repr-cf`
  - 负责防止模型只是学会换个说法
- `retain`
  - 负责不让模型整体漂移

### 中期推荐

在上述基础上，把 `repr-cf` 的单点目标逐步扩展成：

- `prototype contrastive loss`

从而把 suppression 从：

- 单一表征

升级成：

- 语义簇级别的抑制

---

## 13. 一句话总结

最值得保留的用户想法有三点：

- 用 `first-k token` 提高 forget objective 的稳定性；
- 用分层 `gamma` 折扣强调后层机制；
- 用语义平均向量 / prototype 覆盖不同表述。

最有价值的改造是两点：

- 不只“远离原始表征”，而是“靠近反事实或正 prototype”；
- 不只保留负 prototype，而是做正负对比。

因此，最终推荐的方向不是单一技巧，而是：

> **稳定的 first-k 行为层忘却 + 层加权的机制约束 + 语义 prototype 扩展**

这三者结合，才最接近“既不只是学会不说，又能训练得动”的目标。

---

## 14. 当前已实现的第一版

目前代码里已经先实现了一个**第一版可插拔机制 loss**，目标是把最容易稳定落地的部分先接到原版和 `v2` 里。

### 14.1 当前实现文件

- 共享模块：
  - `mechanistic_objectives.py`
- 接入位置：
  - 原版 `unlearn.py`
  - `v2/unlearn.py`

### 14.2 当前已实现的两个部件

#### A. `first-k` forget objective

当前 forget loss 已支持：

- 如果 `forget_k_tokens == 0`
  - 保持原始 full-step forget 行为
- 如果 `forget_k_tokens > 0`
  - 只对目标 step 的前 `k` 个 target token 计算 forget objective

这对应本文件前面讨论的：

- `first-k NPO` 稳定基线

#### B. 层加权 representation similarity penalty

当前已支持一个辅助机制项：

- 对 forget step 的 hidden states
- 与 reference/oracle 模型在**同一输入**上的 hidden states 做余弦相似度比较
- 对目标 token 做平均
- 只取最后若干层
- 再按 `gamma` 从后往前折扣
- 最后将该项加回当前 forget loss

也就是说，这一版实现的是：

\[
L
=
L_{\text{forget}}
+
\lambda_{\text{repr}} L_{\text{repr-sim}}
+
L_{\text{retain}}
\]

其中当前 `L_repr-sim` 更接近：

- “别再像原来那样表示这个 step”

而不是更强的：

- “靠近 removed-step 的 counterfactual 表征”
- 或“远离 forget prototype、靠近 retain prototype”

### 14.3 当前开关

原版和 `v2` 现在都支持以下参数：

- `--forget_k_tokens`
  - 只对目标 step 的前 `k` 个 token 计算 forget objective
- `--repr_loss`
  - 是否开启 representation similarity penalty
- `--repr_lambda`
  - 机制项权重
- `--repr_last_layers`
  - 取最后多少层做机制项
- `--repr_gamma`
  - 分层折扣系数
- `--repr_k_tokens`
  - 机制项里只取前 `k` 个 target token
- `--repr_auto_scale`
  - 是否按当前 forget loss 尺度自动缩放机制项

### 14.4 当前实现的定位

这一版不是最终推荐主线，而是：

- 可切换
- 可对照实验
- 与原 NPO 兼容
- 工程风险较低

的第一版机制约束实现。

它最接近本文前面保留下来的两点：

1. `first-k` token 稳定 forget
2. 层加权 hidden-state 机制项

### 14.5 当前还没有实现的内容

本文件中更长期推荐的两条线，目前还没有真正落代码：

1. `counterfactual representation loss`
   - 也就是把目标从“远离原始表征”升级为“靠近 removed-step 反事实表征”

2. `prototype contrastive loss`
   - 也就是把单一平均语义向量升级为正负 prototype 对比

因此当前这版应被理解为：

> 第一版工程实现：稳定 baseline + 机制项入口  
> 而不是最终语义最强的训练目标。

### 14.6 推荐的当前使用方式

如果要给合作者一个清晰的实验入口，当前最值得先跑的三组是：

1. 原始基线

```bash
--method npo_KL
```

2. 稳定 forget 基线

```bash
--method npo_KL --forget_k_tokens 4
```

3. 稳定 forget + 机制项

```bash
--method npo_KL \
--forget_k_tokens 4 \
--repr_loss \
--repr_lambda 0.1 \
--repr_last_layers 4 \
--repr_gamma 0.9 \
--repr_auto_scale
```

这样最容易回答三个问题：

- `first-k` 是否比 full-step 更稳；
- 只加稳定 forget 是否已经足够；
- 加上机制项后，行为变化与内部变化是否更一致。
