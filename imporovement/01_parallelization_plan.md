# 并行化与缓存优化计划

这份计划只覆盖 `improvement.md` 中的第一点，即在尽量不改变当前实验语义的前提下，系统性优化训练与评估的执行效率。当前项目的主要问题不是单个算子太慢，而是存在大量串行 Python 控制流、重复模型前向、重复 tokenization 与重复评估编码。

## 1. 目标

这一轮优化的目标分为四个层次：

1. 保持当前 `stepwise=True` 主路径的实验语义不变。
2. 优先减少重复计算，再做批处理和向量化。
3. 把 GPU 主要时间从“小 batch 串行调用”转移到“批量模型前向”。
4. 让后续的 `non-stepwise` 修复和更强 loss 改造有一个更稳固的工程基础。

## 1.1 当前实施约束

为了保证 `v1` / `v2` 可对照、也避免工程分叉失控，这一轮第一点优化额外遵守以下约束：

- 只修改 `parametric-faithfulness/v2/` 中**从原版复制出来的原始文件**；
- 当前只推进：
  - `v2/data.py`
  - `v2/unlearn.py`
  - `v2/evaluate.py`
- 不再新增 `*_v2.py` 一类平行入口；
- 尽量不改原有对外函数签名，优先在现有函数体内部完成缓存、批处理和向量化；
- 在完成第一点之前，不把工作扩散到新的训练框架或新的实验接口。

## 2. 现阶段主要瓶颈

结合当前代码，瓶颈大致来自以下几类：

- `unlearn.py` 中每个 target step 都重新加载 `model` 与 `oracle_model`。
- `compute_loss` 中对冻结 `oracle_model` 的前向重复且未缓存。
- `data.py` 中 `qcot_encoder`、prompt 拼接、mask 构造放在 `__getitem__`，导致每次取样都重复做预处理。
- `evaluate.py` 中大量 `generate()` 和 answer probability 计算以单样本方式执行。
- `segment.py` 和 POS 对齐逻辑目前偏单条样本预处理。
- `num_targets()` 会在正式训练前再次触发完整数据编码流程。

## 3. 总体策略

优化顺序建议如下：

1. 先做“缓存确定不变的量”。
2. 再做“批量化 teacher-forcing 前向”。
3. 然后做“批量化 generation / evaluation”。
4. 最后再处理更复杂的 POS 对齐和分任务选项数差异。

原因是：

- 缓存通常最稳，改动小、收益高。
- 向量化如果太早做，很容易和当前粗糙数据流耦合在一起，导致改动范围失控。
- 评估路径最慢，但也最容易引入行为变化，适合在训练主路径稳定后再推进。

## 3.1 新的核心提速方向：不要再把每个 target step 当成一个完全独立的小训练任务

当前最主要的问题不是单个 batch 太小，而是训练组织方式本身非常碎：

- 每次只拿一个 target 样本；
- 每个样本在 `stepwise=True` 下只处理一个 step；
- 每个 step 又要重复训练很多个 epoch；
- 每个 step 基本都被当成一次独立小实验运行。

这会导致：

- GPU 吞吐极低；
- 模型加载和准备成本被放大很多次；
- 相同 retain / oracle / 评估逻辑被反复执行。

因此，后续优化不应只局限于“给现有单样本路径提速”，还应显式引入**新的批量训练模式**。

### 3.1.1 多样本联合训练模式

一个更激进但很自然的方向是：

- 不再一次只训练一个 forget step；
- 而是一次性取多个 forget target；
- 按长度分桶后组成 batch；
- 用同一个可训练模型同时更新。

这样做的核心收益是：

- 训练吞吐会显著提升；
- 可训练模型的前向和反向不再被单步独占；
- retain 约束也可以天然 batch 化。

但要注意，这会改变实验语义：

- 当前 `stepwise=True` 的问题是“单独删除这一步会怎样”；
- 多样本联合训练后，问题更像“同时删除一批 step 会怎样”。

所以这一模式更适合作为：

- 面向效率的训练模式；
- 或面向更大规模 unlearning 的近似方案；

而不是原始单步因果 probing 的严格复现。

### 3.1.2 分桶策略：比“相同步数”更重要的是“相近长度”

为了让多样本 batch 真正高效，分桶逻辑不应只看 step 数，更应看：

- prompt token 长度；
- prefix token 长度；
- completion token 长度；
- 总序列长度。

“相同步数”是一个有用启发，因为相同步数往往意味着 prefix 长度相近；但真正决定 padding 浪费和显存占用的是 token 长度，而不是逻辑步数本身。

因此更合理的 bucket key 可以是：

- `step_idx`
- 再叠加 token length bucket

例如：

- 第 1 步 + 短前缀；
- 第 2-3 步 + 中前缀；
- 第 4 步及以后 + 长前缀。

### 3.1.3 LoRA 并行是第二阶段而不是第一阶段

另一个很有吸引力的方向是：

- 共享一个 base model；
- 为不同 target 初始化很多组独立 LoRA 参数；
- 用 adapter 路由的方式，让 batch 内不同样本更新不同 LoRA；
- 相当于并行地训练很多个“独立编辑后的小模型”。

这个方向理论上很适合当前项目，因为它能避免反复复制完整模型权重。但它实现复杂度更高，主要难点包括：

- batch 内样本如何路由到不同 adapter；
- 梯度如何严格隔离，只更新各自 LoRA；
- optimizer state 会随 adapter 数量增长；
- 需要重新设计结果导出与评估接口。

因此更合理的推进顺序是：

1. 先做普通的多样本联合 batch 训练；
2. 再评估是否值得进一步做多 adapter LoRA 并行。

简化地说：

> “多样本 batch + 分桶”应当是第一阶段；  
> “多组 LoRA 并行独立实验”应当是第二阶段。

## 3.2 retain 端的新思路：从“单 retain 样本”改成“全量平均 + 异步刷新”

当前 retain 端的一个根本问题是：

- 它通常只选到很少量甚至固定的 retain 样本；
- 这既不稳定，也容易让 retain 正则带有偶然性。

如果暂时不把重点放在最终输出答案，而只强调“不要破坏模型主体知识”或“保持输入语义对齐”，那么 retain 端可以考虑更强的平均化近似。

### 3.2.1 全量 retain 平均

一个直接想法是：

- 对所有 retain 样本都计算 retain loss；
- 再对它们做平均；
- 用这个平均 retain loss 作为统一的稳定项。

和当前做法相比，它的优点很明显：

- 波动更小；
- 不依赖某一个 retain 样本的偶然性；
- 更接近“保持整体知识分布不变”的目标；
- 对 batch 训练更友好。

缺点是：

- 单次更新成本更高；
- 若每步都完整重算，会带来额外前向开销。

### 3.2.2 异步 retain 平均缓存

为了解决“全量 retain 平均太贵”的问题，可以借鉴 DQN/target network 一类的思路，不追求每一步都用最新 retain 平均，而是：

- 周期性地在后台或间隔若干步，重新计算一次 retain 平均损失或 retain 参考统计；
- 将这个平均项缓存下来；
- 连续若干次参数更新都复用这个较旧的平均 retain 项；
- 到下一个刷新点再整体更新。

可缓存的对象可以分层考虑：

1. 最简单：缓存 oracle 侧 retain log-probs。
2. 更进一步：缓存某一轮下的平均 retain target statistics。
3. 若要更激进：缓存完整的平均 retain loss 近似项。

### 3.2.3 这种方案成立所依赖的假设

这类异步平均方案隐含了一个很强的假设：

> 遗忘某个局部 CoT step 的训练，不会在短时间内大幅改变模型主体知识，  
> 也不会显著改变 retain 侧的参考分布。

也就是你说的那个核心前提：

- 当前干预主要是局部性的；
- 不会立刻触发大规模灾难性遗忘；
- 因此允许 retain 端在若干步内使用“陈旧但稳定”的平均近似。

如果这个假设成立，那么 retain 端完全没有必要每一次更新都重新精确计算。

### 3.2.4 风险与验证

这个方向虽然很有吸引力，但必须额外验证：

- 如果遗忘更新实际上会较快扰动主体分布，那么陈旧 retain 平均可能失真；
- 如果使用的是当前模型侧 retain 平均而不是 oracle 侧参考，那么 stale cache 的偏差会更明显；
- 如果刷新间隔太长，retain 项可能失去约束作用。

因此需要引入两个超参数：

- retain 平均使用多少样本；
- retain cache 每隔多少步/多少 epoch 刷新一次。

并至少监控：

- retain loss 的漂移幅度；
- specificity 指标是否在刷新间隔增大后明显恶化；
- forget/retain tradeoff 是否失衡。

## 4. 第一阶段：缓存优先

这一阶段不改实验定义，只减少无意义重复计算。

### 4.1 `oracle_model` 相关缓存

文件：

- `unlearn.py`

计划：

- 在 `stepwise=True` 的默认主路径下，`forget_inputs` 固定，因此 `forget_loss_oracle` 可以在 epoch 循环前计算一次并缓存。
- 对 `npo_KL` 分支，若 retain 样本固定，则 `oracle_model` 在 retain 上的 logits / log-probs 也应缓存。
- 若后续 retain 采样改成真正轮换，则 oracle retain cache 需要改成按 retain sample key 缓存。

预期收益：

- 明显减少冻结参考模型前向次数。
- 降低 `compute_loss` 的重复开销。

### 4.2 数据编码缓存

文件：

- `data.py`
- `segment.py`

计划：

- 将 `prompt + prefix + completion` 的编码结果预先构造成张量，而不是在 `SegmentOTFDataset.__getitem__()` 中实时调用 `qcot_encoder`。
- 对 `pos_filter=False` 主路径，直接缓存：
  - `input_ids`
  - `labels`
  - `attention_mask`
  - `target_token_count`
- 对 `pos_filter=True` 路径，也至少缓存 POS 对齐结果，避免重复调用 tokenizer 与 spaCy。

预期收益：

- 降低 DataLoader 每次取样时的 CPU 开销。
- 为后续 batch 化 collator 做准备。

### 4.3 评估输入缓存

文件：

- `evaluate.py`

计划：

- 缓存每个评估样本的：
  - answer prompt
  - cot prompt
  - answer letter token ids
  - 固定 CoT 的编码结果
- 对 held-out specificity split，建立一次性预处理结构，而不是每次 `evaluate()` 重新构造 prompt。

预期收益：

- 明显减少频繁评估时的 prompt 组装与 tokenizer 开销。

### 4.4 删除或改写重复编码路径

文件：

- `data.py`
- `unlearn.py`

计划：

- 改写 `num_targets()`，不再通过 `self[idx]` 触发完整编码流程。
- 对 step 长度、有效 target token 数等信息，采用预先统计。

预期收益：

- 避免训练前完整重跑一遍数据管线。

## 5. 第二阶段：训练路径向量化

这一阶段重点是 teacher-forcing 路径，而不是 generation。

### 5.1 将 `SegmentOTFDataset` 改成预展平数据集

文件：

- `data.py`

计划：

- 预先把 `forget` / `retain` 样本展平成统一结构，不在 `__getitem__` 里做逻辑判断与字符串拼接。
- 对 `stepwise=True`，数据集可以直接只保留一个 forget target，而 retain 部分只返回已编码候选。
- 对 `stepwise=False`，直接把所有 step 预编码成列表。

收益：

- 提高 DataLoader 稳定性。
- 降低 Python 逻辑分支成本。

### 5.2 支持批量编码和批量 collate

文件：

- `data.py`

计划：

- 让 `FRCollator` 只负责 padding 和 device 搬运，不再承担任何编码逻辑。
- 在 retain 样本长度允许的前提下，为 `stepwise=False` 训练支持真正的 `batch_size > 1`。
- 在 `stepwise=True` 主路径中，即使 batch 仍为 1，也应保留批量接口一致性，方便后续统一。

收益：

- 为训练循环中的向量化前向打通接口。

### 5.3 减少无必要的模型重载

文件：

- `unlearn.py`

计划：

- 在同一 target instance 的多个 step 之间，考虑只重置可训练权重而不是整模型重新从磁盘加载。
- 若严格复现实验要求必须重新开始，则至少可复用 `oracle_model` 常驻内存，减少重复加载。
- 后续可评估是否引入“原始状态快照 + in-memory restore”的机制，替代反复 `from_pretrained`。

收益：

- 缩短单个 target step 之间的准备时间。
- 减少磁盘读取和 GPU 权重搬运开销。

## 6. 第三阶段：评估路径向量化

这是最有潜力但也最容易引入行为差异的部分。

### 6.1 `answer_probabilities` 批处理

文件：

- `evaluate.py`

计划：

- 将同一任务内的 answer prompt 做 padding 后批量 `generate()` 或批量前向。
- 对 `A/B/C/D/E` 的读取按任务分组，避免不同选项数直接混 batch。

难点：

- 不同任务选项数不同。
- 不同 prompt 长度不同。

收益：

- specificity 评估会显著提速。

### 6.2 `complete` / `generation_fixed_cot` 批处理

文件：

- `evaluate.py`

计划：

- 将多个 CoT prompt 合并批量生成。
- 对固定 CoT 再打分的路径，优先改成 teacher-forcing 批量前向，而不是逐条 `generate()`。

收益：

- 降低每次 epoch 评估成本。

### 6.3 评估频率与评估模式分层

文件：

- `unlearn.py`

计划：

- 将评估拆成“轻量评估”和“完整评估”。
- 每个 epoch 只跑轻量项：
  - forget step probability
  - final answer probability
- 最终 epoch 再跑：
  - specificity
  - new CoT generation
  - fixed CoT probability

收益：

- 在不丢掉结果的前提下显著降低中间 epoch 的评估时间。

## 7. 文件级实施顺序

建议按下面顺序推进，避免同时改太多路径：

1. `unlearn.py`
   - 缓存 `forget_loss_oracle`
   - 评估频率分层
   - 减少模型重复加载
2. `data.py`
   - 预编码 forget / retain
   - 改造 `SegmentOTFDataset`
   - 精简 `FRCollator`
3. `evaluate.py`
   - prompt/token id 缓存
   - specificity 路径批处理
   - CoT generation 批处理
4. `segment.py`
   - POS 对齐与 tokenizer 路径的并行预处理

## 8. 每一步改完后的验证

每次优化后都应做最小行为验证，而不是一次性大改。

建议检查：

- 相同 seed 下，`stepwise=True` 主路径结果是否基本一致。
- `forget_loss`、`retain_loss` 数值是否与改动前一致或仅有浮点级差异。
- 训练样本数、target token 数、retain 选择逻辑是否改变。
- `evaluate()` 输出字段是否保持兼容。

性能指标建议记录：

- 单个 target step 的训练总耗时
- 单个 epoch 耗时
- 单次 `evaluate()` 耗时
- GPU 显存占用峰值
- `oracle_model` 前向次数

## 9. 本阶段不建议立即做的事

当前先不建议把下面几件事和并行化混在一起做：

- 修改 NPO / KL 的数学定义
- 引入 RL 式 unlearning
- 大规模重写 `non-stepwise` 语义
- 将 faithfulness 评估标准整体替换

这些都属于更上层的方法改造，应该建立在当前数据流与训练/评估路径先跑得足够快、足够稳定之后。

## 10. 一个更具体的执行结论

这一轮并行化的最优先任务不是“到处加 batch”，而是：

1. 把冻结 oracle 的重复前向缓存掉。
2. 把数据编码从 `__getitem__` 中移出去。
3. 把高频评估改成轻量版并缓存输入。
4. 再逐步把 evaluation 中的单样本 `generate()` 批处理化。

也就是说，当前最值得优先实现的是：

> 先缓存，再向量化；  
> 先 teacher-forcing 主路径，再 generation 评估路径。
