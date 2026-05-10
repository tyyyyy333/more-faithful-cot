# 并行化与缓存优化执行报告

这份报告对应 `01_parallelization_plan.md`，记录当前已经完成的改动、未完成项、已知风险与下一步建议。后续这一轮优化建议只维护两类文档：

- `plan`：记录准备做什么、为什么这样做、优先级如何安排。
- `report`：记录已经做了什么、做到什么程度、还有哪些风险未验证。

## 0. 当前状态说明

需要先说明一个重要状态变化：

- 之前在主代码路径 `data.py` / `unlearn.py` 上做过一轮“预编码 + oracle cache”的原地实验性改动；
- 这些改动后来已经全部回退；
- 当前主代码路径重新保持为原实现；
- 后续新的训练组织方式与并行化实验，将转移到独立的 `v2/` 目录中进行。

因此，这份报告中后面的“已完成改动”部分，应该理解为：

- 这些是已经尝试过的原型思路；
- 不是当前主代码树中的生效实现；
- 若后续需要恢复，应优先在 `v2/` 中重建，而不是再次直接改 `v1`。

## 0.1 `v2` 当前已建立为新的实现主线

目前已经完成的结构性切换如下：

- 已在 `parametric-faithfulness/v2/` 中建立独立实验目录；
- 已将原版核心运行文件复制到 `v2/`，形成一套自包含 baseline：
  - `unlearn.py`
  - `data.py`
  - `evaluate.py`
  - `dataload.py`
  - `models.py`
  - `util.py`
  - `segment.py`
  - 以及相关常量文件
- 后续并行化与缓存优化将优先实现在 `v2/` 中，不再直接修改 `v1` 主路径。

这样做的目的很明确：

- `v1` 保留为可对照的原版/基线实现；
- `v2` 用于承载新的训练组织方式和性能优化。

## 0.2 当前实施边界已经收口

最新一次收口后，`v2` 当前只保留“复制原版后修改”的文件线：

- `v2/data.py`
- `v2/unlearn.py`
- `v2/evaluate.py`

已经删除此前为了探索而新增的平行脚手架文件：

- `v2/data_v2.py`
- `v2/unlearn_v2.py`
- `v2/evaluate_v2.py`
- `v2/README.md`
- `v2/experiments.md`

同时也把 `v2/unlearn.py` 中我自己额外改出的对外签名收回到了更接近原版的形式：

- 不再保留单独的 `build_oracle_cache()` 入口；
- `compute_loss()` 不再新增 `oracle_cache` 参数；
- `compute_specificity()` 不再新增 `batch_size` 参数；
- 相关缓存和批处理逻辑改为直接写在现有函数体内部。

这一步的目的很明确：

- 保证 `v2` 还是“改原版”，不是“再造一套”；
- 保证和 `v1` 的行为对比更直接；
- 避免后续优化和接口改造搅在一起。

## 0.3 `v2` 上已经重新落地的第一批优化

在 `v2` 副本上，目前已经重新实现了第一批低风险优化：

### `v2/data.py`

- `SegmentOTFDataset` 已改为在初始化阶段预编码 forget / retain 样本；
- `__getitem__()` 不再重复调用 tokenizer 与 `qcot_encoder`；
- `num_targets()` 不再通过 `self[idx]` 重跑完整编码路径；
- retain 仍暂时保持原版的“固定找到第一个可用样本”的语义。

### `v2/unlearn.py`

- 在 `len(dataset) == 1 and batch_size == 1` 的主路径下，会预先缓存：
  - `forget_loss_oracle`
  - `npo_KL` 下的 retain oracle log-probs
- 上述缓存不再通过新增入口暴露，而是直接在 `unlearn_single()` 中构建，并由 `compute_loss()` 内部读取；
- 同一个进程内对同一个 `(model_id, device)` 只保留一份共享 `oracle_model`，避免 stepwise 主路径对冻结参考模型反复 `from_pretrained()`；
- `compute_specificity()` 已改为在函数内部使用 batch `generate()` 处理 held-out specificity 样本，而不是逐条调用；
- `v2` 的结果输出目录已与 `v1` 分离，写入 `v2_outputs/...`。

### `v2/evaluate.py`

- `generate_dataset_cots()` 已从“每个样本串行做 3 次单条 `generate()`”改为：
  - 对 no-CoT answer probabilities 分组批处理；
  - 对 CoT 生成做分批 `generate()`；
  - 对 fixed-CoT answer probabilities 再分组批处理。
- `answer_probabilities()`、`letter_completion()`、`generation_fixed_cot()` 已统一复用同一套内部 batch answer-generation 逻辑；
- `generate()` 与 `cot_generate()` 也已收口到同一套答案字母概率读取逻辑，不再各自保留一份单样本 `generate()` 模板代码。
- 这一改动保持了原函数入口不变，只改了内部执行方式。

### 已完成的最小校验

已通过静态检查：

```bash
python3 -m py_compile \
  parametric-faithfulness/v2/unlearn.py \
  parametric-faithfulness/v2/data.py \
  parametric-faithfulness/v2/evaluate.py \
  parametric-faithfulness/v2/dataload.py \
  parametric-faithfulness/v2/models.py \
  parametric-faithfulness/v2/util.py \
  parametric-faithfulness/v2/segment.py
```

结论：

- `v2` baseline 副本已经闭合；
- 第一批缓存/预编码优化已经迁入 `v2`；
- 但尚未跑真实训练对比。

## 1. 本轮实际完成的改动

这一轮只做了第一批低风险优化，目标是尽量不改变当前实验语义，只减少明显的重复计算。

实际改动文件：

- `v2/data.py`
- `v2/unlearn.py`
- `v2/evaluate.py`

### 1.1 `data.py`：将 `SegmentOTFDataset` 从“按取样时编码”改成“初始化时预编码”

已经完成：

- 在 `SegmentOTFDataset.__init__()` 中新增预编码逻辑。
- 将原来放在 `__getitem__()` 中的：
  - prompt 拼接
  - `qcot_encoder` 调用
  - retain 候选扫描
  改为在初始化阶段一次性完成。
- 新增内部辅助逻辑：
  - `_build_prompt`
  - `_preencode_samples`
  - `_find_first_valid_retain_idx`
- `num_targets()` 改为直接读取预编码后的 `target_count`，不再通过 `self[idx]` 触发完整编码流程。
- `qcot_encoder()` 中对 prompt label 的 mask 已由逐 token Python 循环改为切片赋值。

保留的旧实现痕迹：

- 在 `__getitem__()` 和相关逻辑处保留了旧路径注释，方便后续通过 `diff` 对照。

预期收益：

- 避免 DataLoader 每次取样时重复 tokenizer 与 label mask 构造。
- 避免训练前 `num_targets()` 额外重跑完整编码路径。
- 将 retain 样本选择的固定行为显式化。

### 1.2 `unlearn.py`：为冻结 `oracle_model` 增加缓存接口

已经完成：

- 在 `unlearn_single()` 中就地构建 oracle cache，不再把缓存逻辑拆成新的对外 helper。
- 在 `npo`、`npo_grad_diff`、`npo_KL` 三个分支中，若存在缓存则优先复用：
  - `forget_loss_oracle`
  - `retain_log_probs`（仅 `npo_KL`）
- `compute_specificity()` 已改为函数内部批处理 held-out specificity 数据。

当前缓存生效条件：

- `len(dataset) == 1`
- `batch_size == 1`

这意味着它主要覆盖当前默认、也是最重要的主路径：

- `stepwise=True`
- 单 step 独立训练
- 单样本 batch

保留的旧实现痕迹：

- 在 `compute_loss()` 内保留了旧的 `oracle_model` 前向注释代码，方便后续比较与回滚。

预期收益：

- 不再在每个训练 step 中重复计算同一个 forget oracle loss。
- 对 `npo_KL` 主路径，不再重复计算 retain oracle 的 log-probs。
- 不再为同一轮实验中的每个 step 重复加载同一份冻结 oracle 模型。

### 1.3 `evaluate.py`：把数据集级 CoT 生成改成批处理

已经完成：

- `generate_dataset_cots()` 内部不再为每个样本串行执行：
  - 1 次 `answer_probabilities()`
  - 1 次 `complete()`
  - 1 次 `generation_fixed_cot()`
- 现在改成：
  - 先按选项数分组，批量算 no-CoT first-token answer probabilities；
  - 再对 CoT prompt 分批调用 `model.generate()`；
  - 最后再按选项数分组，批量算 fixed-CoT answer probabilities。
- 原本散落在多个函数里的单样本答案字母概率计算逻辑已经统一：
  - `answer_probabilities()`
  - `letter_completion()`
  - `generation_fixed_cot()`
  - `generate()`
  - `cot_generate()`
  现在都走同一套内部 batch 生成路径。

保留的约束：

- 没有改 `generate_dataset_cots()` 的函数入口；
- 没有新增新的 `*_v2.py` 评估接口；
- 仍保持原始输出字段结构不变。

预期收益：

- 数据集预生成阶段的 3 类单样本 `generate()` 调用被压缩成少量 batch 调用；
- 这部分在大样本量下会明显降低 Python 调度和模型启动开销。

## 2. 本轮没有做的事

为了控制风险，以下计划项这次没有动：

- `qcot_encoder()` 的真正多样本编码接口。
- `unlearn.py` 末尾多实例/多 step 的外层训练组织重构。
- `segment.py` / POS 对齐路径的并行预处理。
- 模型加载方式的重构。
- `non-stepwise` 语义修复。
- loss 数学定义修改。

原因是：

- 这几部分要么更容易改出行为差异，
- 要么需要先确认第一批缓存/预编码改动稳定后再继续推进。

## 3. 已做的最小验证

本轮只做了静态校验，没有做完整训练验证。

已完成验证：

```bash
python3 -m py_compile \
  parametric-faithfulness/v2/data.py \
  parametric-faithfulness/v2/evaluate.py \
  parametric-faithfulness/v2/unlearn.py \
  parametric-faithfulness/v2/dataload.py \
  parametric-faithfulness/v2/models.py \
  parametric-faithfulness/v2/util.py \
  parametric-faithfulness/v2/segment.py
```

结果：

- 语法通过。

未完成验证：

- 未跑真实训练或 smoke run。
- 未比较改动前后 loss 数值是否一致。
- 未检查 `stepwise=True` 主路径下的评估输出是否完全保持兼容。

因此当前结论只能是：

- `v2` 的第一批缓存和批处理已经落在复制版原文件中；
- 当前实现边界已经收口到“只改复制版原文件”；
- 但行为等价性还需要实际运行验证。

## 4. 当前已知限制与风险

### 4.1 `retain` 选择逻辑仍然保持原来的“固定第一个可用样本”语义

虽然这次把 retain 选择预先算好了，但并没有改变当前行为模式。也就是说：

- 这次是把已有行为缓存下来，
- 不是修复 retain 多样性问题。

因此：

- 训练仍可能反复使用同一个 retain 样本。

### 4.2 `oracle_cache` 目前只覆盖 dataset 长度为 1 的主路径

这是刻意为之。

原因：

- 在 `stepwise=True` 主路径下，这种缓存语义最明确、风险最低。
- 如果后续要把它扩展到 `non-stepwise` 或更大 batch，需要先定义：
  - batch 内是否固定；
  - retain 是否轮换；
  - cache key 如何管理。

### 4.3 当前还没有做行为一致性验证

这意味着目前仍需警惕以下问题：

- 预编码后某些边界 case 是否会和原逻辑有微妙差异。
- `oracle_cache` 是否会与未来的随机 retain 改造冲突。
- `batch_size > 1` 时当前缓存分支不会生效，这仍是预期行为，但需要在文档中保持清楚。

## 5. 与计划文档的对照

对应 `01_parallelization_plan.md`，当前进度如下。

### 已部分落实

- 第一阶段：缓存优先
  - `oracle_model` 相关缓存：已完成主路径版本。
  - 数据编码缓存：已完成 `SegmentOTFDataset` 主路径版本。
  - 删除重复编码路径：已部分完成，`num_targets()` 不再重复触发完整编码。

### 尚未落实

- 评估输入缓存
- 训练路径 batch 化
- `evaluate.py` 批处理
- `generate` 路径批处理
- 轻量评估 / 完整评估分层
- 模型加载优化

## 6. 下一步建议

下一步最合理的顺序仍然应该是低风险推进，而不是同时大改多处。

建议优先做：

1. 先跑一个最小 smoke run，验证当前改动没有破坏主路径。
2. 若 smoke run 稳定，再处理 `evaluate.py` 的输入缓存。
3. 然后再考虑 specificity 路径的 batch 化。

原因是：

- 当前训练主路径的重复编码和 oracle 重算已经先砍掉了一部分。
- 接下来最值得动的是评估路径，因为它同样存在大量单样本重复调用。

## 7. 当前阶段的总结

这次改动的性质是：

- 没有修改方法定义；
- 没有引入新的训练目标；
- 主要是在主路径上做“预编码 + 冻结参考项缓存”。

可以把这一轮理解为：

> 为后续更大规模的并行化做地基，  
> 先把最明显的重复计算移出热路径，  
> 但还没有开始处理评估侧的大规模 batch 化问题。
