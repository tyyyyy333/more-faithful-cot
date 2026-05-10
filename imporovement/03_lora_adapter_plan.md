# LoRA Adapter 并行化计划

这份文档承接第一阶段的工程加速收尾，进入下一阶段的结构性改造：不再把 `stepwise` 视为“很多次完整小微调实验”，而是改成：

- 一个共享的 base model；
- 多个独立的 LoRA adapter；
- 每个 `(instance, step)` 任务对应一套独立 adapter 参数；
- 最后保存的是 adapter，而不是一份份完整主模型。

## 1. 目标

这一阶段的核心目标不是继续优化旧脚本，而是改训练组织方式：

1. 传入一个 base model，返回一个维护多组 LoRA 参数的对象。
2. 对一批 stepwise 任务，一次性创建多组 adapter。
3. 训练时不再反复加载庞大的主模型，只在共享 base model 上切换/更新不同 adapter。
4. 训练结束后保存每个任务对应的 LoRA 参数，并保留任务到 adapter 的映射关系。
5. 推理/评估时，只需要加载 base model，再挂上指定 adapter。

## 2. 为什么这比当前方式更治本

当前 `stepwise` 即使完成了第一阶段加速，本质上仍然是：

- 一个 `(instance, step)` 一个独立微调任务；
- 每个任务都要构造自己的可训练模型状态；
- 只是现在重复计算和评估开销被压下去了。

这仍然没有改变根问题：

> 你在跑很多个彼此独立的小编辑实验。

LoRA 多 adapter 的目标是改变这件事：

- 主模型只加载一次；
- 不同任务的差异只体现在低秩增量参数上；
- 最终保存的也是小体积 adapter；
- 从“很多次完整模型编辑”变成“一个共享底座上的很多次局部独立编辑”。

## 3. 第一版实现边界

第一版不做多进程异步，但会直接支持：

- **共享一个 base model**
- **支持很多个独立 adapter**
- **训练多个 adapter 时不重复加载 base model**
- **batch 内不同 sample 可以绑定不同 adapter**
- **trainer 可以把多个任务合成一个更新批次**

这意味着第一版的收益主要来自：

- 避免反复 `from_pretrained()` 主模型；
- 避免为每个任务复制完整可训练参数；
- 让“很多个 stepwise 任务”共享一套底座；
- 直接具备 adapter-level batch 并行的基本能力。

## 4. 对象模型

### 4.1 `LoRAAdapterManager`

第一版的核心对象是：

- 输入：一个已经加载好的 base model
- 输出：一个 adapter manager 对象

该对象负责：

- 找到目标线性层并注入 LoRA wrapper；
- 创建新 adapter；
- 激活/停用某个 adapter；
- 返回某个 adapter 的可训练参数；
- 保存/加载指定 adapter；
- 维护“有哪些 adapter、分别对应哪些层”的状态。

### 4.2 `MultiAdapterLinear`

LoRA 真正挂载在线性层上，因此需要一个包装层，概念上类似：

- 保留冻结的 base linear；
- 每个 adapter 一组 `(A, B)` 低秩参数；
- forward 时：
  - 先走 base linear；
  - 若当前 active adapter 存在，再加上该 adapter 的低秩增量。

### 4.3 `AdapterTrainer`

trainer 负责：

- 接受 base model、oracle model、adapter manager；
- 接受一批任务；
- 为每个任务分配 adapter id；
- 在训练该任务时只更新当前 adapter；
- 训练完成后把 adapter state 存起来。

第一版 trainer 的关键词不是“真并行”，而是：

- shared-base
- multi-adapter
- no repeated base-model loading

## 5. 训练任务的数据抽象

一个 stepwise 任务最小需要描述：

- `adapter_id`
- `target`
- `step_idx`
- `dataset`
- `collator`
- `epochs`
- `lr`
- `loss_type`

后续如果要并行化，再往这个对象上加：

- bucket id
- adapter group
- scheduler state
- evaluation metadata

## 6. 训练策略分阶段

### 6.1 第一版：共享底座，多 adapter batch 并行

第一版的目标是：

- 一次创建很多个 adapter；
- 一个 batch 中不同样本可以走不同 adapter；
- trainer 可以把多个任务的数据合并成一个更新批次；
- base model 始终不重载。

优点：

- 已经切中主要结构问题；
- 已经能显著减少主模型加载和内存复制。

缺点：

- 仍未引入多进程异步；
- 仍需要对任务分组做一定约束，例如相同超参数、相容的 batch 组织方式。

### 6.2 第二版：多进程异步

可选方向：

- 一个主进程持有 base model；
- 多个工作器分别持有 adapter 状态；
- 异步调度训练任务。

这个方向工程复杂度较高，且对 GPU/模型访问模式要求比较严。

### 6.3 第三版：更强的调度与异构任务混合

当 sample-wise adapter routing 已经具备后，后续再往前走的重点会变成：

- 不同任务长度和步数的更智能分桶；
- 异构任务混合调度；
- adapter 级 optimizer / scheduler 设计；
- 更细粒度的并发执行策略。

## 7. 保存与加载策略

这一阶段的一个明确收益就是：

- 不再保存完整模型；
- 只保存 adapter state。

因此需要定义：

1. adapter 参数文件格式；
2. 任务 id 到 adapter id 的映射；
3. adapter 的元信息：
   - rank
   - alpha
   - dropout
   - target modules
   - 对应 target / step 信息

推理时流程应当是：

- 加载 base model；
- 创建/恢复 adapter manager；
- 加载指定 adapter；
- 激活该 adapter；
- 再做评估或生成。

## 8. 第一版实施顺序

按最稳的顺序，第一版分成四步：

1. 实现 `MultiAdapterLinear`
2. 实现 `LoRAAdapterManager`
3. 实现支持 sample-wise adapter routing 的 `AdapterTrainer`
4. 把现有 `v2/unlearn.py` 的 stepwise 任务流接入 batched adapter trainer

## 9. 第一版不做什么

为了避免目标失控，第一版明确不做：

- 不做多进程异步训练；
- 不改当前 loss 数学定义；
- 不同时处理 `non-stepwise` 语义修复。

## 10. 当前实现决策

当前 repo 没有现成的 `peft`/LoRA 训练集成痕迹，因此第一版优先采用：

- **自定义 lightweight LoRA 注入实现**

而不是先引入新的外部训练框架。

原因：

- 更容易控制与现有 `v2` 训练流的兼容性；
- 更容易按这个项目的 stepwise 任务组织方式改；
- 不必先处理新的依赖管理和框架适配问题。
