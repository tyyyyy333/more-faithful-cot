# LoRA Adapter 并行化执行报告

这份报告对应 `03_lora_adapter_plan.md`，记录当前 LoRA 多 adapter 主线已经完成的代码能力，以及仍然存在的边界。

## 1. 当前已完成的能力

### 1.1 `LoRAAdapterManager` 与 `MultiAdapterLinear`

文件：

- `v2/lora_adapter.py`

当前已实现：

- 传入一个 base model，返回一个管理多组 LoRA 参数的对象；
- 自动查找目标线性层并注入 `MultiAdapterLinear` wrapper；
- 每个 adapter 独立维护 `(A, B)` 低秩参数；
- 支持：
  - 创建 adapter；
  - 激活单个 adapter；
  - 激活 batch 内按 sample 路由的 adapter；
  - 保存 adapter；
  - 加载 adapter。

关键点：

- 同一个 base model 上可以同时维护很多组 adapter；
- sample-wise adapter routing 已经落代码，不是顺序切换 adapter 冒充并行。

### 1.2 `AdapterTrainer`

文件：

- `v2/adapter_trainer.py`

当前已实现：

- `AdapterTrainingJob` 数据结构；
- trainer 接受：
  - base model
  - oracle model
  - adapter manager
  - 多个 adapter jobs
- 支持两类训练方式：
  - `sequential`
  - `batched`
- `batched` 模式下：
  - 多个 job 的 batch 会被合并；
  - batch 内不同 sample 会路由到不同 adapter；
  - 多个 adapter 参数在同一个优化器里一起更新。

此外还已实现：

- merged batch 的 pad/concat；
- epoch 级 callback；
- scheduler builder 接口。

### 1.3 `Sk` jobs 主流程接入

文件：

- `v2/unlearn.py`

当前已实现：

- 当前主入口不再走“每个 step 一个完整模型微调”的旧路径；
- 现在会先展开：
  - `S` 个样本
  - 每个样本 `k_i` 个 step
  - 总共形成 `Sk` 个显式 adapter jobs
- 每个 job 都会构造：
  - `adapter_id`
  - `dataset`
  - `collator`
  - `epochs`
  - `lr`
  - `loss_type`
  - metadata
- 这些 jobs 会按长度排序后切分成 group；
- 每个 group 交给 `AdapterTrainer(..., mode="batched")` 训练。

关键点：

- base model 在整轮实验中只加载一次；
- oracle model 在整轮实验中只加载一次；
- 不再为每个 `(instance, step)` 反复加载完整可训练主模型。

### 1.4 adapter 保存与记录链路

文件：

- `v2/unlearn.py`
- `v2/adapter_runtime.py`

当前已实现：

- 每个 job 训练完成后，会保存：
  - adapter 参数文件：`v2_outputs/adapters/.../*.pt`
  - adapter 记录文件：`v2_outputs/adapter_records/.../*.json`
- 记录中包含：
  - `base_model_id`
  - `adapter_id`
  - `adapter_path`
  - metadata

也就是说，训练产物已经从：

- 完整主模型

切换成：

- base model + adapter record + adapter 参数

### 1.5 adapter 恢复与评估链路

文件：

- `v2/adapter_runtime.py`

当前已实现：

- `AdapterRecord`
- `load_adapter_record(...)`
- `AdapterRuntime.from_base_model(...)`
- `AdapterRuntime.from_adapter_record(...)`
- `AdapterRuntime.load_adapter(...)`
- `AdapterRuntime.activate(...)`
- `AdapterRuntime.evaluate(...)`
- `evaluate_record(...)`
- `load_runtime_for_records(...)`

这意味着后续调用时可以做到：

- 加载 base model；
- 根据 record 加载指定 adapter；
- 激活该 adapter；
- 直接跑现有评估函数。

## 2. 已完成的最小验证

### 2.1 静态检查

已通过：

```bash
python3 -m py_compile \
  parametric-faithfulness/v2/lora_adapter.py \
  parametric-faithfulness/v2/adapter_trainer.py \
  parametric-faithfulness/v2/adapter_runtime.py \
  parametric-faithfulness/v2/unlearn.py
```

### 2.2 结构烟测

已通过两个小型 smoke 测试：

1. `LoRAAdapterManager`
   - 多 adapter 挂到同一个 toy base model；
   - sample-wise adapter routing 生效；
   - save/load 后输出一致。

2. `AdapterTrainer`
   - 两个 toy jobs 在 `mode="batched"` 下可以合并训练；
   - trainer 可以驱动多 adapter batch 更新；
   - 训练 history 正常返回。

## 3. 当前边界

这条主线已经完成了“架构闭环”，但仍有几个边界需要明确：

- 当前选择的是 **batch 并行**，没有实现多进程异步；
- `mmlu/gsm` 依赖的外部 `lm_eval` 全模型导出路径，在这套 adapter-only 流程下还没有重接；
- `unlearn_single()` 和旧单模型训练函数仍然保留在文件里，但主入口 `main()` 已经切到 adapter jobs 路径；
- 当前 group 切分主要是：
  - 相同全局训练超参数
  - 相近长度
  - 固定 `adapter_group_size`
  还没有更细粒度的异构任务调度。

## 4. 当前结论

如果按这轮规划来衡量，当前代码已经具备：

1. base model 只加载一次；
2. `Sk` 个 stepwise 任务显式展开；
3. 多 adapter 管理；
4. batch 内 sample-wise adapter routing；
5. batched adapter trainer；
6. adapter-only 存储；
7. record-based 恢复与评估。

也就是说，核心闭环已经落代码，不再停留在骨架层。
