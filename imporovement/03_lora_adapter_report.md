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
  - reference/oracle handle
  - adapter manager
  - 多个 adapter jobs
- 支持两类训练方式：
  - `sequential`
  - `batched`
- `batched` 模式下：
  - 多个 job 的 batch 会被合并；
  - batch 内不同 sample 会路由到不同 adapter；
  - 多个 adapter 参数在同一个优化器里一起更新。
- `train_jobs(..., mode="batched")` 会先按训练签名自动分组，再分别做 batched 训练；
- 因此调用端可以直接交任意数量的 jobs，而不必先自己手工分好完全同构的小批。

此外还已实现：

- merged batch 的 pad/concat；
- epoch 级 callback；
- scheduler builder 接口。

### 1.3 `Sk` jobs 主流程接入

文件：

- `v2/unlearn.py`

当前已实现：

- 当前主入口不再走“每个 step 一个完整模型微调”的旧路径；
- 当前这条 LoRA 多 adapter 主线明确只支持 `stepwise`；
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
- CoT 预生成路径会复用同一份 base model，不再额外起一份生成模型；
- reference/oracle 前向不再单独加载第二份大模型，而是复用同一个 base model，在 oracle 前向时临时停用 adapter；
- 不再为每个 `(instance, step)` 反复加载完整可训练主模型。

### 1.4 adapter 保存与记录链路

文件：

- `v2/unlearn.py`
- `v2/adapter_runtime.py`

当前已实现：

- 每个 job 训练完成后，会保存：
  - adapter 参数文件：`v2_outputs/adapters/.../*.pt`
  - adapter 记录文件：`v2_outputs/adapter_records/.../*.json`
  - run 级索引文件：`v2_outputs/adapter_records/.../run_manifest.json`
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

此外当前还支持：

- 从 `run_manifest.json` 一次性恢复同一轮实验下的多组 adapter；
- record 中的 adapter 路径按 record 所在目录解析，避免恢复时依赖调用目录。

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

3. batched trainer 自动分组
   - 不同训练签名的 toy jobs 可以直接一起交给 `train_jobs(..., mode="batched")`；
   - trainer 会按签名自动拆组后训练；
   - 调用端不再需要手工先分组。

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
7. record-based 恢复与评估；
8. run-manifest-based 批量恢复。

也就是说，核心闭环已经落代码，不再停留在骨架层。

## 5. 总更新日志

这一轮围绕 `stepwise` 的 LoRA 多 adapter 主线，累计完成了以下改造：

1. 新增 `v2/lora_adapter.py`
   - 实现 `LoRAConfig`、`MultiAdapterLinear`、`LoRAAdapterManager`。
   - 支持一个 base model 上同时维护多组 adapter。
   - 支持单 adapter 激活和 batch 内 sample-wise adapter routing。
   - 支持 adapter state 的保存与加载。

2. 新增 `v2/adapter_trainer.py`
   - 引入 `AdapterTrainingJob` 抽象。
   - 实现 `AdapterTrainer`，支持 `sequential` 与 `batched` 两种训练模式。
   - `batched` 模式下可合并多 job batch，并按 sample 路由不同 adapter。
   - 进一步支持按训练签名自动分组，调用端不需要手工先拆批。

3. 重写 `v2/unlearn.py` 主入口
   - LoRA 主线明确限定为 `stepwise`。
   - 将 `S` 个样本、每个样本的 `k_i` 个 step 展开成 `Sk` 个显式 adapter jobs。
   - 主模型只加载一次；CoT 预生成复用同一份 base model。
   - reference/oracle 前向不再依赖第二份大模型，而是通过临时停用 adapter 取得基座分布。
   - 训练结束后只保存 adapter 参数和对应记录。

4. 完善 adapter 恢复链路
   - 新增/补全 `v2/adapter_runtime.py`。
   - 支持从 per-adapter JSON record 恢复 `base model + adapter`。
   - 支持从 `run_manifest.json` 批量恢复同一轮实验的多组 adapter。
   - 统一了 record / manifest 中相对路径的保存与解析方式。

5. 打通运行时评估接口
   - `AdapterRuntime` 现在可直接调用：
     - `evaluate`
     - `answer_probabilities`
     - `letter_completion`
     - `generate`
     - `generate_cot`
     - `cot_generate`
     - `generation_fixed_cot`
     - `completion_probabilities`

6. 同步更新数据与评估辅助路径
   - `v2/data.py` 和 `v2/evaluate.py` 支持在需要生成 CoT 时复用外部传入的 base model。
   - `v2/models.py` 在 tokenizer 缺少 `pad_token` 时自动补齐，保证恢复后的生成路径一致。

7. 文档同步
   - 更新 `03_lora_adapter_plan.md`
   - 更新 `03_lora_adapter_report.md`
   - 让计划、实现边界、恢复方式、验证结果与当前代码保持一致。
