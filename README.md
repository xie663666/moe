<<<<<<< HEAD
# MoE 相似样本专家迁移实验项目

这是一个从零整理的新项目，用来研究下面这个问题：

> 对目标样本集 B，固定使用一部分来自相似样本集 A 的常用专家，并让剩余专家继续通过 router 动态选择时，准确率和运行时间会发生什么变化？

项目完整实现了你指定的流程：

1. 对每个相似对 `(A, B)`、每组 `(E, K)`：
   - 在 A 上正常训练 source MoE，得到 `source_ckpt`
   - 用 A 提取平均稀疏门控分布 `avg_gate_A`
2. 对 `m in [0, 1, ..., K]`：
   - 从同一个 `source_ckpt` 初始化 B 模型
   - 构造 `fixed_profile(m)`
   - 冻结固定专家
   - 在 B 上训练 / 评估
   - 记录准确率、总 step 时间、router 时间、expert 时间、router 占比
3. 再反向做一次 `B -> A`

---

## 1. 数据集与实验对

项目使用 CIFAR-100 的六对相似 superclass：

- `vehicles_1 ↔ vehicles_2`
- `aquatic_mammals ↔ fish`
- `flowers ↔ trees`
- `insects ↔ non_insect_invertebrates`
- `small_mammals ↔ medium_sized_mammals`
- `large_carnivores ↔ large_omnivores_and_herbivores`

每个 superclass 视为一个 **5 类分类任务**。
也就是说：

- A 上训练的是 A 内部 5 个细类的分类器
- B 上训练的是 B 内部 5 个细类的分类器
- 从 source checkpoint 迁移时，只加载 `backbone + router + experts`，**不加载分类头**

这样就避免了 A/B 标签空间不同带来的 head 冲突。

---

## 2. 模型设计

- **Backbone**：轻量 CNN，输出 `hidden_dim` 维特征
- **Router**：线性层 `Linear(hidden_dim, num_experts)`
- **Experts**：若干个 MLP expert
- **MoE 路由**：top-k sparse routing

### 固定专家 + 动态专家混合路由

对目标任务 B：

- 固定专家个数为 `m`
- 总 top-k 为 `K`
- 固定专家从 `avg_gate_A` 中取最常用的前 `m` 个
- 剩余 `K-m` 个专家仍由 B 的 router 动态选择

默认固定概率质量采用：

```text
fixed_mass = m / K
```

这样：

- `m = 0`：完全普通 MoE，对照实验
- `m = K`：完全固定专家，不再运行 router

项目默认实现：

- 固定专家内部按 `avg_gate_A` 的相对大小分配固定权重
- 动态部分只在剩余专家上 softmax + top-k
- 最终概率和严格等于 1

---

## 3. 时间统计

项目会统计并输出：

- `step_ms`：完整 batch 训练 / 评估时间
- `router_ms`：router 阶段时间
- `dispatch_ms`：将样本分发到各 expert 的时间
- `expert_forward_ms`：expert 前向计算时间
- `combine_ms`：expert 输出加权合并与最终归一化时间
- `expert_ms`：expert 阶段总时间，等于 `dispatch_ms + expert_forward_ms + combine_ms`
- `moe_ms`：MoE 整体前向时间
- `router_ratio = router_ms / step_ms`
- `expert_ratio = expert_ms / step_ms`

GPU 计时使用了 `torch.cuda.synchronize()`，避免异步误差。

---

## 4. 项目结构

```text
moe_transfer_project/
├── README.md
├── requirements.txt
├── run_pair_experiments.py
└── src/
    ├── __init__.py
    ├── cifar100_groups.py
    ├── datasets.py
    ├── model.py
    ├── routing_profile.py
    └── train_utils.py
```

---

## 5. 安装依赖

```bash
pip install -r requirements.txt
```

如果你已经有 PyTorch 环境，只需要额外确保：

```bash
pip install torchvision tqdm
```

---

## 6. 运行方法

### 6.1 跑全部六对、双向、多个 E/K

```bash
python run_pair_experiments.py \
  --data_root ./data \
  --output_dir ./outputs \
  --pairs all \
  --num_experts_list 8,16 \
  --top_k_list 2,4 \
  --source_epochs 30 \
  --target_epochs 20 \
  --batch_size 128 \
  --eval_batch_size 256 \
  --freeze_fixed_experts \
  --timing_interval 10
```

### 6.2 只跑一对

```bash
python run_pair_experiments.py \
  --pairs aquatic_mammals->fish \
  --num_experts_list 8 \
  --top_k_list 2 \
  --source_epochs 20 \
  --target_epochs 10 \
  --freeze_fixed_experts
```

### 6.3 复用已经训练好的 source checkpoint

如果之前已经跑过 source 训练并保存了 `best.pt` 和 `avg_gate.pt`，可以加：

```bash
--resume_source
```

---

## 7. 主要输出

结果会保存到：

```text
outputs/
├── results.csv
└── <pair_name>/
    └── <direction>/
        └── E{E}_K{K}/
            ├── source_train/
            │   ├── best.pt
            │   ├── last.pt
            │   └── epoch_*.json
            ├── profiles/
            │   └── avg_gate.pt
            └── transfer_runs/
                ├── fixed_0/
                ├── fixed_1/
                ├── ...
                └── fixed_K/
```

其中：

- `results.csv`：汇总表，适合直接做统计和画图
- `summary.json`：每个 transfer run 的单独摘要
- `avg_gate.pt`：从 source 数据集提取的平均门控分布

---

## 8. CSV 字段说明

`results.csv` 中包含：

- 基本实验信息：
  - `pair_name`
  - `direction`
  - `source_group`
  - `target_group`
  - `num_experts`
  - `top_k`
  - `num_fixed`
- 准确率：
  - `best_val_acc`
  - `best_test_acc`
- 时间统计：
  - `train_step_ms`
  - `train_router_ms`
  - `train_dispatch_ms`
  - `train_expert_forward_ms`
  - `train_combine_ms`
  - `train_expert_ms`
  - `train_router_ratio`
  - `val_step_ms`
  - `val_router_ms`
  - `val_dispatch_ms`
  - `val_expert_forward_ms`
  - `val_combine_ms`
  - `val_expert_ms`
  - `val_router_ratio`
  - `test_step_ms`
  - `test_router_ms`
  - `test_dispatch_ms`
  - `test_expert_forward_ms`
  - `test_combine_ms`
  - `test_expert_ms`
  - `test_router_ratio`

---

## 9. 默认实现细节

### 9.1 固定专家冻结策略

默认加上 `--freeze_fixed_experts` 后：

- 固定专家参数不更新
- 非固定专家继续更新
- backbone / head / 动态路由相关部分继续更新

### 9.2 load balance loss

项目提供了可选的轻量负载均衡项：

```bash
--load_balance_coef 0.01
```

在混合固定路由模式下，它只对 **动态可选专家集合** 计算，不会错误约束固定专家。

### 9.3 source / target 学习率分开设置

- `--source_lr`
- `--target_lr`

因为 source 正常训练和 target 迁移训练的稳定性需求可能不同。

---

## 10. 最建议的首轮实验

建议先做一个最小可运行版本验证逻辑：

```bash
python run_pair_experiments.py \
  --pairs aquatic_mammals->fish \
  --num_experts_list 8 \
  --top_k_list 2 \
  --source_epochs 10 \
  --target_epochs 10 \
  --freeze_fixed_experts
```

先确认三件事：

1. `fixed_0 / fixed_1 / fixed_2` 都能正常跑通
2. `fixed_2 (=K)` 时 `router_ms` 明显下降
3. 准确率没有完全崩掉

然后再扩展到六对、双向、多组 E/K。

---

## 11. 可继续扩展的方向

这个项目现在是一个结构清晰、可直接跑的基线版本。你后续可以继续往下扩展：

1. 换更强 backbone（ResNet18 / ResNet34）
2. 让 `avg_gate_A` 改成按类别提取，而不是整组提取
3. 比较“固定概率质量 = m/K”和“直接使用 raw profile 质量”两种策略
4. 再增加 `fixed expert weight` 是否可学习的对照实验
5. 输出更多路由分析指标，例如：
   - 平均熵
   - expert usage histogram
   - dominant expert consistency

---

## 12. 注意事项

1. 这个项目是**新建的独立实验工程**，不依赖你之前杂乱的旧项目。
2. A/B 是两个不同的 5 类分类任务，因此迁移时只加载 trunk，不加载 head。
3. 如果你想完全对齐你过去那套实验风格，可以再把 backbone 替换成你熟悉的结构。
4. 该项目优先保证实验逻辑清晰、可复现、可扩展，而不是追求最大精度。
=======

