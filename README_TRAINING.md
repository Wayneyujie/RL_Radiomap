# 训练使用说明

## 问题说明

`scripts/visualize_training.py` 只是用于**可视化单个episode**，不会进行真正的训练。它可能会"卡住"是因为：
1. 它只是运行一个episode并显示可视化
2. 没有训练循环，不会更新神经网络参数
3. 主要用于调试和查看环境

## 真正的训练

使用 `train_with_yaml.py` 进行**真正的强化学习训练**：

### 基本用法

```bash
python train_with_yaml.py --yaml mpcom_ral_1220/map_editor_output.yaml --num-robots 2 --total-timesteps 100000
```

### 参数说明

- `--yaml`: YAML配置文件路径（必需）
- `--num-robots`: 机器人数量（默认: 3）
- `--total-timesteps`: 总训练步数（默认: 100000）
- `--eval-freq`: 评估频率，每N步评估一次（默认: 5000）
- `--use-iqn`: 使用IQN算法（默认）
- `--use-dqn`: 使用DQN算法（与--use-iqn互斥）
- `--device`: 设备，cpu或cuda（默认: cpu）
- `--load-model`: 预训练模型路径（可选）

### 训练示例

#### 快速测试（小规模训练）
```bash
python train_with_yaml.py \
    --yaml mpcom_ral_1220/map_editor_output.yaml \
    --num-robots 2 \
    --total-timesteps 10000 \
    --eval-freq 2000
```

#### 完整训练
```bash
python train_with_yaml.py \
    --yaml mpcom_ral_1220/map_editor_output.yaml \
    --num-robots 3 \
    --total-timesteps 1000000 \
    --eval-freq 50000 \
    --device cpu
```

#### 从预训练模型继续训练
```bash
python train_with_yaml.py \
    --yaml mpcom_ral_1220/map_editor_output.yaml \
    --num-robots 3 \
    --total-timesteps 1000000 \
    --load-model pretrained_models/IQN/seed_9
```

## 训练输出

训练过程中会：
1. 显示训练进度和episode信息
2. 定期进行评估并保存结果
3. 保存模型到 `pretrained_models/training_YYYY-MM-DD-HH-MM-SS/seed_X/`
4. 保存评估数据到 `evaluations.npz`

## 可视化训练过程

如果需要可视化训练过程，可以：
1. 先进行训练保存模型
2. 然后使用 `visualize_training.py` 加载模型进行可视化

```bash
# 训练
python train_with_yaml.py --yaml mpcom_ral_1220/map_editor_output.yaml --num-robots 2 --total-timesteps 50000

# 可视化（使用训练好的模型）
python scripts/visualize_training.py \
    --yaml mpcom_ral_1220/map_editor_output.yaml \
    --num-robots 2 \
    --model pretrained_models/training_XXX/seed_0
```

## 注意事项

1. **训练需要时间**：真正的训练需要大量时间步数才能看到效果
2. **内存使用**：训练会使用replay buffer，注意内存使用
3. **设备选择**：如果有GPU，使用 `--device cuda` 可以加速训练
4. **YAML配置**：确保YAML文件中的障碍物和机器人配置正确

