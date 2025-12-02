# 常见问题排查

## 问题1: YAML定义了多个机器人，但只看到部分机器人在移动

### 原因
当使用 `--num-robots` 参数时，代码会限制只使用指定数量的机器人。即使YAML文件中有更多机器人，也只会使用前N个。

### 解决方案
**方案1（推荐）**: 不指定 `--num-robots` 参数，让代码自动使用YAML中的所有机器人：
```bash
python scripts/visualize_training.py \
    --yaml mpcom_ral_1220/map_editor_output.yaml \
    --model pretrained_models/training_2025-12-02-11-15-12/seed_0
```

**方案2**: 指定正确的机器人数量：
```bash
python scripts/visualize_training.py \
    --yaml mpcom_ral_1220/map_editor_output.yaml \
    --num-robots 3 \
    --model pretrained_models/training_2025-12-02-11-15-12/seed_0
```

### 修复说明
代码已更新，现在当YAML中的机器人数量大于 `--num-robots` 时，会自动使用YAML中的所有机器人，并显示提示信息。

---

## 问题2: 为什么seed 0和seed 1都能看到可视化？

### 原因
1. **可视化脚本中seed被硬编码**: 之前的代码中，无论你指定什么seed，可视化脚本都使用 `seed=0`。
2. **YAML配置优先**: 当使用YAML配置时，机器人的位置和障碍物位置都是从YAML文件读取的，不是随机生成的，所以seed的影响很小。
3. **Seed的作用范围**: Seed主要用于：
   - 随机生成额外的机器人（如果YAML中的机器人数量不足）
   - 随机生成障碍物（如果YAML中没有障碍物）
   - 其他随机生成的内容（如vortex等）

### 解决方案
现在代码已更新，支持 `--seed` 参数：
```bash
python scripts/visualize_training.py \
    --yaml mpcom_ral_1220/map_editor_output.yaml \
    --model pretrained_models/training_2025-12-02-11-15-12/seed_0 \
    --seed 1
```

**注意**: 如果YAML文件完整（包含所有机器人和障碍物），seed对可视化结果的影响很小，因为所有位置都是从YAML读取的。

---

## 如何检查YAML中有多少个机器人？

```bash
python -c "import yaml; f=open('mpcom_ral_1220/map_editor_output.yaml'); d=yaml.safe_load(f); print(f'YAML中有 {len(d.get(\"robots\", []))} 个机器人')"
```

---

## 其他常见问题

### "too long episode" 是什么意思？
这表示一个episode执行了太长时间（超过了最大步数限制）。可能的原因：
- 机器人无法找到到达目标的路径
- 环境太复杂，需要更多步数
- 策略不够好，机器人绕圈或卡住

解决方法：
- 增加 `--max-steps` 参数
- 检查起点和终点是否合理
- 检查障碍物配置是否合理
- 训练更长时间以获得更好的策略

### 如何减少训练次数？
修改 `train_with_yaml.py` 中的 `--total-timesteps` 参数：
```bash
python train_with_yaml.py \
    --yaml mpcom_ral_1220/map_editor_output.yaml \
    --num-robots 3 \
    --total-timesteps 50000  # 减少这个值
```

