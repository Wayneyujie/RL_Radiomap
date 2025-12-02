#!/usr/bin/env python3
"""
使用YAML配置场景进行训练
支持从YAML文件加载场景配置，并进行真正的强化学习训练
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import numpy as np
import argparse
import json
from datetime import datetime
from marinenav_env.envs.marinenav_env import MarineNavEnv2
from policy.agent import Agent
from policy.trainer import Trainer

def create_training_config(yaml_path=None, num_robots=3, total_timesteps=100000, 
                          eval_freq=5000, use_iqn=True, device="cpu"):
    """创建训练配置"""
    
    # 基础训练配置
    config = {
        "seed": 0,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "use_iqn": use_iqn,
        "save_dir": "pretrained_models",
        "training_schedule": {
            "timesteps": [0, total_timesteps],
            "num_cooperative": [num_robots, num_robots],
            "num_non_cooperative": [0, 0],
            "num_cores": [0, 0],  # 不使用vortex
            "num_obstacles": [0, 0],  # 从YAML加载，不使用随机生成
            "min_start_goal_dis": [10.0, 10.0]
        },
        "eval_schedule": {
            "num_episodes": [10],
            "num_cooperative": [num_robots],
            "num_non_cooperative": [0],
            "num_cores": [0],
            "num_obstacles": [0],
            "min_start_goal_dis": [10.0]
        }
    }
    
    return config

def train_with_yaml(yaml_path, num_robots=3, total_timesteps=100000, 
                    eval_freq=5000, use_iqn=True, device="cpu", 
                    load_model=None, visualize=False):
    """
    使用YAML配置进行训练
    
    Args:
        yaml_path: YAML配置文件路径
        num_robots: 机器人数量
        total_timesteps: 总训练步数
        eval_freq: 评估频率
        use_iqn: 是否使用IQN
        device: 设备（cpu/cuda）
        load_model: 预训练模型路径（可选）
        visualize: 是否可视化（暂不支持）
    """
    
    print("=" * 60)
    print("多机器人强化学习训练 - YAML场景配置")
    print("=" * 60)
    print(f"YAML配置: {yaml_path}")
    print(f"机器人数量: {num_robots}")
    print(f"总训练步数: {total_timesteps}")
    print(f"评估频率: {eval_freq}")
    print(f"使用IQN: {use_iqn}")
    print(f"设备: {device}")
    print("=" * 60 + "\n")
    
    # 创建训练配置
    config = create_training_config(
        yaml_path=yaml_path,
        num_robots=num_robots,
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        use_iqn=use_iqn,
        device=device
    )
    
    # 创建实验目录
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    exp_dir = os.path.join(config["save_dir"], 
                          f"training_{timestamp}",
                          f"seed_{config['seed']}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(exp_dir, "training_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ 训练配置已保存: {config_file}\n")
    
    # 创建训练环境（使用YAML配置）
    print("初始化训练环境...")
    train_env = MarineNavEnv2(seed=config["seed"], 
                              schedule=config["training_schedule"],
                              yaml_config_path=yaml_path)
    train_env.num_cooperative = num_robots
    train_env.num_non_cooperative = 0
    train_env.num_cores = 0  # 不使用vortex
    train_env.num_obs = 0    # 从YAML加载障碍物
    
    # 创建评估环境
    print("初始化评估环境...")
    eval_env = MarineNavEnv2(seed=253, 
                            yaml_config_path=yaml_path)
    eval_env.num_cooperative = num_robots
    eval_env.num_non_cooperative = 0
    eval_env.num_cores = 0
    eval_env.num_obs = 0
    
    # 创建智能体
    print("初始化智能体...")
    cooperative_agent = Agent(
        cooperative=True,
        device=device,
        use_iqn=use_iqn,
        seed=config["seed"] + 100,
        training=True
    )
    
    # 加载预训练模型（如果提供）
    if load_model and os.path.exists(load_model):
        print(f"加载预训练模型: {load_model}")
        cooperative_agent.load_model(load_model, "cooperative", device)
    
    # 创建训练器
    print("创建训练器...\n")
    trainer = Trainer(
        train_env=train_env,
        eval_env=eval_env,
        eval_schedule=config["eval_schedule"],
        cooperative_agent=cooperative_agent,
        non_cooperative_agent=None,
    )
    
    # 保存评估配置
    trainer.save_eval_config(exp_dir)
    
    # 开始训练
    print("=" * 60)
    print("开始训练...")
    print("=" * 60 + "\n")
    
    trainer.learn(
        total_timesteps=config["total_timesteps"],
        eval_freq=config["eval_freq"],
        eval_log_path=exp_dir
    )
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"模型保存在: {exp_dir}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RL agents with YAML scene configuration')
    
    parser.add_argument('--yaml', type=str, required=True,
                       help='YAML配置文件路径')
    parser.add_argument('--num-robots', type=int, default=3,
                       help='机器人数量 (default: 3)')
    parser.add_argument('--total-timesteps', type=int, default=100000,
                       help='总训练步数 (default: 100000)')
    parser.add_argument('--eval-freq', type=int, default=5000,
                       help='评估频率 (default: 5000)')
    parser.add_argument('--use-iqn', action='store_true', default=True,
                       help='使用IQN (default: True)')
    parser.add_argument('--use-dqn', action='store_true', default=False,
                       help='使用DQN而不是IQN')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='设备 (default: cpu)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='预训练模型路径（可选）')
    
    args = parser.parse_args()
    
    # 处理IQN/DQN选择
    use_iqn = args.use_iqn and not args.use_dqn
    
    # 检查YAML文件是否存在
    if not os.path.exists(args.yaml):
        print(f"错误: YAML文件不存在: {args.yaml}")
        sys.exit(1)
    
    # 开始训练
    train_with_yaml(
        yaml_path=args.yaml,
        num_robots=args.num_robots,
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        use_iqn=use_iqn,
        device=args.device,
        load_model=args.load_model
    )

