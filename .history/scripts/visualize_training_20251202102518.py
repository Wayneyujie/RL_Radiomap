#!/usr/bin/env python3
"""
实时训练可视化脚本
显示机器人轨迹、障碍物、起点终点和训练进度
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from marinenav_env.envs.marinenav_env import MarineNavEnv2
from policy.agent import Agent
import time

class TrainingVisualizer:
    def __init__(self, env, agent=None, update_interval=10, save_animation=False, output_dir="training_visualization"):
        """
        Args:
            env: 训练环境
            agent: 训练中的智能体（可选）
            update_interval: 每N步更新一次可视化
            save_animation: 是否保存动画
            output_dir: 输出目录
        """
        self.env = env
        self.agent = agent
        self.update_interval = update_interval
        self.save_animation = save_animation
        self.output_dir = output_dir
        
        if save_animation:
            os.makedirs(output_dir, exist_ok=True)
        
        # 创建图形
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.ax.set_xlim(0, env.width)
        self.ax.set_ylim(0, env.height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)', fontsize=12)
        self.ax.set_ylabel('Y (m)', fontsize=12)
        self.ax.set_title('Multi-Robot Training Visualization', fontsize=14, fontweight='bold')
        
        # 存储轨迹
        self.trajectories = [[] for _ in range(len(env.robots))]
        self.episode_count = 0
        self.step_count = 0
        
        # 颜色列表
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(env.robots)))
        
        # 绘制静态元素
        self.draw_static_elements()
        
        plt.tight_layout()
    
    def draw_static_elements(self):
        """绘制静态障碍物、起点和终点"""
        # 清除之前的静态元素（除了障碍物）
        for patch in self.ax.patches:
            if not isinstance(patch, patches.Rectangle) or patch.get_facecolor()[0] != 0.5:  # 不是障碍物
                patch.remove()
        
        # 绘制障碍物
        for obs in self.env.obstacles:
            circle = patches.Circle((obs.x, obs.y), obs.r, 
                                  color='gray', alpha=0.5, zorder=1)
            self.ax.add_patch(circle)
        
        # 绘制机器人起点和终点
        for i, rob in enumerate(self.env.robots):
            # 起点
            start_circle = patches.Circle((rob.start[0], rob.start[1]), 0.5,
                                        color=self.colors[i], alpha=0.6, zorder=5)
            self.ax.add_patch(start_circle)
            self.ax.text(rob.start[0], rob.start[1], 'S', 
                         ha='center', va='center', fontsize=10, fontweight='bold', zorder=6)
            
            # 终点
            goal_circle = patches.Circle((rob.goal[0], rob.goal[1]), 0.5,
                                        color=self.colors[i], alpha=0.6, zorder=5)
            self.ax.add_patch(goal_circle)
            self.ax.text(rob.goal[0], rob.goal[1], 'G', 
                         ha='center', va='center', fontsize=10, fontweight='bold', zorder=6)
    
    def update(self, states, actions, rewards, dones, infos):
        """更新可视化"""
        self.step_count += 1
        
        if self.step_count % self.update_interval != 0:
            return
        
        # 清除之前的机器人位置和轨迹
        for line in self.ax.lines:
            line.remove()
        for patch in list(self.ax.patches):
            if isinstance(patch, patches.Circle) and patch.get_radius() < 0.6:  # 机器人标记
                patch.remove()
        
        # 更新轨迹并绘制
        for i, rob in enumerate(self.env.robots):
            if not rob.deactivated:
                # 添加当前位置到轨迹
                self.trajectories[i].append([rob.x, rob.y])
                
                # 绘制轨迹
                if len(self.trajectories[i]) > 1:
                    traj = np.array(self.trajectories[i])
                    self.ax.plot(traj[:, 0], traj[:, 1], 
                               color=self.colors[i], alpha=0.6, linewidth=2, 
                               label=f'Robot {i+1}' if i == 0 else '', zorder=3)
                
                # 绘制当前机器人位置
                robot_circle = patches.Circle((rob.x, rob.y), 0.3,
                                            color=self.colors[i], alpha=0.8, zorder=4)
                self.ax.add_patch(robot_circle)
                
                # 绘制机器人方向
                dx = 0.5 * np.cos(rob.theta)
                dy = 0.5 * np.sin(rob.theta)
                self.ax.arrow(rob.x, rob.y, dx, dy,
                            head_width=0.2, head_length=0.15,
                            fc=self.colors[i], ec=self.colors[i], zorder=4)
        
        # 更新标题显示训练进度
        self.ax.set_title(f'Multi-Robot Training - Episode: {self.episode_count}, Step: {self.step_count}', 
                         fontsize=14, fontweight='bold')
        
        # 添加图例（只在第一次）
        if self.step_count == self.update_interval:
            self.ax.legend(loc='upper right')
        
        plt.draw()
        plt.pause(0.01)
    
    def reset_episode(self):
        """重置episode，清除轨迹"""
        self.episode_count += 1
        self.trajectories = [[] for _ in range(len(self.env.robots))]
        self.step_count = 0
        self.draw_static_elements()
    
    def save_frame(self, frame_num):
        """保存当前帧"""
        if self.save_animation:
            filename = os.path.join(self.output_dir, f"frame_{frame_num:06d}.png")
            self.fig.savefig(filename, dpi=100, bbox_inches='tight')


def visualize_training_episode(env, agent=None, max_steps=1000, use_rl=True, use_iqn=True):
    """
    可视化单个训练episode
    
    Args:
        env: 环境
        agent: 智能体（可选，如果为None则使用随机动作）
        max_steps: 最大步数
        use_rl: 是否使用RL
        use_iqn: 是否使用IQN
    """
    visualizer = TrainingVisualizer(env, agent, update_interval=5, save_animation=False)
    
    states, _, _ = env.reset()
    visualizer.reset_episode()
    
    for step in range(max_steps):
        # 获取动作
        actions = []
        for i, rob in enumerate(env.robots):
            if rob.deactivated:
                actions.append(None)
                continue
            
            if agent is not None and use_rl:
                if use_iqn:
                    a, _, _ = agent.act(states[i], eps=0.0)
                else:
                    a, _ = agent.act_dqn(states[i], eps=0.0)
            else:
                # 如果没有agent，使用随机动作
                import random
                a = random.randint(0, 8)  # 9个离散动作
            actions.append(a)
        
        # 执行动作
        next_states, rewards, dones, infos = env.step(actions)
        
        # 更新可视化
        visualizer.update(states, actions, rewards, dones, infos)
        
        # 检查是否结束
        if env.check_all_deactivated() or step >= max_steps - 1:
            break
        
        states = next_states
    
    plt.show()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training process')
    parser.add_argument('--yaml', type=str, default=None, help='YAML config file path')
    parser.add_argument('--model', type=str, default=None, help='Pretrained model path')
    parser.add_argument('--num-robots', type=int, default=3, help='Number of robots')
    parser.add_argument('--max-steps', type=int, default=500, help='Maximum steps per episode')
    
    args = parser.parse_args()
    
    # 创建环境
    if args.yaml:
        env = marinenav_env.MarineNavEnv2(seed=0, yaml_config_path=args.yaml)
        env.num_cooperative = args.num_robots
        env.num_non_cooperative = 0
    else:
        env = marinenav_env.MarineNavEnv2(seed=0)
        env.num_cooperative = args.num_robots
        env.num_non_cooperative = 0
    
    # 创建智能体
    agent = Agent(cooperative=True, device="cpu", use_iqn=True, training=False)
    if args.model:
        agent.load_model(args.model, "cooperative", "cpu")
    
    # 可视化
    visualize_training_episode(env, agent, max_steps=args.max_steps)

