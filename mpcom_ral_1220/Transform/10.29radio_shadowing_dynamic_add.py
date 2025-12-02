"""
你的原始代码修改版本
展示如何在 query_radio() 函数中添加动态 Cube

主要修改：
1. 在 scene.add(tx) 之后添加立方体
2. 添加了 add_dynamic_obstacles() 辅助函数
3. 可视化时标记立方体位置
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
import random
from sionna.rt import load_scene, Transmitter, PlanarArray, Camera
import os
import datetime

gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def add_dynamic_obstacles(scene, obstacles_config=None):
    """
    向场景中添加动态障碍物（立方体）
    
    Parameters:
    -----------
    scene : Sionna Scene object
        已加载的场景对象
    obstacles_config : list of dict, optional
        障碍物配置列表，每个字典包含：
        - name: 障碍物名称
        - position: [x, y, z] 位置
        - size: [width, depth, height] 尺寸
        - material: 材质名称
        - orientation: [roll, pitch, yaw] 旋转（可选）
    
    Returns:
    --------
    int : 成功添加的障碍物数量
    """
    
    if obstacles_config is None:
        # 默认配置：添加一些示例障碍物
        obstacles_config = [
            {
                "name": "wall1",
                "position": [5.0, 5.0, 2.0],
                "size": [0.3, 8.0, 4.0],  # 薄墙
                "material": "itu_concrete",
                "orientation": [0, 0, 0]
            },
            {
                "name": "cabinet",
                "position": [10.0, -5.0, 1.0],
                "size": [2.0, 1.0, 2.0],
                "material": "itu_metal",
                "orientation": [0, 0, 0]
            },
            {
                "name": "pillar",
                "position": [-8.0, 8.0, 2.0],
                "size": [1.0, 1.0, 4.0],
                "material": "itu_concrete",
                "orientation": [0, 0, 0]
            }
        ]
    
    added_count = 0
    
    try:
        from sionna.rt import Box
        
        print("\n" + "="*60)
        print("添加动态障碍物到场景")
        print("="*60)
        
        for config in obstacles_config:
            try:
                # 创建立方体
                cube = Box(
                    name=config["name"],
                    position=config["position"],
                    size=config["size"],
                    material=config.get("material", "itu_concrete")
                )
                
                # 如果指定了旋转角度，设置旋转
                if "orientation" in config:
                    cube.orientation = config["orientation"]
                
                # 添加到场景
                scene.add(cube)
                added_count += 1
                
                print(f"✓ {config['name']:15s} | "
                      f"位置: {config['position']} | "
                      f"尺寸: {config['size']} | "
                      f"材质: {config.get('material', 'itu_concrete')}")
                
            except Exception as e:
                print(f"✗ 无法添加 {config['name']}: {e}")
        
        print("="*60)
        print(f"总计成功添加 {added_count} 个障碍物\n")
        
    except ImportError:
        print("\n⚠ 警告: Box 类不可用")
        print("请升级 Sionna 版本: pip install --upgrade sionna")
        print("或检查 Sionna 版本: pip show sionna\n")
        
    return added_count


def query_radio_with_obstacles(index, color='red', recurrence=False, add_obstacles=True):
    """
    修改后的 query_radio 函数，支持添加动态障碍物
    
    Parameters:
    -----------
    index : int
        IoT 设备索引
    color : str
        绘图颜色
    recurrence : bool
        是否重复使用数据
    add_obstacles : bool
        是否添加动态障碍物
    
    Returns:
    --------
    data : numpy array
        覆盖图数据
    """
    
    index1 = []
    index1.append(index)
    case = index
    
    # 加载无线电地图数据
    data = np.load(f"./radio_maps/radio_case{case}.npy")
    radio_map_height = data.shape[1]
    radio_map_width = data.shape[2]
    
    # 获取 IoT 位置（假设有这个函数）
    # location = get_location_from_txt("../iot1.txt", index1)
    # 这里用示例值替代
    irsim_robot = [0, 0]
    translation = [13.9, -12.6]
    radio_robot_pos = [irsim_robot[0] + translation[0], irsim_robot[1] + translation[1]]
    
    # 转换为网格索引
    cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
    cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
    iot_orientation = [0, 1.57, 1.57]
    iot_position = [cellindex_x - 32, cellindex_y - 12.6, 3]
    
    # 加载场景
    scene = load_scene('INVS2/INVS.xml')
    
    # 配置天线阵列
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="tr38901", polarization="V"
    )
    
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1,
        vertical_spacing=0.5, horizontal_spacing=0.5,
        pattern="dipole", polarization="cross"
    )
    
    # 添加发射器
    tx = Transmitter(
        name="tx",
        position=iot_position,
        orientation=iot_orientation,
        color=[1, 0, 0]
    )
    scene.add(tx)
    
    # ========================================
    # 关键修改：添加动态障碍物
    # ========================================
    
    
    
    # ========================================
    # 继续原始代码
    # ========================================
    
    # 设置频率
    scene.frequency = 2.14e9
    scene.synthetic_array = True
    
    # 生成覆盖图
    print("\n生成覆盖图...")
    cm = scene.coverage_map(
        max_depth=10,
        diffraction=True,
        cm_cell_size=(0.5, 0.5),
        combining_vec=None,
        precoding_vec=None,
        num_samples=int(5e6)
    )
    
    # 处理覆盖图数据
    cm_numpy = 10. * np.log10(cm._path_gain.numpy())
    data_2d = cm_numpy.reshape(-1, cm_numpy.shape[-1])
    data_2d = np.nan_to_num(data_2d, neginf=-200)
    data = data_2d
    
    # 可视化
    plt.figure(figsize=(14, 10))
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label="Path gain (dB)")
    plt.title(f"Coverage Map with Dynamic Obstacles (IoT {index})")
    plt.xlabel("Cell index (X-axis)")
    plt.ylabel("Cell index (Y-axis)")
    
    # 标记 IoT 发射器位置
    plt.scatter(
        (iot_position[0] + 32) * 2,
        (iot_position[1] + 12.6) * 2,
        s=100, c='r', marker='o',
        label='IoT Transmitter',
        edgecolors='white', linewidths=2
    )
    
    # 如果添加了障碍物，也标记它们的位置
    if add_obstacles and obstacles_added > 0:
        for obs in obstacles:
            obs_pos = obs["position"]
            plt.scatter(
                (obs_pos[0] + 32) * 2,
                (obs_pos[1] + 12.6) * 2,
                s=150, c='blue', marker='s',
                label=obs["name"],
                edgecolors='white', linewidths=1.5,
                alpha=0.7
            )
    
    plt.legend(loc='upper right')
    
    # 保存图片
    current_time = datetime.datetime.now().time()
    plt.savefig(
        f'../plot/coverage_with_obstacles_{current_time}.pdf',
        bbox_inches='tight', dpi=300
    )
    plt.show()
    plt.close()
    
    return data


def example_custom_obstacles():
    """
    示例：使用自定义障碍物配置
    """
    
    # 定义一个复杂的障碍物场景
    custom_obstacles = [
        # 外墙
        {
            "name": "outer_wall_north",
            "position": [0.0, 15.0, 2.5],
            "size": [30.0, 0.3, 5.0],
            "material": "itu_concrete"
        },
        {
            "name": "outer_wall_south",
            "position": [0.0, -15.0, 2.5],
            "size": [30.0, 0.3, 5.0],
            "material": "itu_concrete"
        },
        
        # 室内柱子
        {
            "name": "pillar_1",
            "position": [-10.0, 0.0, 2.0],
            "size": [1.0, 1.0, 4.0],
            "material": "itu_concrete"
        },
        {
            "name": "pillar_2",
            "position": [10.0, 0.0, 2.0],
            "size": [1.0, 1.0, 4.0],
            "material": "itu_concrete"
        },
        
        # 金属设备柜
        {
            "name": "server_rack",
            "position": [5.0, 5.0, 1.0],
            "size": [2.0, 1.0, 2.0],
            "material": "itu_metal"
        },
        
        # 木质隔断
        {
            "name": "wooden_partition",
            "position": [0.0, 0.0, 1.5],
            "size": [8.0, 0.2, 3.0],
            "material": "itu_wood",
            "orientation": [0, 0, np.pi/4]  # 旋转 45 度
        }
    ]
    
    # 加载场景并添加障碍物
    scene = load_scene('INVS2/INVS.xml')
    
    # 配置天线
    scene.tx_array = PlanarArray(1, 1, 0.5, 0.5, "tr38901", "V")
    scene.rx_array = PlanarArray(1, 1, 0.5, 0.5, "dipole", "cross")
    
    # 添加发射器
    tx = Transmitter("tx", [0, 0, 3], [0, 1.57, 1.57], [1, 0, 0])
    scene.add(tx)
    
    # 添加自定义障碍物
    #add_dynamic_obstacles(scene, custom_obstacles)
    
    # 生成覆盖图
    scene.frequency = 2.14e9
    scene.synthetic_array = True
    
    cm = scene.coverage_map(
        max_depth=5,
        diffraction=True,
        cm_cell_size=(1, 1),
        num_samples=int(1e6)
    )
    
    cm.show()
    
    return scene, cm


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Sionna RT - 在你的代码中添加动态障碍物")
    print("="*70 + "\n")
    
    # 示例 1: 使用修改后的 query_radio 函数
    try:
        print("示例 1: 使用修改后的 query_radio 函数\n")
        
        # 调用修改后的函数，启用障碍物
        data = query_radio_with_obstacles(
            index=7,
            color='blue',
            recurrence=False,
            add_obstacles=True  # 启用障碍物
        )
        
        print("\n✓ 覆盖图生成成功！")
        
    except FileNotFoundError as e:
        print(f"\n✗ 文件未找到: {e}")
        print("请确保以下文件/目录存在:")
        print("  - ./radio_maps/radio_case*.npy")
        print("  - INVS2/INVS.xml")
        print("  - ../plot/ (输出目录)")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 示例 2: 使用自定义障碍物配置（可选）
    # print("\n示例 2: 使用自定义障碍物配置\n")
    # scene, cm = example_custom_obstacles()
    
    print("\n" + "="*70)
    print(" 完成!")
    print("="*70 + "\n")


# ============================================
# 快速参考指南
# ============================================

"""
快速添加障碍物到你的现有代码：

1. 在你的函数中，在 scene.add(tx) 之后添加：

    try:
        from sionna.rt import Box
        
        cube = Box(
            name="my_obstacle",
            position=[x, y, z],      # 位置（米）
            size=[w, d, h],          # 尺寸（米）
            material="itu_concrete"  # 材质
        )
        scene.add(cube)
        print("✓ 障碍物已添加")
        
    except ImportError:
        print("⚠ Box 类不可用，请升级 Sionna")


2. 可用材质:
   - itu_concrete  (混凝土)
   - itu_brick     (砖墙)
   - itu_metal     (金属)
   - itu_wood      (木材)
   - itu_glass     (玻璃)


3. 坐标系统:
   - X: 东西方向
   - Y: 南北方向
   - Z: 高度（向上）
   - 单位: 米


4. 建议:
   - 在调用 coverage_map() 前添加所有对象
   - 使用有意义的名称标识不同障碍物
   - 合理设置尺寸以匹配实际场景
   - 选择适当的材质以获得准确的电磁特性
"""