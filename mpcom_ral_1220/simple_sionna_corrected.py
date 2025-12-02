#!/usr/bin/env python3
"""
简化版 SiOnNA Radio Map 生成器 (带裁剪功能)
只生成指定区域的 Path Gain numpy 数组
支持从YAML文件自动转换为XML
"""

import numpy as np
import matplotlib.pyplot as plt
from sionna.rt import load_scene, Transmitter, PlanarArray
import os
import yaml
from yaml_to_xml_converter import YAMLToXMLConverter

# GPU 配置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def crop_data(data, full_bounds, crop_bounds, resolution):
    """
    根据物理坐标裁剪 numpy 数组
    data: 原始 2D 数组 [height, width]
    full_bounds: 原始数据的物理范围 [min_x, max_x, min_y, max_y]
    crop_bounds: 想要保留的物理范围 [target_min_x, target_max_x, target_min_y, target_max_y]
    resolution: 单元格大小 (米)
    """
    min_x_full, _, min_y_full, _ = full_bounds
    target_min_x, target_max_x, target_min_y, target_max_y = crop_bounds
    
    # 1. 计算 X 轴的索引范围 (对应数组的列)
    idx_x_start = int(np.round((target_min_x - min_x_full) / resolution))
    idx_x_end   = int(np.round((target_max_x - min_x_full) / resolution))
    
    # 2. 计算 Y 轴的索引范围 (对应数组的行)
    # 注意：imshow origin='lower' 时，行索引 0 对应 min_y
    idx_y_start = int(np.round((target_min_y - min_y_full) / resolution))
    idx_y_end   = int(np.round((target_max_y - min_y_full) / resolution))
    
    # 3. 边界安全检查 (防止索引越界)
    height, width = data.shape
    idx_x_start = max(0, min(idx_x_start, width))
    idx_x_end   = max(0, min(idx_x_end, width))
    idx_y_start = max(0, min(idx_y_start, height))
    idx_y_end   = max(0, min(idx_y_end, height))
    
    print(f"✂️  裁剪索引: X[{idx_x_start}:{idx_x_end}], Y[{idx_y_start}:{idx_y_end}]")
    
    # 4. 执行切片 [行(Y), 列(X)]
    cropped_data = data[idx_y_start:idx_y_end, idx_x_start:idx_x_end]
    return cropped_data

def generate_path_gain_map(yaml_file='map_editor_output.yaml', xml_file=None):
    """
    生成 Path Gain 地图
    
    Args:
        yaml_file: YAML配置文件路径 (默认: 'map_editor_output.yaml')
        xml_file: XML场景文件路径 (可选，如果提供则直接使用，否则从YAML转换)
    """
    print("🚀 开始生成 Path Gain 地图")
    print("=" * 60)
    
    # 加载YAML配置以获取发射器位置和世界边界
    yaml_config = None
    if os.path.exists(yaml_file):
        try:
            with open(yaml_file, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
            print(f"✓ 已加载YAML配置: {yaml_file}")
        except Exception as e:
            print(f"⚠ 无法加载YAML文件 {yaml_file}: {e}")
            print("  使用默认值")
    
    # 确定使用的XML文件
    if xml_file is None:
        # 从YAML文件名生成XML文件名
        if yaml_file.endswith('.yaml'):
            xml_file = yaml_file.replace('.yaml', '.xml')
        elif yaml_file.endswith('.yml'):
            xml_file = yaml_file.replace('.yml', '.xml')
        else:
            xml_file = yaml_file + '.xml'
    
    # 如果XML文件不存在，或YAML文件更新，则进行转换
    yaml_mtime = os.path.getmtime(yaml_file) if os.path.exists(yaml_file) else 0
    xml_mtime = os.path.getmtime(xml_file) if os.path.exists(xml_file) else 0
    
    if not os.path.exists(xml_file) or (os.path.exists(yaml_file) and yaml_mtime > xml_mtime):
        print(f"🔄 转换YAML到XML: {yaml_file} -> {xml_file}")
        converter = YAMLToXMLConverter()
        if not converter.convert(yaml_file, xml_file):
            print(f"❌ YAML转换失败，尝试使用现有XML文件: {xml_file}")
            if not os.path.exists(xml_file):
                print("❌ XML文件不存在，无法继续")
                return None
    else:
        print(f"✓ 使用现有XML文件: {xml_file}")
    
    # 1. 加载场景
    scene = load_scene(xml_file)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="dipole", polarization="cross")
    
    # 从YAML提取发射器位置，如果没有则使用默认值
    if yaml_config and 'robots' in yaml_config:
        robot_state = yaml_config['robots'].get('state', [2, 2, 0.5, 0])
        tx_position = [float(robot_state[0]), float(robot_state[1]), 3.0]
        print(f"📡 从YAML读取发射器位置: ({tx_position[0]:.2f}, {tx_position[1]:.2f}, {tx_position[2]:.2f})")
    else:
        tx_position = [10, 10, 3]
        print(f"📡 使用默认发射器位置: ({tx_position[0]:.2f}, {tx_position[1]:.2f}, {tx_position[2]:.2f})")
    
    tx = Transmitter(name="tx", position=tx_position, orientation=[0, 1.57, 1.57])
    scene.add(tx)
    scene.frequency = 2.14e9
    scene.synthetic_array = True
    
    # 2. 生成覆盖图 (保持不变)
    resolution = 0.5 # 分辨率
    print("🗺️  生成覆盖图...")
    cm = scene.coverage_map(
        max_depth=7,
        diffraction=True,
        cm_cell_size=(resolution, resolution), 
        num_samples=int(5e6)
    )
    
    # 3. 计算原始全图的物理边界
    center = cm.center.numpy()
    size = cm.size.numpy()
    full_min_x = center[0] - size[0] / 2
    full_max_x = center[0] + size[0] / 2
    full_min_y = center[1] - size[1] / 2
    full_max_y = center[1] + size[1] / 2
    
    full_bounds = [full_min_x, full_max_x, full_min_y, full_max_y]
    print(f"📏 原始地图范围: X[{full_min_x:.1f}, {full_max_x:.1f}], Y[{full_min_y:.1f}, {full_max_y:.1f}]")
    
    # 4. 获取原始数据
    path_gain_raw = cm._path_gain.numpy()
    path_gain_db = 20 * np.log10(np.abs(path_gain_raw))
    path_gain_db = np.nan_to_num(path_gain_db, nan=-150, neginf=-150, posinf=0)
    
    if len(path_gain_db.shape) == 3:
        path_gain_2d = path_gain_db[0]
    else:
        path_gain_2d = path_gain_db

    # ==========================================
    # 👇 新增：裁剪逻辑
    # ==========================================
    
    # 从YAML提取世界边界，如果没有则使用默认值
    if yaml_config and 'world' in yaml_config:
        world_width = float(yaml_config['world'].get('width', 20))
        world_height = float(yaml_config['world'].get('height', 20))
        crop_bounds = [0, world_width, 0, world_height]
        print(f"📏 从YAML读取世界大小: {world_width}m × {world_height}m")
    else:
        crop_bounds = [0, 20, 0, 20]  # 默认值
        print(f"📏 使用默认世界大小: 20m × 20m")
    
    print(f"\n🔪正在裁剪数据到区域: X[{crop_bounds[0]}, {crop_bounds[1]}], Y[{crop_bounds[2]}, {crop_bounds[3]}]...")
    
    cropped_map = crop_data(path_gain_2d, full_bounds, crop_bounds, resolution)
    
    print(f"📊 裁剪后形状: {cropped_map.shape}")
    
    # ==========================================
    
    # 5. 保存裁剪后的数据
    np.save("path_gain_map_cropped.npy", cropped_map)
    np.savetxt("path_gain_map_cropped.txt", cropped_map, fmt='%.6f')
    print(f"💾 已保存裁剪后的数据 (path_gain_map_cropped.npy)")

    # 6. 可视化 (传入裁剪后的数据和裁剪后的边界)
    plot_path_gain(cropped_map, tx_position, crop_bounds)
    
    print("\n✅ 完成!")
    return cropped_map

def plot_path_gain(path_gain_2d, tx_position, bounds):
    """绘制 Path Gain 地图"""
    print("\n📈 生成可视化图...")
    
    min_x, max_x, min_y, max_y = bounds
    
    plt.figure(figsize=(10, 8))
    
    # 绘制热图 (extent 使用裁剪后的边界)
    im = plt.imshow(path_gain_2d, 
                   cmap='viridis', 
                   origin='lower',
                   extent=[min_x, max_x, min_y, max_y],
                   aspect='equal')
    
    plt.colorbar(im, label='Path Gain (dB)', fraction=0.046, pad=0.04)
    
    # 标记发射器
    plt.scatter(tx_position[0], tx_position[1], c='red', s=300, marker='*',
               label='Tx', edgecolors='white', linewidth=2, zorder=10)
    
    # 标记障碍物 (从YAML加载，如果可用)
    # 注意：这里只做可视化标记，实际障碍物已经在XML场景中
    
    plt.title('Path Gain Map (Cropped Data)', fontsize=16, fontweight='bold')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    output_filename = "path_gain_map_cropped.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    import sys
    # 支持命令行参数指定YAML文件
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else 'map_editor_output.yaml'
    generate_path_gain_map(yaml_file=yaml_file)