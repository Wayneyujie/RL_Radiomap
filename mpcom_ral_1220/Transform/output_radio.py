#radio_case8.npy'
#

import numpy as np

# 读取 .npy 文件
data = np.load('./radio_maps/radio_case15.npy')
radio_map_height = data.shape[1]
radio_map_width = data.shape[2]

# 将 3D 数组展平为 2D 数组
data_2d = data.reshape(-1, data.shape[-1])  # 或者 data.flatten().reshape(-1, data.shape[-1])

# 将数据保存为 .txt 文件
#np.savetxt('output.txt', data_2d)

#output shape
print(f"output 数据的形状: {data_2d.shape}")

#output radiomap value
#print(data_2d[4][4])

txt_path = 'point_list_ref.txt'  # 修改为你的文件路径
output_path ='index2radio.txt'
values_list = []

# 读取txt文件
with open(txt_path, 'r') as f:
    for line in f:
        # 每行按空格分隔，提取x, y
        x, y, _ = map(float, line.split())
        translation = [13.9, -12.6]
        radio_robot_pos = [x + translation[0], y + translation[1]]
        cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
        cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
        path_loss = data[0, cellindex_y, cellindex_x]
        values_list.append(path_loss)

print(values_list)

with open(output_path, 'w') as f:
    for value in values_list:
        f.write(f"{value}\n")

# 打印成功消息
print("数据已成功保存为 index2radio.txt")