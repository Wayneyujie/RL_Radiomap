#radio_case8.npy'
# Transform  radiomap.npy  to   radiomap.txt

import numpy as np

# 读取 .npy 文件
#data = np.load('radio_Tjunc.npy')
data = np.load('radio_case5.npy')

# 将 3D 数组展平为 2D 数组
data_2d = data.reshape(-1, data.shape[-1])  # 或者 data.flatten().reshape(-1, data.shape[-1])

# 将数据保存为 .txt 文件
np.savetxt('output.txt', data_2d)

#output shape
print(f"output 数据的形状: {data_2d.shape}")

#
#print(data_2d[4][4])


# 打印成功消息
#print("数据已成功保存为 output.txt")