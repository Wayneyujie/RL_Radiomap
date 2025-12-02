import numpy as np

# 从txt文件读取数据
file_path = 'test.txt'  # 替换成你自己的文件路径

data = np.loadtxt(file_path)

# 计算欧几里得距离
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# 创建邻接矩阵
num_points = len(data)
adjacency_matrix = np.zeros((num_points, num_points))

# 填充邻接矩阵
for i in range(num_points):
    for j in range(num_points):
        if i != j:
            adjacency_matrix[i, j] = euclidean_distance(data[i], data[j])

# 保存邻接矩阵到txt文件
np.savetxt("euclidean_distance.txt", adjacency_matrix, fmt='%.4f')
