import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 读取 test.txt 文件中的 22x66 数据矩阵
data = np.loadtxt('output.txt')

# 打印数据的形状，确保它是 22x66 的矩阵
print("Data shape:", data)

print(data[0][0])

# # 聚类数目 K，假设你选择 K=3
# K = 3

# # 创建 KMeans 模型
# kmeans = KMeans(n_clusters=K, random_state=0)

# # 训练模型
# kmeans.fit(data)

# # 获取聚类中心（K个点）
# centroids = kmeans.cluster_centers_

# # 获取每个点所属的聚类标签
# labels = kmeans.labels_

# # 输出聚类中心的二维坐标
# print("Cluster Centers (K points):")
# print(centroids)

# print("\nLabels for each point:")
# print(labels)

# # 可视化聚类结果
# plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
# plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', s=100, label='Centroids')
# plt.title('KMeans Clustering')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.legend()
# plt.show()
