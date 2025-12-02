import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans





def calculate_carla_anchor(cluster_centers,radio_map_height,radio_map_width):
    values_list = []  # 用于存储每个聚类中心的路径损失值
    goal_data = []

    print('The radio map has a size of:', radio_map_height, 'x', radio_map_width)
    for center in cluster_centers:
        cellindex_x=int(center[0])
        cellindex_y=int(center[1])

        # input a robot pose (anchor point) at irsim 
        irsim_robot_pos = [0, 0]
        radio_robot_pos = [0,0]



        # convert robot pose in irsim to radio map for futher query
        translation = [13.9, -12.6]


        radio_robot_pos[0] = cellindex_x-radio_map_width/2
        radio_robot_pos[1] = cellindex_y-radio_map_height/2
        irsim_robot_pos[0] = radio_robot_pos[0]-translation[0]
        irsim_robot_pos[1] = radio_robot_pos[1]-translation[1]

        values_list.append([irsim_robot_pos[0],irsim_robot_pos[1]])


        x= irsim_robot_pos[0]*10 - 136.5
        y= irsim_robot_pos[1] *10 -38
        
    
        goal_data.append({
                'x': x,
                'y': y,
                'yaw': 96,  # 假设 yaw 的变化为递增，具体情况可根据需要修改
                'comment': f'MOVE: Goal1'  # 每个目标的 comment 都不一样
                
            })
            
    output_path ='./Transform/txt/select_random_radio.txt'
    with open(output_path, 'w') as f:
        for value in values_list:
            f.write(f"{value}\n")

    return values_list
    



def generate_DBSCAN(index):
    #data = np.load('radio_Tjunc.npy')
    #data_radio = np.load(f'./Transform/radio_maps/radio_case{index}.npy')
    #data_radio = np.load(f'./radio_maps/radio_case{index}.npy')
    data_radio = np.load(f'./radio_maps/radio_{index}.npy')

    radio_map_height = data_radio.shape[1]
    radio_map_width = data_radio.shape[2]

    # 将 3D 数组展平为 2D 数组
    data_2d = data_radio.reshape(-1, data_radio.shape[-1])  # 或者 data.flatten().reshape(-1, data.shape[-1])

    # Load the matrix data from the file
    data = data_2d
    data = np.nan_to_num(data, nan=-500)
    print(data)

    # Display the shape of the data
    print("Matrix shape:", data.shape)

    # Reshape the matrix into a 2D array of (row, column) coordinates for clustering
    rows, cols = data.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    coordinates = np.vstack([x.ravel(), y.ravel()]).T

    # Flatten the matrix to use as input for DBSCAN clustering
    intensities = data.ravel()

    # Scale the intensities to influence the DBSCAN algorithm more
    scaled_intensities = intensities - intensities.min()  # Normalize intensities
    scaled_intensities = scaled_intensities / scaled_intensities.max()  # Scale between 0 and 1

    # Apply DBSCAN clustering with a metric weighted by intensities
    # We will use a weighted distance by multiplying coordinates with intensity values
    weighted_coordinates = coordinates * (1 + 5 * scaled_intensities[:, np.newaxis])

    # DBSCAN clustering with a higher eps for density-based clustering
    db = DBSCAN(eps=5, min_samples=5, metric='euclidean')
    db.fit(weighted_coordinates)

    # Get the cluster labels
    labels = db.labels_

    # Get unique clusters and their centers (just for visualization purposes)
    unique_labels = set(labels)
    cluster_centers = []

    for label in unique_labels:
        if label != -1:  # Ignore noise points labeled as -1
            cluster_points = coordinates[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            cluster_centers.append(cluster_center)

    cluster_centers = np.array(cluster_centers)

    filtered_centers = []
    for center in cluster_centers:
        x, y = center
        filtered_centers.append(center)

    cluster_centers = np.array(filtered_centers)



    # Plot the original data as a heatmap
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label="Path gain (dB)")
    plt.title("Electromagnetic Map")
    plt.xlabel("Cell index (X-axis)")
    plt.ylabel("Cell index (Y-axis)")

    # Plot the cluster centers
    # Print the cluster center coordinates
    print("Cluster centers (coordinates):")
    print(cluster_centers)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', marker='x', label='Cluster centers')
    plt.legend()

    # Show the plot
    plt.show()

    
    return calculate_carla_anchor(cluster_centers,radio_map_height,radio_map_width)

def generate_KMeans(index, n_clusters=5):
    # 加载数据
    data_radio = np.load(f'./radio_maps/radio_{index}.npy')

    radio_map_height = data_radio.shape[1]
    radio_map_width = data_radio.shape[2]

    # 将 3D 数组展平为 2D 数组
    data_2d = data_radio.reshape(-1, data_radio.shape[-1])

    # 处理缺失值
    data = np.nan_to_num(data_2d, nan=-500)
    print(data)

    # 显示数据的形状
    print("Matrix shape:", data.shape)

    # 将矩阵重塑为 (row, column) 坐标的 2D 数组
    rows, cols = data.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    coordinates = np.vstack([x.ravel(), y.ravel()]).T

    # 将矩阵展平以用于 K-means 聚类
    intensities = data.ravel()

    # 对强度进行缩放以影响 K-means 算法
    scaled_intensities = intensities - intensities.min()  # 归一化强度
    scaled_intensities = scaled_intensities / scaled_intensities.max()  # 缩放到 0 到 1 之间

    # 使用强度值加权坐标
    weighted_coordinates = coordinates * (1 + 5 * scaled_intensities[:, np.newaxis])

    # 使用 K-means 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(weighted_coordinates)

    # 获取聚类标签
    labels = kmeans.labels_

    # 获取聚类中心
    cluster_centers = kmeans.cluster_centers_

    # 过滤聚类中心
    filtered_centers = []
    for center in cluster_centers:
        x, y = center
        if not (x > 60.6 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 20.32 or x < 47 or y < 2):
            filtered_centers.append(center)

    cluster_centers = np.array(filtered_centers)

    # 绘制原始数据的热图
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label="Path gain (dB)")
    plt.title("Electromagnetic Map")
    plt.xlabel("Cell index (X-axis)")
    plt.ylabel("Cell index (Y-axis)")

    # 绘制聚类中心
    print("Cluster centers (coordinates):")
    print(cluster_centers)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', marker='x', label='Cluster centers')
    plt.legend()

    # 显示图像
    plt.show()

    return calculate_carla_anchor(cluster_centers, radio_map_height, radio_map_width)

def process_radio_map(index):
    # 加载 .npy 文件
    #data_radio = np.load(f'./radio_maps/radio_{index}.npy')
    data_radio = np.load(f'./Transform/radio_maps/radio_case{index}.npy')
    #data_radio = np.nan_to_num(data_radio, nan=-500)
    print(data_radio.shape)

    # 获取地图的高度和宽度
    radio_map_height = data_radio.shape[1]
    radio_map_width = data_radio.shape[2]
    data_radio = data_radio.reshape(26, 66)
    data_radio = np.flip(data_radio, axis=0)
    data_radio = np.where(np.isneginf(data_radio), -200, data_radio)

    # 将 3D 数组展平为 2D 数组（假设最后一个维度是强度值）
    #intensity_data = data_radio.reshape(-1, data_radio.shape[-1])  # 取最后一个维度的均值作为强度
    intensity_data = data_radio
    #intensity_data = intensity_data.reshape(radio_map_height, radio_map_width)  # 重新调整为 2D 形状

    # 反转强度值（假设强度值越大表示信号越强）
    processed_data = intensity_data
    processed_data = intensity_data.max() + intensity_data
    print(processed_data)
    save_array_to_txt(processed_data, 'processed_data.txt')

    # 设置阈值进行区域划分（高强度、中强度、低强度区域）
    high_intensity_threshold = np.percentile(processed_data, 70)  # 高强度阈值（前 20%）
    low_intensity_threshold = np.percentile(processed_data, 50)   # 低强度阈值（前 60%）

    print(f"High intensity threshold: {high_intensity_threshold}")
    print(f"Low intensity threshold: {low_intensity_threshold}")

    # 聚类点的数量设置
    high_intensity_clusters = 5  # 高强度区域更多的聚类点
    medium_intensity_clusters = 3  # 中强度区域中等数量的聚类点
    low_intensity_clusters = 10   # 低强度区域较少的聚类点

    # 对高强度区域进行聚类
    high_intensity_area = processed_data > high_intensity_threshold
    high_intensity_pixels = processed_data[high_intensity_area].reshape(-1, 1)



    if len(high_intensity_pixels) > 0:
        kmeans_high = KMeans(n_clusters=high_intensity_clusters, random_state=0).fit(high_intensity_pixels)
        cluster_centers_high = kmeans_high.cluster_centers_

    # 对中强度区域进行聚类
    medium_intensity_area = (processed_data <= high_intensity_threshold) & (processed_data >= low_intensity_threshold)
    medium_intensity_pixels = processed_data[medium_intensity_area].reshape(-1, 1)

    if len(medium_intensity_pixels) > 0:
        kmeans_medium = KMeans(n_clusters=medium_intensity_clusters, random_state=0).fit(medium_intensity_pixels)
        cluster_centers_medium = kmeans_medium.cluster_centers_

    # 对低强度区域进行聚类
    low_intensity_area = processed_data < low_intensity_threshold
    low_intensity_pixels = processed_data[low_intensity_area].reshape(-1, 1)

    if len(low_intensity_pixels) > 0:
        kmeans_low = KMeans(n_clusters=low_intensity_clusters, random_state=0).fit(low_intensity_pixels)
        cluster_centers_low = kmeans_low.cluster_centers_



    Anchor_lists=[]
    # 绘制高强度区域的聚类中心（红色）
    for idx, center in enumerate(cluster_centers_high, start=1):
        coords = np.argwhere(np.abs(processed_data - int(center)) == np.min(np.abs(processed_data - int(center))))  # 获取聚类中心的坐标
        print("coord:", coords[0][0])
        Anchor_lists.append([coords[0][1],coords[0][0]])
        if coords.size > 0:
            
            coord = coords[np.random.randint(len(coords))]  # 随机选择一个坐标


    # 绘制中强度区域的聚类中心（绿色）
    for idx, center in enumerate(cluster_centers_medium, start=1):
        coords = np.argwhere(np.abs(processed_data - int(center)) == np.min(np.abs(processed_data - int(center))))  # 获取聚类中心的坐标
        Anchor_lists.append([coords[0][1],coords[0][0]])
        if coords.size > 0:
            print("coord:", coords)
            coord = coords[np.random.randint(len(coords))]  # 随机选择一个坐标


    # 绘制低强度区域的聚类中心（蓝色）
    for idx, center in enumerate(cluster_centers_low, start=1):
        coords = np.argwhere(np.abs(processed_data - int(center)) == np.min(np.abs(processed_data - int(center))))  # 获取聚类中心的坐标
        Anchor_lists.append([coords[0][1],coords[0][0]])
        if coords.size > 0:
            print("coord:", coords)
            coord = coords[np.random.randint(len(coords))]  # 随机选择一个坐标

    
    
    


    print("high_intensity_centers:", cluster_centers_high)
    print("medium_intensity_centers:", cluster_centers_medium)
    print("low_intensity_centers:", cluster_centers_low)
    

    filtered_centers = []

    for center in Anchor_lists:
        x, y = center
        if not (x > 64  or y > 20.32 or x < 47 or y < 1):
            filtered_centers.append(center)

    Anchor_lists = np.array(filtered_centers)
    print("Anchor_lists:", Anchor_lists)
    return calculate_carla_anchor(Anchor_lists,radio_map_height,radio_map_width)

def process_radio_map_plot(index):
    # 加载 .npy 文件
    #data_radio = np.load(f'./radio_maps/radio_{index}.npy')
    data_radio = np.load(f'./Transform/radio_maps/radio_case{index}.npy')
    print(data_radio.shape)

    # 获取地图的高度和宽度
    radio_map_height = data_radio.shape[1]
    radio_map_width = data_radio.shape[2]
    data_radio = data_radio.reshape(26, 66)
    data_radio = np.flip(data_radio, axis=0)
    data_radio = np.where(np.isneginf(data_radio), -200, data_radio)

    # 将 3D 数组展平为 2D 数组（假设最后一个维度是强度值）
    #intensity_data = data_radio.reshape(-1, data_radio.shape[-1])  # 取最后一个维度的均值作为强度
    intensity_data = data_radio
    #intensity_data = intensity_data.reshape(radio_map_height, radio_map_width)  # 重新调整为 2D 形状

    # 反转强度值（假设强度值越大表示信号越强）
    processed_data = intensity_data
    #processed_data = intensity_data.max() + intensity_data
    print(processed_data)
    save_array_to_txt(processed_data, 'processed_data.txt')

    # 设置阈值进行区域划分（高强度、中强度、低强度区域）
    high_intensity_threshold = np.percentile(processed_data, 70)  # 高强度阈值（前 30%）
    low_intensity_threshold = np.percentile(processed_data, 50)   # 低强度阈值（前 50%）

    print(f"High intensity threshold: {high_intensity_threshold}")
    print(f"Low intensity threshold: {low_intensity_threshold}")

    # 聚类点的数量设置
    high_intensity_clusters = 15  # 高强度区域更多的聚类点
    medium_intensity_clusters = 7  # 中强度区域中等数量的聚类点
    low_intensity_clusters = 3   # 低强度区域较少的聚类点

    # 对高强度区域进行聚类
    high_intensity_area = processed_data > high_intensity_threshold
    high_intensity_pixels = processed_data[high_intensity_area].reshape(-1, 1)

    if len(high_intensity_pixels) > 0:
        kmeans_high = KMeans(n_clusters=high_intensity_clusters, random_state=0).fit(high_intensity_pixels)
        cluster_centers_high = kmeans_high.cluster_centers_

    # 对中强度区域进行聚类
    medium_intensity_area = (processed_data <= high_intensity_threshold) & (processed_data >= low_intensity_threshold)
    medium_intensity_pixels = processed_data[medium_intensity_area].reshape(-1, 1)

    if len(medium_intensity_pixels) > 0:
        kmeans_medium = KMeans(n_clusters=medium_intensity_clusters, random_state=0).fit(medium_intensity_pixels)
        cluster_centers_medium = kmeans_medium.cluster_centers_

    # 对低强度区域进行聚类
    low_intensity_area = processed_data < low_intensity_threshold
    low_intensity_pixels = processed_data[low_intensity_area].reshape(-1, 1)

    if len(low_intensity_pixels) > 0:
        kmeans_low = KMeans(n_clusters=low_intensity_clusters, random_state=0).fit(low_intensity_pixels)
        cluster_centers_low = kmeans_low.cluster_centers_

    # 可视化结果，包含高、中、低强度区域的聚类中心
    plt.figure(figsize=(26, 66))

    # 显示处理后的电磁强度图
    plt.imshow(processed_data)  # 使用灰度显示处理后的图像
    plt.title("处理后的电磁强度图")
    plt.axis('off')

    Anchor_lists=[]
    # 绘制高强度区域的聚类中心（红色）
    for idx, center in enumerate(cluster_centers_high, start=1):
        coords = np.argwhere(np.abs(processed_data - int(center)) == np.min(np.abs(processed_data - int(center))))  # 获取聚类中心的坐标
        print("coord:", coords[0][0])
        Anchor_lists.append([coords[0][1],coords[0][0]])
        if coords.size > 0:
            
            coord = coords[np.random.randint(len(coords))]  # 随机选择一个坐标
            plt.scatter(coord[1], coord[0], color='red', s=200, edgecolor='black')  # 调整点的大小
            plt.text(coord[1], coord[0], str(idx), color='white', fontsize=16, ha='center', va='center')  # 调整序号字体大小

    # 绘制中强度区域的聚类中心（绿色）
    for idx, center in enumerate(cluster_centers_medium, start=1):
        coords = np.argwhere(np.abs(processed_data - int(center)) == np.min(np.abs(processed_data - int(center))))  # 获取聚类中心的坐标
        Anchor_lists.append([coords[0][1],coords[0][0]])
        if coords.size > 0:
            print("coord:", coords)
            coord = coords[np.random.randint(len(coords))]  # 随机选择一个坐标
            plt.scatter(coord[1], coord[0], color='green', s=200, edgecolor='black')  # 调整点的大小
            plt.text(coord[1], coord[0], str(idx), color='white', fontsize=16, ha='center', va='center')  # 调整序号字体大小

    # 绘制低强度区域的聚类中心（蓝色）
    for idx, center in enumerate(cluster_centers_low, start=1):
        coords = np.argwhere(np.abs(processed_data - int(center)) == np.min(np.abs(processed_data - int(center))))  # 获取聚类中心的坐标
        Anchor_lists.append([coords[0][1],coords[0][0]])
        if coords.size > 0:
            print("coord:", coords)
            coord = coords[np.random.randint(len(coords))]  # 随机选择一个坐标
            plt.scatter(coord[1], coord[0], color='blue', s=200, edgecolor='black')  # 调整点的大小
            plt.text(coord[1], coord[0], str(idx), color='white', fontsize=16, ha='center', va='center')  # 调整序号字体大小
    
    
    
    plt.tight_layout()
    output_image_path = f'processed_radio_map_{index}.png'  # 保存的文件名
    plt.savefig(output_image_path, bbox_inches='tight')  # 保存图像
    #plt.show()   #这一段一旦加入就会影响irsim的作图过程

    print("high_intensity_centers:", cluster_centers_high)
    print("medium_intensity_centers:", cluster_centers_medium)
    print("low_intensity_centers:", cluster_centers_low)
    print("Anchor_lists_raw:", Anchor_lists)

    filtered_centers = []

    for center in Anchor_lists:
        x, y = center
        filtered_centers.append([x,26-y])
        

    Anchor_lists = np.array(filtered_centers)
    print("Anchor_lists:", Anchor_lists)
    plt.close()
    return calculate_carla_anchor(Anchor_lists,radio_map_height,radio_map_width)

def save_array_to_txt(array, filename):
    """
    将NumPy数组保存为txt文件
    
    参数:
    array (numpy.ndarray): 需要保存的NumPy数组
    filename (str): 输出的txt文件名
    """
    np.savetxt(filename, array, fmt='%f')  # %f 用于保存为浮点数格式，可以根据需要修改格式

#generate_DBSCAN("case7")
#generate_KMeans("case15")

#process_radio_map_plot("case1")

#calculate_path_loss(cluster_centers, data_radio, radio_map_height, radio_map_width)
#calculate_carla_anchor(cluster_centers,radio_map_height,radio_map_width)

# print(cluster_centers.shape)
# for center in cluster_centers:
#         # x = center[0]
#         # y = center[1]
#         print(center[0])



