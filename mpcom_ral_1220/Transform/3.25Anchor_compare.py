import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from scipy.interpolate import interp2d
from scipy.ndimage import zoom
import random
from sionna.rt import load_scene, Transmitter, PlanarArray, Camera
import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import datetime
from openai import  OpenAI

## configuration:
save_cm_numpy = True
cm_cell_size = (1, 1)   # covermap cell size
camera_position = [22,0,40]
camera_look_at = [22,0,0]

import math

def db_to_linear(db_value):
    """
    将dB值转换为线性值。
    
    参数:
    db_value -- 路径损耗的dB值。
    
    返回:
    对应的线性值。
    """
    return 10 ** (db_value / 10)

def calculate_Fik(Bk, RMk, pk, sigma_squared):
    # 计算公式中的括号部分
    bracket_part = 1 + (RMk * pk) / sigma_squared
    # 使用math.log函数计算以2为底的对数
    log_part = math.log(bracket_part, 2)
    # 最终结果
    Fik = Bk * log_part
    return Fik

def calculate_carla_anchor(radio_map_height,radio_map_width):
    index=1
    index1=[]
    index1.append(index)
    case = index
    data = np.load(f"./radio_maps/radio_case{case}.npy")
    radio_map_height = data.shape[1]
    radio_map_width = data.shape[2]
    location = get_location_from_txt("../iot1.txt",index1)
    x= location[0][0]
    y= location[0][1]
    irsim_robot =[x,y]
    translation = [13.9, -12.6]
    radio_robot_pos = [irsim_robot[0] + translation[0], irsim_robot[1] + translation[1]]

    # convert radio map position to cell index ([0,0] at the upper left corner)
    cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
    cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
    iot_orientation = [0, 1.57, 1.57]
    iot_position = [cellindex_x-32, cellindex_y-12.6, 3]

   


    scene = load_scene('INVS2/INVS.xml') # load mesh

    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")


    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="cross")

    # Add transmitter instance to scene
    tx = Transmitter(name="tx",position=iot_position, orientation=iot_orientation, color=[1, 0, 0],)


    scene.add(tx)
    scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)


    cm = scene.coverage_map(max_depth=5,
                            diffraction=True, # Disable to see the effects of diffraction
                            cm_cell_size=cm_cell_size, # Grid size of coverage map cells in m
                            combining_vec=None,
                            precoding_vec=None,
                            num_samples=int(1e6)) # Reduce if your hardware does not have enough memory



    cm.show()

    print('The radio map has a size of:', radio_map_height, 'x', radio_map_width)

    #values_list.append(f"{start_x:.3f} {start_y:.3f}") 

    #start_x =start_x*10 - 136.5
    #start_y =start_y*10 -38
    # goal_data.append({
    #             'x': start_x,
    #             'y': start_y,
    #             'yaw': 96,  # 假设 yaw 的变化为递增，具体情况可根据需要修改
    #             'comment': f'MOVE: Goal1'  # 每个目标的 comment 都不一样
                
    #         })
    #input_file = "selectbyhand.txt"  #if invoked by yourself
    #input_file = f"./Transform/radio_maps/F{index}.txt"
    input_file = "./selectbyhand.txt"   # if invoked by sim_main_proposed.py
    with open(input_file, 'r') as f:
        lines = f.readlines()
    top_intensity =lines
    print("top_intensity:", top_intensity)

    original_lines = []
    

    sum_path_loss = 0
    count = 0
    for line in top_intensity:
        # Extract the x and y values from the line using string manipulation
        if "x=" in line and "y=" in line:
            # Find the x and y values using splitting
            parts = line.split(',')
            x_part = float(parts[0].split('=')[1].strip())
            y_part = float(parts[1].split('=')[1].strip())
            path_loss = parts[2].split('=')[1].strip()
            sum_path_loss+= float(path_loss)
            
            # Convert the x and y values to integers (if you need to use them in calculations)
            cellindex_x = int(x_part)
            cellindex_y = int(y_part)
            x=cellindex_x
            y=cellindex_y
            if cellindex_x ==60:
                cellindex_x = cellindex_x +1 
            if cellindex_x ==50 and cellindex_y== 7:
                cellindex_x = cellindex_x +1 
            if cellindex_x ==49 and cellindex_y< 7:
                cellindex_x = cellindex_x +1 
            if cellindex_x <58 and cellindex_y == 7:
                cellindex_y = cellindex_y -2 
            if cellindex_x ==58 and cellindex_y < 7:
                cellindex_x = cellindex_x +1   
            if cellindex_x >48 and cellindex_x<59 and cellindex_y ==19:
                cellindex_y = cellindex_y +1.5  
            if cellindex_x ==47:
                cellindex_x = cellindex_x +1   
            if cellindex_x== 49 and cellindex_y<8:
                cellindex_x = cellindex_x +1   
        #if not (x > 64 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2):
        if True:
            count += 1
            original_lines.append([cellindex_x, cellindex_y])
            plt.scatter(cellindex_x, cellindex_y, color='red', marker='x', label='Cluster centers')   
    print("original_lines=", original_lines)
    print("Sum of Path Losses:", sum_path_loss/10)
    save_v_all_a("Anchor_compare.txt",sum_path_loss/10,"Sum")  
    #save_v_all_a("Anchor_compare.txt", 111111111111111111111)  
    current_time=datetime.datetime.now()
    current_time= current_time.time()
    plt.savefig(f'../plot/{current_time}.png', bbox_inches='tight')
    # 显示图表
    plt.show()
    plt.close()   
        

         

def read_files_from_txt(input_file):
    with open(input_file, 'r') as file:
        F_vector = file.read()
    return F_vector

def generate_output2(iot_list,anchor_num=10, output_filename='../LLM/prompt.txt'):
    

    
    with open('./selectbyhand.txt', 'w') as output_file:
    # Loop through each file in the list
        for value in iot_list:
            with open(f'./radio_maps/F{value}.txt', 'r') as file:
                # Read the content of the current file and write it to the output file
                content = file.read()
                # output_file.write(f"F{value}:\n")
                output_file.write(content)
                # output_file.write("\n")  # Optionally, add a newline to separate the contents
    
    F_vector = read_files_from_txt("./selectbyhand.txt")



    # 启动目标文本内容
    
    #start_point = "{'x': -127.2, 'y': 29.5, 'yaw': 33, 'comment': 'MOVE: Goal1'}"
#     anchor_points = """self.points = [
#     {'x': -127.2, 'y': 29.5, 'yaw': 33, 'comment': 'MOVE: Goal1'},
#     {'x': -109.5, 'y': 39, 'yaw': 2, 'comment': 'MOVE: Goal1'},
#     {'x': -88.3, 'y': 38.5, 'yaw': -6, 'comment': 'MOVE: Goal1'},
#     {'x': -50.6, 'y': 36.2, 'yaw': 3, 'comment': 'MOVE: Goal1'},
#     {'x': -31.3, 'y': 37.3, 'yaw': 4, 'comment': 'MOVE: Goal1'},
#     {'x': -13.9, 'y': 54.4, 'yaw': 80, 'comment': 'MOVE: Goal1'},
#     {'x': -10.2, 'y': 82.9, 'yaw': 86, 'comment': 'MOVE: Goal1'},
#     {'x': -8.6, 'y': 110.8, 'yaw': 86, 'comment': 'MOVE: Goal1'},
#     {'x': -8.6, 'y': 139.6, 'yaw': 96, 'comment': 'MOVE: Goal1'},
#     {'x': -32.2, 'y': 157.7, 'yaw': -170, 'comment': 'MOVE: Goal1'},
#     {'x': -58.8, 'y': 158.4, 'yaw': -179, 'comment': 'MOVE: Goal1'},
#     {'x': -90, 'y': 158.2, 'yaw': -171, 'comment': 'MOVE: Goal1'},
#     {'x': -93.9, 'y': 156.4, 'yaw': -177, 'comment': 'MOVE: Goal1'},
#     {'x': -121.9, 'y': 156.9, 'yaw': 179, 'comment': 'MOVE: Goal1'},
#     {'x': -126.3, 'y': 144.2, 'yaw': -105, 'comment': 'MOVE: Goal1'},
#     {'x': -132.4, 'y': 120.3, 'yaw': -105, 'comment': 'MOVE: Goal1'},
#     {'x': -136.4, 'y': 88, 'yaw': -97, 'comment': 'MOVE: Goal1'}
# ]"""
    iot_points = "[-41.500, 39.000]"

    # 在这里，你可以选择符合要求的 anchor 点，判断路径并选择合适的点
    selected_points = "v_all[2:3]=0, v_all[5]=0, v_all[9] =0"
    num = int(anchor_num/len(iot_list))



#Each IoT's F matrix must select at least 3 points
    # 输出格式化内容
    output_content = f"""
Task Description:
Select the {anchor_num} most suitable anchor points from the provided data (F1, F2, F3) for a robot to complete a signal collection task efficiently. The robot must:

Start and end at the point (x=48, y=7).

Each IoT's F matrix must select at least {num} points

When the number of IoTs is greater than 1, the Euclidean distance between the selected anchor points should not be less than 3.

Travel through the selected anchor points while avoiding obstacles.

Maximize the communication rate (higher path loss values indicate better communication rates).

Constraints:
Path Loss Priority: Select anchor points with a relatively higher loss values to ensure the best communication rates. But at the same time, we also try to balance the number of anchor points selected for each IoT to avoid some IoTs not having any points, resulting in too little data collection.

Obstacle Avoidance: Ensure the selected anchor points do not overlap with vertical or horizontal obstacle areas or intersect with obstacle line segments. For example, Rank 9: x=58, y=5, Path Loss=-70.39161682128906 is on the obstacle boundary (58,0)->(58,6).

Start and End Point: The robot must start and end at (x=48, y=7).

Output Format: Strictly adhere to the template format provided below. Do not include any additional text, explanations, or deviations.

Output Template:(Do not separate lines)(Strictly adhere to the template format provided below with out any other information)
Fi: x=59, y=14, Path Loss=3.60979461669922 
Fi: x=51, y=3, Path Loss=3.61632537841797
...  
(Continue selecting {anchor_num} points)  

Data:
Start points:
x=48, y=7


{F_vector}

Obstacles:
Vertical Obstacles:
(49,0)->(49,6)
(58,0)->(58,6)
(49,8)->(49,19)
(58,8)->(58,19)
(60,1)->(60,19)

Horizontal Obstacles:
(49,6)->(58,6)
(49,8)->(58,8)
(49,19)->(58,19)
"""
    with open(output_filename, 'w') as output_file:
        output_file.write(output_content)
    print(f"Output written to {output_filename}")

def read_input_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        # 读取txt文件中的每一行，去掉换行符
        return file.read().strip()

def deepseek_api():
    input = read_input_from_file('../LLM/prompt.txt')
    client = OpenAI(api_key="sk-4f72ccad7abc440f8bd6c5abcd54c5d1", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": f"{input}"},
        ],
        stream=False
    )

    output_path = f'./selectbyhand_record.txt'
    with open(output_path, 'a') as f:
        f.write(response.choices[0].message.content)  # Write the entire content as a single line
        f.write(f"\n\n")
    output_path = f'./selectbyhand.txt'
    save_lines_to_txt(output_path,response.choices[0].message.content)
    print(response.choices[0].message.content)
    return(response.choices[0].message.content)

def get_location_from_txt(txt_file, index_list):
    location = []

    with open(txt_file, 'r') as file:
        for line in file:
            # 拆分每一行
            parts = line.strip().split()
            
            # 获取当前行的第一个值，并检查是否在给定的index_list中
            index = int(parts[0])
            if index in index_list:
                # 提取对应的第二列和第三列，转换为浮点数
                x = float(parts[1])
                y = float(parts[2])
                location.append([x, y])  # 将位置添加到列表中

    return location

def save_lines_to_txt(output_path,lines):
    with open(output_path, 'w') as f:
        f.write(lines)  # Write the entire content as a single line

def save_lines_to_txt1(output_path,lines):
    with open(output_path, 'w') as f:
        for value in lines:
            f.write(f"{value}\n")

def save_v_all_a(file,v_all,method):
    with open(file, "a") as f:
        f.write(f"method:{method}\n")
        f.write(f"pathloss:{v_all}\n")


def convert_coordinates(x, y, translation=[66/2, 26/2]):
    # 假设这是你想要的转换公式
    translation1 = [13.9, -12.6]
    radio_robot_pos = [x - translation[0], y - translation[1]]
    irsim_robot_pos = [radio_robot_pos[0] - translation1[0], radio_robot_pos[1] - translation1[1]]
    return irsim_robot_pos


def generate_DBSCAN1(index):
    index1=[]
    index1.append(index)
    case = index
    data = np.load(f"./radio_maps/radio_case{case}.npy")
    radio_map_height = data.shape[1]
    radio_map_width = data.shape[2]
    location = get_location_from_txt("../iot1.txt",index1)
    x= location[0][0]
    y= location[0][1]
    irsim_robot =[x,y]
    translation = [13.9, -12.6]
    radio_robot_pos = [irsim_robot[0] + translation[0], irsim_robot[1] + translation[1]]

    # convert radio map position to cell index ([0,0] at the upper left corner)
    cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
    cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
    iot_orientation = [0, 1.57, 1.57]
    iot_position = [cellindex_x-32, cellindex_y-12.6, 3]

   


    scene = load_scene('INVS2/INVS.xml') # load mesh

    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")


    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="cross")

    # Add transmitter instance to scene
    tx = Transmitter(name="tx",position=iot_position, orientation=iot_orientation, color=[1, 0, 0],)


    scene.add(tx)
    scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)


    cm = scene.coverage_map(max_depth=7,
                            diffraction=True, # Disable to see the effects of diffraction
                            cm_cell_size=(0.5,0.5), # Grid size of coverage map cells in m
                            combining_vec=None,
                            precoding_vec=None,
                            num_samples=int(5e6)) # Reduce if your hardware does not have enough memory



    #cm.show()

    #data = np.load('radio_Tjunc.npy')
    #data_radio = np.load(f'./Transform/radio_maps/radio_case{index}.npy')
    #data_radio = np.load(f'./radio_maps/radio_case{index}.npy')
    data_radio = np.load(f'./radio_maps/radio_case{case}.npy')

    radio_map_height = data_radio.shape[1]
    radio_map_width = data_radio.shape[2]

    # 将 3D 数组展平为 2D 数组
    data_2d = data_radio.reshape(-1, data_radio.shape[-1])  # 或者 data.flatten().reshape(-1, data.shape[-1])

    # Load the matrix data from the file
    data = data_2d
    data = np.nan_to_num(data, nan=-300)
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
            if len(cluster_centers) > 10:
                break

    cluster_centers = np.array(cluster_centers)
    data = np.load(f'./radio_maps/radio_case{case}.npy')
    filtered_centers = []
    sum_path_loss = 0
    count = 0
    F = []
    for center in cluster_centers:
        x, y = center
        path_loss = data[0, int(y), int(x)]
        f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
        F.append(f)
        sum_path_loss += path_loss
        count += 1
        filtered_centers.append(center)
    if count !=0:
        save_v_all_a("Anchor_compare.txt",sum_path_loss/count,"DBSCAN")
        print("Sum of Path Losses:", sum_path_loss/count)
    cluster_centers = np.array(filtered_centers)
    output_path = f'./radio_maps/F{index}.txt'
    save_lines_to_txt1(output_path,F)

    

    # Plot the original data as a heatmap
    #plt.imshow(data, cmap='viridis', origin='lower')
    #plt.colorbar(label="Path gain (dB)")
    plt.title("Electromagnetic Map")
    plt.xlabel("Cell index (X-axis)")
    plt.ylabel("Cell index (Y-axis)")

    # Plot the cluster centers
    # Print the cluster center coordinates
    print("Cluster centers (coordinates):")
    # 将坐标四舍五入到1位小数
    rounded_centers = np.round(cluster_centers, 1)
    converted_centers = [convert_coordinates(x[0], x[1]) for x in rounded_centers]

    # 格式化输出
    formatted_centers = [f"[{x[0]}, {x[1]}]" for x in converted_centers]
    original_list = [f"[{x[0]}, {x[1]}]" for x in rounded_centers]
    anchor_list = ", ".join(formatted_centers)
    print(f"original_list= [{', '.join(original_list)}]")
    print(f"anchor_list=[{anchor_list}]")

    cluster_centers[:, 0] *= 2  # Multiply the x-coordinates by 2
    cluster_centers[:, 1] *= 2  # Multiply the y-coordinates by 2

    cm_numpy = 20.*np.log10(cm._path_gain.numpy())
    data_2d = cm_numpy.reshape(-1, cm_numpy.shape[-1]) 
    #data_2d = data_radio.reshape(-1, data_radio.shape[-1])
    data_2d = np.nan_to_num(data_2d, neginf=-200)
    data =data_2d
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label="Path gain (dB)")   

    if count!=0:
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=15, color='blue', marker='^', label='Cluster centers')
    plt.scatter((iot_position[0]+32)*2, (iot_position[1]+12.6)*2, s=30, c='r', marker='o', label='Sensor')
    #plt.scatter(48*2, 7*2, s=10, c='b', marker='s', label='Sensor')
    #plt.legend()

    current_time=datetime.datetime.now()
    current_time= current_time.time()
    plt.savefig(f'../plot/{current_time}.pdf', bbox_inches='tight')
    # 显示图表
    plt.show()
    plt.close()

    
    #return calculate_carla_anchor(cluster_centers,radio_map_height,radio_map_width)

def debug_kmeans_randomness(index):
    """调试K-means随机性的函数"""
    
    # 加载数据部分保持不变...
    data_radio = np.load(f'./radio_maps/radio_case{index}.npy')
    data_2d = data_radio[0]  # 取26*66的矩阵
    data_2d = np.nan_to_num(data_2d, nan=-300)
    
    # 筛选强信号区域
    strong_signal_mask = data_2d > -70
    strong_signal_coords = np.where(strong_signal_mask)
    y_coords, x_coords = strong_signal_coords
    
    print(f"Found {len(x_coords)} points with signal > -70dB")
    print(f"Data range: {np.min(data_2d):.2f} to {np.max(data_2d):.2f}")
    
    if len(x_coords) < 10:
        # 自适应调整阈值
        sorted_signals = np.sort(data_2d.ravel())[::-1]
        threshold = sorted_signals[min(49, len(sorted_signals)-1)]
        print(f"Adjusting threshold to {threshold:.2f}dB")
        strong_signal_mask = data_2d > threshold
        strong_signal_coords = np.where(strong_signal_mask)
        y_coords, x_coords = strong_signal_coords
        print(f"After adjustment: {len(x_coords)} points")
    
    # 原始坐标（不加权）
    coordinates = np.column_stack((x_coords, y_coords))
    signal_strengths = data_2d[y_coords, x_coords]
    
    print(f"Coordinate range: X({np.min(x_coords)}-{np.max(x_coords)}), Y({np.min(y_coords)}-{np.max(y_coords)})")
    print(f"Signal strength range: {np.min(signal_strengths):.2f} to {np.max(signal_strengths):.2f}")
    
    # 测试不同random_state的影响
    from sklearn.cluster import KMeans
    
    print("\n=== 测试不同random_state对结果的影响 ===")
    
    # 方法1: 直接对原始坐标聚类
    print("\n方法1: 原始坐标聚类")
    for rs in [42, 123, 456, 789]:
        kmeans = KMeans(n_clusters=10, random_state=rs, n_init=1, init='random')
        labels = kmeans.fit_predict(coordinates)
        centers = kmeans.cluster_centers_
        print(f"Random state {rs}: Center[0] = [{centers[0,0]:.2f}, {centers[0,1]:.2f}]")
    
    # 方法2: 加权坐标聚类
    print("\n方法2: 加权坐标聚类")
    weights = (signal_strengths - np.min(signal_strengths)) + 1
    weighted_coordinates = coordinates * weights[:, np.newaxis]
    
    for rs in [42, 123, 456, 789]:
        kmeans = KMeans(n_clusters=10, random_state=rs, n_init=1, init='random')
        labels = kmeans.fit_predict(weighted_coordinates)
        centers = kmeans.cluster_centers_
        print(f"Random state {rs}: Weighted Center[0] = [{centers[0,0]:.2f}, {centers[0,1]:.2f}]")
    
    # 方法3: 增加数据多样性（添加微小随机扰动）
    print("\n方法3: 添加随机扰动")
    np.random.seed(42)  # 为了可重现性
    noise_factor = 0.1
    
    for rs in [42, 123, 456, 789]:
        np.random.seed(rs)  # 使用random_state作为噪声种子
        noisy_coordinates = coordinates + np.random.normal(0, noise_factor, coordinates.shape)
        kmeans = KMeans(n_clusters=10, random_state=rs, n_init=1, init='random')
        labels = kmeans.fit_predict(noisy_coordinates)
        centers = kmeans.cluster_centers_
        print(f"Random state {rs}: Noisy Center[0] = [{centers[0,0]:.2f}, {centers[0,1]:.2f}]")

def generate_Kmeans_improved(index, random_state=42, method='weighted',recurrence=False):
    """改进的K-means函数，支持不同的聚类方法"""
    
    # 前面的数据加载和预处理代码保持不变...
    index1=[]
    index1.append(index)
    case = index
    data = np.load(f"./radio_maps/radio_case{case}.npy")
    radio_map_height = data.shape[1]
    radio_map_width = data.shape[2]
    location = get_location_from_txt("../iot1.txt",index1)
    x= location[0][0]
    y= location[0][1]
    irsim_robot =[x,y]
    translation = [13.9, -12.6]
    radio_robot_pos = [irsim_robot[0] + translation[0], irsim_robot[1] + translation[1]]

    cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
    cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
    iot_orientation = [0, 1.57, 1.57]
    iot_position = [cellindex_x-32, cellindex_y-12.6, 3]

    # ... 场景设置代码保持不变 ...
    scene = load_scene('INVS2/INVS.xml')
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                horizontal_spacing=0.5, pattern="tr38901", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                horizontal_spacing=0.5, pattern="dipole", polarization="cross")
    tx = Transmitter(name="tx",position=iot_position, orientation=iot_orientation, color=[1, 0, 0],)
    scene.add(tx)
    scene.frequency = 2.14e9
    scene.synthetic_array = True

    cm = scene.coverage_map(max_depth=7, diffraction=True, cm_cell_size=(0.5,0.5),
                            combining_vec=None, precoding_vec=None, num_samples=int(5e6))

    # 数据处理
    data_radio = np.load(f'./radio_maps/radio_case{case}.npy')
    data_2d = data_radio[0]
    data_2d = np.nan_to_num(data_2d, nan=-300)
    
    # 筛选强信号区域
    strong_signal_mask = data_2d > -68
    strong_signal_coords = np.where(strong_signal_mask)
    y_coords, x_coords = strong_signal_coords
    
    if len(x_coords) < 10:
        sorted_signals = np.sort(data_2d.ravel())[::-1]
        threshold = sorted_signals[min(49, len(sorted_signals)-1)]
        print(f"Adjusting threshold to {threshold:.2f}dB")
        strong_signal_mask = data_2d > threshold
        strong_signal_coords = np.where(strong_signal_mask)
        y_coords, x_coords = strong_signal_coords
    
    coordinates = np.column_stack((x_coords, y_coords))
    signal_strengths = data_2d[y_coords, x_coords]
    
    # 根据选择的方法进行聚类
    from sklearn.cluster import KMeans
    
    if method == 'original':
        # 方法1: 原始坐标
        cluster_data = coordinates
        
    elif method == 'weighted':
        # 方法2: 加权坐标
        weights = (signal_strengths - np.min(signal_strengths)) + 1
        cluster_data = coordinates * weights[:, np.newaxis]
        
    elif method == 'noisy':
        # 方法3: 添加随机扰动
        np.random.seed(random_state)
        noise_factor = 0.1
        cluster_data = coordinates + np.random.normal(0, noise_factor, coordinates.shape)
        
    else:
        raise ValueError("Method must be 'original', 'weighted', or 'noisy'")
    
    # K-means聚类 - 关键参数调整
    kmeans = KMeans(
        n_clusters=10, 
        random_state=random_state, 
        n_init=1,           # 只运行一次，不取最佳结果
        init='random',      # 使用随机初始化而不是k-means++
        max_iter=300,
        tol=1e-4
    )
    
    cluster_labels = kmeans.fit_predict(cluster_data)
    
    # 计算真实的cluster centers
    if method == 'weighted':
        # 对于加权方法，需要计算原始坐标系下的center
        cluster_centers = []
        for i in range(10):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                cluster_coords = coordinates[cluster_mask]
                cluster_weights = (signal_strengths[cluster_mask] - np.min(signal_strengths)) + 1
                weighted_center = np.average(cluster_coords, axis=0, weights=cluster_weights)
                cluster_centers.append(weighted_center)
        cluster_centers = np.array(cluster_centers)
    else:
        # 对于其他方法，直接使用kmeans的centers
        if method == 'noisy':
            # 对于噪声方法，需要将centers映射回原始数据范围
            cluster_centers = kmeans.cluster_centers_
        else:
            cluster_centers = kmeans.cluster_centers_
    
    print(f"Using method: {method}, random_state: {random_state}")
    print(f"First anchor position: [{cluster_centers[0,0]:.2f}, {cluster_centers[0,1]:.2f}]")
    
    # 后续处理代码保持不变...
    filtered_centers = []
    sum_path_loss = 0
    count = 0
    F = []
    
    for i, center in enumerate(cluster_centers):
        x, y = center
        x_int = int(np.clip(x, 0, radio_map_width-1))
        y_int = int(np.clip(y, 0, radio_map_height-1))
        
        path_loss = data_2d[y_int, x_int]
        f = f"F{index}_Anchor{i+1}: x={x:.1f}, y={y:.1f}, Path Loss={path_loss:.2f}dB"
        F.append(f)
        sum_path_loss += path_loss
        count += 1
        filtered_centers.append(center)
    
    if count != 0:
        avg_path_loss = sum_path_loss / count
        save_v_all_a("Anchor_compare.txt", avg_path_loss, f"K-means_{method}")
        print("Average Path Loss of Anchors:", avg_path_loss)
    
    # 保存和可视化代码保持不变...
    cluster_centers = np.array(filtered_centers)
    output_path = f'./radio_maps/F{index}_Kmeans_{method}_{random_state}.txt'
    save_lines_to_txt1(output_path, F)

    # 绘图
    #plt.figure(figsize=(12, 6))
    cm_numpy = 20.*np.log10(cm._path_gain.numpy())
    data_2d_plot = cm_numpy.reshape(-1, cm_numpy.shape[-1]) 
    data_2d_plot = np.nan_to_num(data_2d_plot, neginf=-200)
    
    plt.imshow(data_2d_plot, cmap='viridis', origin='lower')
    plt.colorbar(label="Path gain (dB)")
    #plt.title(f"K-means Anchors (Method: {method}, Random State: {random_state})")
    # plt.xlabel("Cell index (X-axis)")
    # plt.ylabel("Cell index (Y-axis)")

    if not recurrence:
        if count != 0:
            plot_centers = cluster_centers.copy()
            print("Scaled X-coordinates:", plot_centers[:, 0])
            for i, center in enumerate(plot_centers):
                if center[0] <40:
                    plot_centers[i, 0] = 39-center[0]+48
                if center[0] <46:
                    plot_centers[i, 0] = 47-center[0]+48
            original_lines = [f"[{x[0]:.3f}, {x[1]:.3f}]" for x in plot_centers]
            plot_centers[:, 0] *= 2
            
            plot_centers[:, 1] *= 2
            plt.scatter(plot_centers[:, 0], plot_centers[:, 1], s=15, color='blue', 
                    marker='^')
        
        print(f"original_lines=[{', '.join(original_lines)}]")
    else:
        original_lines=[[49.7, 6.6], [56.8, 7.1], [53.6, 9.7], [56.0, 7.2], [57.9, 7.7], [54.9, 13.4], [54.3, 6.2], [58.1, 6.9], [58.4, 7.1], [52.3, 5.8]]
        original_lines=[[52.8, 6.8], [53.8, 6.9], [58.2, 6.8], [54.9, 8.6], [58.4, 7.1], [57.8, 7.0], [57.9, 7.7], [55.5, 6.1], [53.9, 9.1], [50.0, 7.1]]
        #IoT 4:
        original_lines=[[52.825, 6.769], [53.769, 6.914], [58.173, 6.809], [54.927, 8.611], [58.377, 7.096], [57.793, 7.035], [57.880, 7.720], [55.550, 6.105], [53.886, 9.145], [50.025, 7.053]]
        #IoT 19:
        original_lines=[[58.903, 9.837], [60.512, 9.981], [60.682, 9.530], [59.470, 10.087],  [57.534, 9.971]]
        #IoT 16:
        original_lines=[ [50.187, 19.083],  [50.283, 18.750], [48.641, 17.626], [50.984, 16.419], [50.240, 18.698], [50.099, 19.033], [49.894, 17.911]]
        #IoT 11:
        original_lines=[[50.018, 7.941],[50.429, 8.598], [49.371, 8.730],[51.320, 9.322], [51.375, 9.166]]

        plot_centers = original_lines.copy()
        doubled_lines = []
        for pair in original_lines:
            doubled_pair = [x * 2 for x in pair]  # 或者写成：doubled_pair = [pair[0] * 2, pair[1] * 2]
            doubled_lines.append(doubled_pair)
        x_values = [point[0] for point in doubled_lines]  # 所有 x
        y_values = [point[1] for point in doubled_lines]  # 所有 y
        
        # plot_centers[:, 0] *= 2
            
        # plot_centers[:, 1] *= 2
        plt.scatter(x_values, y_values, s=15, color='blue', 
                    marker='^')
        #print(f"original_lines=[{', '.join(original_lines)}]")
    
    
    plt.scatter((iot_position[0]+32)*2, (iot_position[1]+12.6)*2, s=30, c='r', 
               marker='o')
    #plt.legend()
    
    current_time = datetime.datetime.now().strftime("%H%M%S")
    plt.savefig(f'../plot/kmeans_{method}_{random_state}_{current_time}.pdf', 
                bbox_inches='tight')
    plt.show()
    plt.close()

    return cluster_centers

def generate_DBSCAN(index):
    index1=[]
    index1.append(index)
    case = index
    data = np.load(f"./radio_maps/radio_case{case}.npy")
    radio_map_height = data.shape[1]
    radio_map_width = data.shape[2]
    location = get_location_from_txt("../iot1.txt",index1)
    x= location[0][0]
    y= location[0][1]
    irsim_robot =[x,y]
    translation = [13.9, -12.6]
    radio_robot_pos = [irsim_robot[0] + translation[0], irsim_robot[1] + translation[1]]

    # convert radio map position to cell index ([0,0] at the upper left corner)
    cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
    cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
    iot_orientation = [0, 1.57, 1.57]
    iot_position = [cellindex_x-32, cellindex_y-12.6, 3]

    scene = load_scene('INVS2/INVS.xml') # load mesh

    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="cross")

    # Add transmitter instance to scene
    tx = Transmitter(name="tx",position=iot_position, orientation=iot_orientation, color=[1, 0, 0],)

    scene.add(tx)
    scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)

    cm = scene.coverage_map(max_depth=7,
                            diffraction=True, # Disable to see the effects of diffraction
                            cm_cell_size=(0.5,0.5), # Grid size of coverage map cells in m
                            combining_vec=None,
                            precoding_vec=None,
                            num_samples=int(5e6)) # Reduce if your hardware does not have enough memory

    # Load radio map data
    data_radio = np.load(f'./radio_maps/radio_case{case}.npy')
    radio_map_height = data_radio.shape[1]
    radio_map_width = data_radio.shape[2]

    # 将 3D 数组展平为 2D 数组 (取第一个频率/天线组合)
    data_2d = data_radio[0]  # 取第一个slice，得到26*66的矩阵
    
    # 处理NaN值
    data_2d = np.nan_to_num(data_2d, nan=-300)
    print("Matrix shape:", data_2d.shape)
    print("Data range:", np.min(data_2d), "to", np.max(data_2d))

    # 筛选出电磁强度 > -70 的区域
    strong_signal_mask = data_2d > -60
    
    # 获取满足条件的坐标点
    strong_signal_coords = np.where(strong_signal_mask)
    y_coords, x_coords = strong_signal_coords
    
    # 如果满足条件的点数少于10个，调整阈值
    if len(x_coords) < 10:
        print(f"Warning: Only {len(x_coords)} points found with signal > -70dB")
        # 找到最强的信号值并适当调整阈值
        sorted_signals = np.sort(data_2d.ravel())[::-1]  # 降序排列
        if len(sorted_signals) >= 50:
            threshold = sorted_signals[49]  # 取前50强的最低值作为阈值
        else:
            threshold = sorted_signals[-1]  # 取最低值
        print(f"Adjusting threshold to {threshold:.2f}dB")
        strong_signal_mask = data_2d > threshold
        strong_signal_coords = np.where(strong_signal_mask)
        y_coords, x_coords = strong_signal_coords
    
    print(f"Found {len(x_coords)} points with strong signal")
    
    if len(x_coords) < 10:
        print("Error: Not enough strong signal points for clustering")
        return
    
    # 准备K-means的输入数据：坐标点
    coordinates = np.column_stack((x_coords, y_coords))
    signal_strengths = data_2d[y_coords, x_coords]
    
    # 可选：根据信号强度对坐标进行加权
    # 信号强度越强(越接近0)，权重越大
    weights = (signal_strengths - np.min(signal_strengths)) + 1  # 避免负权重
    weighted_coordinates = coordinates * weights[:, np.newaxis]
    
    # 使用K-means聚类，设置k=10
    from sklearn.cluster import KMeans
    
    # 设置随机种子以获得可重复的结果
    kmeans = KMeans(n_clusters=10, random_state=30, n_init=10)
    
    # 对加权坐标进行聚类（也可以直接对coordinates聚类）
    cluster_labels = kmeans.fit_predict(weighted_coordinates)
    
    # 获取cluster centers（需要转换回原始坐标系）
    cluster_centers = kmeans.cluster_centers_
    
    # 如果使用了加权坐标，需要将centers转换回原始坐标
    # 计算每个cluster的平均权重
    cluster_centers_original = []
    for i in range(10):
        cluster_mask = cluster_labels == i
        if np.any(cluster_mask):
            cluster_coords = coordinates[cluster_mask]
            cluster_weights = weights[cluster_mask]
            # 使用加权平均计算真实的center位置
            weighted_center = np.average(cluster_coords, axis=0, weights=cluster_weights)
            cluster_centers_original.append(weighted_center)
    
    cluster_centers = np.array(cluster_centers_original)
    
    # 计算每个cluster center的path loss并保存结果
    filtered_centers = []
    sum_path_loss = 0
    count = 0
    F = []
    
    for i, center in enumerate(cluster_centers):
        x, y = center
        # 确保坐标在有效范围内
        x_int = int(np.clip(x, 0, radio_map_width-1))
        y_int = int(np.clip(y, 0, radio_map_height-1))
        
        path_loss = data_2d[y_int, x_int]
        f = f"F{index}_Anchor{i+1}: x={x:.1f}, y={y:.1f}, Path Loss={path_loss:.2f}dB"
        F.append(f)
        sum_path_loss += path_loss
        count += 1
        filtered_centers.append(center)
    
    if count != 0:
        avg_path_loss = sum_path_loss / count
        save_v_all_a("Anchor_compare.txt", avg_path_loss, "K-means")
        print("Average Path Loss of Anchors:", avg_path_loss)
    
    cluster_centers = np.array(filtered_centers)
    output_path = f'./radio_maps/F{index}_Kmeans.txt'
    save_lines_to_txt1(output_path, F)

    # 绘图部分
    plt.figure(figsize=(12, 6))
    
    # 使用coverage map数据进行可视化
    cm_numpy = 20.*np.log10(cm._path_gain.numpy())
    data_2d_plot = cm_numpy.reshape(-1, cm_numpy.shape[-1]) 
    data_2d_plot = np.nan_to_num(data_2d_plot, neginf=-200)
    
    plt.imshow(data_2d_plot, cmap='viridis', origin='lower')
    plt.colorbar(label="Path gain (dB)")
    plt.title("Electromagnetic Map with K-means Anchors")
    plt.xlabel("Cell index (X-axis)")
    plt.ylabel("Cell index (Y-axis)")

    # 绘制anchor点 (需要调整坐标到绘图坐标系)
    if count != 0:
        plot_centers = cluster_centers.copy()
        plot_centers[:, 0] *= 2  # 根据原代码调整x坐标
        plot_centers[:, 1] *= 2  # 根据原代码调整y坐标
        plt.scatter(plot_centers[:, 0], plot_centers[:, 1], s=50, color='blue', 
                   marker='^', label='K-means Anchors', edgecolors='white', linewidth=1)
    
    # 绘制传感器位置
    plt.scatter((iot_position[0]+32)*2, (iot_position[1]+12.6)*2, s=60, c='r', 
               marker='o', label='Sensor', edgecolors='white', linewidth=1)
    
    plt.legend()
    
    # 保存图片
    current_time = datetime.datetime.now()
    current_time = current_time.time()
    plt.savefig(f'../plot/kmeans_{current_time}.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

    # 输出anchor坐标信息
    print("K-means Anchor Centers (coordinates):")
    rounded_centers = np.round(cluster_centers, 1)
    converted_centers = [convert_coordinates(x[0], x[1]) for x in rounded_centers]
    
    formatted_centers = [f"[{x[0]}, {x[1]}]" for x in converted_centers]
    original_list = [f"[{x[0]}, {x[1]}]" for x in rounded_centers]
    anchor_list = ", ".join(formatted_centers)
    print(f"original_list= [{', '.join(original_list)}]")
    print(f"anchor_list=[{anchor_list}]")


# def generate_DBSCAN(index):
#     index1 = []
#     index1.append(index)
#     case = index
#     data = np.load(f"./radio_maps/radio_case{case}.npy")
#     radio_map_height = data.shape[1]
#     radio_map_width = data.shape[2]
#     location = get_location_from_txt("../iot1.txt", index1)
#     x = location[0][0]
#     y = location[0][1]
#     irsim_robot = [x, y]
#     translation = [13.9, -12.6]
#     radio_robot_pos = [irsim_robot[0] + translation[0], irsim_robot[1] + translation[1]]

#     # convert radio map position to cell index ([0,0] at the upper left corner)
#     cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
#     cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
#     iot_orientation = [0, 1.57, 1.57]
#     iot_position = [cellindex_x-32, cellindex_y-12.6, 3]

#     # [之前的scene setup代码保持不变...]
#     scene = load_scene('INVS2/INVS.xml')
    
#     scene.tx_array = PlanarArray(num_rows=1,
#                                 num_cols=1,
#                                 vertical_spacing=0.5,
#                                 horizontal_spacing=0.5,
#                                 pattern="tr38901",
#                                 polarization="V")

#     scene.rx_array = PlanarArray(num_rows=1,
#                                 num_cols=1,
#                                 vertical_spacing=0.5,
#                                 horizontal_spacing=0.5,
#                                 pattern="dipole",
#                                 polarization="cross")

#     tx = Transmitter(name="tx", position=iot_position, orientation=iot_orientation, color=[1, 0, 0])
#     scene.add(tx)
#     scene.frequency = 2.14e9
#     scene.synthetic_array = True

#     cm = scene.coverage_map(max_depth=7,
#                             diffraction=True,
#                             cm_cell_size=(0.5,0.5),
#                             combining_vec=None,
#                             precoding_vec=None,
#                             num_samples=int(5e6))

#     # 加载数据
#     data_radio = np.load(f'./radio_maps/radio_case{case}.npy')
#     data_2d = data_radio.reshape(-1, data_radio.shape[-1])
#     data_2d = np.nan_to_num(data_2d, nan=-300)
    
#     print(f"Original data shape: {data_2d.shape}")
#     print(f"Path loss range: {data_2d.min()} to {data_2d.max()}")

#     # 方法1: 直接基于高质量信号区域进行聚类
#     def method1_signal_quality_clustering():
#         """基于信号质量的聚类方法"""
#         print("\n=== Method 1: Signal Quality Based Clustering ===")
        
#         # 将path loss转换为信号强度（负数越大越好，所以取负号）
#         signal_strength = data_2d- data_2d.min()  # 现在正数越大越好
        
#         # 设置信号强度阈值，只考虑较好的信号区域
#         threshold = np.percentile(signal_strength, 75)  # 取前25%的高质量信号
#         print(f"Signal strength threshold: {threshold}")
        
#         # 创建坐标网格
#         rows, cols = signal_strength.shape
#         y_coords, x_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        
#         # 找到高质量信号的位置
#         high_quality_mask = signal_strength >= threshold
#         high_quality_coords = np.column_stack([
#             x_coords[high_quality_mask], 
#             y_coords[high_quality_mask]
#         ])
#         high_quality_strengths = signal_strength[high_quality_mask]
        
#         print(f"High quality points: {len(high_quality_coords)}")
        
#         if len(high_quality_coords) < 10:
#             print("Too few high quality points, lowering threshold")
#             threshold = np.percentile(signal_strength, 60)
#             high_quality_mask = signal_strength >= threshold
#             high_quality_coords = np.column_stack([
#                 x_coords[high_quality_mask], 
#                 y_coords[high_quality_mask]
#             ])
#             high_quality_strengths = signal_strength[high_quality_mask]
        
#         # 使用信号强度加权坐标
#         weights = (high_quality_strengths - high_quality_strengths.min()) / (high_quality_strengths.max() - high_quality_strengths.min())
#         weighted_coords = high_quality_coords * (1 + 2 * weights[:, np.newaxis])
        
#         # DBSCAN聚类
#         db = DBSCAN(eps=3, min_samples=3, metric='euclidean')
#         labels = db.fit_predict(weighted_coords)
        
#         unique_labels = set(labels)
#         print(f"Found {len(unique_labels - {-1})} clusters")
        
#         cluster_centers = []
#         for label in unique_labels:
#             if label != -1:
#                 cluster_points = high_quality_coords[labels == label]
#                 cluster_strengths = high_quality_strengths[labels == label]
#                 # 使用加权平均计算聚类中心
#                 weights = cluster_strengths / cluster_strengths.sum()
#                 center = np.average(cluster_points, axis=0, weights=weights)
#                 cluster_centers.append(center)
        
#         return np.array(cluster_centers)

#     # 方法2: 多尺度聚类
#     def method2_multiscale_clustering():
#         """多尺度聚类方法"""
#         print("\n=== Method 2: Multi-scale Clustering ===")
        
#         signal_strength = -data_2d
#         rows, cols = signal_strength.shape
#         y_coords, x_coords = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
#         all_coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
#         all_strengths = signal_strength.ravel()
        
#         all_centers = []
        
#         # 使用不同的eps值进行多次聚类
#         eps_values = [2, 3, 4, 5]
#         min_samples_values = [3, 4, 5]
        
#         for eps in eps_values:
#             for min_samples in min_samples_values:
#                 # 只使用高质量点
#                 threshold = np.percentile(all_strengths, 70)
#                 mask = all_strengths >= threshold
#                 coords = all_coords[mask]
#                 strengths = all_strengths[mask]
                
#                 if len(coords) < min_samples * 2:
#                     continue
                
#                 # 加权坐标
#                 weights = (strengths - strengths.min()) / (strengths.max() - strengths.min() + 1e-8)
#                 weighted_coords = coords * (1 + weights[:, np.newaxis])
                
#                 db = DBSCAN(eps=eps, min_samples=min_samples)
#                 labels = db.fit_predict(weighted_coords)
                
#                 for label in set(labels):
#                     if label != -1:
#                         cluster_points = coords[labels == label]
#                         cluster_strengths = strengths[labels == label]
#                         center = np.average(cluster_points, axis=0, weights=cluster_strengths)
#                         all_centers.append(center)
        
#         # 去除重复的聚类中心
#         if len(all_centers) > 0:
#             all_centers = np.array(all_centers)
#             # 使用简单的距离阈值去重
#             unique_centers = []
#             for center in all_centers:
#                 is_unique = True
#                 for existing in unique_centers:
#                     if np.linalg.norm(center - existing) < 3:
#                         is_unique = False
#                         break
#                 if is_unique:
#                     unique_centers.append(center)
#             return np.array(unique_centers)
#         return np.array([])

#     # 方法3: 基于局部最大值的方法
#     def method3_local_maxima():
#         """基于局部最大值的方法"""
#         print("\n=== Method 3: Local Maxima Based ===")
        
#         from scipy.ndimage import maximum_filter
#         from scipy.ndimage import label
        
#         signal_strength = -data_2d
        
#         # 使用形态学操作找到局部最大值
#         local_maxima = maximum_filter(signal_strength, size=3) == signal_strength
        
#         # 设置最小信号强度阈值
#         threshold = np.percentile(signal_strength, 80)
#         strong_signal_mask = signal_strength >= threshold
        
#         # 结合局部最大值和强信号区域
#         candidate_mask = local_maxima & strong_signal_mask
        
#         # 获取候选点坐标
#         candidate_coords = np.column_stack(np.where(candidate_mask))
#         candidate_coords = candidate_coords[:, [1, 0]]  # swap to (x, y)
        
#         print(f"Found {len(candidate_coords)} candidate points")
        
#         if len(candidate_coords) == 0:
#             return np.array([])
        
#         # 如果候选点太多，选择信号最强的前N个
#         if len(candidate_coords) > 15:
#             candidate_strengths = signal_strength[candidate_mask]
#             top_indices = np.argsort(candidate_strengths)[-15:]
#             candidate_coords = candidate_coords[top_indices]
        
#         # 对候选点进行空间聚类以避免过于密集
#         if len(candidate_coords) > 5:
#             db = DBSCAN(eps=4, min_samples=2)
#             labels = db.fit_predict(candidate_coords)
            
#             cluster_centers = []
#             for label in set(labels):
#                 if label != -1:
#                     cluster_points = candidate_coords[labels == label]
#                     center = np.mean(cluster_points, axis=0)
#                     cluster_centers.append(center)
#                 else:
#                     # 添加噪声点（孤立的高质量点）
#                     noise_points = candidate_coords[labels == -1]
#                     cluster_centers.extend(noise_points)
            
#             return np.array(cluster_centers)
#         else:
#             return candidate_coords

#     # 尝试所有方法并合并结果
#     centers1 = method1_signal_quality_clustering()
#     centers2 = method2_multiscale_clustering()
#     centers3 = method3_local_maxima()
    
#     # 合并所有聚类中心
#     all_centers = []
#     for centers in [centers1, centers2, centers3]:
#         if len(centers) > 0:
#             all_centers.extend(centers)
    
#     if len(all_centers) == 0:
#         print("No cluster centers found!")
#         return []
    
#     # 最终去重和筛选
#     all_centers = np.array(all_centers)
    
#     # 更严格的去重
#     final_centers = []
#     for center in all_centers:
#         is_unique = True
#         for existing in final_centers:
#             if np.linalg.norm(center - existing) < 2.5:
#                 is_unique = False
#                 break
#         if is_unique:
#             final_centers.append(center)
    
#     final_centers = np.array(final_centers)
    
#     # 限制anchor数量（如果太多的话）
#     if len(final_centers) > 12:
#         # 基于信号质量选择最好的12个
#         center_qualities = []
#         for center in final_centers:
#             x, y = int(center[0]), int(center[1])
#             if 0 <= x < data_2d.shape[1] and 0 <= y < data_2d.shape[0]:
#                 quality = -data_2d[y, x]  # 转为正数，越大越好
#                 center_qualities.append(quality)
#             else:
#                 center_qualities.append(-1000)  # 边界外的点给低分
        
#         top_indices = np.argsort(center_qualities)[-12:]
#         final_centers = final_centers[top_indices]
    
#     print(f"\nFinal number of anchors: {len(final_centers)}")
    
#     # 计算平均path loss并保存结果
#     filtered_centers = []
#     sum_path_loss = 0
#     count = 0
#     F = []
    
#     for center in final_centers:
#         x, y = int(center[0]), int(center[1])
#         if 0 <= x < data_2d.shape[1] and 0 <= y < data_2d.shape[0]:
#             path_loss = data_2d[y, x]
#             f = f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
#             F.append(f)
#             sum_path_loss += path_loss
#             count += 1
#             filtered_centers.append(center)
    
#     if count != 0:
#         avg_path_loss = sum_path_loss / count
#         save_v_all_a("Anchor_compare.txt", avg_path_loss, "DBSCAN_Multi")
#         print("Average Path Loss:", avg_path_loss)
    
#     final_centers = np.array(filtered_centers)
#     output_path = f'./radio_maps/F{index}.txt'
#     save_lines_to_txt1(output_path, F)
    
#     # 可视化结果
#     plt.figure(figsize=(12, 8))
    
#     # 获取coverage map数据用于显示
#     cm_numpy = 20.*np.log10(cm._path_gain.numpy())
#     cm_2d = cm_numpy.reshape(-1, cm_numpy.shape[-1])
#     cm_2d = np.nan_to_num(cm_2d, neginf=-200)
    
#     plt.imshow(cm_2d, cmap='viridis', origin='lower')
#     plt.colorbar(label="Path gain (dB)")
#     plt.title(f"Electromagnetic Map with {len(final_centers)} Anchors")
#     plt.xlabel("Cell index (X-axis)")
#     plt.ylabel("Cell index (Y-axis)")
    
#     # 绘制聚类中心
#     if len(final_centers) > 0:
#         # 调整坐标以匹配显示
#         display_centers = final_centers.copy()
#         display_centers[:, 0] *= 2
#         display_centers[:, 1] *= 2
        
#         plt.scatter(display_centers[:, 0], display_centers[:, 1], 
#                    s=100, color='red', marker='^', 
#                    label=f'Anchors ({len(final_centers)})', 
#                    edgecolors='white', linewidths=1)
    
#     # 绘制传感器位置
#     plt.scatter((iot_position[0]+32)*2, (iot_position[1]+12.6)*2, 
#                s=150, c='blue', marker='o', label='Sensor',
#                edgecolors='white', linewidths=2)
    
#     plt.legend()
    
#     current_time = datetime.datetime.now()
#     current_time = current_time.time()
#     plt.savefig(f'../plot/{current_time}_multi_anchors.pdf', bbox_inches='tight')
#     plt.show()
#     plt.close()
    
#     # 输出结果
#     if len(final_centers) > 0:
#         rounded_centers = np.round(final_centers, 1)
#         converted_centers = [convert_coordinates(x[0], x[1]) for x in rounded_centers]
#         formatted_centers = [f"[{x[0]}, {x[1]}]" for x in converted_centers]
#         original_list = [f"[{x[0]}, {x[1]}]" for x in rounded_centers]
#         anchor_list = ", ".join(formatted_centers)
        
#         print(f"original_list= [{', '.join(original_list)}]")
#         print(f"anchor_list=[{anchor_list}]")
    
#     return final_centers


def select_random_voro(index,anchor_num, recurrence=True):
    index1=[]
    index1.append(index)
    case = index
    data = np.load(f"./radio_maps/radio_case{case}.npy")
    radio_map_height = data.shape[1]
    radio_map_width = data.shape[2]
    location = get_location_from_txt("../iot1.txt",index1)
    x= location[0][0]
    y= location[0][1]
    irsim_robot =[x,y]
    translation = [13.9, -12.6]
    radio_robot_pos = [irsim_robot[0] + translation[0], irsim_robot[1] + translation[1]]

    # convert radio map position to cell index ([0,0] at the upper left corner)
    cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
    cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
    iot_orientation = [0, 1.57, 1.57]
    iot_position = [cellindex_x-32, cellindex_y-12.6, 3]

   


    scene = load_scene('INVS2/INVS.xml') # load mesh

    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")


    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="cross")

    # Add transmitter instance to scene
    tx = Transmitter(name="tx",position=iot_position, orientation=iot_orientation, color=[1, 0, 0],)


    scene.add(tx)
    scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)


    cm = scene.coverage_map(max_depth=5,
                            diffraction=True, # Disable to see the effects of diffraction
                            cm_cell_size=(0.5,0.5), # Grid size of coverage map cells in m
                            combining_vec=None,
                            precoding_vec=None,
                            num_samples=int(5e6)) # Reduce if your hardware does not have enough memory



    #cm.show()

    cm_numpy = 10.*np.log10(cm._path_gain.numpy())
    data_2d = cm_numpy.reshape(-1, cm_numpy.shape[-1]) 
    #print(cm_numpy)
    np.savetxt('cm_numpy.txt',data_2d)



    # 定义一个列表来存储符合条件的坐标行
    valid_lines = []
    values_list=[]
    data_radio = np.load(f'./radio_maps/radio_case{case}.npy')

    radio_map_height = data_radio.shape[1]
    radio_map_width = data_radio.shape[2]

    # 将 3D 数组展平为 2D 数组
    data_2d = data_radio.reshape(-1, data_radio.shape[-1])  # 或者 data.flatten().reshape(-1, data.shape[-1])

    # Load the matrix data from the file
    data = data_2d
    data = np.nan_to_num(data, nan=-500)

    # 打开文件并读取每一行
    case =0
    if case == 0:
        with open("./coordinates5.txt", "r") as file:
            for line in file:
                # 每行读取x, y
                x, y = map(float, line.split())  # 将每行的x和y值转换为浮动数
                x1=(x -555.375)/2.165
                y1=(y-202.703)/1.984

                x1=(x -555.375)/2.165
                y1=(y-202.703)/1.984

                radiox=(x1+600)/647*66
                radioy=(y1+34.4)/251.5*26
                cellindex_x=int(radiox)
                cellindex_y=int(radioy)
                x= cellindex_x
                y= cellindex_y
                
                
                # 判断条件是否满足
                #if not (x > 59 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 20 or x < 47 or y < 2 or x==57):
                if not (x > 64 or(x < 59 and y > 8 and x > 48 and y < 19) or y > 20 or x < 47 or y < 2 or x==57):
                    valid_lines.append((x, y))

        # 随机选择 10 条符合条件的坐标
        selected_lines = random.sample(valid_lines, anchor_num) if len(valid_lines) >= anchor_num else valid_lines
        print("selected_lines:", selected_lines)

        # 打印选中的坐标
        sum_path_loss = 0
        count = 0
        data = np.load(f'./radio_maps/radio_case{case}.npy')
        F=[]
        
        if not recurrence:
            for x, y in selected_lines:
                print(x,y)
                cellindex_x = int(x)
                cellindex_y = int(y)
                path_loss = data[0, cellindex_y, cellindex_x]
                sum_path_loss += path_loss
                count += 1
                f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
                F.append(f)
                # if cellindex_x ==60:
                #     cellindex_x = cellindex_x +1 
                # if cellindex_x ==50 and cellindex_y== 7:
                #     cellindex_x = cellindex_x +1 
                # if cellindex_x ==49 and cellindex_y< 7:
                #     cellindex_x = cellindex_x +1 
                # if cellindex_x <58 and cellindex_y == 7:
                #     cellindex_y = cellindex_y -2 
                # if cellindex_x ==58 and cellindex_y < 7:
                #     cellindex_x = cellindex_x +1 
                # if cellindex_x ==47:
                #     cellindex_x = cellindex_x +1 
                # if cellindex_x >48 and cellindex_x<59 and cellindex_y ==19:
                #     cellindex_y = cellindex_y +1.2 
                # if cellindex_x >48 and cellindex_x<59 and cellindex_y ==20:
                #     cellindex_y = cellindex_y +0.2 


                # # input a robot pose (anchor point) at irsim 
                # irsim_robot_pos = [0, 0]
                # radio_robot_pos = [0,0]



                # # convert robot pose in irsim to radio map for futher query
                # translation = [13.9, -12.6]


                # radio_robot_pos[0] = cellindex_x-66/2
                # radio_robot_pos[1] = cellindex_y-26/2
                # irsim_robot_pos[0] = radio_robot_pos[0]-translation[0]
                # irsim_robot_pos[1] = radio_robot_pos[1]-translation[1]
                values_list.append([cellindex_x,cellindex_y])
            print("Sum of Path Losses:", sum_path_loss/count)
            save_v_all_a("Anchor_compare.txt",sum_path_loss/count, "Voronoi")
            # Plot the original data as a heatmap
            output_path = f'./radio_maps/F{index}.txt'
            save_lines_to_txt1(output_path,F)
        else:
            #IoT 15
            original_lines=[[59, 12], [51, 8], [52, 5], [59, 11], [59, 8], [48, 4], [54, 20], [52, 4], [56, 3], [59, 2]]
            #IoT 4
            original_lines=[[47, 12], [51, 6], [54, 3], [50, 5], [47, 20], [52, 4], [48, 15], [59, 4], [47, 15], [50, 8]]


            for point in original_lines:
                x, y = point  # 每个 point 是一个包含两个浮点数的列表，如 [1.1, 14.6]
                values_list.append((x, y))
        original_lines = [f"[{x}, {y}]" for x, y in values_list]
        converted_lines = [f"[{convert_coordinates(x, y)[0]}, {convert_coordinates(x, y)[1]}]" for x, y in selected_lines]

# 输出结果
        print(f"original_lines=[{', '.join(original_lines)}]")
        print(f"anchor_list=[{', '.join(converted_lines)}]")
        

   
    


    values_array = np.array(values_list)
    #plt.imshow(data, cmap='viridis', origin='lower')
    #plt.colorbar(label="Path gain (dB)")
    plt.title("Electromagnetic Map")
    plt.xlabel("Cell index (X-axis)")
    plt.ylabel("Cell index (Y-axis)")

    # Plot the cluster centers
    # Print the cluster center coordinates
    values_array[:, 0] *= 2  # Multiply the x-coordinates by 2
    values_array[:, 1] *= 2  # Multiply the y-coordinates by 2

    cm_numpy = 20.*np.log10(cm._path_gain.numpy())
    data_2d = cm_numpy.reshape(-1, cm_numpy.shape[-1]) 
    #data_2d = data_radio.reshape(-1, data_radio.shape[-1])
    data_2d = np.nan_to_num(data_2d, neginf=-200)
    data =data_2d
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label="Path gain (dB)")   

    plt.scatter(values_array[:, 0], values_array[:, 1], s=15, color='black', marker='*', label='Cluster centers')
    plt.scatter((iot_position[0]+32)*2, (iot_position[1]+12.6)*2, s=30, c='r', marker='o', label='Sensor')
    #plt.scatter(48*2, 7*2, s=10, c='b', marker='s', label='Sensor')
    #plt.legend()

    current_time=datetime.datetime.now()
    current_time= current_time.time()
    plt.savefig(f'../plot/{current_time}.pdf', bbox_inches='tight')
    # 显示图表
    plt.show()
    plt.close()

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def query_radio(index, recurrence=True):
    index1=[]
    index1.append(index)
    case = index
    data = np.load(f"./radio_maps/radio_case{case}.npy")
    radio_map_height = data.shape[1]
    radio_map_width = data.shape[2]
    location = get_location_from_txt("../iot1.txt",index1)
    x= location[0][0]
    y= location[0][1]
    irsim_robot =[x,y]
    translation = [13.9, -12.6]
    radio_robot_pos = [irsim_robot[0] + translation[0], irsim_robot[1] + translation[1]]

    # convert radio map position to cell index ([0,0] at the upper left corner)
    cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
    cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
    iot_orientation = [0, 1.57, 1.57]
    iot_position = [cellindex_x-32, cellindex_y-12.6, 3]

   


    scene = load_scene('INVS2/INVS.xml') # load mesh

    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")


    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="cross")

    # Add transmitter instance to scene
    tx = Transmitter(name="tx",position=iot_position, orientation=iot_orientation, color=[1, 0, 0],)


    scene.add(tx)
    scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)


    cm = scene.coverage_map(max_depth=10,
                            diffraction=True, # Disable to see the effects of diffraction
                            cm_cell_size=(0.5,0.5), # Grid size of coverage map cells in m
                            combining_vec=None,
                            precoding_vec=None,
                            num_samples=int(5e6)) # Reduce if your hardware does not have enough memory


    #print(cm)
    #cm.show()

        # 打开并读取coordinates.txt文件
    transformed_lines = []
    intensity_values = []
    with open("coordinates5.txt", "r") as file:
        for line in file:
            # 每行读取x, y
            x, y = map(float, line.split())  # 将每行的x和y值转换为浮动数
            

            x1=(x -555.375)/2.165
            y1=(y-202.703)/1.984

            x1=(x -555.375)/2.165
            y1=(y-202.703)/1.984

            radiox=(x1+600)/647*66
            radioy=(y1+34.4)/251.5*26
            cellindex_x=int(radiox)
            cellindex_y=int(radioy) 
            if cellindex_x ==57:
                cellindex_x = cellindex_x +1
            # if cellindex_x ==50 and cellindex_y== 7:
            #     cellindex_x = cellindex_x +1 
            # if cellindex_x ==49 and cellindex_y< 7:
            #     cellindex_x = cellindex_x +1 
            # if cellindex_x <58 and cellindex_y == 7:
            #     cellindex_y = cellindex_y -2 
            # if cellindex_x ==58 and cellindex_y < 7:
            #     cellindex_x = cellindex_x +1   
            # if cellindex_x >48 and cellindex_x<59 and cellindex_y ==19:
            #     cellindex_y = cellindex_y +1
            # if cellindex_x ==47:
            #     cellindex_x = cellindex_x +1   
            # if cellindex_x== 49 and cellindex_y<8:
            #     cellindex_x = cellindex_x +1  
            x= cellindex_x
            y= cellindex_y
            
            transformed_lines.append(f"{cellindex_x:.3f} {cellindex_y:.3f}\n")

            data = np.load(f"./radio_maps/radio_case{case}.npy")  # 加载数据

            if not (x > 63 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 23 or x < 45 or y < 1): # x better >46
                path_loss = data[0, cellindex_y, cellindex_x]  # 获取路径损耗强度值
                point1 = [48,7]
                point2 =[x,y]
                distance = euclidean_distance(point1, point2)
                beta = 0.5
                # 将坐标和路径损耗存储到列表中
                intensity_values.append(((1-beta)*np.exp((path_loss+75)/10)-distance*beta,path_loss, cellindex_x, cellindex_y))
            # 打印转换后的结果
            print(f"x1: {cellindex_x}, y1: {cellindex_y}")






    # Sort the intensity values in descending order based on the first element (intensity)
    intensity_values.sort(reverse=True, key=lambda x: x[0])

    # Initialize a set to track unique points
    unique_points = set()
    top_intensity = []
    selected_points = []
    selected_points1 = []


    # Iterate through the sorted intensity values
    # First, select 5 points from intensity range [-50, -60)




    ####################################################Exp1
    
    if not recurrence:
        for point in intensity_values:
            if -55 < point[1] <= -40:
                selected_points1.append(point)

        # If there are enough points, randomly select 2
        selected_points1 = random.sample(selected_points1, 3)
        selected_points.extend(selected_points1)  # Use all points if there are fewer than 2

        for point in intensity_values:
            if -59 < point[1] <= -48:
                selected_points1.append(point)

        # If there are enough points, randomly select 2
        selected_points1 = random.sample(selected_points1, 5)
        selected_points.extend(selected_points1)  # Use all points if there are fewer than 2


        for point in intensity_values:  #-63
            if -70 < point[1] <= -55:
                selected_points1.append(point)

        # If there are enough points, randomly select 2
        selected_points1 = random.sample(selected_points1, 3)
        selected_points.extend(selected_points1)  # Use all points if there are fewer than 2

    else:
        #RadioNav Single IoT
        
        
        
        #Multi IoT [19,16,11] RadioNav
        original_lines = [
        [59, 15],
        [58, 12],
        [59, 8],
        [60, 8],
        [58, 6],
        [52, 19],
        [51, 20],
        [53, 19],
        [54, 19],
        [47, 20],
        [50, 4],
        [52, 7],
        [48, 8],
        [47, 7],
        [54, 7]
    ]
        ##Multi IoT [19,16,11] DBscan 
        original_list= [[25.9, 13.4], [21.4, 1.4], [39.8, 0.2], [52.3, 3.1], [61.6, 2.9], [46.2, 0.6], [18.5, 5.0], [62.2, 6.2], [47.7, 7.7], [61.9, 8.4], [53.1, 13.3]]
        original_list= [[28.4, 10.4], [53.9, 8.4], [53.4, 11.3], [48.7, 15.5], [29.7, 14.3], [53.7, 15.1], [53.5, 14.0], [48.9, 17.9], [55.3, 16.3], [33.4, 18.8], [36.0, 19.4]]
        original_list= [[29.1, 13.8], [30.3, 1.2], [53.1, 1.5], [30.0, 0.8], [35.6, 0.9], [33.5, 1.0], [34.3, 5.0], [49.0, 4.6], [25.7, 3.3], [52.0, 3.9], [24.0, 5.0]]
        
        ##Multi IoT [19,16,11] DBscan  Selected Points
        original_list =[[53.1, 1.5],[49.0, 4.6],[52.0, 3.9],[61.6, 2.9],[62.2, 6.2],[61.9, 8.4],[48, 15.5],[48.9, 18.9],[47.2, 0.6],[53.1, 13.3],[55.3, 16.3],[53.5, 14.0],[53.9, 8.4], [53.4, 11.3]]
        
        ##Multi IoT [19,16,11] Voronoi
        original_lines=[[53, 6], [52, 2], [50, 8], [59, 16], [56, 3], [51, 7], [48, 18], [50, 5], [52, 8], [48, 2]]
        original_lines=[[48, 4], [54, 6], [51, 20], [53, 2], [53, 20], [50, 4], [53, 20], [52, 7], [55, 8], [50, 7]]
        original_lines=[[51, 7], [55, 20], [53, 19], [50, 7], [50, 20], [48, 6], [48, 19], [51, 3], [59, 16], [51, 5]]

        ##Multi IoT [19,16,11] Voronoi  Selected Points
        original_lines = [
        [48, 18],
        [51, 7],
        [52, 8],
        [50, 8],
        [59, 16],
        [51, 20],
        [53, 20],
        [64, 8],
        [50, 7],
        [48, 4],
        [51, 3],
        [50, 20],
        [48, 19],
        [59, 16],
        [51, 5],
    ]
        #IoT 4
        original_lines=[[59,11], [59,6], [53,8], [60,10], [59,4], [58,3], [60,8], [60,1], [49,1], [53,4]]

        #IoT 15
        original_lines=[[48,16], [46,16], [52,20], [51,19], [48,13], [48,20],  [48,16], [57,19],  [55,19]]


        original_lines= [[48, 16], [46, 16], [51, 20.5], [48, 20], [48, 13], [50, 1], [48, 3], [53, 1], [51, 3], [56, 2], [63, 6], [62, 5], [55, 5], [59, 6], [59, 10]]

        #Voronoi Anchors #2
        original_lines= [[58, 13], [51, 3], [48, 11], [51, 20], [60, 19], [61, 9], [59, 15], [62, 16], [63, 12], [48, 5], [63, 20], [61, 2], [48, 10], [61, 15], [50, 5]]

        original_lines = [
        [59, 15],
        [58, 12],
        [59, 8],
        [61, 8],
        [58, 6],
        [52, 19],
        [51, 20],
        [53, 19],
        [54, 19],
        [47, 20],
        [50, 4],
        [52, 6],
        [48, 8],
        [47, 7],
        [54, 8]
    ]
        original_lines= [[48, 16], [46, 16], [59, 21], [47, 20], [48, 13], [50, 1], [48, 3], [46, 1], [51, 3], [56, 2], [63, 6], [62, 5], [55, 5], [59, 6], [61, 16]]

        # #IoT 6
        # original_lines=[[50,1], [48,3], [48,6], [53,1], [50,1], [51,3], [56,2], [53,5]]

        # #IoT 23
        # original_lines=[[63,6], [59,8], [62,5], [55,7], [58,6], [59,10]]

    #     #RadioNav Multi IoT [19,16,11] Voronoi
    #     original_lines = [
    #     [59, 15],
    #     [58, 12],
    #     [59, 8],
    #     [60, 8],
    #     [58, 6],
    #     [52, 19],
    #     [51, 20],
    #     [53, 19],
    #     [54, 19],
    #     [47, 20],
    #     [50, 4],
    #     [52, 7],
    #     [48, 8],
    #     [47, 7],
    #     [54, 7]
    # ]

        intensity_values = []  # 假设这是一个事先定义好的列表

        for point in original_lines:
            x, y = point  # 每个 point 是一个包含两个浮点数的列表，如 [1.1, 14.6]
            selected_points.append((0, 0, x, y))


    ####################################################3


    # for point in intensity_values:
    #     if -59 < point[1] <=55:
    #         selected_points.append(point)

    # # If there are enough points, randomly select 2
    # if len(selected_points) >= 7:
    #     selected_points = random.sample(selected_points, 5)
    # else:
    #     selected_points = selected_points  # Use all points if there are fewer than 2



####################################################


    # for point in intensity_values:
    #     if -50 < point[1] <= -42 and len(selected_points) < 2:
    #         if point not in unique_points:
    #             selected_points.append(point)
    #             unique_points.add(point)

    # # Then, select 3 points from intensity range [-60, -70)
    # for point in intensity_values:
    #     if -59 < point[1] <=55  and len(selected_points) < 7:
    #         if point not in unique_points:
    #             selected_points.append(point)
    #             unique_points.add(point)
            
    # for point in intensity_values:
    #     if -63 < point[1] <= -58 and len(selected_points) < 10:
    #         if point not in unique_points:
    #             selected_points.append(point)
    #             unique_points.add(point)

####################################################Exp1  Recurrence


    







    # for point in intensity_values:
    #     if -50 < point[1] <= -30 and len(selected_points) < 2:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)

    # # Then, select 3 points from intensity range [-60, -70)
    # for point in intensity_values:
    #     if -60 < point[1] <= -55 and len(selected_points) < 4:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)
    # for point in intensity_values:
    #     if -70 < point[1] <= -65 and len(selected_points) < 6:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)

    # for point in intensity_values:
    #     if -80 < point[1] <= -75 and len(selected_points) < 8:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)

    # for point in intensity_values:
    #     if -85 < point[1] <= -80 and len(selected_points) < 10:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)




# First, select 5 points from intensity range [-50, -60)
    # for point in intensity_values:
    #     if -50 < point[1] <= -30 and len(selected_points) < 5:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)

    # # Then, select 3 points from intensity range [-60, -70)
    # for point in intensity_values:
    #     if -65 < point[1] <= -55 and len(selected_points) < 10:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)



    # Finally, select 2 points from intensity range [-70, -80)
    # for point in intensity_values:
    #     if -90 < point[0] <= -80 and len(selected_points) < 10:
    #         if point not in unique_points:
    #             selected_points.append(point)
    #             unique_points.add(point)

    # Ensure that we have exactly 10 points selected

    # if len(selected_points) == 10:
    #     top_intensity = selected_points


    # 打印最强的10个点及其路径损耗
    #calculate_carla_anchor(top_intensity,26,66,10,2)

    F=[]


    top_intensity=selected_points

    #ADD START POINT
    start_x=10
    start_y=2
    data_radio = np.load(f"./radio_maps/radio_case{case}.npy")
    
    
    # F.append(f"Start: x={start_x}, y={start_y}, Path Loss={path_loss}")

    sum_path_loss = 0
    for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1):
        plt.scatter(x*2, y*2,s=15, color='red', marker='x', label='Cluster centers')
        #path_loss=data[0, y, x]

        #path_loss =db_to_linear(path_loss)  # 将路径损耗从dB转换为线性值
        #path_loss = calculate_Fik(1, path_loss, 10, 10 ** (-6))  # 使用calculate_Fik函数计算Fik值
        f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
        F.append(f)
        sum_path_loss += path_loss
        print(f)
    output_path = f'./radio_maps/F{index}.txt'
    save_lines_to_txt1(output_path,F)

    original_lines = [f"[{x},{y}]" for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1)]
    converted_lines = [f"[{convert_coordinates(x, y)[0]}, {convert_coordinates(x, y)[1]}]" for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1)]

# 输出结果
    print(f"original_lines=[{', '.join(original_lines)}]")
    print(f"anchor_list=[{', '.join(converted_lines)}]")


    print("Sum of Path Losses:", sum_path_loss/10)
    save_v_all_a("Anchor_compare.txt",sum_path_loss/10,"Radionav")
    output_path ='index2radio.txt'
    with open(output_path, 'w') as f:
        for value in F:
            f.write(f"{value}\n")

    cm_numpy = 20.*np.log10(cm._path_gain.numpy())
    data_2d = cm_numpy.reshape(-1, cm_numpy.shape[-1]) 
    #data_2d = data_radio.reshape(-1, data_radio.shape[-1])
    data_2d = np.nan_to_num(data_2d, neginf=-200)
    data =data_2d
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label="Path gain (dB)")   

    plt.scatter((iot_position[0]+32)*2, (iot_position[1]+12.6)*2, s=30, c='r', marker='o', label='Sensor')
    #plt.scatter(48*2, 7*2, s=10, c='b', marker='s', label='Sensor')

    current_time=datetime.datetime.now()
    current_time= current_time.time()
    plt.savefig(f'../plot/{current_time}.pdf', bbox_inches='tight')
    # 显示图表
    plt.show()
    plt.close()


#index=[12,18,7]
#index=[2,14,17]
index=[3,1,4]
#index=[15,6,1]
index=[25,6,3]
#index=[11]
index=[4]
for i in index:
    query_radio(i, recurrence=False)
    #generate_Kmeans_improved(i,recurrence=False)
    #generate_DBSCAN(i)
    #select_random_voro(i,10, recurrence=False)


# generate_output2(index,15)
# deepseek_api()
# calculate_carla_anchor(26,66)

# generate_DBSCAN(index)
# select_random_voro(index,10)