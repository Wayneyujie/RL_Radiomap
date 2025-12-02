import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
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
        if not (x > 64 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2):
            count += 1
            plt.scatter(cellindex_x, cellindex_y, color='red', marker='x', label='Cluster centers')   
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




#Each IoT's F matrix must select at least 3 points
    # 输出格式化内容
    output_content = f"""
Task Description:
Select the {anchor_num} most suitable anchor points from the provided data (F1, F2, F3) for a robot to complete a signal collection task efficiently. The robot must:

Start and end at the point (x=48, y=7).

Each IoT's F matrix must select at least 3 points

When the number of IoTs is greater than 1, the Euclidean distance between the selected anchor points should not be less than 3.

Travel through the selected anchor points while avoiding obstacles.

Maximize the communication rate (higher path loss values indicate better communication rates).

Constraints:
Path Loss Priority: Select anchor points with a relatively higher loss values to ensure the best communication rates. But at the same time, we also try to balance the number of anchor points selected for each IoT to avoid some IoTs not having any points, resulting in too little data collection.

Obstacle Avoidance: Ensure the selected anchor points do not overlap with vertical or horizontal obstacle areas or intersect with obstacle line segments. For example, Rank 9: x=58, y=5, Path Loss=-70.39161682128906 is on the obstacle boundary (58,0)->(58,6).

Start and End Point: The robot must start and end at (x=48, y=7).

Output Format: Strictly adhere to the template format provided below. Do not include any additional text, explanations, or deviations.

Output Template:(Do not separate lines)(Strictly adhere to the template format provided below with out any other information)
F1: x=59, y=14, Path Loss=-53.60979461669922 
F1: x=51, y=3, Path Loss=-63.61632537841797
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
    anchor_list = ", ".join(formatted_centers)
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
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], color='red', marker='x', label='Cluster centers')
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

def select_random_voro(index,anchor_num):
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
                if not (x > 64 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2):
                    valid_lines.append((x, y))

        # 随机选择 10 条符合条件的坐标
        selected_lines = random.sample(valid_lines, anchor_num) if len(valid_lines) >= anchor_num else valid_lines

        # 打印选中的坐标
        sum_path_loss = 0
        count = 0
        data = np.load(f'./radio_maps/radio_case{case}.npy')
        F=[]
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
        converted_lines = [f"[{convert_coordinates(x, y)[0]}, {convert_coordinates(x, y)[1]}]" for x, y in selected_lines]

# 输出结果
        print(f"anchor_list=[{', '.join(converted_lines)}]")
        print("Sum of Path Losses:", sum_path_loss/count)
        save_v_all_a("Anchor_compare.txt",sum_path_loss/count, "Voronoi")
        # Plot the original data as a heatmap
        output_path = f'./radio_maps/F{index}.txt'
        save_lines_to_txt1(output_path,F)

   
    


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

    plt.scatter(values_array[:, 0], values_array[:, 1], s=20, color='red', marker='x', label='Cluster centers')
    plt.scatter((iot_position[0]+32)*2, (iot_position[1]+12.6)*2, s=30, c='r', marker='o', label='Sensor')
    #lt.scatter(48*2, 7*2, s=10, c='b', marker='s', label='Sensor')
    #plt.legend()

    current_time=datetime.datetime.now()
    current_time= current_time.time()
    plt.savefig(f'../plot/{current_time}.pdf', bbox_inches='tight')
    # 显示图表
    plt.show()
    plt.close()

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def query_radio(index, color = 'red', recurrence=False):
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
                            cm_cell_size=(1,1), # Grid size of coverage map cells in m
                            combining_vec=None,
                            precoding_vec=None,
                            num_samples=int(5e6)) # Reduce if your hardware does not have enough memory



    if  not recurrence:
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

                if not (x > 60 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 21 or x < 47 or y < 2): # x better >46
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
        for point in intensity_values:
            if -50 < point[1] <= -40:
                selected_points1.append(point)

        # If there are enough points, randomly select 2
        selected_points1 = random.sample(selected_points1, 2)
        selected_points.extend(selected_points1)  # Use all points if there are fewer than 2

        for point in intensity_values:
            if -59 < point[1] <= -55:
                selected_points1.append(point)

        # If there are enough points, randomly select 2
        selected_points1 = random.sample(selected_points1, 5)
        selected_points.extend(selected_points1)  # Use all points if there are fewer than 2


        for point in intensity_values:  #-63
            if -63 < point[1] <= -58:
                selected_points1.append(point)

        # If there are enough points, randomly select 2
        selected_points1 = random.sample(selected_points1, 3)
        selected_points.extend(selected_points1)  # Use all points if there are fewer than 2

        F=[]


        top_intensity=selected_points

        #ADD START POINT
        start_x=10
        start_y=2
        data_radio = np.load(f"./radio_maps/radio_case{case}.npy")
        
        path_loss=data[0, start_y, start_x]
        # F.append(f"Start: x={start_x}, y={start_y}, Path Loss={path_loss}")

        sum_path_loss = 0
        for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1):
            #plt.scatter(x*2, y*2,s=35, color=color, marker='x', label='Cluster centers')
            f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
            F.append(f)
            sum_path_loss += path_loss
            print(f)
        output_path = f'./radio_maps/F{index}.txt'
        save_lines_to_txt1(output_path,F)

        converted_lines = [f"[{convert_coordinates(x, y)[0]}, {convert_coordinates(x, y)[1]}]" for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1)]

    # 输出结果
        print(f"anchor_list=[{', '.join(converted_lines)}]")


        print("Sum of Path Losses:", sum_path_loss/10)
        save_v_all_a("Anchor_compare.txt",sum_path_loss/10,"Radionav")
        output_path ='index2radio.txt'
        with open(output_path, 'w') as f:
            for value in F:
                f.write(f"{value}\n")

    cm_numpy = 10.*np.log10(cm._path_gain.numpy())
    data_2d = cm_numpy.reshape(-1, cm_numpy.shape[-1]) 
    #data_2d = data_radio.reshape(-1, data_radio.shape[-1])
    data_2d = np.nan_to_num(data_2d, neginf=-200)
    data =data_2d
    plt.scatter((iot_position[0]+32)*2, (iot_position[1]+12.6)*2, s=30, c='r', marker='o', label='Sensor')
    return data   #for [28,7,16] #data+50

def rda_radio(index):
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
    path_loss = data[0, cellindex_y, cellindex_x]
    print(f"F{index}: x={cellindex_x}, y={cellindex_y}, Path Loss={path_loss}")




def view(data):
    print(data)
    np.savetxt("data.txt", data, fmt="%s")
    plt.imshow(data, cmap='viridis', origin='lower')
    plt.colorbar(label="Path gain (dB)")   


    values_list = []
    original_lines=[[47, 12], [51, 6], [54, 3], [50, 5], [47, 20], [52, 4], [48, 15], [59, 4], [47, 15], [50, 8]]
    
    

    original_lines =[[53.1, 1.5],[49.0, 4.6],[52.0, 3.9],[61.6, 2.9],[62.2, 6.2],[61.9, 8.4],[48, 15.5],[48.9, 18.9]]

    #DBSCAN Anchors
    original_lines =[[53.1, 1.5],[49.0, 4.6],[52.0, 3.9],[61.6, 2.9],[62.2, 6.2],[61.9, 8.4],[48, 15.5],[48.9, 18.9],[47.2, 0.6],[53.1, 13.3],[55.3, 16.3],[53.5, 14.0],[53.9, 8.4], [53.4, 11.3]]

    original_lines = [
    [48, 18],
    [51, 7],
    [52, 8],
    [50, 8],
    [59, 16],
    [51, 20],
    [53, 20],
    [55, 8],
    [50, 7],
    [48, 4],
    [51, 3],
    [50, 20],
    [48, 19],
    [59, 16],
    [51, 5],
]   
    ##Voronoi Anchors
    original_lines = [
    [60, 4],
    [64, 19],
    [62, 15],
    [47, 16],
    [55, 8],
    [47, 12],
    [59, 14],
    [58, 20],
    [63, 7],
    [48, 8],
    [50, 3],
    [48, 14],
    [48, 5],
    [48, 20],
    [60, 16],
]   
    original_lines= [[48, 16], [46, 16], [59, 21], [47, 20], [48, 13], [50, 1], [48, 3], [46, 1], [51, 3], [56, 2], [63, 6], [62, 5], [55, 5], [59, 6], [61, 16]]

    original_lines=[ ]

    #Radionav Anchors
    # original_lines = [
    #     [59, 15],
    #     [58, 12],
    #     [59, 8],
    #     [61, 8],
    #     [58, 6],
    #     [52, 19],
    #     [51, 20],
    #     [53, 19],
    #     [54, 19],
    #     [47, 20],
    #     [50, 4],
    #     [52, 6],
    #     [48, 8],
    #     [47, 7],
    #     [54, 8]
    # ]

    #Voronoi Anchors #2
    #original_lines= [[58, 13], [51, 3], [48, 11], [51, 20], [60, 19], [61, 9], [59, 15], [62, 16], [63, 12], [48, 5], [63, 20], [61, 2], [48, 10], [61, 15], [50, 5]]
   
    #Kmeans Anchors #2
    #original_lines= [[58.903, 9.837], [60.512, 9.981], [60.682, 9.530], [59.470, 10.087],  [57.534, 9.971], [50.187, 19.083],  [50.283, 18.750], [48.641, 17.626], [50.984, 16.419], [50.240, 18.698], [50.099, 19.033], [49.894, 17.911],[50.018, 7.941],[50.429, 8.598], [49.371, 8.730],[51.320, 9.322], [51.375, 9.166]]

    for point in original_lines:
        x, y = point  # 每个 point 是一个包含两个浮点数的列表，如 [1.1, 14.6]
        values_list.append((x, y))
        #plt.scatter(x*2, y*2,s=15, color='r', marker='x', label='Cluster centers') # 'x' RadioNav '^' DBSCAN
        #plt.scatter(x*2, y*2,s=15, color='blue', marker='^', label='Cluster centers')  
        plt.scatter(x*2, y*2,s=15, color='black', marker='*', label='Cluster centers')
        
    #plt.scatter(48*2, 7*2, s=10, c='b', marker='s', label='Sensor')

    current_time=datetime.datetime.now()
    current_time= current_time.time()
    #plt.savefig(f'../plot/{current_time}.pdf', bbox_inches='tight')
    # 显示图表
    plt.show()
    plt.close()
    

def fusion():
    #index=[12,18,7]
    #index=[2,14,17]
    #index=[8]

    #two iot better use index=[1,16]
    #three iot better use index=[19,16,11]
    recur = True

    data1=query_radio(3,'blue', recurrence=recur)  # recurrence=True to avoid reloading data

    #data2=query_radio(16,'green', recurrence=recur)  # recurrence=True to avoid reloading data

    


    #data3=query_radio(7,'red', recurrence=recur)  # recurrence=True to avoid reloading data

    
    # data2=data2+50
    # data1=data1+50
    # data3= data3+50    


    def compute_weight(data):
        return 1 / (np.abs(data) + 1e-5)  # 加上一个小值避免除以零

    # 计算权重
    weight1 = compute_weight(data1)
    # weight2 = compute_weight(data2)
    # weight3 = compute_weight(data3)


    

    # 使用加权平均进行计算
    #data = (data1 * weight1 + data2 * weight2) / (weight1 + weight2)
    data = data1
    #data = (data1 * weight1 + data2 * weight2+data3 * weight3) / (weight1 + weight2+weight3)
    #data3= (data1 + data2)/2


    #for [28,7,16]
    
    data[3:6, 51:56] -= 22
    # data[17:42, 115:121] -= 22
    # data[14:17, 115:121] -= 15
    # data[10:14, 115:121] -= 13
    # data[7:10, 115:121] -= 8
    # data[0:6, 115:121] -= 4
    # data=data-50
    view(data)

fusion()
#rda_radio(15)


# for i in index:
#     query_radio(i)
    #generate_DBSCAN(i)
    #select_random_voro(i,10)
#generate_output2(index,10)
#deepseek_api()
#calculate_carla_anchor(26,66)

# generate_DBSCAN(index)
# select_random_voro(index,10)