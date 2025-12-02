import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
# Import Sionna
try:
    import sionna
except ImportError as e:
    # Install Sionna if package is not already installed
    import os
    os.system("pip install sionna")
    import sionna

import matplotlib.pyplot as plt
import numpy as np
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, PlanarArray, Camera

## configuration:
save_cm_numpy = True
cm_cell_size = (1, 1)   # covermap cell size
camera_position = [22,0,40]
camera_look_at = [22,0,0]

# These cases are the red points in rm_1215
# case = 5
# case = 6
# case = 7
case = 11

if case == 0:
    # case map center 
    iot_position = [0, 0, 3]
    iot_orientation = [0, 0, 0]

elif case == 1:
    # case mpcom
    iot_position = [26.2, -11.5, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 2:
    # case open
    iot_position = [-3, -3, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 3:
    # case room
    iot_position = [20.5, -9.0, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 4:
    # case corridor
    iot_position = [26.1, -5.2, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 5:
    # case T junction near conference door
    iot_position = [15.9, -5.1, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 6:
    # case T junction near F1015 door
    iot_position = [15.9, -12.6, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 7:
    # case corridor corner
    iot_position = [26.2, -12.6, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 8:
    # case F1015 corner left
    iot_position = [26.2, -5.1, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 9:
    # case corridor middle
    iot_position = [15+13.9, 19-12.6, 2]
    iot_orientation = [0, 1.57, 1.57]

elif case == 10:
    # case conference room
    iot_position = [28.9, -7.6, 3]
    iot_orientation = [0, 0, 0]

elif case == 11:
    # case conference room
    iot_position = [18, 6.4, 3]
    iot_orientation = [0, 1.57, 1.57]


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
tx = Transmitter(name="tx",
                 position=iot_position, orientation=iot_orientation, color=[1, 0, 0],)


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


def calculate_carla_anchor(top_intensity,radio_map_height,radio_map_width,start_x,start_y):
    values_list = []  # 用于存储每个聚类中心的路径损失值
    goal_data = []

    print('The radio map has a size of:', radio_map_height, 'x', radio_map_width)

    values_list.append(f"{start_x:.3f} {start_y:.3f}") 

    start_x =start_x*10 - 136.5
    start_y =start_y*10 -38
    goal_data.append({
                'x': start_x,
                'y': start_y,
                'yaw': 96,  # 假设 yaw 的变化为递增，具体情况可根据需要修改
                'comment': f'MOVE: Goal1'  # 每个目标的 comment 都不一样
                
            })
    for i, (path_loss, x, y) in enumerate(top_intensity, 1):
        cellindex_x=int(x)
        cellindex_y=int(y)

        # input a robot pose (anchor point) at irsim 
        irsim_robot_pos = [0, 0]
        radio_robot_pos = [0,0]



        # convert robot pose in irsim to radio map for futher query
        translation = [13.9, -12.6]


        radio_robot_pos[0] = cellindex_x-radio_map_width/2
        radio_robot_pos[1] = cellindex_y-radio_map_height/2
        irsim_robot_pos[0] = radio_robot_pos[0]-translation[0]
        irsim_robot_pos[1] = radio_robot_pos[1]-translation[1]

        values_list.append(f"{irsim_robot_pos[0]:.3f} {irsim_robot_pos[1]:.3f}") 


        x= irsim_robot_pos[0]*10 - 136.5
        y= irsim_robot_pos[1] *10 -38
        
    
        goal_data.append({
                'x': x,
                'y': y,
                'yaw': 96,  # 假设 yaw 的变化为递增，具体情况可根据需要修改
                'comment': f'MOVE: Goal1'  # 每个目标的 comment 都不一样
                
            })
            
    output_path ='irsim_anchor.txt'
    with open(output_path, 'w') as f:
        for value in values_list:
            f.write(f"{value}\n")


    output_path ='carla_anchor.txt'
    with open(output_path, 'w') as f:
        for i, goal in enumerate(goal_data):
            if i != len(goal_data) - 1:  # 如果不是最后一个目标，末尾加逗号
                f.write(f"{goal},\n")
            else:  # 最后一个目标，不加逗号
                f.write(f"{goal}\n")



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
        x= cellindex_x
        y= cellindex_y
        
        transformed_lines.append(f"{cellindex_x:.3f} {cellindex_y:.3f}\n")

        data = np.load("./radio_maps/radio_case16.npy")  # 加载数据

        if not (x > 60.6 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 20.32 or x < 44.55 or y < 2):
            path_loss = data[0, cellindex_y, cellindex_x]  # 获取路径损耗强度值

            # 将坐标和路径损耗存储到列表中
            intensity_values.append((path_loss, cellindex_x, cellindex_y))
        # 打印转换后的结果
        print(f"x1: {cellindex_x}, y1: {cellindex_y}")






# Sort the intensity values in descending order based on the first element (intensity)
intensity_values.sort(reverse=True, key=lambda x: x[0])

# Initialize a set to track unique points
unique_points = set()
top_intensity = []

# Iterate through the sorted intensity values
for point in intensity_values:
    # If the point is not already in the set, add it to the result list and the set
    if point not in unique_points:
        top_intensity.append(point)
        unique_points.add(point)
    
    # If we have 15 unique points, stop the iteration
    if len(top_intensity) == 15:
        break


# 打印最强的10个点及其路径损耗
calculate_carla_anchor(top_intensity,26,66,10,2)

F=[]



#ADD START POINT
start_x=10
start_y=2
data = np.load("./radio_maps/radio_invsdoor.npy")
path_loss=data[0, start_y, start_x]
F.append(f"Start: x={start_x}, y={start_y}, Path Loss={path_loss}")


for i, (path_loss, x, y) in enumerate(top_intensity, 1):
    plt.scatter(x, y, color='red', s=50, label='标注点')
    f=f"Rank {i}: x={x}, y={y}, Path Loss={path_loss}"
    F.append(f)
    print(f)
output_path ='index2radio.txt'
with open(output_path, 'w') as f:
    for value in F:
        f.write(f"{value}\n")

plt.show()