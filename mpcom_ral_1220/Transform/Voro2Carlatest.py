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
import random
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

def save_lines_to_txt(output_path,lines):
    with open(output_path, 'w') as f:
        for value in lines:
            f.write(f"{value}\n")

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def query_radio(index):
    # 打开并读取coordinates.txt文件
    transformed_lines = []
    intensity_values = []
    with open("./Transform/coordinates5.txt", "r") as file:
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

            data = np.load(f"./Transform/radio_maps/radio_case{index}.npy")  # 加载数据

            if not (x > 64 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2):
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

    # Select points based on intensity ranges
    selected_points = []

    # First, select 5 points from intensity range [-50, -60)
    for point in intensity_values:
        if -50 < point[1] <= -30 and len(selected_points) < 2:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)

    # Then, select 3 points from intensity range [-60, -70)
    for point in intensity_values:
        if -60 < point[1] <= -55 and len(selected_points) < 4:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)
    for point in intensity_values:
        if -70 < point[1] <= -65 and len(selected_points) < 6:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)

    for point in intensity_values:
        if -80 < point[1] <= -75 and len(selected_points) < 8:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)

    for point in intensity_values:
        if -85 < point[1] <= -80 and len(selected_points) < 10:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)

    # for point in intensity_values:
    #     if -50 < point[1] <= -30 and len(selected_points) < 3:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)

    # # Then, select 3 points from intensity range [-60, -70)
    # for point in intensity_values:
    #     if -60 < point[1] <= -55 and len(selected_points) < 6:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)
    # for point in intensity_values:
    #     if -70 < point[1] <= -65 and len(selected_points) < 9:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)

    # for point in intensity_values:
    #     if -80 < point[1] <= -75 and len(selected_points) < 12:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)

    # for point in intensity_values:
    #     if -85 < point[1] <= -80 and len(selected_points) < 15:
    #         if point not in unique_points:
    #             print(point)
    #             selected_points.append(point)
    #             unique_points.add(point)
            
    # for point in intensity_values:
    #     if -90 < point[0] <= -75 and len(selected_points) < 15:
    #         if point not in unique_points:
    #             selected_points.append(point)
    #             unique_points.add(point)

    # Finally, select 2 points from intensity range [-70, -80)
    # for point in intensity_values:
    #     if -90 < point[0] <= -80 and len(selected_points) < 10:
    #         if point not in unique_points:
    #             selected_points.append(point)
    #             unique_points.add(point)

    # Ensure that we have exactly 10 points selected
    top_intensity = selected_points

    F=[]


    for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1):
        #plt.scatter(x, y, color='red', s=50, label='标注点')
        f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
        F.append(f)
        print(f)

    output_path = f'./Transform/radio_maps/F{index}.txt'
    save_lines_to_txt(output_path,F)

    #plt.show()

# query_radio("invsdoor")
# query_radio("Tjunc")
# query_radio("toilet")


def calculate_Fik(Bk, RMk, pk, sigma_squared):
    # 计算公式中的括号部分
    bracket_part = 1 + (RMk * pk) / sigma_squared
    # 使用math.log函数计算以2为底的对数
    log_part = math.log(bracket_part, 2)
    # 最终结果
    Fik = Bk * log_part
    return Fik

def query_radio913(index):
    # 打开并读取coordinates.txt文件
    transformed_lines = []
    intensity_values = []
    with open("./Transform/coordinates5.txt", "r") as file:
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

            data = np.load(f"./Transform/radio_maps/radio_case{index}.npy")  # 加载数据

            if not (x > 63 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 23 or x < 46 or y < 1): #exclude invs room "or (x>48 and x<59 and y<7)""
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

    # Select points based on intensity ranges
    selected_points = []
    selected_points1 = []


    # Iterate through the sorted intensity values
    # First, select 5 points from intensity range [-50, -60)
    for point in intensity_values:
        if -55 < point[1] <= -40 and point not in selected_points and point not in selected_points1:
            selected_points1.append(point)

    # 如果有足够的点，随机选择3个
    if len(selected_points1) >= 3:
        selected_points1 = random.sample(selected_points1, 3)
    else:
        selected_points1 = selected_points1  # 如果少于3个，使用所有点

    selected_points.extend(selected_points1)  # 添加选中的点到selected_points中

    # 清空selected_points1，为下一个条件做准备
    selected_points1 = []

    # 第二个条件
    for point in intensity_values:
        if -59 < point[1] <= -48 and point not in selected_points and point not in selected_points1:
            selected_points1.append(point)

    # 如果有足够的点，随机选择4个
    if len(selected_points1) >= 4:
        selected_points1 = random.sample(selected_points1, 4)
    else:
        selected_points1 = selected_points1  # 如果少于4个，使用所有点

    selected_points.extend(selected_points1)  # 添加选中的点到selected_points中

    # 清空selected_points1，为下一个条件做准备
    selected_points1 = []

    # 第三个条件
    for point in intensity_values:
        if -77 < point[1] <= -55 and point not in selected_points and point not in selected_points1:
            selected_points1.append(point)

    # 如果有足够的点，随机选择3个
    if len(selected_points1) >= 3:
        selected_points1 = random.sample(selected_points1, 3)
    else:
        selected_points1 = selected_points1  # 如果少于3个，使用所有点

    selected_points.extend(selected_points1)  # 添加选中的点到selected_points中
    
            
    
    top_intensity = selected_points

    F=[]


    for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1):
        #plt.scatter(x, y, color='red', s=50, label='标注点')
        path_loss =db_to_linear(path_loss)  # 将路径损耗从dB转换为线性值
        path_loss = calculate_Fik(1, path_loss, 10, 10 ** (-6))  # 使用calculate_Fik函数计算Fik值
        f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
        F.append(f)
        print(f)

    output_path = f'./Transform/radio_maps/F{index}.txt'
    save_lines_to_txt(output_path,F)

def query_radio416(index):
    # 打开并读取coordinates.txt文件
    transformed_lines = []
    intensity_values = []
    with open("./Transform/coordinates5.txt", "r") as file:
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

            data = np.load(f"./Transform/radio_maps/radio_case{index}.npy")  # 加载数据

            if not (x > 59 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2 ): #exclude invs room "or (x>48 and x<59 and y<7)""
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

    # Select points based on intensity ranges
    selected_points = []
    selected_points1 = []


    # Iterate through the sorted intensity values
    # First, select 5 points from intensity range [-50, -60)
    for point in intensity_values:
        if -50 < point[1] <= -40 and point not in selected_points and point not in selected_points1:
            selected_points1.append(point)

    # 如果有足够的点，随机选择3个
    if len(selected_points1) >= 3:
        selected_points1 = random.sample(selected_points1, 3)
    else:
        selected_points1 = selected_points1  # 如果少于3个，使用所有点

    selected_points.extend(selected_points1)  # 添加选中的点到selected_points中

    # 清空selected_points1，为下一个条件做准备
    selected_points1 = []

    # 第二个条件
    for point in intensity_values:
        if -59 < point[1] <= -55 and point not in selected_points and point not in selected_points1:
            selected_points1.append(point)

    # 如果有足够的点，随机选择4个
    if len(selected_points1) >= 4:
        selected_points1 = random.sample(selected_points1, 4)
    else:
        selected_points1 = selected_points1  # 如果少于4个，使用所有点

    selected_points.extend(selected_points1)  # 添加选中的点到selected_points中

    # 清空selected_points1，为下一个条件做准备
    selected_points1 = []

    # 第三个条件
    for point in intensity_values:
        if -63 < point[1] <= -58 and point not in selected_points and point not in selected_points1:
            selected_points1.append(point)

    # 如果有足够的点，随机选择3个
    if len(selected_points1) >= 3:
        selected_points1 = random.sample(selected_points1, 3)
    else:
        selected_points1 = selected_points1  # 如果少于3个，使用所有点

    selected_points.extend(selected_points1)  # 添加选中的点到selected_points中
    
            
    
    top_intensity = selected_points

    F=[]


    for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1):
        #plt.scatter(x, y, color='red', s=50, label='标注点')
        path_loss =db_to_linear(path_loss)  # 将路径损耗从dB转换为线性值
        path_loss = calculate_Fik(1, path_loss, 10, 10 ** (-6))  # 使用calculate_Fik函数计算Fik值
        f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
        F.append(f)
        print(f)

    output_path = f'./Transform/radio_maps/F{index}.txt'
    save_lines_to_txt(output_path,F)



def query_radio1(index):
    # 打开并读取coordinates.txt文件
    transformed_lines = []
    intensity_values = []
    with open("./Transform/coordinates5.txt", "r") as file:
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

            data = np.load(f"./Transform/radio_maps/radio_case{index}.npy")  # 加载数据

            if not (x > 64 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2):
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

    # Select points based on intensity ranges
    selected_points = []

    # First, select 5 points from intensity range [-50, -60)
    for point in intensity_values:
        if -50 < point[1] <= -30 and len(selected_points) < 5:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)

    # Then, select 3 points from intensity range [-60, -70)
    for point in intensity_values:
        if -65 < point[1] <= -55 and len(selected_points) < 10:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)
    
            
    
    top_intensity = selected_points

    F=[]


    for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1):
        #plt.scatter(x, y, color='red', s=50, label='标注点')
        f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
        F.append(f)
        print(f)

    output_path = f'./Transform/radio_maps/F{index}.txt'
    save_lines_to_txt(output_path,F)

def query_radio416(index):
    # 打开并读取coordinates.txt文件
    transformed_lines = []
    intensity_values = []
    with open("./Transform/coordinates5.txt", "r") as file:
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

            data = np.load(f"./Transform/radio_maps/radio_case{index}.npy")  # 加载数据

            if not (x > 59 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2 ): #exclude invs room "or (x>48 and x<59 and y<7)""
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

    # Select points based on intensity ranges
    selected_points = []
    selected_points1 = []


    # Iterate through the sorted intensity values
    # First, select 5 points from intensity range [-50, -60)
    for point in intensity_values:
        if -50 < point[1] <= -40 and point not in selected_points and point not in selected_points1:
            selected_points1.append(point)

    # 如果有足够的点，随机选择3个
    if len(selected_points1) >= 3:
        selected_points1 = random.sample(selected_points1, 3)
    else:
        selected_points1 = selected_points1  # 如果少于3个，使用所有点

    selected_points.extend(selected_points1)  # 添加选中的点到selected_points中

    # 清空selected_points1，为下一个条件做准备
    selected_points1 = []

    # 第二个条件
    for point in intensity_values:
        if -59 < point[1] <= -55 and point not in selected_points and point not in selected_points1:
            selected_points1.append(point)

    # 如果有足够的点，随机选择4个
    if len(selected_points1) >= 4:
        selected_points1 = random.sample(selected_points1, 4)
    else:
        selected_points1 = selected_points1  # 如果少于4个，使用所有点

    selected_points.extend(selected_points1)  # 添加选中的点到selected_points中

    # 清空selected_points1，为下一个条件做准备
    selected_points1 = []

    # 第三个条件
    for point in intensity_values:
        if -63 < point[1] <= -58 and point not in selected_points and point not in selected_points1:
            selected_points1.append(point)

    # 如果有足够的点，随机选择3个
    if len(selected_points1) >= 3:
        selected_points1 = random.sample(selected_points1, 3)
    else:
        selected_points1 = selected_points1  # 如果少于3个，使用所有点

    selected_points.extend(selected_points1)  # 添加选中的点到selected_points中
    
            
    
    top_intensity = selected_points

    F=[]


    for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1):
        #plt.scatter(x, y, color='red', s=50, label='标注点')
        path_loss =db_to_linear(path_loss)  # 将路径损耗从dB转换为线性值
        path_loss = calculate_Fik(1, path_loss, 10, 10 ** (-6))  # 使用calculate_Fik函数计算Fik值
        f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
        F.append(f)
        print(f)

    output_path = f'./Transform/radio_maps/F{index}.txt'
    save_lines_to_txt(output_path,F)

def query_radio1(index):
    # 打开并读取coordinates.txt文件
    transformed_lines = []
    intensity_values = []
    with open("./Transform/coordinates5.txt", "r") as file:
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

            data = np.load(f"./Transform/radio_maps/radio_case{index}.npy")  # 加载数据

            if not (x > 64 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2):
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

    # Select points based on intensity ranges
    selected_points = []

    # First, select 5 points from intensity range [-50, -60)
    for point in intensity_values:
        if -50 < point[1] <= -30 and len(selected_points) < 5:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)

    # Then, select 3 points from intensity range [-60, -70)
    for point in intensity_values:
        if -65 < point[1] <= -55 and len(selected_points) < 10:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)
    
            
    
    top_intensity = selected_points

    F=[]


    for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1):
        #plt.scatter(x, y, color='red', s=50, label='标注点')
        f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
        F.append(f)
        print(f)

    output_path = f'./Transform/radio_maps/F{index}.txt'
    save_lines_to_txt(output_path,F)



def query_radio2(index):
    # 打开并读取coordinates.txt文件
    transformed_lines = []
    intensity_values = []
    with open("./Transform/coordinates5.txt", "r") as file:
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

            data = np.load(f"./Transform/radio_maps/radio_case{index}.npy")  # 加载数据

            if not (x > 64 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2):
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

    # Select points based on intensity ranges
    selected_points = []

    # First, select 5 points from intensity range [-50, -60)
    for point in intensity_values:
        if -50 < point[1] <= -30 and len(selected_points) < 3:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)

    # Then, select 3 points from intensity range [-60, -70)
    for point in intensity_values:
        if -70 < point[1] <= -65 and len(selected_points) < 6:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)

    for point in intensity_values:
        if -80 < point[1] <= -75 and len(selected_points) < 10:
            if point not in unique_points:
                print(point)
                selected_points.append(point)
                unique_points.add(point)
    
            
    # for point in intensity_values:
    #     if -90 < point[0] <= -75 and len(selected_points) < 15:
    #         if point not in unique_points:
    #             selected_points.append(point)
    #             unique_points.add(point)

    # Finally, select 2 points from intensity range [-70, -80)
    # for point in intensity_values:
    #     if -90 < point[0] <= -80 and len(selected_points) < 10:
    #         if point not in unique_points:
    #             selected_points.append(point)
    #             unique_points.add(point)

    # Ensure that we have exactly 10 points selected
    top_intensity = selected_points

    F=[]


    for i, (impact_factor,path_loss, x, y) in enumerate(top_intensity, 1):
        #plt.scatter(x, y, color='red', s=50, label='标注点')
        f=f"F{index}: x={x}, y={y}, Path Loss={path_loss}"
        F.append(f)
        print(f)

    output_path = f'./Transform/radio_maps/F{index}.txt'
    save_lines_to_txt(output_path,F)


def query_radio3(index):
    # 打开并读取coordinates.txt文件
    transformed_lines = []
    intensity_values = []
    with open("./Transform/coordinates5.txt", "r") as file:
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

            data = np.load(f"./Transform/radio_maps/radio_case{index}.npy")  # 加载数据

            if not (x > 64 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2):
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

    # Select points based on intensity ranges
    selected_points = []

    # First, select 5 points from intensity range [-50, -60)
    for point in intensity_values:
        if -50 < point[0] <= -30 and len(selected_points) < 5:
            if point not in unique_points:
                selected_points.append(point)
                unique_points.add(point)

    # Then, select 3 points from intensity range [-60, -70)
    for point in intensity_values:
        if -65 < point[0] <= -55 and len(selected_points) < 10:
            if point not in unique_points:
                selected_points.append(point)
                unique_points.add(point)
            
    # for point in intensity_values:
    #     if -90 < point[0] <= -75 and len(selected_points) < 15:
    #         if point not in unique_points:
    #             selected_points.append(point)
    #             unique_points.add(point)

    # Finally, select 2 points from intensity range [-70, -80)
    # for point in intensity_values:
    #     if -90 < point[0] <= -80 and len(selected_points) < 10:
    #         if point not in unique_points:
    #             selected_points.append(point)
    #             unique_points.add(point)

    # Ensure that we have exactly 10 points selected
    if len(selected_points) == 10:
        top_intensity = selected_points
    else:
        print("Error: Not enough points were found in the specified ranges.")

    F=[]


    for i, (path_loss, x, y) in enumerate(top_intensity, 1):
        #plt.scatter(x, y, color='red', s=50, label='标注点')
        f=f"Rank {i}: x={x}, y={y}, Path Loss={path_loss}"
        F.append(f)
        print(f)

    output_path = f'./Transform/radio_maps/F{index}.txt'
    save_lines_to_txt(output_path,F)
    #plt.show()

# query_radio("invsdoor")
# query_radio("Tjunc")
# query_radio("toilet")