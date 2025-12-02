import os
gpu_num = 0 # Use "" to use the CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
import time
import random

# Import Sionna RT components
#from sionna.rt import load_scene, Transmitter, PlanarArray, Camera


def select_random_voro(anchor_num):
    # 定义一个列表来存储符合条件的坐标行
    valid_lines = []
    values_list=[]

    # 打开文件并读取每一行
    case =0
    if case == 0:
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
                
                
                # 判断条件是否满足
                if not (x > 60 or (x < 59 and y > 8 and x > 48 and y < 19) or y > 22 or x < 47 or y < 2):
                    valid_lines.append((x, y))

        # 随机选择 10 条符合条件的坐标
        selected_lines = random.sample(valid_lines, anchor_num) if len(valid_lines) >= anchor_num else valid_lines

        # 打印选中的坐标
        for x, y in selected_lines:
            print(x,y)
            cellindex_x = int(x)
            cellindex_y = int(y)
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
            if cellindex_x ==47:
                cellindex_x = cellindex_x +1 
            if cellindex_x >48 and cellindex_x<59 and cellindex_y ==19:
                cellindex_y = cellindex_y +1.2 
            if cellindex_x >48 and cellindex_x<59 and cellindex_y ==20:
                cellindex_y = cellindex_y +0.2 
            # input a robot pose (anchor point) at irsim 
            irsim_robot_pos = [0, 0]
            radio_robot_pos = [0,0]



            # convert robot pose in irsim to radio map for futher query
            translation = [13.9, -12.6]


            radio_robot_pos[0] = cellindex_x-66/2
            radio_robot_pos[1] = cellindex_y-26/2
            irsim_robot_pos[0] = radio_robot_pos[0]-translation[0]
            irsim_robot_pos[1] = radio_robot_pos[1]-translation[1]
            values_list.append([irsim_robot_pos[0],irsim_robot_pos[1]])

        output_path = f'./Transform/txt/select_random_voro.txt'
        with open(output_path, 'a') as f:
            for value in values_list:
                f.write(f"{value}\n")

    elif case == 1:
        with open("./Transform/txt/select_random_voro1.txt", "r") as file:
            for line in file:
                line_data = eval(line.strip())  # 使用 eval 将字符串转换为列表
                values_list.append(line_data)

    return values_list





   