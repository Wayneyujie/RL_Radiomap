# RadioNav: RA-L
# Yujie Wan, Guoliang Li, Shuai Wang

from ir_sim.env import EnvBase
import sys
import numpy as np
import scipy.io
import yaml
import time
import os
import argparse

from rda_planner.mpc import MPC
from collections import namedtuple
from curve_generator import curve_generator
from numpy.linalg import norm
from tsp_path import tsp_path, tsp_path1
import matplotlib.pyplot as plt

from task_planner.algorithm import iterative_local_search, iterative_local_search1
from global_planner.astar_planner import run_astar, Point
import math
from Transform.radio2irsim import calculate_carla_anchor
from Transform.select_RandomVoro import select_random_voro
from Transform.select_Kmeansradio import generate_DBSCAN, process_radio_map,process_radio_map_plot
from LLM.ollama import deepseek_api
from Transform.Voro2Carlatest import query_radio, query_radio1, query_radio2, query_radio416, query_radio913
from LLM.promp_gene1 import generate_output, generate_output1, generate_output2
from plot.plot_sumdata import record_param, record_sum
import shutil

# environment
env = EnvBase('basic_example.yaml', save_ani=True, display=True, full=False)
car = namedtuple('car', 'G h cone_type wheelbase max_speed max_acce')

# static methods

def save_path_to_txt(path, filename="path.txt"):
    with open(filename, "w") as file:
        for point in path:
            file.write(f"{point[0]} {point[1]}\n")

def save_lines_to_txt_a(output_path,lines):
    with open(output_path, 'a') as f:
        f.write(lines)  # Write the entire content as a single line

def read_file(file):
    with open(file, "r") as f:
        content = f.read()  # 读取文件的全部内容
    return content

def db_to_linear(db_value):
    """
    将dB值转换为线性值。
    
    参数:
    db_value -- 路径损耗的dB值。
    
    返回:
    对应的线性值。
    """
    return 10 ** (db_value / 10)

def calculate_distance(point1, point2):
    """
    计算两个航点之间的欧几里得距离。
    
    :param point1: 第一个航点，(x1, y1) 形式的元组。
    :param point2: 第二个航点，(x2, y2) 形式的元组。
    :return: 两点间的距离。
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def trajectory_length(waypoints):
    """
    计算轨迹的总长度。
    
    :param waypoints: 航点列表，每个元素是一个表示坐标的元组。
    :return: 轨迹的总长度。
    """
    total_length = 0.0
    for i in range(len(waypoints) - 1):
        total_length += calculate_distance(waypoints[i], waypoints[i+1])
    return total_length

## proposed methods

def formula(xt, yt, ak, bk, alpha):
    # 计算 L2 范数 ||(xt - ak, yt - bk)||
    norm = np.sqrt((xt - ak)**2 + (yt - bk)**2)
    
    # 返回 L2 范数的 -alpha 次方
    return norm**(-alpha)

def radio_map_generation(anchor_list, iot_list, pre_gen=False):    

    M = len(anchor_list)   # M = num_anchor
    K = len(iot_list)  # K = num_iot
    RM = np.zeros([K, M])

    if pre_gen == True:
        # load radio map
        radio_map_list = []
        for k in range(K):
            if k == 0:
                #radio_map = np.load("./Transform/radio_maps/radio_case9.npy")
                radio_map = np.load("./radio_maps/radio_Tjunc.npy")  # map T_junction
            if k == 1:
                radio_map = np.load("./radio_maps/radio_invsdoor.npy")  # map invsdoor
            if k == 2:
                radio_map = np.load("./radio_maps/radio_toilet.npy")  # map toilet
            radio_map_list.append(radio_map)

        for k in range(K):
            for m, anchor in enumerate(anchor_list):
                pathloss_dB = query_radio_map(radio_map_list[k], anchor, iot_list[k])
                path_loss_linear = db_to_linear(pathloss_dB)
                RM[k,m] = path_loss_linear
    elif pre_gen == False:
        radio_map_list = []
        for k in range(K):
            radio_map = np.load(f"./Transform/radio_maps/radio_case{iot_list[k]}.npy")  # map toilet
            radio_map_list.append(radio_map)

        iot_location = get_location_from_txt("./iot1.txt",iot_list)    


        for k in range(K):
            for m, anchor in enumerate(anchor_list):
                pathloss_dB = query_radio_map(radio_map_list[k], anchor, iot_list[k])
                path_loss_linear = db_to_linear(pathloss_dB)
                RM[k,m] = path_loss_linear
                print("linear:",path_loss_linear)
                #(2.776956207811318e-07, -0.6527304207098054)


                ##############RM[k,m] Reality Formulation##################
                #RM[k,m] = 0.25*formula(anchor[0],anchor[1],iot_location[k][0],iot_location[k][1], 4)
                print("Distance RM:",RM[k,m])
                #RM[k,m] = 0.000202*formula(anchor[0],anchor[1],iot_location[k][0],iot_location[k][1], 3.9)
                #RM[k,m] = formula(anchor[0],anchor[1],iot_list[k][0],iot_list[k][1], 2)
    else:
        print("Generating radio maps according to anchor list and iot list ... ")
        # To do list
    
    assert sum(sum(RM - np.zeros([K, M]))) != 0

    return RM, radio_map_list


def task_plan(anchor_list, iot_list, RM, alpha, dist_case,tsp_case=False, algorithm_case=0):    

    print('*** Computing the task graph ***')

    anchor_list_array = np.array(anchor_list)
    iot_list_array = np.array(iot_list)
    num_anchor = len(anchor_list)
    num_iot = len(iot_list)
    K = num_iot
    M = num_anchor

    dist_matrix = np.zeros([num_anchor, num_anchor])

    if dist_case == 0:
        # direct distance between any two anchors
        for i, anchor_i in enumerate(anchor_list_array):
            for j, anchor_j in enumerate(anchor_list_array):
                dist_matrix[i,j] = norm(anchor_i-anchor_j)
    elif dist_case == 1:
        # load distance from carla
        # txt_path = sys.path[0] + '/astar/array.txt'
        # D_array = np.loadtxt(txt_path)
        # dist_matrix = D_array * 0.1 # carla/irsim = 10/1

        #load static distance
        txt_path = sys.path[0] + '/array.txt'
        D_array = np.loadtxt(txt_path)
        dist_matrix = D_array # carla/irsim = 10/1



    elif dist_case == 2:
        # compute distance using astar
        for i in range(M):
            dist_matrix[i, i] = 0  # 直接为对角线上的值赋 0
            for j in range(i + 1, M):  # 只计算上三角矩阵部分
                # 如果 dist_matrix[0, j] 已经为 100000，则跳过计算
                if dist_matrix[0, j] == 100000 or dist_matrix[0,i] == 100000:
                    continue

                point_start = anchor_list[i]
                point_end = anchor_list[j]
                print("point_start:", point_start)
                print("point_end:", point_end)
                irsim_start = Point(point_start[0], point_start[1])
                irsim_end = Point(point_end[0], point_end[1])
                path, irsim_path, cropped_image = run_astar(irsim_start, irsim_end)

                # Downsample the global_path
                if irsim_path:
                    path_length = trajectory_length(irsim_path[::100])
                    dist_matrix[i, j] = path_length
                    dist_matrix[j, i] = path_length  # 利用对称性，直接赋值
                else:
                    dist_matrix[i, j] = 100000
                    dist_matrix[j, i] = 100000  # 同样赋值

                # 如果 dist_matrix[0, j] 为 100000，则设置所有 [k, j] 和 [j, k] 为 100000
                if i == 0 and dist_matrix[0, j] == 100000:
                    for k in range(M):
                        dist_matrix[k, j] = 100000
                        dist_matrix[j, k] = 100000

                print(f"The path from anchor {i} to anchor {j} is {dist_matrix[i, j]}")


    elif dist_case ==3:
        for i in range(M):
            dist_matrix[i, i] = 0  # 直接为对角线上的值赋 0
            for j in range(i + 1, M):  # 只计算上三角矩阵部分
                # 如果 dist_matrix[0, j] 已经为 100000，则跳过计算
                if dist_matrix[0, j] == 100000 or dist_matrix[0,i] == 100000:
                    continue

                point_start = anchor_list[i]
                point_end = anchor_list[j]
                irsim_start = Point(point_start[0], point_start[1])
                irsim_end = Point(point_end[0], point_end[1])
                path, irsim_path, cropped_image = run_astar(irsim_start, irsim_end)

                # Downsample the global_path
                if irsim_path:
                    path_length = trajectory_length(irsim_path[::100])
                    dist_matrix[i, j] = path_length
                    dist_matrix[j, i] = path_length  # 利用对称性，直接赋值
                else:
                    dist_matrix[i, j] = 100000
                    dist_matrix[j, i] = 100000  # 同样赋值

                # 如果 dist_matrix[0, j] 为 100000，则设置所有 [k, j] 和 [j, k] 为 100000
                if i == 0 and dist_matrix[0, j] == 100000:
                    for k in range(M):
                        dist_matrix[k, j] = 100000
                        dist_matrix[j, k] = 100000

                print(f"The path from anchor {i} to anchor {j} is {dist_matrix[i, j]}")

        v_all = np.zeros(num_anchor)
        num_ones=2
        #num_zeros=np.random.randint(num_anchor-3,num_anchor-2)
            # 随机选择 num_zeros 个索引
        zero_indices = np.random.choice(num_anchor, num_ones, replace=False)
        # 将选中的索引位置的值设为 0
        v_all[zero_indices]=1
        while np.any(dist_matrix[0][zero_indices] == 100000):
            # 找到 dist[0][zero_indices] 中值为 10000 的索引
            invalid_indices = zero_indices[dist_matrix[0][zero_indices] == 100000]
            
            # 将这些索引重新置为 1
            v_all[invalid_indices] = 0
            
            # 重新选择新的索引置为 0
            new_zero_indices = np.random.choice(num_anchor, len(invalid_indices), replace=False)
            v_all[new_zero_indices] = 1
            
            # 更新 zero_indices
            zero_indices = np.setdiff1d(zero_indices, invalid_indices)
            zero_indices = np.concatenate((zero_indices, new_zero_indices))
        v_all[0] =1
        print("v_all:", v_all)
        save_v_all_a("v_all.txt", v_all)



        RM, radio_map_list = radio_map_generation(anchor_list, iot_list, False)
        Bk = 1.0 * np.ones((K, 1)) # 假设带宽为1, 单位MHz
        Pk = 10.0 * np.ones((K, 1)) # 假设发射功率为10 mW单位
        sigma_squared = 10 ** (-6)  # 单位mW, 假设噪声功率为 -60 dBm

        transfer_list=[]
        for k in range(K):
            transfer=[]
            radio_map = radio_map_list[k]
            count=0
            for m in range(M):
                if(v_all[m]==1):
                    irsim_robot_pos =[anchor_list[m][0],anchor_list[m][1]]
                    pathloss = query_radio_map(radio_map, irsim_robot_pos, iot_list[k])
                    data_amount = communication_perf(pathloss)  # bits
                    transfer_speed = calculate_Fik(Bk[k], RM[k,m], Pk[k], sigma_squared)
                    transfer.append(transfer_speed)
                    count+=1
                else:
                    continue
            transfer_list.append(sum(transfer)/count)
        save_v_all("Anchor_transfer.txt",transfer_list)
        anchor_list_array = anchor_list_array.T
        tsp_path(anchor_list_array, iot_list_array, dist_matrix, v_all, num_anchor)
        np.savetxt('array.txt',dist_matrix)
        print("The distance matrix in ir-sim is:", dist_matrix)
        return np.zeros(M), count
        

    np.savetxt('array.txt',dist_matrix)
    print("The distance matrix in ir-sim is:", dist_matrix)

    if dist_case != 3:
        anchor_list_array = anchor_list_array.T

        ref_speed = 1 # velocity in m/s
        Bk = 1.0 * np.ones((K, 1)) # 假设带宽为1, 单位MHz
        Pk = 10.0 * np.ones((K, 1)) # 假设发射功率为10 mW单位
        sigma_squared = 10 ** (-6)  # 单位mW, 假设噪声功率为 -60 dBm

        
        if False:
            #v_opt =
            # count =  sum(1 for x in v_opt if x==1)
            tsp_path1(anchor_list_array, iot_list_array, dist_matrix, v_opt, num_anchor)
            return np.zeros(M), count


        
        if tsp_case==True:
            v_all = np.ones(num_anchor)  # full path
            v_opt = v_all
            count = sum(1 for x in v_opt if x==1)
            save_v_all_a("v_all.txt", v_opt)
            tsp_path1(anchor_list_array, iot_list_array, dist_matrix, v_opt, num_anchor)
            return np.zeros(M), count

        else:

            if algorithm_case ==1:
                best_E_total = float('inf')  # 初始化最优总时间为无穷大
                best_result = None  # 初始化最优结果
                ITER_MAX = 10
                v0_list=Initial_v0(5)
                #v0_list= [np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]), np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0]), np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])]
                best_value=float('inf')
                for i, v0 in enumerate(v0_list):
                    v0=np.insert(v0,0,1)
                    Tm0, Tcom0, t_list, E_ls, v_ls, E_iter = iterative_local_search(K, M, dist_matrix, ref_speed, RM, Bk, Pk, sigma_squared, alpha, ITER_MAX,v0.reshape(-1, 1),best_value)
                    
                    if E_ls < best_E_total:
                        best_E_total = E_ls
                        t_best =t_list
                        v_best = v_ls
                        best_value = E_ls

                v_opt = v_best[:, 0]
                #v_opt =[1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
                #v_opt =[1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                #v_opt= [1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1]
                #v_opt =[1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]
                save_v_all_a("v_all.txt", v_opt)
                save_v_all_a("v_all_record.txt", v_opt)
                v_all = np.ones(num_anchor)  # full path
                count = sum(1 for x in v_opt if x==1)
                tsp_path1(anchor_list_array, iot_list_array, dist_matrix, v_opt, num_anchor)
                with open("t_list.txt","w") as file:
                    file.write(" ".join(map(str, t_best)))
                return t_best, count
                # save_path(map["UGV_location_all"], map["user_location_all"], map["D_all"], v_all, map["M"])
            elif algorithm_case ==0:
                ITER_MAX = 30 #80
                Tm0, Tcom0, t_list, E_ls, v_ls, E_iter = iterative_local_search1(K, M, dist_matrix, ref_speed, RM, Bk, Pk, sigma_squared, alpha, ITER_MAX)
                v_opt = v_ls[:, 0]
                #v_opt=[1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
                #v_opt=[1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                #v_opt = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
                # v_opt =[1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                # v_opt = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                # v_opt = [1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
                # v_opt = [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
                #v_opt =[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                # v_opt = [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
                # v_opt = [1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0]
                save_v_all_a("v_all.txt", v_opt)
                save_v_all_a("v_all_record.txt", v_opt)
                v_all = np.ones(num_anchor)  # full path
                count = sum(1 for x in v_opt if x==1)
                tsp_path1(anchor_list_array, iot_list_array, dist_matrix, v_opt, num_anchor)
                with open("t_list.txt","w") as file:
                    file.write(" ".join(map(str, t_list)))
                return t_list, count
                # save_path(map["UGV_location_all"], map["user_location_all"], map["D_all"], v_all, map["M"])

def Initial_v0(v0_num):
    all_data = []

    with open('./Transform/selectbyhand.txt', 'r') as file:
        for line in file:
            # 按空格分割每一行
            parts = line.strip().split(', ')
            # 提取组名 (F1, F6, F15)
            group = parts[0].split(':')[0]
            # 提取x, y, Path Loss
            x = int(float(parts[0].split('=')[1]))
            y = int(float(parts[1].split('=')[1]))
            path_loss = float(parts[2].split('=')[1])
            
            # 将提取的数据添加到all_data中
            all_data.append((group, x, y, path_loss))

    print(all_data)


    # 生成五个v0，每次交替选中F1、F6、F15组中的一个数据点
    v0_list = []
    selected_groups = list(set(group for group, _, _, _ in all_data))  # 需要交替选择的组
    group_counter = 0  # 用于标记当前选择的组

    for _ in range(v0_num):
        while True:
            v0 = np.zeros(len(all_data), dtype=int)  # 初始化v0为全0

            # 找到当前组的所有索引
            for group in selected_groups:
                # # 找到当前组的所有数据点索引
                # group_indices = [i for i, (g, _, _, _) in enumerate(all_data) if g == group]
                # # 随机选择一个索引
                # selected_index = np.random.choice(group_indices)
                # v0[selected_index] = 1  # 将选中的数据点标记为1
                random_indices = np.random.choice(len(all_data), size=len(all_data)-3, replace=False)  # 从 0 到 7 中随机选择 5 个不重复的索引
                v0[random_indices] = 1  # 将这些索引对应的位置设置为 1

            print(v0)
            # 交替选择下一个组
            #group_counter = (group_counter + 1) % len(selected_groups)
            if not any(np.array_equal(v0, existing_v0) for existing_v0 in v0_list):
                v0_list.append(v0)  # 如果不重复，则添加到v0_list
                break  # 跳出while循环，继续生成下一个v0


    # 输出v0_list查看生成的五个v0
    for i, v0 in enumerate(v0_list):
        print(f"v0_{i+1}: {v0}")
    return v0_list


def global_plan(point_list): 

    if(len(point_list[0])>1):
        print("The number of selected anchors:", len(point_list))
        global_path = []

        print("point_list:"+str(point_list))

        for i in range(len(point_list)-1):
            point_start = point_list[i]
            point_end = point_list[i+1]
            irsim_start = Point(point_start[0], point_start[1])
            irsim_end = Point(point_end[0], point_end[1])
            path, irsim_path, cropped_image = run_astar(irsim_start, irsim_end)
            global_path.extend(irsim_path)
        
        # Downsample the global_path
        downsampled_global_path = global_path[::10]  # Adjust the step size as needed
        print(downsampled_global_path)
        save_path_to_txt(downsampled_global_path, filename="global_path.txt")
    else:
        downsampled_global_path = point_list
        save_path_to_txt(downsampled_global_path, filename="global_path.txt")


    return downsampled_global_path


def smooth_path_angles(path_list):
    """确保路径角度连续，避免跳变"""
    if len(path_list) < 2:
        return path_list
    
    for i in range(1, len(path_list)):
        # 计算相邻点之间的角度差
        diff = path_list[i][2, 0] - path_list[i-1][2, 0]
        
        # 使用wraptopi确保差值在[-π, π]范围内
        while diff > np.pi:
            path_list[i][2, 0] -= 2 * np.pi
            diff = path_list[i][2, 0] - path_list[i-1][2, 0]
        while diff < -np.pi:
            path_list[i][2, 0] += 2 * np.pi
            diff = path_list[i][2, 0] - path_list[i-1][2, 0]
    
    return path_list


def fix_reference_path_angles(path_list):
    """
    根据路径点的实际位置重新计算角度
    这是解决'raw=0.0°'问题的关键
    """
    if len(path_list) < 2:
        return path_list
    
    print(f"\n🔧 修复参考路径角度，共{len(path_list)}个点")
    
    # 第一遍：计算基于位置的角度
    for i in range(len(path_list)):
        if i == len(path_list) - 1:
            # 最后一个点，使用倒数第二个点的角度
            path_list[i][2, 0] = path_list[i-1][2, 0]
        else:
            # 计算从当前点到下一个点的方向
            dx = path_list[i+1][0, 0] - path_list[i][0, 0]
            dy = path_list[i+1][1, 0] - path_list[i][1, 0]
            
            # 使用更多点来平滑方向（看前方5个点）
            if i + 5 < len(path_list):
                dx = path_list[i+5][0, 0] - path_list[i][0, 0]
                dy = path_list[i+5][1, 0] - path_list[i][1, 0]
            
            angle = np.arctan2(dy, dx)
            path_list[i][2, 0] = angle
    
    # 第二遍：确保角度连续（消除跳变）
    for i in range(1, len(path_list)):
        diff = path_list[i][2, 0] - path_list[i-1][2, 0]
        
        # 将差值调整到[-π, π]范围
        while diff > np.pi:
            path_list[i][2, 0] -= 2 * np.pi
            diff = path_list[i][2, 0] - path_list[i-1][2, 0]
        while diff < -np.pi:
            path_list[i][2, 0] += 2 * np.pi
            diff = path_list[i][2, 0] - path_list[i-1][2, 0]
        
        # 限制单步最大角度变化
        max_step_change = 0.3  # 约17度
        if abs(diff) > max_step_change:
            path_list[i][2, 0] = path_list[i-1][2, 0] + np.sign(diff) * max_step_change
    
    # 第三遍：高斯平滑（可选，进一步平滑）
    window_size = 5
    for i in range(window_size, len(path_list) - window_size):
        angles = [path_list[j][2, 0] for j in range(i - window_size, i + window_size + 1)]
        path_list[i][2, 0] = np.mean(angles)
    
    print(f"✅ 角度修复完成")
    return path_list

def local_plan(ref_path_list, iot_list, radio_map_list, anchor_case, str):
    ref_path_list = fix_reference_path_angles(ref_path_list)
    
    robot_info = env.get_robot_info()
    car_tuple = car(robot_info.G, robot_info.h, robot_info.cone_type, robot_info.shape[2], [1, 1], [3, 0.5])
    obstacle_template_list = [ {'edge_num': 4, 'obstacle_num': 4, 'cone_type': 'Rpositive'},{'edge_num': 3, 'obstacle_num': 0, 'cone_type': 'norm2'}]
    
    mpc_opt = MPC(car_tuple, ref_path_list, receding=15, sample_time=0.1, process_num=4, iter_num=1, ro1=50, obstacle_template_list=obstacle_template_list, ws=5, wu=5)
    
    mpc_opt.update_parameter(slack_gain=4, max_sd=0.5, min_sd=0.1)  

    K = len(iot_list)
    data_amount_list = [[] for _ in range(K)]
    
    for i in range(5000):
        robot_pos = np.array([env.robot.state[0], env.robot.state[1]])
        
        # 获取障碍物
        obs_list = env.get_obstacle_list()
        nearby_obs = []
        # for obs in obs_list:
        #     obs_pos = np.array(obs.center[0:2]).flatten()
        #     distance = np.linalg.norm(robot_pos - obs_pos)
        #     if distance < 3.0:
        #         nearby_obs.append(obs)
        nearby_obs = obs_list[-2:]
        
        # MPC控制
        if len(ref_path_list) == 1:
            opt_vel = np.zeros((2, 1))
        else:
            opt_vel, info = mpc_opt.control(
                env.robot.state, 
                0.8,
                nearby_obs,
                obs_list=nearby_obs
            )
            
            if info['arrive']:
                print('✅ 到达目标！')
                record_sum('finish', anchor_case, K, str)
                break
            
            env.draw_trajectory(info['opt_state_list'], 'g', refresh=True)

        # === 记录位置（用于通信计算） ===
        car_location_before = [float(env.robot.state[0]), float(env.robot.state[1])]

        env.step(opt_vel, stop=False)
        env.render(show_traj=True, show_trail=True, traj_type='-r')
        print(f"🎮 控制量: 速度={opt_vel[0,0]:.2f}, 转向={opt_vel[1,0]:.2f}")

        if env.done():
            env.render_once(show_traj=True, show_trail=True)
            break

        # === 通信性能计算 ===
        car_location_after = [float(env.robot.state[0]), float(env.robot.state[1])]
        irsim_robot_pos = [(car_location_before[0] + car_location_after[0]) / 2, 
                           (car_location_before[1] + car_location_after[1]) / 2]

        for k in range(K):
            radio_map = radio_map_list[k]
            print(f"The robot is at ({irsim_robot_pos})")
            pathloss = query_radio_map(radio_map, irsim_robot_pos, iot_list[k])
            data_amount = communication_perf(pathloss)
            print(f"📶 IoT设备{k}在该时间步传输speed: {data_amount/0.1/10**6} bits")
            data_amount_list[k].append(data_amount)

            # 保存数据
            if anchor_case == 0:
                result_folder = './results'
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)
                data_amount_list_path = os.path.join(result_folder, f"data_amount_list_voro{k}.txt")
                np.savetxt(data_amount_list_path, data_amount_list[k], fmt='%f')
            elif anchor_case == 2:
                result_folder = './results'
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)
                data_amount_list_path = os.path.join(result_folder, f"data_amount_list_nav{k}.txt")
                np.savetxt(data_amount_list_path, data_amount_list[k], fmt='%f')
            elif anchor_case == 1:
                result_folder = './results'
                if not os.path.exists(result_folder):
                    os.makedirs(result_folder)
                data_amount_list_path = os.path.join(result_folder, f"data_amount_list_{k}.txt")
                np.savetxt(data_amount_list_path, data_amount_list[k], fmt='%f')  

    env.end(ani_name='path_track', show_traj=True, show_trail=True, ending_time=2, keep_len=200, ani_kwargs={'subrectangles':True})


def modify_radio_map_region(radio_map, region, value_change):
    """
    修改无线电地图的特定区域
    
    参数:
    radio_map: 原始无线电地图数据 (3D数组)
    region: 要修改的区域 [y_start:y_end, x_start:x_end]
    value_change: 要增加/减少的值
    """
    # 复制数据避免修改原始数组
    modified_map = radio_map.copy()
    
    # 修改指定区域
    modified_map[region] += value_change
    
    return modified_map

def query_radio_map(radio_map, irsim_robot_pos, iot_pos):
    data = radio_map.copy()
    
    # 记录修改前的值
    original_value = data[0, 3:6, 51:56].mean()
    print(f"修改前区域平均路径损耗: {original_value:.2f} dB")
    
    # 应用修改
    data[0,3:6, 51:56] -= 22
    
    # 记录修改后的值
    modified_value = data[0, 3:6, 51:56].mean()
    print(f"修改后区域平均路径损耗: {modified_value:.2f} dB")
    print(f"变化量: {modified_value - original_value:.2f} dB")

    # check the size of radio map
    radio_map_height = data.shape[1]
    radio_map_width = data.shape[2]
    print('The radio map has a size of:', radio_map_height, 'x', radio_map_width)

    # convert robot pose in irsim to radio map for futher query
    translation = [13.9, -12.6]
    radio_robot_pos = [irsim_robot_pos[0] + translation[0], irsim_robot_pos[1] + translation[1]]

    # convert radio map position to cell index ([0,0] at the upper left corner)
    cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
    cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
    rm_index = [0, cellindex_y, cellindex_x]

    # load the path_loss
    path_loss = data[0, cellindex_y, cellindex_x]

    print("Path loss from", iot_pos, "to", irsim_robot_pos, "in irsim:",path_loss)
    #print(path_loss) 

    return path_loss
     

def communication_perf(pathloss):
    tau = 0.1  # time step
    robot_power = 10 # 10 mW
    noise_power = 10**(-6) # mW, noise power, -80 dBm
    pathloss_linear = 10 ** (pathloss/10)
    iot_B = 1000000 # 0.1MHz

    Iot_com = tau * iot_B * np.log(1 + robot_power * pathloss_linear / noise_power) /np.log(2)
    
    return Iot_com

def calculate_Fik(Bk, RMk, pk, sigma_squared):
    # 计算公式中的括号部分
    bracket_part = 1 + (RMk * pk) / sigma_squared
    # 使用math.log函数计算以2为底的对数
    log_part = math.log(bracket_part, 2)
    # 最终结果
    Fik = Bk * log_part
    return Fik

def save_matrix_to_txt(matrix, filename):
    with open(filename, 'w') as file:
        for i in range(len(matrix)):
                file.write(f"{matrix[i][0][0]} {matrix[i][1][0]} {matrix[i][2][0]}\n")

def save_t_list(t_list):
    with open('./results/success_rate_data.txt', 'a') as file:
        # for anchor in anchor_list:
        #     # 写入每个子列表，格式化为字符串
        #     file.write(f"{anchor[0]}, {anchor[1]}\n")
        file.write(str(t_list))

def save_anchor(anchor_list):
    with open('./Transform/selectbyhand_record.txt', 'a') as file:
        # for anchor in anchor_list:
        #     # 写入每个子列表，格式化为字符串
        #     file.write(f"{anchor[0]}, {anchor[1]}\n")
        file.write(str(anchor_list))

def save_v_all(file,v_all):
    with open(file,"w") as file:
        file.write(" ".join(map(str, v_all)))
        #file.write(str(v_all))

def save_v_all_a(file,v_all):
    with open(file,"a") as file:
        #file.write(" ".join(map(str, v_all)))
        file.write(str(v_all))

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



if __name__ == '__main__':

    
     # Parse command line arguments manually to handle flexible format
    import sys
    import ast

    # 直接解析所有参数
    all_args = sys.argv[1:]

    # 检查是否有包含列表的参数（包含 '[' 的参数）
    anchor_list_from_args = None
    main_params = []

    i = 0
    while i < len(all_args):
        if '[' in all_args[i]:
            # 找到包含列表的参数
            list_str = all_args[i]
            # 如果列表被分割到多个参数中，合并它们
            while i + 1 < len(all_args) and ']' not in all_args[i]:
                i += 1
                list_str += ' ' + all_args[i]
            
            try:
                # 使用 ast.literal_eval 安全地解析列表
                anchor_list_from_args = ast.literal_eval(list_str)
                print(f"成功解析 anchor_list: {anchor_list_from_args}")
            except:
                print(f"解析列表失败: {list_str}")
                anchor_list_from_args = None
        else:
            main_params.append(all_args[i])
        i += 1

    # 如果没有找到列表参数，尝试从最后一个参数解析
    if anchor_list_from_args is None and main_params:
        last_param = main_params[-1]
        if '[' in last_param:
            try:
                anchor_list_from_args = ast.literal_eval(last_param)
                main_params = main_params[:-1]  # 从主参数中移除
                print(f"从最后一个参数解析 anchor_list: {anchor_list_from_args}")
            except:
                pass

    print(f"main_params: {main_params}")
    print(f"anchor_list_from_args: {anchor_list_from_args}")

    # Parse main parameters from main_params
    params = main_params
    
    # Parse parameters with flexible iot_list length
    # Find the position of tsp_option (True/False) to determine iot_list length
    tsp_index = -1
    for i, param in enumerate(params):
        if param in ['True', 'False']:
            tsp_index = i
            break
    
    if tsp_index == -1:
        raise ValueError("tsp_option (True/False) not found in parameters")
    
    # Extract parameters based on tsp_option position
    iot_list = [int(x) for x in params[:tsp_index-4]]  # Everything before the 4 fixed params before tsp_option
    anchor_case = int(params[tsp_index-4])
    dist_case = int(params[tsp_index-3]) 
    anchor_number = int(params[tsp_index-2])
    alpha = int(params[tsp_index-1])
    tsp_option = params[tsp_index] == 'True'
    algorithm_case = int(params[tsp_index+1])
    anchor_fixed = int(params[tsp_index+2])
    
    # Create args object similar to argparse
    class Args:
        def __init__(self):
            self.iot_list = iot_list
            self.anchor_case = anchor_case
            self.dist_case = dist_case
            self.anchor_number = anchor_number
            self.alpha = alpha
            self.tsp_option = 'True' if tsp_option else 'False'
            self.algorithm_case = algorithm_case
            self.anchor_fixed = anchor_fixed
            self.anchor_list_from_args = anchor_list_from_args
        
        def __str__(self):
            return f"iot_list={self.iot_list}, anchor_case={self.anchor_case}, dist_case={self.dist_case}, anchor_number={self.anchor_number}, alpha={self.alpha}, tsp_option={self.tsp_option}, algorithm_case={self.algorithm_case}, anchor_fixed={self.anchor_fixed}"
    
    args = Args()


    with open('./results/cmd_vel1.txt', 'w') as file:
        pass  # 不做任何操作，直接关闭文件，即清空文件内容
    with open('./results/cmd_vel.txt', 'a') as file:
        file.write(f"{args}\n\n\n")
    with open("./results/traj.txt", "a") as file:
        # 将坐标和方向写入文件，以空格分隔，并以换行符结束
        file.write(f"{args}\n\n\n")
    with open("./results/Carlatraj.txt", "a") as file:
        # 将坐标和方向写入文件，以空格分隔，并以换行符结束
        file.write(f"{args}\n\n\n")
    with open("./Transform/selectbyhand_record.txt", "a") as file:
        # 将坐标和方向写入文件，以空格分隔，并以换行符结束
        file.write(f"{args}\n\n\n")
    with open("E_iter.txt", "a") as file:
        # 将坐标和方向写入文件，以空格分隔，并以换行符结束
        file.write(f"{args}\n\n\n")
    with open("v_all_record.txt", "a") as file:
        # 将坐标和方向写入文件，以空格分隔，并以换行符结束
        file.write(f"{args}\n\n\n")
    with open("Iter_timecost.txt", "a") as file:
        # 将坐标和方向写入文件，以空格分隔，并以换行符结束
        file.write(f"{args}\n\n\n")



    iot_list = args.iot_list
    K = len(iot_list)
    anchor_case = args.anchor_case
    dist_case = args.dist_case
    algorithm_case = args.algorithm_case
    anchor_number = args.anchor_number
    alpha = args.alpha * np.ones((K, 1)) # Mbits
    anchor_fixed=args.anchor_fixed
    #alpha =[300,200]
    tsp_option = args.tsp_option == 'True'
    print(f"iot_list:{iot_list}",f"anchor_case:{anchor_case}",f"dist_case:{dist_case}")
    iot_location = get_location_from_txt("./iot1.txt",iot_list)

                        
    #dist_case=2
    #anchor_case = 0
    #iot_list = [3]
    #anchor_number =2
    #alpha = 100 * np.ones((K, 1)) # Mbits
    #tsp_option =False
    
    repeat = False
    pre_gen = False
    
    

    if repeat:
        # 初始化一个空列表来存储数据
        anchor_list = []

        # 读取文件
        with open('anchors.txt', 'r') as file:
            for line in file:
                # 处理每一行，去掉末尾的换行符并将逗号分隔的值转换为浮点数
                anchor = list(map(float, line.strip().split(', ')))
                anchor_list.append(anchor)

        txt_path = sys.path[0] + '/point_list_ref.txt'
        point_array = np.loadtxt(txt_path)
        point_list = []
        for index, point in enumerate(point_array):
            point_np_tmp = np.c_[point_array[index]]
            point_list.append(point_np_tmp)

        txt_path = sys.path[0] + '/global_path.txt'
        point_array = np.loadtxt(txt_path)
        point_list = []

        for index, point in enumerate(point_array):
                point_array_3d = np.append(point_array[index], 0)
                point_np_tmp = np.c_[point_array_3d]
                point_list.append(point_np_tmp)

        
        # 输出结果
        cg = curve_generator()
        ref_path_list = cg.generate_curve(curve_style='line', way_points=point_list, step_size=0.5, min_radius=0.2)
        #env.draw_trajectory(ref_path_list, traj_type='-c', label='trajectory', linewidth=1)
        print(anchor_list)
        RM, radio_map_list = radio_map_generation(anchor_list, iot_list, pre_gen)

        anchor_list_array = np.array(anchor_list)
        iot_list_array = np.array(iot_list)
        num_anchor = len(anchor_list)
        anchor_list_array = anchor_list_array.T
        with open("v_all.txt", "r") as file:
        # 读取文件内容并转回列表
            v_opt = list(map(float, file.read().split()))
        txt_path = sys.path[0] + '/array.txt'
        D_array = np.loadtxt(txt_path)
        dist_matrix = D_array
        plt.plot([], [], color='r', linestyle='--', label='Red Dashed Line')
        tsp_path1(anchor_list_array, iot_list_array, dist_matrix, v_opt, num_anchor)
        #tsp_path(anchor_list_array, iot_list_array, dist_matrix, v_opt, num_anchor)

        local_plan(ref_path_list, iot_list, radio_map_list, anchor_case)
    else:
        if anchor_case == 0:
            # SELECT RANDOM VORONOI
            anchor_list =  select_random_voro(anchor_number)
            anchor_list.insert(0, [1.5, 7.5]) 
            #anchor_list=[[1.5, 7.5], [5.1, 4.6], [1.0999999999999996, 15.6], [14.1, 15.6], [14.1, 7.6], [4.1, 20.6], [11.1, 20.6], [1.0999999999999996, 5.6], [1.0999999999999996, 11.6], [12.1, 7.6], [3.0999999999999996, 2.5999999999999996], [14.1, 13.6], [1.0999999999999996, 18.6], [1.0999999999999996, 20.6], [8.1, 7.6], [8.1, 3.5999999999999996], [3.0999999999999996, 21.6], [4.1, 4.6], [5.1, 1.5999999999999996]]
            #anchor_list.insert(0, [10, 5.5]) 
            save_anchor(anchor_list)
            print("anchorlist:" , anchor_list)
        elif anchor_case == 1:
            # SELECT DBSCAN RADIOMAP 
            anchor_list=[[1.5, 7.5]]
            for num in iot_list:
                list1 = process_radio_map_plot(num)
                anchor_list= anchor_list + list1
            save_anchor(anchor_list)
            print("anchorlist:" , anchor_list)
        elif anchor_case == 2:
            # OURS RADIONAV
            radionav_case =anchor_fixed
            if radionav_case == 0:
                #deepseek_api()######
                print("=== DEBUG ANCHOR POINTS ===")
                print(f"anchor_case: {anchor_case}")
                print(f"anchor_list_from_args: {anchor_list_from_args}")
                
                if anchor_list_from_args is not None:
                    anchor_list = anchor_list_from_args
                else:
                    anchor_list =calculate_carla_anchor(26,66)
                    anchor = read_file("./Transform/selectbyhand.txt")
                    save_lines_to_txt_a("./Transform/selectbyhand_record.txt", anchor)
                    #anchor_list =[[1.5, 7.5], [10.1, 0.5999999999999996],[10.1, 2.5999999999999996],  [6.1, 4.6], [6.1, 5.6], [5.1, 4.6], [1.0999999999999996, 18.6], [3.0999999999999996, 19.6], [1.0999999999999996, 17.6]]
                    #anchor_list = [[11.9,0.5],[9.1,5.6],[3.1,19.9]]
                    #anchor_list=[[10.1, 0.5999999999999996], [12.1, 2.5999999999999996], [12.1, 2.5999999999999996], [14.1, 1.5999999999999996], [10.1, 2.5999999999999996], [12.1, 5.6], [12.1, 6.6], [6.1, 4.6], [6.1, 5.6], [5.1, 4.6], [1.0999999999999996, 18.6], [3.0999999999999996, 19.6], [1.0999999999999996, 17.6], [-0.9000000000000004, 18.6], [1.0999999999999996, 19.6]]
                    anchor_list.insert(0, [1.5, 7.5]) 
                #anchor_list =[[1.5, 7.5], [8.1, 3.5999999999999996], [8.1, 19.799999999999997], [1.0999999999999996, 4.6], [1.0999999999999996, 7.6], [1.0999999999999996, 7.6], [8.1, 4.6], [14.1, 3.5999999999999996], [3.0999999999999996, 21.6], [5.1, 19.799999999999997], [9.1, 2.5999999999999996]]
                print("anchorlist:" , anchor_list)

                # for num in iot_list:
                #     query_radio(num)
                #     anchor_list =calculate_carla_anchor(26,66,num)
                # #anchor_list =calculate_carla_anchor(26,66)
                # anchor_list=anchor_list[:5]
                # anchor_list.insert(0, [1.5, 7.5]) 
                # print("anchorlist:" , anchor_list)
            elif radionav_case==1:
                for num in iot_list:
                    query_radio416(num)
                generate_output2(iot_list,anchor_number)
                anchor_list =calculate_carla_anchor(26,66)
                anchor = read_file("./Transform/selectbyhand.txt")
                save_lines_to_txt_a("./Transform/selectbyhand_record.txt", anchor)
                #anchor_list =calculate_carla_anchor(26,66)
                anchor_list.insert(0, [1.5, 7.5]) 
                print("anchorlist:" , anchor_list)

                # for num in iot_list:
                #     query_radio(num)
                #     anchor_list =calculate_carla_anchor(26,66,num)
                # #anchor_list =calculate_carla_anchor(26,66)
                # anchor_list=anchor_list[:5]
                # anchor_list.insert(0, [1.5, 7.5]) 
                # print("anchorlist:" , anchor_list)
            elif radionav_case==2:
                
                if anchor_list_from_args is not None:
                    anchor_list = anchor_list_from_args
                else:
                    for num in iot_list:
                        query_radio913(num)
                    
                    #generate_output()
                    generate_output1(iot_list,anchor_number)
                    deepseek_api()
                    anchor_list =calculate_carla_anchor(26,66)
                    anchor_list.insert(0, [1.5, 7.5]) 
                save_anchor(anchor_list)
                print("anchorlist:" , anchor_list)

        #Record all params 
        record_param(iot_list,anchor_case,anchor_number,alpha,tsp_option,dist_case,anchor_list)


        
        
        print("Radio map generation...")
        RM, radio_map_list = radio_map_generation(anchor_list, iot_list, pre_gen)
        
        print("Load radio map to task planner...")
        t_list, count=task_plan(anchor_list, iot_list, RM, alpha, dist_case, tsp_option, algorithm_case)
        save_t_list(t_list)
        
        
        Bk = 1.0 * np.ones((K, 1)) # 假设带宽为1, 单位MHz
        Pk = 10.0 * np.ones((K, 1)) # 假设发射功率为10 mW单位
        sigma_squared = 10 ** (-6)  # 单位mW, 假设噪声功率为 -60 dBm


        M = len(anchor_list)
        data_amount_list=[]
        transfer_list=[]
        for k in range(K):
            sum_collect =0
            transfer=[]
            radio_map = radio_map_list[k]
            for m in range(M):
                if(t_list[m]>0.1):
                    irsim_robot_pos =[anchor_list[m][0],anchor_list[m][1]]
                    pathloss = query_radio_map(radio_map, irsim_robot_pos, iot_list[k])
                    data_amount = communication_perf(pathloss)  # bits
                    transfer_speed = float(calculate_Fik(Bk[k], RM[k,m], Pk[k], sigma_squared))
                    
                    transfer.append(transfer_speed)
                    sum_collect += transfer_speed * t_list[m]
                    #print(f"Collect data from sensor at ({iot_list[k][0]}, {iot_list[k][1]}) with data amount: {data_amount/10**6} Mbits")
                else:
                    continue
            data_amount_list.append(sum_collect)
            transfer_list.append(sum(transfer)/count)
        save_v_all("Anchor_collect.txt",data_amount_list)
        if dist_case !=3:
            save_v_all("Anchor_transfer.txt",transfer_list)
        




        print("Load the goal list from the task graph...")
        txt_path = sys.path[0] + '/point_list_ref.txt'
        point_array = np.loadtxt(txt_path)
        point_list = []

        for index, point in enumerate(point_array):
            point_np_tmp = np.c_[point_array[index]]
            point_list.append(point_np_tmp)

        print("point_list_length:"+str(len(point_list[0])))
        print("point_list:"+str(point_list))

        if len(point_list[0]) !=1:
            gpath = global_plan(point_list)
            print("Load the trajectory from the global planner...")
            txt_path = sys.path[0] + '/global_path.txt'
            point_array = np.loadtxt(txt_path)
            point_list = []

            for index, point in enumerate(point_array):
                point_array_3d = np.append(point_array[index], 0)
                point_np_tmp = np.c_[point_array_3d]
                point_list.append(point_np_tmp)
            cg = curve_generator()

        if len(point_list[0]) ==1:
            with open('point_list_ref.txt', 'r') as file:
                first_line = file.readline()
            print(first_line)
            ref_path_list = first_line
            #ref_path_list = [point_list]
            data_amount_list = [[] for _ in range(K)]
            for k in range(K):
                radio_map = radio_map_list[k]
                irsim_robot_pos =[point_list[0],point_list[1]]
                print(f"The robot is at ({irsim_robot_pos})")
                pathloss = query_radio_map(radio_map, irsim_robot_pos, iot_list[k])

                data_amount = communication_perf(pathloss)  # bits
                data_amount = data_amount/10**6
                #print(f"Collect data from sensor at ({iot_list[k][0]}, {iot_list[k][1]}) with data amount: {data_amount/10**6} Mbits")


                data_amount_list[k].append(alpha/data_amount)
            print("Stay Static and Transfer the data with time:" , data_amount_list)
            record_sum('static',anchor_case,k)


        else:
            ref_path_list = cg.generate_curve(curve_style='line', way_points=point_list, step_size=0.5, min_radius=0.2)
            ref_path_list = smooth_path_angles(ref_path_list)  # 添加这一行
            
            print(ref_path_list)
            print('The total number of waypoints is:', len(ref_path_list))

            if len(point_list[0]) !=1:
                env.draw_trajectory(ref_path_list, traj_type='-c', label='trajectory', linewidth=1) #轨迹颜色为c色，线条加粗

            for k in range(len(iot_location)):
                env.draw_point((iot_location[k][0], iot_location[k][1]), 'Sensor', 8, 'r')

            print("Start executing the task using the local planner...")
            local_plan(ref_path_list, iot_list, radio_map_list, anchor_case, str(args))