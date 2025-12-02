from ir_sim.env import EnvBase
import sys
import numpy as np
import scipy.io
import yaml
import time
import os

from rda_planner.mpc import MPC
from collections import namedtuple
from curve_generator import curve_generator
from numpy.linalg import norm
from tsp_path import tsp_path
import matplotlib.pyplot as plt

# environment
env = EnvBase('ral_1220.yaml', save_ani=True, display=True, full=False)
car = namedtuple('car', 'G h cone_type wheelbase max_speed max_acce')

def global_plan():    

    print('*** Computing the joint communication and navigation path ***')

    anchor_list = [[10, 2.0], [1.6,6.5], [1.6,7.5], [12, 8], [12,19.5], [1.8, 19.5]]
    anchor_list_array = np.array(anchor_list)

    num_anchor = len(anchor_list)

    D1 = np.zeros([num_anchor, num_anchor])


    # directly compute the distance
    for i, anchor_i in enumerate(anchor_list_array):
        for j, anchor_j in enumerate(anchor_list_array):
            D1[i,j] = norm(anchor_i-anchor_j)

    # load the results from Astar algorithm
    txt_path = sys.path[0] + '/astar/array.txt'
    D_array = np.loadtxt(txt_path)
    D2 = D_array * 0.1 # carla/irsim = 10/1
    
    print("The distance matrix D1 is:", D1)
    print("The distance matrix D2 is:", D2)

    v_all = np.ones(num_anchor)  # full path

    iot_list = [ [12.3, 7.5]]
    iot_list_array = np.array(iot_list)

    anchor_list_array = anchor_list_array.T
    v_opt = v_all

    tsp_path(anchor_list_array, iot_list_array, D2, v_opt, num_anchor)

def local_plan():
    
    robot_info = env.get_robot_info()
    car_tuple = car(robot_info.G, robot_info.h, robot_info.cone_type, robot_info.shape[2], [1, 1], [3, 0.5])
    obstacle_template_list = [ {'edge_num': 4, 'obstacle_num': 15, 'cone_type': 'Rpositive'},{'edge_num': 3, 'obstacle_num': 0, 'cone_type': 'norm2'}]
    mpc_opt = MPC(car_tuple, ref_path_list, receding=15, sample_time=0.1, process_num=4, iter_num=1, ro1=200, obstacle_template_list=obstacle_template_list, ws=5, wu=20)
    
    mpc_opt.update_parameter(slack_gain=4, max_sd=0.5, min_sd=0.1)  

    # load radio map
    radio_map = np.load("radio_Tjunc.npy")  # map T_junction
    data_amount_list = []

    for i in range(1000):   
        obs_list = env.get_obstacle_list() # 获取障碍物列表

        opt_vel, info = mpc_opt.control(env.robot.state, 0.8, obs_list)
        env.draw_trajectory(info['opt_state_list'], 'g', refresh=True)
  
        car_location_before=[float(env.robot.state[0]),float(env.robot.state[1])]
        env.step(opt_vel, stop=False)
        env.render(show_traj=True, show_trail=True, traj_type='-r')

        if env.done():
            env.render_once(show_traj=True, show_trail=True)
            break

        if info['arrive']:
            print('arrive at the goal') 
            break
        
        car_location_after = [float(env.robot.state[0]),float(env.robot.state[1])]

        irsim_robot_pos = [(car_location_before[0] + car_location_after[0]) / 2, 
                   (car_location_before[1] + car_location_after[1]) / 2]


        print(f"The robot is at ({irsim_robot_pos})")
        pathloss = query_radio_map(radio_map, irsim_robot_pos)

        data_amount = communication_perf(pathloss)  # bits
        print(f"Collect data from sensor at ({iot_x}, {iot_y}) with data amount: {data_amount/10**6} Mbits")

        data_amount_list.append(data_amount)

        # Save data_amount_list to the result folder
        result_folder = './results'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)
        data_amount_list_path = os.path.join(result_folder, 'data_amount_list.txt')
        np.savetxt(data_amount_list_path, data_amount_list, fmt='%f')

    env.end(ani_name='path_track', show_traj=True, show_trail=True, ending_time=10, keep_len=100, ani_kwargs={'subrectangles':True})


def query_radio_map(radio_map, irsim_robot_pos):
    data = radio_map
    # check the size of radio map
    radio_map_height = data.shape[1]
    radio_map_width = data.shape[2]
    print('The radio map has a size of:', radio_map_height, 'x', radio_map_width)

    # T junction, iot poses at irsim and radio maps, respectively
    irsim_iot_pos = [12.3, 7.5]

    # convert robot pose in irsim to radio map for futher query
    translation = [13.9, -12.6]
    radio_robot_pos = [irsim_robot_pos[0] + translation[0], irsim_robot_pos[1] + translation[1]]

    # convert radio map position to cell index ([0,0] at the upper left corner)
    cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
    cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
    rm_index = [0, cellindex_y, cellindex_x]

    # load the path_loss
    path_loss = data[0, cellindex_y, cellindex_x]

    print("Path loss from", irsim_iot_pos, "to", irsim_robot_pos, "in irsim:")
    print(path_loss) 

    return path_loss
     

def communication_perf(pathloss):
    tau = 0.1  # time step
    robot_power = 10 # 10 mW
    noise_power = 10**(-8) # mW, noise power, -80 dBm
    pathloss_linear = 10 ** (pathloss/10)

    iot_file_path = "./iot_Tjunc.txt" 
    with open(iot_file_path, "r") as file:
        for line in file:
            iot_coords = line.strip().split(",")
            iot_x, iot_y, iot_B = map(float, iot_coords)

    Iot_com = tau * iot_B * np.log(1 + robot_power * pathloss_linear / noise_power) /np.log(2)
    
    return Iot_com


def save_matrix_to_txt(matrix, filename):
    with open(filename, 'w') as file:
        for i in range(len(matrix)):
                file.write(f"{matrix[i][0][0]} {matrix[i][1][0]} {matrix[i][2][0]}\n")

if __name__ == '__main__':

    global_plan()

    # load the results from communication aware task planner
    txt_path = sys.path[0] + '/point_list_ref.txt'
    point_array = np.loadtxt(txt_path)
    point_list = []

    for index, point in enumerate(point_array):
        point_np_tmp = np.c_[point_array[index]]
        point_list.append(point_np_tmp)

    cg = curve_generator()
    ref_path_list = cg.generate_curve(curve_style='line',way_points=point_list, step_size=0.2, min_radius=0.2, )
    print('The total number of waypoints is:', len(ref_path_list))
    
    with open('./iot_Tjunc.txt', 'r') as file:
        for line in file:
            line = line.strip()
            iot_x, iot_y, _ = line.split(',')
            iot_x = float(iot_x)
            iot_y = float(iot_y)
            env.draw_point((iot_x, iot_y), 'Sensor', 8, 'r')

    local_plan()