import numpy as np

# load radio map
# data = np.load("radio_Tjunc.npy")  # map T_junction
# # check the size of radio map
# radio_map_height = data.shape[1]
# radio_map_width = data.shape[2]
# print('The radio map has a size of:', radio_map_height, 'x', radio_map_width)

# cellindex_x=59
# cellindex_y=19

# # input a robot pose (anchor point) at irsim 
# irsim_robot_pos = [0, 0]
# radio_robot_pos = [0,0]



# # convert robot pose in irsim to radio map for futher query
# translation = [13.9, -12.6]


# radio_robot_pos[0] = cellindex_x-radio_map_width/2
# radio_robot_pos[1] = cellindex_y-radio_map_height/2
# irsim_robot_pos[0] = radio_robot_pos[0]-translation[0]
# irsim_robot_pos[1] = radio_robot_pos[1]-translation[1]
# print(irsim_robot_pos)



def calculate_carla_anchor(radio_map_height,radio_map_width):
    values_list = []  # 用于存储每个聚类中心的路径损失值
    goal_data = []

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
    input_file = "./Transform/selectbyhand.txt"   # if invoked by sim_main_proposed.py
    with open(input_file, 'r') as f:
        lines = f.readlines()
    top_intensity =lines
    print("top_intensity:", top_intensity)
    for line in top_intensity:
        # Extract the x and y values from the line using string manipulation
        if "x=" in line and "y=" in line:
            # Find the x and y values using splitting
            parts = line.split(',')
            x_part = float(parts[0].split('=')[1].strip())
            y_part = float(parts[1].split('=')[1].strip())
            
            # Convert the x and y values to integers (if you need to use them in calculations)
            cellindex_x = int(x_part)
            cellindex_y = int(y_part)
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
        # input a robot pose (anchor point) at irsim 
        irsim_robot_pos = [0, 0]
        radio_robot_pos = [0,0]



        # convert robot pose in irsim to radio map for futher query
        translation = [13.9, -12.6]


        radio_robot_pos[0] = cellindex_x-radio_map_width/2
        radio_robot_pos[1] = cellindex_y-radio_map_height/2
        irsim_robot_pos[0] = radio_robot_pos[0]-translation[0]
        irsim_robot_pos[1] = radio_robot_pos[1]-translation[1]

        #values_list.append(f"{irsim_robot_pos[0]:.3f} {irsim_robot_pos[1]:.3f}") 
        values_list.append([irsim_robot_pos[0],irsim_robot_pos[1]])

        # x= irsim_robot_pos[0]*10 - 136.5
        # y= irsim_robot_pos[1] *10 -38
        
    
        # goal_data.append({
        #         'x': x,
        #         'y': y,
        #         'yaw': 96,  # 假设 yaw 的变化为递增，具体情况可根据需要修改
        #         'comment': f'MOVE: Goal1'  # 每个目标的 comment 都不一样
                
        #     })

    #print("values_list:", values_list)
    return values_list        
    # output_path ='irsim_anchor.txt'
    # with open(output_path, 'w') as f:
    #     for value in values_list:
    #         f.write(f"{value}\n")


    # output_path ='carla_anchor.txt'
    # with open(output_path, 'w') as f:
    #     for i, goal in enumerate(goal_data):
    #         if i != len(goal_data) - 1:  # 如果不是最后一个目标，末尾加逗号
    #             f.write(f"{goal},\n")
    #         else:  # 最后一个目标，不加逗号
    #             f.write(f"{goal}\n")



#calculate_carla_anchor(26,66)