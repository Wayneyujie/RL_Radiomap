import numpy as np

# load radio map
data = np.load("radio_Tjunc.npy")  # map T_junction

# check the size of radio map
radio_map_height = data.shape[1]
radio_map_width = data.shape[2]
print('The radio map has a size of:', radio_map_height, 'x', radio_map_width)

# T junction, iot poses at irsim and radio maps, respectively
irsim_iot_pos = [12.3, 7.5]
radio_iot_pos = [26.2, -5.1]

# input a robot pose (anchor point) at irsim 
irsim_robot_pos = [12, 0]

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
