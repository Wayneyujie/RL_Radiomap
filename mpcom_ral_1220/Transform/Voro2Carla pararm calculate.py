#用于计算Voro转Carla的系数
import numpy as np

# x1 = w * x + b
# y1 = u * y + c

# Set up the system of equations for x1 and y1
# x1 equations: [w * x + b = x1]
# y1 equations: [u * y + c = y1]

# Equations for x1: w * x + b = x1
x_points = np.array([-14.5, -26.7, 0.5])
x_values = np.array([525, 497, 556])

# Equations for y1: u * y + c = y1
y_points = np.array([162, 146.2, -22.1])
y_values = np.array([526, 491, 159])

# Setting up matrices for solving the system
# For x: Ax = B where A is the matrix of x points and B is the x_values
A_x = np.vstack([x_points, np.ones(len(x_points))]).T
B_x = x_values

# For y: Ay = B where A is the matrix of y points and B is the y_values
A_y = np.vstack([y_points, np.ones(len(y_points))]).T
B_y = y_values

# Solve the system using least squares (Ax = B => x = inv(A^T * A) * A^T * B)
params_x = np.linalg.lstsq(A_x, B_x, rcond=None)[0]
params_y = np.linalg.lstsq(A_y, B_y, rcond=None)[0]

print(params_x, params_y)




# with open("coordinates.txt", "r") as file:
#     for line in file:
#         # 每行读取x, y
#         x, y = map(float, line.split())  # 将每行的x和y值转换为浮动数
        

#         x1=(x -555.375)/2.165
#         y1=(y-202.703)/1,984

#         x2=x*2.165+555.375
#         y2=y*1.984+202.703
#         print(f"x1: {x1}, y1: {y1}")
#         print(f"x2: {x2}, y2: {y2}")


# with open("coordinates.txt", "r") as file:
#     for line in file:
#         # 每行读取x, y
#         x, y = map(float, line.split())  # 将每行的x和y值转换为浮动数
        

#         x1=(x -555.375)/2.165
#         y1=(y-202.703)/1.984

#         #Carla 2 irsim
#         irsim_robot_pos = [0, 0]
#         irsim_robot_pos[0] =  (x1+136.5)/10
#         irsim_robot_pos[1] = (y1+38)/10
#         translation = [13.9, -12.6]

#         #irsim 2 Radio
#         radio_map_width=26
#         radio_map_height=66
#         radio_robot_pos = [irsim_robot_pos[0] + translation[0], irsim_robot_pos[1] + translation[1]]
#         cellindex_x = int(radio_map_width/2 + radio_robot_pos[0])
#         cellindex_y = int(radio_map_height/2 + radio_robot_pos[1])
#         x= cellindex_x
#         y= cellindex_y
    
#         print(f"x1: {x1}, y1: {y1}")

with open("coordinates.txt", "r") as file:
    for line in file:
        # 每行读取x, y
        x, y = map(float, line.split())  # 将每行的x和y值转换为浮动数
        

        x1=(x -555.375)/2.165
        y1=(y-202.703)/1.984

        radiox=(x1+600)/647*66
        radioy=(y1+34.4)/251.5*26

    
        print(f"x1: {radiox}, y1: {radioy}")