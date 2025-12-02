
# ***********************************************************************
import numpy as np
import matplotlib.pyplot as plt
import sys

# 初始化列表来存储时间、油门和转向数据 a=case2_CAMPC b=case1_CAMPC c=case2_MPC d=case1_MPC
time_list_a = []
time_list_b = []
time_list_c = []
time_list_d = []
speed_list_a = []
speed_list_b = []
speed_list_c = []
speed_list_d = []
steer_list_a = []
steer_list_b = []
steer_list_c = []
steer_list_d = []

# 加载文本文件数据
txt_path = './results/cmd_vel.txt'  # 根据实际文件名修改路径
d_line = 0
c_line = 0
b_line = 0
with open(txt_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
        data = line.split()
        if len(data) == 12:
            d_line += 1
            c_line += 1
            b_line += 1
        elif len(data) == 9:
            c_line += 1
            b_line += 1
        elif len(data) == 6:
            b_line += 1


point_array_a = np.loadtxt(txt_path,usecols=(0,1,2))

# point_array_d = np.loadtxt(txt_path,usecols=(9,10,11),max_rows=d_line)

# 解析数据
for point in point_array_a:
    time_list_a.append(float(point[0]))
    speed_list_a.append(float(point[1]))
    steer_list_a.append(float(point[2]))


# 调整时间序列以匹配数据点数
time_list = [0.5 * i for i in range(0, len(time_list_a))]

# 将列表转换为NumPy数组以便绘图
t_a = np.array(time_list_a)/10
speed_a = np.array(speed_list_a)
steer_a = np.array(steer_list_a)

print(speed_a)
max_sp = max(speed_a)
min_sp = min(speed_a)
max_st = max(steer_a)
min_st = min(steer_a)


# 创建绘图
fig, axs = plt.subplots(2, 1)  # 使用两个子图：一个用于速度，一个用于转向
# 添加其他实验场景的速度数据，设置不同的颜色和标签
# axs.plot(t, speed2, color="green", label="实验场景2")
# axs.plot(t, speed3, color="blue", label="实验场景3")

# 绘制速度数据

axs[0].plot(t_a, speed_a, color="blue", linewidth=3.0,linestyle='-')
# axs[0].plot(t_d, speed_d, color="green", linewidth=3.0,linestyle=':',label="case1_MPC")

# 添加其他实验场景的速度数据，设置不同的颜色和标签
axs[0].set_xlim(0,max(t_a))
# axs[0].set_ylim(min(throttle) - 0.1, max(throttle) + 0.1)
axs[0].set_ylim(min_sp-(max_sp-min_sp)/10, max_sp+(max_sp-min_sp)/10)
# axs[0].set_yticks(np.arange(min_sp-(max_sp-min_sp)/10, max_sp+(max_sp-min_sp)/10, (max_sp-min_sp)/10))
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Speed(m/s)')
axs[0].legend()
axs[0].grid(True)


# 绘制转向数据
axs[1].plot(t_a, steer_a, color="red", linewidth=3.0,linestyle='-')
# axs[1].plot(t_d, steer_d, color="green", linewidth=3.0,linestyle=':',label="case1_MPC")
axs[1].set_xlim(0,max(t_a))
# axs[1].set_ylim(min(steer) - 0.1, max(steer) + 0.1)
axs[1].set_ylim(min_st-(max_st-min_st)/10,max_st+(max_st-min_st)/10)
# axs[1].set_yticks(np.arange(min_st-(max_st-min_st)/10,max_st+(max_st-min_st)/10,(max_st-min_st)/10))
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Steer(rad)')
axs[1].legend()
axs[1].grid(True)

# 调整布局并显示图表
fig.tight_layout()
cur_path = sys.path[0]
plt.savefig(cur_path+ '/'+ 'NLOS1' +'.pdf', bbox_inches='tight')
plt.show()

#nlos1*****************************************************************************
