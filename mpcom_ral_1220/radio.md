## Prerequisite
- Python >= 3.8
- numpy
- cvxpy
- [ir_sim==1.1.9](https://github.com/hanruihua/ir_sim): A python based 2d robot simulator for robotics navigation algorithm. 
- [GenerateCurveTool](https://github.com/hanruihua/GenerateCurveTool): A tool of generating the common curves from way points for the robot path planning, including dubins path, reeds shepp, etc.

# Install cvxpy
```
pip install cvxpy[MOSEK]
```
WARN: MOSEK solver requires licence. Ask for one at: https://www.mosek.com/products/academic-licenses/

# Install ir_sim 
```
pip install ir_sim
```

## ********************
## Generate radio map
see ./radio/readme_radio.md

## Run MPCOM
python3 baseline.py 

## View Evaluation results
The video is stored at: ./animation/xxx.gif

# 1. Communication data amount is stored at: ./results/data_amount_list.txt 
Unit: bits in 0.1s

# 2. Robot trajectory in irsim is stored at: ./results/traj.txt
Transform coordinate system from irsim to carla:
```
bash
python3 tf_irsim_to_carla.py ./results/traj.txt
```
Robot trajectory in carla is stored at: ./carla/carla_traj.txt
NOTE: Need to paste carla_traj.txt to: irsim_to_carla/src/irsim_to_carla/opt_planner_ros/src

# 3. Robot control in irsim is stored at: ./results/cmd_vel.txt
View the control results
```
python3 plot_cmd.py
```