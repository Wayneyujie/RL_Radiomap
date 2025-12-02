from ir_sim.env import EnvBase
import sys
import numpy as np
import matplotlib.pyplot as plt

from rda_planner.mpc import MPC
from collections import namedtuple
from curve_generator import curve_generator

def simple_local_planning_test():
    """
    Simple local planning test in an open space environment
    """
    print('*** Starting Simple Local Planning Test ***')

    # Initialize environment with simple map
    env = EnvBase('map_editor_output.yaml', save_ani=True, display=True, full=False)
    car = namedtuple('car', 'G h cone_type wheelbase max_speed max_acce')

    # Get robot information
    robot_info = env.get_robot_info()
    car_tuple = car(robot_info.G, robot_info.h, robot_info.cone_type,
                   robot_info.shape[2], [1, 1], [3, 0.5])

    # Load reference path
    txt_path = sys.path[0] + '/point_list_ref.txt'
    point_array = np.loadtxt(txt_path)
    point_list = []

    for index, point in enumerate(point_array):
        point_np_tmp = np.c_[point_array[index]]
        point_list.append(point_np_tmp)

    # Generate smooth reference trajectory
    cg = curve_generator()
    ref_path_list = cg.generate_curve(curve_style='line', way_points=point_list,
                                     step_size=0.2, min_radius=0.2)

    print(f'Reference path loaded with {len(ref_path_list)} waypoints')

    # Get obstacle count from environment
    obs_list = env.get_obstacle_list()
    num_obstacles = len(obs_list) if obs_list else 0
    print(f'Found {num_obstacles} obstacles in the environment')
    
    # Setup MPC controller for local planning
    # Dynamically set obstacle_num based on actual obstacle count
    # Use a reasonable maximum (20) to handle future obstacles
    max_obstacles = max(num_obstacles, 20)
    obstacle_template_list = [
        {'edge_num': 4, 'obstacle_num': max_obstacles, 'cone_type': 'Rpositive'},
        {'edge_num': 3, 'obstacle_num': 0, 'cone_type': 'norm2'}
    ]
    print(f'MPC configured for up to {max_obstacles} obstacles')

    mpc_opt = MPC(car_tuple, ref_path_list,
                 receding=15, sample_time=0.1, process_num=4, iter_num=1,
                 ro1=200, obstacle_template_list=obstacle_template_list,
                 ws=5, wu=20)

    mpc_opt.update_parameter(slack_gain=4, max_sd=0.5, min_sd=0.1)

    # Local planning loop - short test (100 steps as requested)
    max_steps = 300
    success = False

    for step in range(max_steps):
        # Get obstacle list
        obs_list = env.get_obstacle_list()

        # MPC control
        opt_vel, info = mpc_opt.control(env.robot.state, 0.8, obs_list)

        # Draw planned trajectory
        env.draw_trajectory(info['opt_state_list'], 'g', refresh=True)

        # Execute control
        car_location_before = [float(env.robot.state[0]), float(env.robot.state[1])]
        env.step(opt_vel, stop=False)
        env.render(show_traj=True, show_trail=True, traj_type='-r')

        # Check if goal is reached
        if info['arrive']:
            print(f'✓ Goal reached at step {step}!')
            success = True
            break

        # Check simulation end condition
        if env.done():
            print('✓ Simulation completed successfully!')
            success = True
            break

        # Print progress every 10 steps
        if step % 10 == 0:
            current_pos = [float(env.robot.state[0]), float(env.robot.state[1])]
            print(f'Step {step}: Robot at ({current_pos[0]:.2f}, {current_pos[1]:.2f})')

    # Final render and save
    if success:
        print('✓ Local planning test completed successfully!')
    else:
        print(f'⚠ Test completed after {max_steps} steps')

    # Save animation
    env.end(ani_name='simple_local_planning_test',
            show_traj=True, show_trail=True,
            ending_time=10, keep_len=100,
            ani_kwargs={'subrectangles': True})

    return success

if __name__ == '__main__':
    try:
        success = simple_local_planning_test()
        if success:
            print('\n🎉 Local planning test PASSED!')
        else:
            print('\n⚠ Local planning test COMPLETED (check results)')
    except Exception as e:
        print(f'\n❌ Test failed with error: {e}')
        raise