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

## configuration:
save_cm_numpy = True
cm_cell_size = (1, 1)   # covermap cell size
camera_position = [22,0,40]
camera_look_at = [22,0,0]

# These cases are the red points in rm_1215
# case = 5
# case = 6
# case = 7
case = 8

if case == 0:
    # case map center 
    iot_position = [0, 0, 3]
    iot_orientation = [0, 0, 0]

elif case == 1:
    # case mpcom
    iot_position = [26.2, -11.5, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 2:
    # case open
    iot_position = [-3, -3, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 3:
    # case room
    iot_position = [20.5, -9.0, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 4:
    # case corridor
    iot_position = [26.1, -5.2, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 5:
    # case T junction near conference door
    iot_position = [15.9, -5.1, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 6:
    # case T junction near F1015 door
    iot_position = [15.9, -12.6, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 7:
    # case corridor corner
    iot_position = [26.2, -12.6, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 8:
    # case F1015 corner left
    iot_position = [26.2, -5.1, 3]
    iot_orientation = [0, 1.57, 1.57]

elif case == 9:
    # case corridor middle
    iot_position = [15+13.9, 19-12.6, 2]
    iot_orientation = [0, 1.57, 1.57]

elif case == 10:
    # case conference room
    iot_position = [28.9, -7.6, 3]
    iot_orientation = [0, 0, 0]


scene = load_scene('INVS2/INVS.xml') # load mesh

# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")


# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="cross")

# Add transmitter instance to scene
tx = Transmitter(name="tx",
                 position=iot_position, orientation=iot_orientation, color=[1, 0, 0],)


scene.add(tx)
scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)


cm = scene.coverage_map(max_depth=5,
                        diffraction=True, # Disable to see the effects of diffraction
                        cm_cell_size=cm_cell_size, # Grid size of coverage map cells in m
                        combining_vec=None,
                        precoding_vec=None,
                        num_samples=int(1e6)) # Reduce if your hardware does not have enough memory


cm.show()

print(cm._path_gain)

if save_cm_numpy:
    cm_numpy = 10.*np.log10(cm._path_gain.numpy())

    map_height = cm_numpy.shape[1]
    map_length = cm_numpy.shape[2]

    gridsize = 1
    cellindex_x = int(map_length/2 + iot_position[0] * gridsize)
    cellindex_y = int(map_height/2 + iot_position[1] * gridsize)

    np.save('radio_case'+str(case), cm_numpy)


my_cam = Camera("my_cam", position=camera_position, orientation=(0, 0, 0), look_at=camera_look_at)
scene.add(my_cam)

scene.render(camera="my_cam", resolution=[650,500], num_samples=512, show_devices=True, show_paths=False, coverage_map=cm)
plt.show()


