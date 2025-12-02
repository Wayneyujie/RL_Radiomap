
# install 
conda create -n sionna python=3.10
pip install sionna

# activate 
conda activate sionna

# run 
python radio_map.py

# view coordinate system 
irsim map position [0,0] see irsim_coordinate.png
radio map position [0,0] see radio_coordinate.png

Note:
radio map position [0,0] located at the center (cell index [13,33])
radio map cell index [0,0] located at the upper left corner