
import sys
import os
import subprocess


iter_nums = [0, 1, 2, 3, 4]
lipids = ["DPPC", "DOPC","DPPE","DOPE"]
angles = [0, 22.5, 45, 67.5, 90]
x_box_size = int(sys.argv[1])

for i in iter_nums:
    for lipid in lipids:
        subprocess.run(['python3', 'init_small.py', str(x_box_size), str(i), lipid])
        subprocess.run(['python3', 'run_small.py', str(x_box_size), str(i), lipid])
        for angle in angles:
            subprocess.run(['python3', 'init_stitched.py', str(x_box_size), str(angle), str(i), lipid])
            subprocess.run(['python3', 'fire_stitched.py', str(x_box_size), str(angle), str(i), lipid])
            subprocess.run(['python3', 'run_stitched.py', str(x_box_size), str(angle), str(i), lipid])

#export LD_LIBRARY_PATH=/home/jamesft2/programs/miniconda/envs/hoomd/lib:libcudart.so.12
