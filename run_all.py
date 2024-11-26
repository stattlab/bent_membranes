
import sys
import os
import subprocess


angles = str(sys.argv[2])
x_box_size = int(sys.argv[1])
lipid = str(sys.argv[4])
i = str(sys.argv[3])


subprocess.run(['python3', 'init_small.py', str(x_box_size), str(i), lipid])
subprocess.run(['python3', 'run_small.py', str(x_box_size), str(i), lipid])
for angle in angles:
    subprocess.run(['python3', 'init_stitched.py', str(x_box_size), str(angle), str(i), lipid])
    subprocess.run(['python3', 'fire_stitched.py', str(x_box_size), str(angle), str(i), lipid])
    subprocess.run(['python3', 'run_stitched.py', str(x_box_size), str(angle), str(i), lipid])
