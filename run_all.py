
import sys
import os
import subprocess


angle = str(sys.argv[2]) # angle in degrees
x_box_size = int(sys.argv[1]) #approximate size of the x dimension of the small membrane patch (equilibrated dimension ~ .6 x_box_size)
lipid = str(sys.argv[4]) #DLPC, DOPC, DPPC, DOPE
i = str(sys.argv[3]) #run number


subprocess.run(['python3', 'init_small.py', str(x_box_size), str(i), lipid])
subprocess.run(['python3', 'run_small.py', str(x_box_size), str(i), lipid])
subprocess.run(['python3', 'init_stitched.py', str(x_box_size), str(angle), str(i), lipid])
subprocess.run(['python3', 'fire_stitched.py', str(x_box_size), str(angle), str(i), lipid])
subprocess.run(['python3', 'run_stitched.py', str(x_box_size), str(angle), str(i), lipid])
