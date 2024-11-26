import numpy as np
import sys
from martini3 import molecules
from martini3 import init_cell
import hoomd
import os
import gsd.hoomd
import quaternionic


def calc_CM(points, box, counts):
    # transform coords from -L/2,L/2 to 0,2pi range
    theta = (points + 0.5 * box) / box * 2.0 * np.pi
    # calculate average
    si = np.sin(theta)
    co = np.cos(theta)
    theta_av = (
        np.arctan2(-np.sum(si, axis=0) / counts, -np.sum(co, axis=0) / counts) + np.pi
    )
    # transform coords back to -L/2,L/2 range
    z_com = box * theta_av / (2.0 * np.pi) - 0.5 * box
    return z_com


def dist_pbc(x0, x1, Box):
    delta = x0 - x1
    delta = np.where(
        delta > 0.5 * Box, delta - Box, np.where(delta < -0.5 * Box, delta + Box, delta)
    )
    return delta
def xz_rot(mat,angle):
    xz_rot_mat = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    return np.matmul(mat,xz_rot_mat)


def main(x_box_size,angle, iter_num, lipid_type="DOPC"):
    path = "data/" + lipid_type + "/" + x_box_size + "/" +iter_num+"/" +angle + "/"
    if not os.path.exists(path):
        os.makedirs(path)
        
    # contents setup
    contents = molecules.Contents()

    init_gsd_path = (
        "data/" + lipid_type + "/" + x_box_size + "/"+iter_num+"/small/"
    )
    frame = gsd.hoomd.open(name=init_gsd_path +  "traj.gsd", mode="r")[-1]
    x_box_small_equil =frame.configuration.box[0]
    y_box_small_equil = frame.configuration.box[1]
    angle_rad = float(angle)*np.pi/180
    
    x_box_size = 12+ x_box_small_equil*(1+np.cos(angle_rad)*2 + 2*np.cos(angle_rad/2))
    y_box_size = y_box_small_equil
    z_box_size = x_box_small_equil*(1+np.sin(angle_rad)+ np.sin(angle_rad/2)) +2+4*np.cos(angle_rad) 

    q5_type_index = frame.particles.types.index("Q5")
    total_lipid = np.sum(frame.particles.typeid == q5_type_index)

    beads_per_lipid = 12 if not "L" in lipid_type else 10
    num_water_x = int(x_box_size*2.1)
    num_water_y = int(y_box_size*2.1)
    grid_upper = np.zeros((num_water_x,num_water_y)) -20
    grid_lower = np.zeros((num_water_x,num_water_y)) + 20

    x_range = np.linspace(-x_box_size/2,x_box_size/2,num_water_x)
    y_range = np.linspace(-y_box_size/2,y_box_size/2,num_water_y)

 
    frozen= []

    #place 4 patches with hardoded angles, x_shift, and z_shift based off of size of equilibrated patch
    num_segments = 4 
    segments = np.linspace(-angle_rad,angle_rad-angle_rad/num_segments*2,num_segments)
    x_shifts = [-x_box_small_equil*(.5 + np.cos(angle_rad/2) ),-x_box_small_equil*(.5),x_box_small_equil*(.5),x_box_small_equil*(.5+np.cos(angle_rad/2))]
    z_shifts = x_box_small_equil*np.sin(angle_rad) + np.array([-x_box_small_equil*(np.sin(angle_rad)/2  + np.sin(angle_rad/2) ),-x_box_small_equil*(np.sin(angle_rad/2)/2 ),-x_box_small_equil*(np.sin(angle_rad/2)/2 ),-x_box_small_equil*(np.sin(angle_rad)/2  + np.sin(angle_rad/2) )])
    

    index = 0
    for seg_num, seg_angle, x,z in zip(range(num_segments), segments, x_shifts,z_shifts):
        lipid_placed = 0
        lipid_actually_placed = 0 

        for i in range(total_lipid):
            if lipid_type=="DOPC":
                lipid = molecules.make_DOPC(contents)
            elif lipid_type=="DOPE":
                lipid = molecules.make_DOPE(contents)
            elif lipid_type == "DPPE":
                lipid = molecules.make_DPPE(contents)
            elif lipid_type == "DLPC":
                lipid = molecules.make_DLPC(contents)
            else:
                lipid = molecules.make_DPPC(contents)
            positions = frame.particles.position[
                lipid_placed * beads_per_lipid : 
                (lipid_placed + 1) * beads_per_lipid,
                :]
            pbc_violation = False
            mean_position = np.mean(positions,axis = 0)
            for position in positions:
                if abs(position[0]-mean_position[0])>2.2:
                    pbc_violation = True

            if pbc_violation :
                for i, position in enumerate(positions):
                    # Check for x boundary crossing and shift to positive side if needed
                    if position[0] < 0:
                        positions[i][0] += x_box_small_equil

            for i, pos in enumerate(positions):
                #add beads that are close to either edge to frozen (and frozen.csv)
                if (seg_num == 0 and pos[0] < -x_box_small_equil/2+1.5) or (seg_num == num_segments-1 and pos[0] >x_box_small_equil/2-2 ) :
                    if (seg_num != num_segments-1  or not np.any(positions[:,0]>x_box_small_equil/2)):
                        frozen.append(index + lipid_actually_placed * beads_per_lipid+i)

            if (seg_num != num_segments-1  or not np.any(positions[:,0]>x_box_small_equil/2)):
                for j in range(beads_per_lipid):
                    pre_shift = np.array(positions[j])
                    percent_across = (pre_shift[0]+x_box_small_equil/2)/x_box_small_equil
                    new_angle = seg_angle + angle_rad/num_segments*2*percent_across
                    shift_x = xz_rot(pre_shift,-new_angle)
                    lipid.position[j] = (shift_x + np.array([x, 0, z])).tolist()
                    x_pos = lipid.position[j][0]
                    y_pos = lipid.position[j][1]
                    z_pos = lipid.position[j][2]

                    #update grid used to place water
                    closest_x = np.abs(x_range-x_pos).argmin()
                    closest_y = np.abs(y_range-y_pos).argmin()
                    if z_pos > grid_upper[closest_x,closest_y]:
                        grid_upper[closest_x,closest_y] = z_pos
                    elif z_pos < grid_lower[closest_x,closest_y]:
                        grid_lower[closest_x,closest_y] = z_pos
                contents.add_molecule(lipid)
                lipid_actually_placed+=1
            lipid_placed+=1
        index = index+lipid_actually_placed*beads_per_lipid

    x_place_water = np.linspace(
        -x_box_size / 2 , x_box_size / 2 , num_water_x
    )
    y_place_water = np.linspace(
        -y_box_size / 2 , y_box_size / 2 , num_water_y
    )
    z_place_water = np.linspace(
        -z_box_size / 2 + 0.2, z_box_size / 2 - 0.2, int(z_box_size * 2.1)
    )

    for x_iter2, x2 in enumerate(x_place_water):
        for y_iter2, y2 in enumerate(y_place_water):
            for z2 in z_place_water:
                dont_place = False
                if x_iter2 > 2 and x_iter2 < len(x_place_water)-3 and y_iter2 > 2 and y_iter2 < len(y_place_water)-3:
                    for l in range(7):
                        for k in range(7):
                            if z2 > grid_lower[x_iter2+l-3,y_iter2+k-3]-.3 and z2<grid_upper[x_iter2+l-3,y_iter2+k-3]+.3:
                                dont_place = True
                else:
                    dont_place = True
                if not dont_place:
                    contents = molecules.add_water(

                        contents, x_shift=x2+np.random.rand()*.1-.05, y_shift=y2+np.random.rand()*.1-.05, z_shift=z2+np.random.rand()*.1-.05)
    np.savetxt(path+'frozen.csv', np.array(frozen,dtype = int),  fmt='%d', delimiter=',')
    lj, coulomb, bond_harmonic, angle_forces, dihedrals,_,_ = init_cell.init_cell(
        contents, path, box_size=[x_box_size, y_box_size, z_box_size], pair_on=False
    )
    # #init cell also saves the gsd


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 conc_bc_in_lipid x_box_size angle iter_num lipid_type")
        sys.exit(1)
    main(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))
