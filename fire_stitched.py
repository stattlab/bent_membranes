import numpy as np
import sys
from martini3 import particles
from martini3 import force_fields
from martini3 import molecules
from martini3 import init_cell
import datetime
import hoomd
import os
import gsd.hoomd


class Status:
    def __init__(self, simulation):
        self.simulation = simulation

    @property
    def seconds_remaining(self):
        try:
            return (
                self.simulation.final_timestep - self.simulation.timestep
            ) / self.simulation.tps
        except ZeroDivisionError:
            return 0

    @property
    def etr(self):
        return str(datetime.timedelta(seconds=self.seconds_remaining))
def get_not_water(file_name,num_beads):
    frame = gsd.hoomd.open(file_name)[-1]
    type_water = np.max(frame.particles.typeid)
    first_water = frame.particles.typeid.tolist().index(type_water)
    print(first_water)
    return range(0,first_water)
def get_not_frozen(path,num_beads):
    unfrozen = []
    frozen = []
    import csv
    with open(path + "frozen.csv","r") as file:
        f = csv.reader(file)
        for line in f:
            frozen.append(int(line[0]))
    for i in range(num_beads):
        if i not in frozen:
            unfrozen.append(i)

    return unfrozen
def main(x_box_size,angle, iter_num, lipid_type="DOPC"):
    path = "data/" + lipid_type + "/" + x_box_size + "/"+ iter_num+"/" + angle + "/" 
    if not os.path.exists(path):
        raise Exception("no data present")

    lj, coulomb, bond_harmonic, angle_forces, dihedrals,impropers,rigid = force_fields.forces_from_gsd(
        path, "init.gsd"
    )
    try:
        sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=int(6) * 500)
    except:
        raise("Not GPU")

    name = "init.gsd"

    sim.create_state_from_gsd(filename=path + name)
    dt = 0.001
    frame = gsd.hoomd.open(path + name)[-1]
    num_beads = frame.particles.N
    rigid_centers_and_free = hoomd.filter.All()
    unfrozen = hoomd.filter.Tags(get_not_frozen(path,num_beads))
    not_water = hoomd.filter.Tags(get_not_water(path+"init.gsd",num_beads))
    temp_filter = hoomd.filter.Intersection(rigid_centers_and_free,unfrozen)
    filter = temp_filter
    fire = hoomd.md.minimize.FIRE(dt=dt,forces = [],
                        force_tol=1e-2,
                        angmom_tol=1e-2,
                        energy_tol=1e-7)
    fire.methods.append(hoomd.md.methods.DisplacementCapped(filter,maximum_displacement=.005))

    dpd = hoomd.md.pair.DPD(nlist = hoomd.md.nlist.Cell(buffer = .4),kT = 3.1,default_r_cut=.6)
    dpd.params[(frame.particles.types,frame.particles.types)] = dict(A=15, gamma=4.5)
    
    fire.forces = [dpd,bond_harmonic,angle_forces]
    sim.operations.integrator = fire
    status = Status(sim)
    logger = hoomd.logging.Logger(categories=["scalar", "string"])
    logger.add(sim, quantities=["timestep", "tps"])
    logger[("Status", "etr")] = (status, "etr", "string")
    table = hoomd.write.Table(
        trigger=hoomd.trigger.Periodic(period=1000), logger=logger
    )
    sim.operations.writers.append(table)
    gsd_writer = hoomd.write.GSD(
        filename=path + "fire.gsd",
        trigger=hoomd.trigger.Periodic(1000),
        filter=hoomd.filter.All(),
        mode="wb",
        dynamic = ['property','particles/image']
    )
    sim.operations.writers.append(gsd_writer)
    dpd.params[(frame.particles.types,frame.particles.types)] = dict(A=20, gamma=4.5)
    fire.forces = [dpd,bond_harmonic,angle_forces]
    sim.operations.integrator=fire
    sim.run(100000)
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 conc_bc_in_lipid x_box_size angle iter_num lipid_type")
        sys.exit(1)
    main(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))

