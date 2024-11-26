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

def main(x_box_size,angle, iter_num,lipid_type="DOPC"):
    path = "data/" + lipid_type + "/" + x_box_size + "/"+iter_num+"/" + angle + "/"
    if not os.path.exists(path):
        raise Exception("no data present")

    lj, coulomb, bond_harmonic, angle_forces, dihedrals,impropers,rigid = force_fields.forces_from_gsd(
        path, "fire.gsd"
    )
    try:
        sim = hoomd.Simulation(device=hoomd.device.GPU(), seed=int(x_box_size) * 500)
    except:
        raise("EXCEPTION: CPU")
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=int(iter_num) * 500)

    name = "fire.gsd"

    sim.create_state_from_gsd(filename=path + name)
    frame = gsd.hoomd.open(path + name)[-1]
    num_beads = frame.particles.N
    # apply NH thermostat with correct stuff
    velocity_rescaling = 1
    pressure_rescaling = 12
    pressure = 0.061019  # 1 unit is 16.388 atm
    mttk = hoomd.md.methods.thermostats.MTTK(kT=2.47, tau=velocity_rescaling)
    rigid_centers_and_free = hoomd.filter.All()
    unfrozen = hoomd.filter.Tags(get_not_frozen(path,num_beads))
    filter = hoomd.filter.Intersection(rigid_centers_and_free,unfrozen)
    cp = hoomd.md.methods.ConstantPressure(
        filter=filter,
        tauS=pressure_rescaling,
        S=[pressure, 0, pressure, 0, 0, 0],
        box_dof = [False,False,True,False,False,False],
        couple="none",
        thermostat=mttk,
    )
    cp.rescale_all = False
    integrator = hoomd.md.Integrator(
        dt=0.002,
        methods=[cp],
        forces=[lj, bond_harmonic, angle_forces, coulomb],
    )
    status = Status(sim)
    logger = hoomd.logging.Logger(categories=["scalar", "string"])
    logger.add(sim, quantities=["timestep", "tps"])
    logger[("Status", "etr")] = (status, "etr", "string")
    table = hoomd.write.Table(
        trigger=hoomd.trigger.Periodic(period=1000), logger=logger
    )
    sim.operations.writers.append(table)
    gsd_writer = hoomd.write.GSD(
        filename=path + "traj.gsd",
        trigger=hoomd.trigger.Periodic(10000),
        filter=hoomd.filter.All(),
        mode="wb",
        dynamic = ['property','particles/image']
    )

    sim.operations.integrator = integrator
    sim.state.thermalize_particle_momenta(filter = hoomd.filter.All(),kT = 3.65)
    print("Initialized")
    sim.run(10000)
    sim.operations.integrator.dt = .02
    print("Updated ts")
    sim.operations.writers.append(gsd_writer)

    sim.run(1000000)
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 conc_bc_in_lipid x_box_size angle iter_num lipid_type")
        sys.exit(1)
    main(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]), str(sys.argv[4]))
