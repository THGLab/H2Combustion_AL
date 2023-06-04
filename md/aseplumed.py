"""
a plumed wrapper for ase
"""

import numpy as np
import os

from ase import units

import plumed


class AsePlumed(object):

    def __init__(
            self, atoms, timestep,
            in_file = 'plumed.dat',
            out_file = 'plumed.out'
        ):

        self.atoms = atoms
        self.timestep = timestep
        self.natoms = len(atoms)
        self.masses = self.atoms.get_masses().copy() # masses cannot change

        self.in_file = in_file
        self.out_file = out_file

        self.worker = self.initialize()

        return

    def initialize(self):
        # os.system(f'cp {self.in_file} {self.in_file}.copy')

        # init

        p_md = plumed.Plumed()
        # ase units: https://wiki.fysik.dtu.dk/ase/ase/units.html
        energyUnits = 96.485  # eV to kJ/mol
        lengthUnits = 0.1  # angstrom to nm
        timeUnits = 1.0 / units.fs * 0.001  # fs to ps


        p_md.cmd("setMDEnergyUnits", energyUnits)
        p_md.cmd("setMDLengthUnits", lengthUnits)
        p_md.cmd("setMDTimeUnits", timeUnits)

        # inp, out
        p_md.cmd("setPlumedDat", self.in_file)
        p_md.cmd("setLogFile", self.out_file)

        # simulation details
        p_md.cmd("setTimestep", self.timestep)
        p_md.cmd("setNatoms", self.natoms)
        p_md.cmd("setMDEngine", 'ase')


        # finally!
        p_md.cmd("init")


        return p_md

    def external_forces(
            self,
            step,
            new_energy = None,
            new_forces = None, # sometimes use forces not attached to self.atoms
            new_virial = None,
            delta_forces = False
        ):
        """return external forces from plumed"""
        # structure info
        positions = self.atoms.get_positions().copy()
        cell = self.atoms.cell[:].copy()

        if new_forces is None:
            forces = self.atoms.get_forces().copy()
        else:
            forces = new_forces.copy()
        original_forces = forces.copy()

        if new_energy is None:
            energy = self.atoms.get_potential_energy()
        else:
            energy = new_energy

        # TODO: get virial

        self.worker.cmd("setStep", step)
        self.worker.cmd("setMasses", self.masses)
        # self.worker.cmd("setForces", forces)
        self.worker.cmd("setPositions", positions)
        # self.worker.cmd("setEnergy", energy)
        self.worker.cmd("setBox", cell)

        forces_bias = np.zeros((self.atoms.get_positions()).shape)
        self.worker.cmd("setForces", forces_bias)
        virial = np.zeros((3, 3))
        self.worker.cmd("setVirial", virial)
        self.worker.cmd("prepareCalc")
        self.worker.cmd("performCalc")
        energy_bias = np.zeros((1,))
        self.worker.cmd("getBias", energy_bias)

        print_detail = False
        if print_detail:
            print('energy bias', energy_bias)
            # print(np.sum(forces_bias))
            print('energy before', energy)
            energy = energy + energy_bias
            print('energy after',energy)

            print('force bias', np.linalg.norm(forces_bias))
            print('forces before', np.linalg.norm(forces,axis=1))
            forces = forces + forces_bias
            print('forces after', np.linalg.norm(forces, axis=1))
        else:
            energy = energy + energy_bias
            forces = forces + forces_bias


        # self.worker.cmd("calc")#, None)



        # implent plumed external forces into momenta
        if delta_forces:
            plumed_forces = forces - original_forces
            plumed_energy = energy_bias
        else:
            plumed_forces = forces
            plumed_energy = energy

        return plumed_forces,plumed_energy

    def finalize(self):
        self.worker.finalize()


if __name__ == '__main__':
    pass