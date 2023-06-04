#!/usr/bin/env python

# @Time: 10/18/21 
# @Author: Nancy Xingyi Guan
# @File: data_sampler.py

import numpy as np
import os
from iodata import load_one
import glob
import pandas as pd

import ase.io
from ase.data import atomic_numbers
from md.aseMD import *
from md.aseplumed import AsePlumed
from combust.utils.utility import combine_rxn_arrays

from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed

import scipy.constants as spc

angstrom =  spc.angstrom / spc.value(u'atomic unit of length')

class DataSampler():
    def __init__(self, calc):
        self.calc = calc
        atomname = ['H', 'O']
        self._coulumbdiag = dict(map(lambda symbol: (
            symbol, atomic_numbers[symbol] ** 2.4 / 2), atomname))

    def run_simulation(self, start_geom, output_dir, temp=300, time_fs=50, stepsize=0.1):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        start_geom = ase.io.read(start_geom)
        md = AseMD(start_geom, self.calc, temp=temp, time_fs=time_fs, stepsize=stepsize)
        md.run_simulation(output_dir, 100)

    def run_metad_simulation(self, start_geom, output_dir, plumed_in_file, repeat=1, temp=300, time_fs=50,
                             stepsize=0.1, ensemble = 'NVE'):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(plumed_in_file, 'r') as f:
            text = f.read()
        text = text.replace('metad/', f'{output_dir}/')
        with open(f'{output_dir}/{plumed_in_file.split("/")[-1]}', 'w') as f:
            f.write(text)
        plumed_in_file = f'{output_dir}/{plumed_in_file.split("/")[-1]}'
        start_geom = ase.io.read(start_geom)
        # start_geom.set_cell(lattice)
        start_geom.center()
        start_geom.set_pbc(False)

        for i in range(repeat):
            ase_plumed = AsePlumed(start_geom, stepsize, plumed_in_file, plumed_in_file.replace('dat', 'out'))
            plumed_calc = PlumedCalculator(ase_plumed)
            metad_calc = Sum_Calc(self.calc, plumed_calc, atoms=start_geom)
            md = AseMD(start_geom, metad_calc, temp=temp, time_fs=time_fs, stepsize=stepsize)
            output_name = f'metad_{time_fs}fs_{temp}K_{i}'
            if not os.path.isdir(f'{output_dir}/{output_name}'):
                os.mkdir(f'{output_dir}/{output_name}')
            os.system(f"cp {plumed_in_file} {output_dir}/{output_name}")
            print(f'Simulation for {output_dir} at {temp}K for {time_fs}fs repeat #{i}')
            md.run_simulation(f'{output_dir}/{output_name}', 100, ensemble=ensemble)
            ase_plumed.finalize()
            os.system(
                f"mv metad/hills metad/COLVAR {plumed_in_file.replace('dat', 'out')} {output_dir}/{output_name}")


    def sample_metad_simulation_by_E_conservation(self, start_geom, output_dir, plumed_in_file, repeat=1, temp=300, time_fs=50,
                             stepsize=0.1,thresh=3,ngeom=10):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        with open(plumed_in_file,'r') as f:
            text = f.read()
        text = text.replace('metad/',f'{output_dir}/')
        with open(f'{output_dir}/{plumed_in_file.split("/")[-1]}','w') as f:
            f.write(text)
        plumed_in_file = f'{output_dir}/{plumed_in_file.split("/")[-1]}'

        start_geom = ase.io.read(start_geom)
        # start_geom.set_cell(lattice)
        start_geom.center()
        start_geom.set_pbc(False)
        geoms = []

        for i in range(repeat):
            ase_plumed = AsePlumed(start_geom, stepsize, plumed_in_file, plumed_in_file.replace('dat', 'out'))
            plumed_calc = PlumedCalculator(ase_plumed)
            metad_calc = SumCalculator([self.calc, plumed_calc], atoms=start_geom)
            md = AseMD(start_geom, metad_calc, temp=temp, time_fs=time_fs, stepsize=stepsize)
            output_name = f'metad_{time_fs}fs_{temp}K_{i}'
            if not os.path.isdir(f'{output_dir}/{output_name}'):
                os.mkdir(f'{output_dir}/{output_name}')
            os.system(f"cp {plumed_in_file} {output_dir}/{output_name}")
            print(f'Simulation for {output_dir} at {temp}K for {time_fs}fs repeat #{i}')
            md.run_simulation(f'{output_dir}/{output_name}', 1, convergence_criteria='energy_conservation',
                              thresh=thresh, ngeom=ngeom)
            geoms.extend(md.geom)
            ase_plumed.finalize()
            os.system(
                f"mv metad/hills metad/COLVAR {plumed_in_file.replace('dat', 'out')} {output_dir}/{output_name}")

        return geoms

    def random_sample(self,max_number=-1,random_seed=90,geoms = None):
        """

        Parameters
        ----------
        max_number: max number of geometry returned. -1 for all.

        Returns
        -------

        """
        if geoms == None:
            geoms = self.calc.geom
        if max_number == -1 or len(geoms) <= max_number:
            return geoms
        else:
            np.random.seed(random_seed)
            first_20pct_idx=int(0.2*max_number)
            geoms = geoms
            #don't ever put geoms into np array, otherwise the Atoms object will become an array of Atom
            selected_geoms = geoms[:first_20pct_idx]
            idx = np.random.choice(len(geoms[first_20pct_idx:]),size=max_number-first_20pct_idx,replace=False)+first_20pct_idx
            selected2 = [geoms[i] for i in idx]
            selected_geoms.extend(selected2)
            for index in sorted(idx, reverse=True):
                del self.calc.geom[index]
            return selected_geoms

    def sample(self, max_number=-1, geoms=None, report_stdev = False):
        """
        sample by forming Coulomb matrix and then use mini batch KMeans to form clusters
        Parameters
        ----------
        max_number: max number of geometry returned.
            -1 for all. 'auto' for using inertia to choose ncluster for Kmeans, maximum is 300 (https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c)
        geoms: None or list of ase.Atoms
            default is None which use self.calc.geom
        report_stdev: bool or list of stdev values

        Returns
        -------

        """
        stdev = []
        downselect = False
        if geoms == None:
            geoms = self.calc.geom
            if report_stdev:
                stdev = self.calc.stdev
        if type(report_stdev) == list or type(report_stdev) == np.ndarray:
            stdev = report_stdev
            assert len(geoms) == stdev

        if type(max_number) == int:
            if max_number == -1 or len(geoms) <= max_number:
                downselect = False
            else:
                downselect = True
        elif max_number == "auto": #auto
            downselect = True
        else:
            raise ValueError(f"max number must be int or 'auto'. You have {max_number}")

        if downselect:
            columb_matrices = [self._calcoulumbmatrix(geom) for geom in geoms]
            idx = DataSampler._clusterdatas(columb_matrices,max_number,n_each=1)
            selected_geoms = [geoms[i] for i in idx]
            for index in sorted(idx, reverse=True):
                del geoms[index]
            if report_stdev:
                selected_std = [stdev[i] for i in idx]
                return selected_geoms,selected_std
            return selected_geoms
        else:
            if report_stdev:
                return geoms, stdev
            else:
                return geoms


    @staticmethod
    def detect_multiplicity(symbols):
        """Calculate multiplicity.
        Parameters
        ----------
        symbols: numpy.ndarray
            The atomic symbols.
        Returns
        -------
        multiplicity: int
            The multiplicity.
        """
        # currently only support charge=0
        # oxygen -> 3
        if symbols == ["O", "O"]:
            return 3
        # calculates the total number of electrons, assumes they are paired as much as possible
        n_total = sum([atomic_numbers[s] for s in symbols])
        return n_total % 2 + 1

    def generate_qchem_input(self, name, dirc, max_number=5000,geoms=None,spin=None,fragmo_atomic_oxygen=True,):
        """

        Parameters
        ----------
        name: str. name of qchem input files to be generated. The final file is named {name}_{i}.in
        dirc: str. path of input files to be stored
        max_number: int. max number of geometry to be generated. If max number is smaller than the provided total geometry,
            will down sample using mini batch kmeans on coulomb matrix
        geoms: list of ase.Atoms or None.
        spin: int or None.
        fragmo_atomic_oxygen: bool. Use fragmo initiation in qchem script if atomic oxygen exist in the geometry


        Returns
        -------
        int: number of structures that qchem input is generated
        """
        if not os.path.isdir(dirc):
            os.mkdir(dirc)
        if geoms == None:
            geoms = self.calc.geom
        structures = self.sample(max_number=max_number,geoms=geoms)
        if len(structures) == 0:
            print("No structure to generate qchem")
            return 0
        print(f"Number of qchem input file: {len(structures)} selected out of {len(geoms)}")
        for i, atoms in enumerate(structures):
            fname = f"{name}_{i}"
            natoms = len(atoms.get_atomic_numbers())
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            if spin == None:
                sp = DataSampler.detect_multiplicity(symbols)
            else:
                sp = spin

            fragmo = False
            if fragmo_atomic_oxygen:
                atomic_o_idx = check_atomic_oxygen(atoms)
                if (len(atomic_o_idx) > 0):
                    fragmo = True

            sp_lines = ["$molecule",
                        f" 0   {sp}"]

            if fragmo:
                # atomic oxygen fragments
                spin_frag_all = 0
                for i,idx in enumerate(atomic_o_idx):
                    if i % 2 == 0:
                        mul_frag = 3
                    else:
                        mul_frag = -3
                    if mul_frag >0:
                        spin_frag_all += (mul_frag - 1) / 2 # spin not mul here
                    else:
                        spin_frag_all += (mul_frag + 1) / 2
                    sp_lines.extend( ["----",
                                f" 0  {mul_frag}"])
                    sp_lines.append(f' {symbols[idx]} {positions[idx][0]} {positions[idx][1]} {positions[idx][2]}')
                # rest of the molecule
                rest_idx = [i for i in range(natoms) if i not in atomic_o_idx]
                if len(rest_idx)>0:
                    sp_lines.append("----")
                    mul = ((sp-1)/2 - spin_frag_all) *2
                    if mul > 0:
                        mul+=1
                    else:
                        mul-=1
                    sp_lines.append(f" 0  {int(mul)}")
                    for idx in rest_idx:
                        sp_lines.append(f' {symbols[idx]} {positions[idx][0]} {positions[idx][1]} {positions[idx][2]}')
            else:
                for k in range(natoms):
                    sp_lines.append(f' {symbols[k]} {positions[k][0]} {positions[k][1]} {positions[k][2]}')
            sp_lines.extend([
                " $end",
                " ",
                " $rem",
                # " ideriv                  1",
                # " incdft                  0",
                # " incfock                 0",
                " jobtype                 force",
                " method                  wB97X-V",
                " unrestricted            True",
                " basis                   cc-pVTZ",
                " internal_stability      True",
                " INTERNAL_STABILITY_ITER    3",
                " INSTABILITY_TRIGGER     6",
                " scf_algorithm           gdm",
                " scf_max_cycles          500",])
            if fragmo:
                sp_lines.append(" scf_guess               fragmo")
            else:
                sp_lines.append(" scf_guess               sad")
            sp_lines.extend([
                " scf_convergence                 8",
                " thresh                  14",
                " xc_grid                 000099000590",
                " symmetry                0",
                " sym_ignore              1",
                # " purecart                1111",
                # " gen_scfman              true",
                # " gen_scfman_final                true",
                # " internal_stability        true",
                # " complex                 false",
                " $end"
            ])
            sp_lines = map(lambda x: x + '\n', sp_lines)
            with open(f"{dirc}/{fname}.in", 'w') as f:
                f.writelines(sp_lines)
        return len(structures)

    def generate_qchem_input_fragmo_and_sad(self, name, dirc, max_number=5000,geoms=None,spin=None,report_stdev=True):
        """
        if atomic oxygen, generate fragmo + sad files. else generate sad file only
        Parameters
        ----------
        name: str. name of qchem input files to be generated. The final file is named {name}_{i}.in
        dirc: str. path of input files to be stored
        max_number: int. max number of geometry to be generated. If max number is smaller than the provided total geometry,
            will down sample using mini batch kmeans on coulomb matrix
        geoms: list of ase.Atoms or None.
        spin: int or None.
        report_stdev: bool or list of stdev values


        Returns
        -------
        int: number of structures that qchem input is generated
        """
        if not os.path.isdir(dirc):
            os.mkdir(dirc)
        if geoms == None:
            geoms = self.calc.geom
        if report_stdev:
            structures,stdev = self.sample(max_number=max_number,geoms=geoms,report_stdev=report_stdev)
        else:
            structures = self.sample(max_number=max_number, geoms=geoms, report_stdev=report_stdev)
        if len(structures) == 0:
            print("No structure to generate qchem")
            return 0
        print(f"Number of qchem input file: {len(structures)} selected out of {len(geoms)}")
        for i, atoms in enumerate(structures):
            fname = f"{name}_{i}"
            natoms = len(atoms.get_atomic_numbers())
            symbols = atoms.get_chemical_symbols()
            positions = atoms.get_positions()
            if spin == None:
                sp = DataSampler.detect_multiplicity(symbols)
            else:
                sp = spin

            # if atomic oxygen, generate fragmo + sad files. else generate sad file onlu
            fragmo = False
            atomic_o_idx = check_atomic_oxygen(atoms)
            if (len(atomic_o_idx) > 0):
                fragmo = True

            #sad file
            sp_lines = ["$molecule",
                        f" 0   {sp}"]
            for k in range(natoms):
                sp_lines.append(f' {symbols[k]} {positions[k][0]} {positions[k][1]} {positions[k][2]}')
            sp_lines.extend([
                " $end",
                " ",
                " $rem",
                " jobtype                 force",
                " method                  wB97X-V",
                " unrestricted            True",
                " basis                   cc-pVTZ",
                " internal_stability      True",
                " INTERNAL_STABILITY_ITER    3",
                " INSTABILITY_TRIGGER     6",
                " scf_algorithm           gdm",
                " scf_max_cycles          500",
                " scf_guess               sad",
                " scf_convergence                 8",
                " thresh                  14",
                " xc_grid                 000099000590",
                " symmetry                0",
                " sym_ignore              1",
                " $end"
            ])
            sp_lines = map(lambda x: x + '\n', sp_lines)
            with open(f"{dirc}/{fname}_sad.in", 'w') as f:
                f.writelines(sp_lines)

            # write fragmo file if atomic oxygen
            if fragmo:
                sp_lines = ["$molecule",
                            f" 0   {sp}"]
                # atomic oxygen fragments
                spin_frag_all = 0
                for i,idx in enumerate(atomic_o_idx):
                    if i % 2 == 0:
                        mul_frag = 3
                    else:
                        mul_frag = -3
                    if mul_frag >0:
                        spin_frag_all += (mul_frag - 1) / 2 # spin not mul here
                    else:
                        spin_frag_all += (mul_frag + 1) / 2
                    sp_lines.extend( ["----",
                                f" 0  {mul_frag}"])
                    sp_lines.append(f' {symbols[idx]} {positions[idx][0]} {positions[idx][1]} {positions[idx][2]}')
                # rest of the molecule
                rest_idx = [i for i in range(natoms) if i not in atomic_o_idx]
                if len(rest_idx)>0:
                    sp_lines.append("----")
                    mul = ((sp-1)/2 - spin_frag_all) *2
                    if mul > 0:
                        mul+=1
                    else:
                        mul-=1
                    sp_lines.append(f" 0  {int(mul)}")
                    for idx in rest_idx:
                        sp_lines.append(f' {symbols[idx]} {positions[idx][0]} {positions[idx][1]} {positions[idx][2]}')

                sp_lines.extend([
                    " $end",
                    " ",
                    " $rem",
                    " jobtype                 force",
                    " method                  wB97X-V",
                    " unrestricted            True",
                    " basis                   cc-pVTZ",
                    " internal_stability      True",
                    " INTERNAL_STABILITY_ITER    3",
                    " INSTABILITY_TRIGGER     6",
                    " scf_algorithm           gdm",
                    " scf_max_cycles          500",
                    " scf_guess               fragmo",
                    " scf_convergence                 8",
                    " thresh                  14",
                    " xc_grid                 000099000590",
                    " symmetry                0",
                    " sym_ignore              1",
                    " $end"
                ])
                sp_lines = map(lambda x: x + '\n', sp_lines)
                with open(f"{dirc}/{fname}_fragmo.in", 'w') as f:
                    f.writelines(sp_lines)
        if report_stdev:
            return len(structures),stdev
        return len(structures)

    def generate_traj(self, name, dirc ,geoms=None):
        """
        save given geomtery as ase traj file
        Parameters
        ----------
        name: name of traj file without suffix
        dirc: output directory
        geoms: list of ase.Atoms

        Returns
        -------
        bool
        """
        if not os.path.isdir(dirc):
            os.mkdir(dirc)
        if geoms == None:
            geoms = self.calc.geom

        if len(geoms) == 0:
            print("No structure to generate traj")
            return False

        ase.io.write(f"{dirc}/{name}.traj",geoms,format = "traj")
        return True



    @staticmethod
    def parse_qchem_output(dir_path, rxn=None, return_idx = False, fragmo_and_sad=False):
        """Load qchem force calculations.

        Parameters
        ----------
        dir_path : str / list
            Path to directory containing AIMD data of a reaction. / List of files
        rxn: str
            data['RXN'] for the returned dictionary
        return_idx: bool
            assuming the parsed qchem output files has index at the end, will return the index of successful runs
            as a list of ints
        fragmo_and_sad: bool
            fragmo and sad output exist in this dir/list. Have to compare 2 and get lowest energy one among them

        Returns
        -------
        data: dict

        """
        # Hartree to kcal/mol conversion factor
        hartree_to_kcal_mol = 627.509
        hartree_to_kcal_mol_ang = 1185.82062
        angstrom_to_bohr = 1.88973

        # pre-computed atomic energies (Hartree)
        energy_h_atom = -0.5004966690
        energy_o_atom = -75.0637742413

        if type(dir_path) == list:
            outfiles = dir_path
        else:
            outfiles = glob.glob(f"{dir_path}/*.out")

        if len(outfiles) ==0:
            raise ValueError(f'No valid file found for {outfiles}')

        if fragmo_and_sad:
            # keep the lower E one among fragmo and sad if a fragmo file exist
            for f in outfiles:
                if f.endswith('fragmo.out'):
                    f_sad = f.replace('fragmo','sad')
                    try:
                        _, _, energy_fragmo, _ = read_sp_file(f)
                        _, _, energy_sad, _ = read_sp_file(f_sad)
                        if energy_fragmo < energy_sad:
                            outfiles.remove(f_sad)
                        else:
                            outfiles.remove(f)
                    except Exception as e:
                        print(e)
                        outfiles.remove(f)
                        outfiles.remove(f_sad)

        # sort by index
        def getint(name):
            num = name.split('.')[-2].split('_')[-1]
            if not num.isnumeric():
                num = name.split('.')[-2].split('_')[-2]
            return int(num)

        try:
            outfiles.sort(key=getint)
        except ValueError:
            outfiles.sort()
        # print(outfiles)
        atnums, atcoords, energies, gradients = [], [], [], []
        indices =[]
        for f in outfiles:
            try:
                nums, coords, energy, gradient = read_sp_file(f)
                coords = coords/angstrom
            except ValueError:
                print("single point force calculation failed:  ",f)
                continue
                # raise
            except AssertionError:
                print(f"single point force calculation unable to parse:  ",f)
                continue
            atnums.append(nums)
            atcoords.append(coords)
            energies.append(energy)
            gradients.append(gradient)
            if return_idx:
                indices.append(int(f.split('.')[-2].split('_')[-1]))

        # list of aimd mols
        aimd_carts = np.array(atcoords) #(N,natoms
        aimd_nums = np.array(atnums)
        raw_energies = np.array(energies)
        gradients = np.array(gradients)

        # sanity checks
        assert aimd_carts.shape[0] == aimd_nums.shape[0],[aimd_carts.shape, aimd_nums.shape]
        assert aimd_carts.shape[1] == aimd_nums.shape[1],[aimd_carts.shape, aimd_nums.shape]
        assert aimd_carts.shape[2] == 3
        # get number of data points and number of atoms
        n_data, n_atoms = aimd_nums.shape
        # get number of oxygen and hydrogen atoms
        # print(aimd_nums)
        n_o_atom = np.count_nonzero(aimd_nums[0] == 8)
        n_h_atom = np.count_nonzero(aimd_nums[0] == 1)

        # convert energies to relative energies in kcal/mol
        aimd_energy = [el - n_h_atom * energy_h_atom - n_o_atom * energy_o_atom for el in raw_energies]
        aimd_energy = np.asarray(aimd_energy)
        aimd_energy = aimd_energy.reshape(-1, 1) * hartree_to_kcal_mol
        assert aimd_energy.ndim == 2
        assert aimd_energy.shape == (n_data, 1)

        # convert Gradients to Nuclear Forces
        forces = []

        for grad in gradients:
            force = [grad[i] * -1 * hartree_to_kcal_mol * angstrom_to_bohr for i in range(len(grad))]
            forces.append(force)

        aimd_forces = np.asarray(forces)
        assert aimd_forces.shape[0] == n_data
        assert aimd_forces.shape[1] == n_atoms * 3
        aimd_forces = aimd_forces.reshape(n_data, n_atoms, 3)

        output = {
            'Z': aimd_nums.astype(int),
            'R': aimd_carts,
            'E': aimd_energy,
            'F': aimd_forces,
            'N': np.repeat(n_atoms, n_data).reshape(-1, 1),
            'RXN': np.repeat(rxn, n_data).reshape(-1, 1),
        }
        data = combine_rxn_arrays([output], 6)
        if return_idx:
            return data,indices
        else:
            return data

    @staticmethod
    def parse_qchem_output_two_spin(list1,list2, rxn=None):
        """Load qchem force calculations.

        Parameters
        ----------
        list1,list2: list of outfiles. Result of glob

        Returns
        -------
        """
        # Hartree to kcal/mol conversion factor
        hartree_to_kcal_mol = 627.509
        hartree_to_kcal_mol_ang = 1185.82062
        angstrom_to_bohr = 1.88973

        # pre-computed atomic energies (Hartree)
        energy_h_atom = -0.5004966690
        energy_o_atom = -75.0637742413


        if len(list1) == 0 or len(list2) == 0:
            raise ValueError(f'No valid file found')

        # sort by index
        def getint(name):
            num = name.split('.')[-2].split('_')[-1]
            if num in ['s','d','t','q']:
                num = name.split('_')[-2]
            return int(num)

        # try:
        #     list1.sort(key=getint)
        #     list2.sort(key=getint)
        # except ValueError:
        #     list1.sort()
        #     list2.sort()
        # max_idx = max(getint(list1[-1]),getint(list2[-1]))
        dict1,dict2 = {},{}
        for name in list1:
            dict1[getint(name)] = name
        for name in list2:
            dict2[getint(name)] = name
        available_idx = set(dict2).intersection(set(dict1))
        available_idx = list(available_idx)
        available_idx.sort()

        atnums, atcoords, energies, gradients = [], [], [], []

        for idx in available_idx:
            try:
                nums1, coords1, energy1, gradient1 = read_sp_file(dict1[idx])
                nums2, coords2, energy2, gradient2 = read_sp_file(dict1[idx])
            except ValueError:
                print("single point force calculation failed for one of:  ")
                print(dict1[idx], dict2[idx])
                continue
                # raise
            np.testing.assert_allclose(coords1,coords2)
            atnums.append(nums1)
            coords1 = coords1 / angstrom
            atcoords.append(coords1)
            if energy1 > energy2:
                energies.append(energy2)
                gradients.append(gradient2)
            else:
                energies.append(energy1)
                gradients.append(gradient1)

        # list of aimd mols
        aimd_carts = np.array(atcoords)  # (N,natoms
        aimd_nums = np.array(atnums)
        raw_energies = np.array(energies)
        gradients = np.array(gradients)

        # sanity checks
        assert aimd_carts.shape[0] == aimd_nums.shape[0], [aimd_carts.shape, aimd_nums.shape]
        assert aimd_carts.shape[1] == aimd_nums.shape[1], [aimd_carts.shape, aimd_nums.shape]
        assert aimd_carts.shape[2] == 3
        # get number of data points and number of atoms
        n_data, n_atoms = aimd_nums.shape
        # get number of oxygen and hydrogen atoms
        # print(aimd_nums)
        n_o_atom = np.count_nonzero(aimd_nums[0] == 8)
        n_h_atom = np.count_nonzero(aimd_nums[0] == 1)

        # convert energies to relative energies in kcal/mol
        aimd_energy = [el - n_h_atom * energy_h_atom - n_o_atom * energy_o_atom for el in raw_energies]
        aimd_energy = np.asarray(aimd_energy)
        aimd_energy = aimd_energy.reshape(-1, 1) * hartree_to_kcal_mol
        assert aimd_energy.ndim == 2
        assert aimd_energy.shape == (n_data, 1)

        # convert Gradients to Nuclear Forces
        forces = []

        for grad in gradients:
            force = [grad[i] * -1 * hartree_to_kcal_mol * angstrom_to_bohr for i in range(len(grad))]
            forces.append(force)

        aimd_forces = np.asarray(forces)
        assert aimd_forces.shape[0] == n_data
        assert aimd_forces.shape[1] == n_atoms * 3
        aimd_forces = aimd_forces.reshape(n_data, n_atoms, 3)

        output = {
            'Z': aimd_nums.astype(int),
            'R': aimd_carts,
            'E': aimd_energy,
            'F': aimd_forces,
            'N': np.repeat(n_atoms, n_data).reshape(-1, 1),
            'RXN': np.repeat(rxn, n_data).reshape(-1, 1),
        }
        data = combine_rxn_arrays([output], 6)
        return data

    def _calcoulumbmatrix(self,atoms):
        """Calculate Coulumb matrix for atoms.
        Parameters
        ----------
        atoms: ase.Atoms
            Atoms to calculate Coulumb matrix.

        Returns
        -------
        numpy.darray (N,)
            The eigenvalues of columb matrix.
        """
        # https://github.com/crcollins/molml/blob/master/molml/utils.py
        top = np.outer(atoms.numbers, atoms.numbers).astype(np.float64)
        r = atoms.get_all_distances(mic=True)
        diag = np.array(
            list(map(self._coulumbdiag.get, atoms.get_chemical_symbols())))
        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(top, r, top)
            np.fill_diagonal(top, diag)
        top[top == np.Infinity] = 0
        top[np.isnan(top)] = 0
        return np.linalg.eigh(top)[0]

    @classmethod
    def _clusterdatas(cls, X, n_clusters, n_each=1):
        """Select data using Mini Batch Kmeans. Select n_each data from each cluster
        Parameters
        ----------
        X: numpy.darray
            The input data.
        n_clusters: int or 'auto'
            The number of clusters.
        n_each: int, optional, default=1
            The number of structures in each cluster.

        Returns
        -------
        numpy.ndarray
            The selected index.
        """
        # https://github.com/tongzhugroup/mddatasetbuilder/blob/master/mddatasetbuilder/datasetbuilder.py
        min_max_scaler = preprocessing.MinMaxScaler()
        X = np.array(min_max_scaler.fit_transform(X))
        if n_clusters == 'auto':
            n_clusters,res = cls.chooseBestKforKMeansParallel(X,range(min(len(X)-1,10),min(len(X),300),10))
            print(f"Selecting {n_clusters} data points for kmeans")
            print(res)
        clus = MiniBatchKMeans(n_clusters=n_clusters, init_size=(
            min(3 * n_clusters, len(X))))
        labels = clus.fit_predict(X)
        choosedidx = []
        for i in range(n_clusters):
            idx = np.where(labels == i)[0]
            if idx.size:
                choosedidx.append(np.random.choice(idx, n_each))
        index = np.concatenate(choosedidx)
        return index

    @classmethod
    def kMeansRes(cls,scaled_data, k, alpha_k=0.0002):
        '''
        Parameters
        ----------
        scaled_data: matrix
            scaled data. rows are samples and columns are features for clustering
        k: int
            current k for applying KMeans
        alpha_k: float
            manually tuned factor that gives penalty to the number of clusters
        Returns
        -------
        scaled_inertia: float
            scaled inertia value for current k
        '''

        inertia_o = np.square((scaled_data - scaled_data.mean(axis=0))).sum()
        # fit k-means
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=0).fit(scaled_data)
        scaled_inertia = kmeans.inertia_ / inertia_o + alpha_k * k
        return scaled_inertia

    @classmethod
    def chooseBestKforKMeansParallel(cls,scaled_data, k_range):
        '''
        Parameters
        ----------
        scaled_data: matrix
            scaled data. rows are samples and columns are features for clustering
        k_range: list of integers
            k range for applying KMeans
        Returns
        -------
        best_k: int
            chosen value of k out of the given k range.
            chosen k is k with the minimum scaled inertia value.
        results: pandas DataFrame
            adjusted inertia value for each k in k_range
        '''

        ans = Parallel(n_jobs=-1, verbose=10)(delayed(cls.kMeansRes)(scaled_data, k) for k in k_range)
        ans = list(zip(k_range, ans))
        results = pd.DataFrame(ans, columns=['k', 'Scaled Inertia']).set_index('k')
        best_k = results.idxmin()[0]
        return best_k, results




def read_sp_file(file):
    """
    from a qchem force calculation get geometry,energy and gradient
    """
    with open(file, 'r') as f:
        lines = f.readlines()
        # check out file is complete
    complete = False
    for l in lines[-10:]:
        if "Thank you very much for using Q-Chem.  Have a nice day." in l:
            complete = True
    if not complete:
        print(f".out file {file} is not complete")
        raise ValueError(f".out file {file} is not complete")
    try:
        mol = load_one(file, fmt='qchemlog')
    except:
        print(f"Problem loading file {file} using loadone")
        raise ValueError(f"Problem loading file {file} using loadone")

    # read gradient
    gradient = []
    for i, l in enumerate(lines):
        if "Gradient of SCF Energy" in l:
            gradient_lines = lines[i + 2:i + 5]
            for line in gradient_lines:
                gradient.append(line.split()[1:])
    gradient = np.array(gradient).T.astype(float).flatten()
    assert len(gradient) == 3*len(mol.atnums)

    # print(mol.energy)
    # print(mol.atnums)
    # print(mol.atcoords)
    # print(gradient)
    return mol.atnums, mol.atcoords, mol.energy, gradient

def run_seml_metad_simulation(committee, start_geom, output_dir, plumed_in_file, repeat=1, temp=300, time_fs=50,
                             stepsize=0.1, charge = 0, multiplicity=1,weight_factor=1,rxn_number=16):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    se_calc = QChem(label=f'rxn_{rxn_number}',
                    # input file name without .in
                    method='PBEh-3c',
                    basis='def2-mSVP',
                    charge=charge,
                    multiplicity=multiplicity,
                    nt=16, np=1)
    ml_calc = MLCommitteeAseCalculator(committee, None)
    calc = ML_SemiEmpirical_AseCalculator(se_calc,ml_calc,weight_factor=weight_factor)
    data_sampler = DataSampler(calc)
    data_sampler.run_metad_simulation(start_geom, output_dir, plumed_in_file, repeat, temp, time_fs, stepsize)
    return data_sampler

def run_seml_simulation(committee, start_geom, output_dir, plumed_in_file, repeat=1, temp=300, time_fs=50,
                             stepsize=0.1, charge = 0, multiplicity=1,weight_factor=1,rxn_number=16):
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    se_calc = QChem(label=f'rxn_{rxn_number}',
                    # input file name without .in
                    method='PBEh-3c',
                    basis='def2-mSVP',
                    charge=charge,
                    multiplicity=multiplicity,
                    nt=16, np=1)
    ml_calc = MLCommitteeAseCalculator(committee, None)
    calc = ML_SemiEmpirical_AseCalculator(se_calc,ml_calc,weight_factor=weight_factor)
    data_sampler = DataSampler(calc)
    data_sampler.run_simulation(start_geom, output_dir, temp, time_fs, stepsize)
    return data_sampler


def check_atomic_oxygen(atoms,split_cutoff=2):
    """

    Parameters
    ----------
    atoms: ase.Atoms
    split_cutoff: cutoff distance for components in angstrom

    Returns
    -------
    indices: list of int
        list of index of each atomic oxygen

    """
    components = ase.build.separate(atoms, scale=split_cutoff)
    components_numbers = [atoms.get_atomic_numbers() for atoms in components]
    idx=0
    indices = []
    for i, component_number in enumerate(components_numbers):
        # check if one piece of component matches atomic oxygen
        # print(component_number)
        if np.array_equal([8], component_number):
            for j,coord in enumerate(atoms.get_positions()):
                if np.array_equal(components[i][0].position, coord):
                    indices.append(j)
                    assert atoms[j].symbol =='O', f'atomic oxygen {j}: {atoms[j]} '

    #remove last if all atoms are atomic oxygen
    if len(indices) == len(atoms.get_atomic_numbers()):
        indices.pop(-1)
    return indices
