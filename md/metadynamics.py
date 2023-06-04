from md.aseMD import *

import glob
import os
import numpy as np

import ase.io



def run_metad_single_TS(model_dir,rxn_number,output_dir,repeat =1,temp =300,time_fs=50,stepsize=0.1,interval = 100,
                        is_up_to_repeat = False, plumed_in_file = None, cpu = True, ensemble = 'NVT',
                        stop_when_committed = False, idxs=None):
    suffix = ""
    if plumed_in_file == None:  # default value
        plumed_in_file = f'metad/rxn{rxn_number}.dat'
    else:
        suffix = plumed_in_file.split('/')[-1].split('.')[-2][5:]
    print(f"Running metad for rxn{rxn_number}")

    start_geom = f'/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/md/irc_geom/TS_{rxn_number}.xyz'
    # output_dir = 'ml_metadynamics/metad_result'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    previous_runs = glob.glob(f'{output_dir}/metad_{rxn_number}{suffix}_{time_fs}fs_{temp}K_*/')
    start = 0
    convergence = None
    if stop_when_committed:
        convergence = 'committed'

    if idxs is None:
        if len(previous_runs)>0:
            previus_idx = [int(d.split('_')[-1].split('/')[0]) for d in previous_runs]
            start = max(previus_idx)+1
            if is_up_to_repeat:
                repeat = repeat - start
                if repeat <= 0:
                    return 0
        idxs = range(start,start+repeat)
    print(idxs)
    for i in idxs:
        molecule = ase.io.read(start_geom)
        # boxlen = 20
        # lattice = np.array([0, 0, boxlen, 0, boxlen, 0, boxlen, 0, 0]).reshape(3, 3)
        # molecule.set_cell(lattice)
        # molecule.center()
        molecule.set_pbc(False)
        MD = AseMLMetadynamics(plumed_in_file,molecule,model_dir,temp = temp, time_fs = time_fs, stepsize = stepsize)

        output_name = f'metad_{rxn_number}{suffix}__{time_fs}fs_{temp}K_{i}'
        if not os.path.isdir(f'{output_dir}/{output_name}'):
            os.mkdir(f'{output_dir}/{output_name}')
        os.system(f"cp {plumed_in_file} {output_dir}/{output_name}")
        print(f'Simulation for {rxn_number} at {temp}K for {time_fs}fs repeat #{i}')
        MD.run_simulation(f'{output_dir}/{output_name}',interval,output_name,convergence_criteria=convergence)
        MD.ase_plumed.finalize()
        os.system(
            f"mv metad/hills metad/COLVAR {plumed_in_file.replace('dat', 'out')} {output_dir}/{output_name}")
    ### In future: generate a json/yml parameter file

def run_qchem_metad_single_TS(plumed_in_file,rxn_number,time_fs,repeat=1,temp=300,interval = 100, is_up_to_repeat = False,
                        stop_when_committed = False, idxs=None, charge = 0, multiplicity=1, stepsize=0.2,
                              output_dir = 'ml_metadynamics/metad_result'):
    data_dir = 'irc_geom' #'../../md/irc_geom'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    previous_runs = glob.glob(f'{output_dir}/metad_{rxn_number}_{time_fs}fs_{temp}K_*/')
    start = 0
    convergence = None
    if stop_when_committed:
        convergence = 'committed'

    if idxs is None:
        if len(previous_runs)>0:
            previus_idx = [int(d.split('_')[-1].split('/')[0]) for d in previous_runs]
            start = max(previus_idx)+1
            if is_up_to_repeat:
                repeat = repeat - start
                if repeat <= 0:
                    return 0
        idxs = range(start,start+repeat)
    print(idxs)
    for i in idxs:
        molecule = ase.io.read(f'{data_dir}/product_{rxn_number}.xyz')
        boxlen = 20
        lattice = np.array([0, 0, boxlen, 0, boxlen, 0, boxlen, 0, 0]).reshape(3, 3)
        # molecule.set_cell(lattice)
        molecule.center()
        molecule.set_pbc(False)
        output_name = f'metad_{rxn_number}_{time_fs}fs_{temp}K_{i}'
        if not os.path.isdir(f'{output_dir}/{output_name}'):
            os.mkdir(f'{output_dir}/{output_name}')
        os.system(f"cp {plumed_in_file} {output_dir}/{output_name}")
        MD = AseQChemMetadynamics(plumed_in_file, molecule, f'rxn_{rxn_number}', temp=temp, time_fs=time_fs,
                                  charge=charge, stepsize= stepsize,
                                  multiplicity=multiplicity, output_dir=f'{output_dir}/{output_name}')
        print(f'Simulation for {rxn_number} at {temp}K for {time_fs}fs repeat #{i}')
        MD.run_simulation(f'{output_dir}/{output_name}',interval,output_name,convergence_criteria=convergence)
        MD.ase_plumed.finalize()
        os.system(f"mv ml_metadynamics/hills ml_metadynamics/COLVAR {plumed_in_file.replace('dat', 'out')} {output_dir}/{output_name}")
    ### In future: generate a json/yml parameter file

if __name__ == '__main__':
    # run_qchem_metad_single_TS('ml_metadynamics/rxn16.dat', 16, 1, 300, 5000, interval=100,multiplicity=2)
    run_metad_single_TS('/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/md/hydrogen_ln','16','ml_md',
                        repeat = 1, temp=300, time_fs= 20000, stepsize= 0.2,plumed_in_file='metad/rxn16.dat',cpu = False,
                        ensemble='NVT')





