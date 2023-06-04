import ase
import ase.io
import ase.build
from ase.io.trajectory import Trajectory
from ase.calculators.qchem import QChem
from md.active_learning.active_learner import CommitteeRegressor
from md.aseMD import MLCommitteeAseCalculator
from md.aseMD import committer_analysis

import numpy as np
import os
import sys
import glob
import pandas as pd

import rmsd

from md.ase_interface import MLAseCalculator
from md.aseMD import AseMD

RXN_MULTIPLICITY= {'01': 4,
                 '02': 3,
                 '03': 2,
                 '04': 3,
                 '05': 1,
                 '06': 3,
                 '06a': 1,
                 '06b': 3,
                 '07': 2,
                 '08': 1,
                 '09': 2,
                 '10': 3,
                 '11': 3,
                 '12': 2,
                 '12a': 2,
                 '12b': 4,
                 '13': 3,
                 '14': 3,
                 '15': 1,
                 '16': 2,
                 '17': 2,
                 '18': 3,
                 '19': 2,
}

# #setting for local runs
# data_dir = '../../irc_geom'
# setting for lrc runs
data_dir = "/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/md/irc_geom"

def RMSD(a,b):
        "align and calculate rmsd, input are nparrays"
        # Manipulate
        A = a - rmsd.centroid(a)
        B = b - rmsd.centroid(b)
        U = rmsd.kabsch(A, B)
        A = np.dot(A, U)
        return rmsd.rmsd(A, B)

def run_rxn_single_TS(rxn_number,repeat,temp,time_fs,output_dir,interval =10,is_up_to_repeat = False, stop_when_committed = False,
                      idxs=None, calc_option = 'ML'):
    """
        Run ase md with starting structure of TS_{rxn_number}.xyz. The md result will be
        stored in {output_dir}/MD_{rxn_number}_{time_fs}fs_{temp}K_{idx}

        Parameters
        ----------
        rxn_number: str
        repeat: int
            The number of times to run the simulation
        temp: int
        time_fs: int
        output_dir: str
            master directory of all simulation. Each repeat with have a separate sub directory here
        interval: int
            Printing interval.
        is_up_to_repeat: bool
            if this is set to true, then if you already have 20 runs with {output_dir}/MD_{rxn_number}_{time_fs}fs_{temp}K_*,
            and repeat=50, then you would only run the other 30 runs. If this is False, then you would run another 50 runs.
        stop_when_committed: bool
            Use True when doing committer analysis
        idxs: list of int
            Can specify the idxs of simulations if needed. Usually set to None
        calc_option: 'ML' or 'QCHEM'

        Returns
        -------

        """

    previous_runs = glob.glob(f'{output_dir}/MD_{rxn_number}_{time_fs}fs_{temp}K_*/')
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
                    print(f'Already have more than {repeat+start} runs for {previous_runs[-1]}')
                    return 0
        idxs = range(start,start+repeat)
    print(idxs)
    for i in idxs:
        molecule = ase.io.read(f'{data_dir}/TS_{rxn_number}.xyz')

        lattice = None
        if calc_option=='ML':
            model_path = '/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/local/hydrogen_ln'
            model_type = 'NewtonNet'
            yml = glob.glob(f'{model_path}/run_scripts/*.yml')
            assert len(yml) ==1, f'{len(yml)} yaml files found in {model_path}/run_scripts'
            yml = yml[0]
            calc = MLAseCalculator(f'{model_path}/models/best_model_state.tar', yml,
                                        model_type = model_type, lattice = lattice)
        elif calc_option =="ML_QBC": # query by committee for active learning trained model
            # #local
            # result_dir = "/Users/nancy/Desktop/THG-research/Reactive-FF/AIMD_H_combustion/active_learning/models/model_al_1kperrxn_bwsl_1"
            #lrc
            result_dir ="/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/local/active_learning/model_al_1kperrxn_bwsl_b"
            committee = CommitteeRegressor.from_dir(result_dir, force_cpu=True, iteration=-1)
            iteration = committee.get_latest_iteration()
            assert type(iteration) == int
            # run ase md simulations and sample points of uncertain
            disagreement_thresh = [2, 100]
            calc = MLCommitteeAseCalculator(committee, disagreement_thresh)
        elif calc_option=='QCHEM':
            #need a qchem input file in the same directory
            in_name = os.path.join(output_dir, f'TS_{rxn_number}')
            if os.path.isfile(in_name+'.in'):
                os.remove(in_name+'.in')
            calc = QChem(label= in_name,  # input file name without .in
                              method='wB97X-V',
                              basis='cc-pVTZ',
                              scf_max_cycles='500',
                              incdft='0',
                              symmetry='false',
                              charge=0,
                              multiplicity=RXN_MULTIPLICITY[rxn_number],
                              nt=16, np=1)
            # calc = QChem(label='calc/ethane', #input file name without .in
            #              method='B3LYP',
            #              basis='6-31+G*')
        MD = AseMD(molecule, calc,temp = temp, time_fs = time_fs)
        output_name = f'MD_{rxn_number}_{time_fs}fs_{temp}K_{i}'
        if not os.path.isdir(f'{output_dir}/{output_name}'):
            os.mkdir(f'{output_dir}/{output_name}')
        print(f'Simulation for {rxn_number} at {temp}K for {time_fs}fs repeat #{i}')
        # print('interval', interval)
        try:
            MD.run_simulation(f'{output_dir}/{output_name}', interval, output_name, convergence_criteria=convergence)
        except Exception as e:
            os.system(f'rm -r {output_dir}/{output_name}')
            print(f"{output_name} skipped because error: ", e)
            # return 0




def energy_conservation_check(logfile, thresh):
    with open(logfile,'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    # print([l.split()[1] for l in lines])
    try:
        energies = [float(l.split()[1]) for l in lines]
    except:
        print(f'{logfile} may contain multiple trajectories')
        # return False
        raise
    if len(energies) == 0:
        raise ValueError(f"{logfile} is an empty trajectory")
    diff = np.max(energies) - np.min(energies)
    if diff > thresh:
        print(f'Energy conservation check failded for {logfile}, with energy difference of {diff}eV')
        return False
    else:
        return True


def run_simulations(repeat,time_fs, output_dir,temp = 300, is_up_to_repeat = False,stop_when_committed = True,
                    calc_option = 'ML', rxns = None):
    if rxns == None:
        tss = glob.glob(f'{data_dir}/TS_*.xyz')
    elif isinstance(rxns,int) or isinstance(rxns,str):
        tss = glob.glob(f'{data_dir}/TS_{rxns}.xyz')
    else: #list
        tss = [f'{data_dir}/TS_{r}.xyz' for r in rxns]

    for ts in sorted(tss):
        # run_rxn_single_TS(ts.split('_')[-1].split('.')[0],repeat,1000,time_fs,is_up_to_repeat,stop_when_committed)
        run_rxn_single_TS(ts.split('_')[-1].split('.')[0],repeat,temp,time_fs,output_dir,interval=5,is_up_to_repeat=is_up_to_repeat,
                          stop_when_committed=stop_when_committed,calc_option = calc_option)

def analyze_committor_all(time_fs,temp = 300, result_dir = '../local/md_result',replace_failed = False,
                thresh = 0.2):
    """
        You need to first run run_simulations() then use this function

        Parameters
        ----------
        time_fs: int
        temp: int or list of ints
        result_dir: str
            The output dir from run_simulations()
        output: str
            name of excel file
        replace_failed: bool
            Sometimes the model is not good enough and the energy conservation is not good. This option allows you to rerun
            those trajectories where the energy conservation fails.
        thresh: float
            The thresh for energy conservation (max_E-min_E) in eV.

        Returns
        -------

        """
    all_simulation = glob.glob(f'{result_dir}/MD_*/')
    tss = glob.glob(f'{data_dir}/TS_*.xyz')
    output = f'{result_dir}/committer_analysis.xls'
    rxns = sorted([ts.split('_')[-1].split('.')[0] for ts in tss])
    if type(temp) == int:
        temperatures = [temp]
    else:
        temperatures = temp
    dfs,dfs_idx = [],[]
    energy_conservation = {rxn:0 for rxn in rxns}
    for temp in temperatures:
        df = pd.DataFrame(0, index=rxns, columns=['reactant', 'product', 'neither'])
        df2 = pd.DataFrame('', index=rxns, columns=['reactant', 'product', 'neither'])
        for simulation in all_simulation:
            if str(temp)+'K' in simulation and str(time_fs)+'fs' in simulation:
                print(simulation)
                rxn = simulation.split('/')[-2].split('_')[1]
                conserved = energy_conservation_check(glob.glob(f'{simulation}/*log')[0],thresh)
                if replace_failed:
                    count = 0
                    while not conserved and count<5:
                        os.system(f'rm -r {simulation}')
                        run_rxn_single_TS(rxn, 1, temp, int(simulation.split('_')[-3][:-2]), result_dir, interval=5,
                                          is_up_to_repeat=False, stop_when_committed=True,
                                          calc_option='ML',idxs=[int(simulation.split('_')[-1][:-1])])
                        conserved = energy_conservation_check(glob.glob(f'{simulation}/*log')[0], thresh)
                        count+=1
                if not conserved:
                    energy_conservation[rxn]+=1
                try:
                    result = committer_analysis(rxn, glob.glob(f'{simulation}/*traj')[0])
                except Exception as e:
                    print(e)
                    continue
                for k,v in result.items():
                    df.at[rxn,k] += len(v)
                    if len(v) > 0:
                        df2.at[rxn,k] += str(simulation.split('_')[-1][:-1]) +','
        dfs.append(df)
        dfs_idx.append(df2)
        print(f'Temperature: {temp}K')
        print(df)
        print('energy conservation failed trajectories', energy_conservation)

    with pd.ExcelWriter(output) as writer:
        for i, df in enumerate(dfs):
            df.to_excel(writer, sheet_name=f'{temperatures[i]}K')
        for i, df in enumerate(dfs_idx):
            df.to_excel(writer, sheet_name=f'{temperatures[i]}K_idx')
        writer.save()

def run_md_simulation(starting_structure,output_dir,temp = 300,time_fs=10, stepsize = 0.1, lattice = None,
                      interval =100, calc_option = 'ML_QBC', ensemble = "NVT"):
    """
        Run ase md with starting structure of TS_{rxn_number}.xyz. The md result will be
        stored in {output_dir}/MD_{rxn_number}_{time_fs}fs_{temp}K_{idx}

        Parameters
        ----------
        starting_structure: xyz file name
        output_dir: str
        temp: int
        time_fs: float
        stepsize: float
        lattice: None or np array that can be reshaped into (3,3)
        interval: int
            Printing interval.
        calc_option: 'ML' or 'ML_QBC'
        ensemble: str
            "NVT" or "NVE"

        Returns
        -------

        """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    molecule = ase.io.read(starting_structure)
    # boxlen = 20
    # lattice = np.array([0, 0, boxlen, 0, boxlen, 0, boxlen, 0, 0]).reshape(3, 3)
    if lattice is not None:
        molecule.set_cell(lattice.reshape(3,3))
        molecule.center()
        molecule.set_pbc(True)
    if calc_option=='ML':
        model_path = '../models/old_single_model'
        model_type = 'NewtonNet'
        yml = glob.glob(f'{model_path}/run_scripts/*.yml')
        assert len(yml) ==1, f'{len(yml)} yaml files found in {model_path}/run_scripts'
        yml = yml[0]
        calc = MLAseCalculator(f'{model_path}/models/best_model_state.tar', yml,
                                    model_type = model_type, lattice = lattice)
    elif calc_option =="ML_QBC": # query by committee for active learning trained model
        result_dir = "../models/model_al_1kperrxn_1"
        committee = CommitteeRegressor.from_dir(result_dir, force_cpu=True, iteration=-1, lattice = lattice)
        iteration = committee.get_latest_iteration()
        assert type(iteration) == int
        # run ase md simulations and sample points of uncertain
        disagreement_thresh = None
        calc = MLCommitteeAseCalculator(committee, disagreement_thresh, lattice=lattice)
    else:
        raise NotImplementedError(f"option {calc_option} not supported")
    MD = AseMD(molecule, calc,temp = temp, time_fs = time_fs, stepsize=stepsize)

    output_name = f'MD_{time_fs}fs_{temp}K'
    if not os.path.isdir(f'{output_dir}/{output_name}'):
        os.mkdir(f'{output_dir}/{output_name}')
    print(f'Simulation for {starting_structure} at {temp}K for {time_fs}fs ')

    MD.run_simulation(f'{output_dir}/{output_name}', interval, output_name,ensemble=ensemble, convergence_criteria=None)

def wrap_PBC_traj(traj_file, out_name = 'md_cell.traj'):
    # wrap PBC into lattice cell for visualization
    W = ase.io.read(traj_file, index=":", format="traj")
    for w in W:
        w.wrap()
    ase.io.write(out_name, W, format="traj")

if __name__ == '__main__':
    ### Committer analysis ###
    # setting for lrc runs
    temp = 500
    output_dir = f'/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/local/committer_analysis_qchem_{temp}K'
    # run_rxn_single_TS('01',1,300,50,output_dir,interval=100,stop_when_committed=True,calc_option = 'ML_QBC')
    # committer_analysis('19',['md_result/MD_19_200fs_1000K_1/MD_19_200fs_1000K_1.traj'])
    run_simulations(100, 50, output_dir, temp=temp, is_up_to_repeat=True, stop_when_committed=True, calc_option='QCHEM')
    analyze_committor_all(50, temp=temp, result_dir=output_dir, replace_failed=False)

