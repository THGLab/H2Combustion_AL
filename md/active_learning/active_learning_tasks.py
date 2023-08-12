#!/usr/bin/env python

# @Time: Aug 2022
# @Author: Nancy Xingyi Guan

import os,sys
import glob
import argparse
import time
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from iodata import load_one
from combust.utils import write_data_npz,get_spin_for_rxn, is_job_in_queue
from md.active_learning.active_learner import CommitteeRegressor
from md.active_learning.data_sampler import DataSampler
from md.aseMD import MLCommitteeAseCalculator, MLCommittee_qchem_hybrid_AseCalculator
from ase.calculators.qchem import QChem
from combust.utils import get_spin_for_rxn
import timeit


########################
### Task functions #####
########################

def evaluate(res_dir,by_rxn=True,cpu = False):
    committee = CommitteeRegressor.from_dir(res_dir, force_cpu=cpu)
    committee.evaluate(write_df=True,by_rxn=by_rxn)

def evaluate_npzs(res_dir,npzs,iteration=-1,cpu = False):
    """

    Parameters
    ----------
    res_dir: model directory
    npzs: list of npz file names
    iteration: int
    cpu: bool
    fname: str. None for no output file

    Returns
    -------
    res_dict: dict {npz:np.array([e_error,f_error, n_data])}
    """
    committee = CommitteeRegressor.from_dir(res_dir, force_cpu=cpu, iteration = iteration)
    res_dict = committee.evaluate_npzs(npzs)
    print(res_dict)
    return res_dict


def initial_training(result_dir,nlearners = 4,added=[]):
    yaml_template = 'config_h2_template.yml'
    committee = CommitteeRegressor.initialize_committee(yaml_template,nlearners,result_dir,added=added)

def metad_production(result_dir, temp = 300, repeat=1,rxn='16',cpu=True, time_fs = 10000, stepsize = 0.1,
                     ensemble='NVT',hybrid = False, plumed_in_file = None, save_uncertain = False, time = False,
                     method = "DFT", iteration = -1, suffix = "", thresh = 10):
    """

    Parameters
    ----------
    result_dir: CommitteeRegressor root directory
    temp: int
    repeat: int
    rxn: str of length 2
    cpu: bool
    time_fs: int
    stepsize: float
    ensemble: "NVT" or "NVE"
    hybrid: bool, will use semiempirical/Qchem when model uncertainty is high
    plumed_in_file: str
    save_uncertain: bool.  If true, will save uncertain steps as a separate ase.traj file
    time: bool. Keep track of total time and time per step if true
    method: str. "SE" or "DFT. Only used together with hybrid = True
    iteration: int. model active learning iteration
    suffix: str. append to end of folder name
    thresh: int/float. thresh for switching to qchem

    Returns
    -------

    """
    if plumed_in_file == None: #default value
        plumed_in_file = f'metad/rxn{rxn}.dat'
    else:
        suffix += plumed_in_file.split('/')[-1].split('.')[-2][5:]
    print(f"Running metad for rxn{rxn}")
    if iteration == -1:
        committee = CommitteeRegressor.from_dir(result_dir, force_cpu=cpu, iteration=-1)
        iteration = committee.get_latest_iteration()
        if type(iteration) != int:
            iteration = int(np.min(iteration))
        # if training in progress, use second to last iteration
        if np.mean(committee.progress_monitor()) < 1800:
            committee = CommitteeRegressor.from_dir(result_dir, force_cpu=cpu, iteration=-2)
            iteration = iteration-1
    else:
        committee = CommitteeRegressor.from_dir(result_dir, force_cpu=cpu, iteration=iteration)
    assert type(iteration) == int
    # run ase md simulations and sample points of uncertain
    disagreement_thresh =  thresh #committee.settings['al']['disagreement_thresh'][0] * 5
    if hybrid:
        if method == "SE":
            qchem_calc = QChem(label=os.path.join('metad/qchem',f"pr_metad_TS{rxn}_SE"), #input file name without .in
                             method='PBEh-3c',
                             basis='def2-mSVP',
                             charge = 0,
                             multiplicity = get_spin_for_rxn(rxn),
                             nt=1,np=1,
                             scf_max_cycles='500',
                             incdft ='0',
                             symmetry='false')
        elif method == "DFT":
            qchem_calc = QChem(label=os.path.join('metad/qchem',f"pr_metad_TS{rxn}_DFT"), #input file name without .in
                         method='wB97X-V',
                         basis='cc-pVTZ',
                         charge = 0,
                         multiplicity = get_spin_for_rxn(rxn),
                         nt=16,np=1,
                         scf_max_cycles='500',
                         incdft='0',
                         symmetry='false')
        else:
            raise ValueError("method needs to be one of ['SE','DFT']")
        calc = MLCommittee_qchem_hybrid_AseCalculator(committee, qchem_calc, disagreement_thresh)
    else:
        calc = MLCommitteeAseCalculator(committee, disagreement_thresh)
    sampler = DataSampler(calc)
    name = f"pr_metad_TS{rxn}_traj_{iteration}{suffix}"
    if time:
        start = timeit.default_timer()
    sampler.run_metad_simulation(
        f'/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/md/irc_geom/TS_{rxn}.xyz', #starting structure
        f'metad/{name}',  # name of metad folder to be generated
        plumed_in_file, #plumed setting file
        repeat=repeat,
        temp=temp,
        time_fs=time_fs,
        ensemble=ensemble,
        stepsize=stepsize)
    if time:
        end = timeit.default_timer()
    with open(f'metad/{name}/info.txt', 'w') as f:
        f.write(f"Rxn: {rxn}\n")
        f.write(f"Model: {result_dir}\n")
        f.write(f"iteration: {iteration}\n")
        f.write(f"Simulation time: {time_fs} fs\n")
        f.write(f"stepsize: {stepsize}\n")
        f.write(f"Temp: {temp}\n")
        f.write(f"ensemble: {ensemble}\n")
        f.write(f"disagreement_thresh: {disagreement_thresh}\n")
        f.write("Plumed input file:\n")
        with open(plumed_in_file,'r') as pf:
            f.writelines(pf.readlines())
        f.write("\n\n\n")

    stdev_pre = sampler.calc.stdev
    print(f"Number of uncertain steps/ steps driven with {method} methods:", len(stdev_pre))
    if hybrid:
        with open("pr_metad_uncertain_steps.txt",'a+') as f:
            f.write(f"{rxn}    {iteration}    {name}    {int(time_fs/stepsize)}    {len(stdev_pre)}\n")
        with open(f'metad/{name}/info.txt', 'a+') as f:
            f.write(f"Number of uncertain steps: {len(stdev_pre)}\n")
            f.write(f"Index of uncertain steps: {calc.uncertain_steps}\n")
            f.write(f"Total steps count in calc: {calc.count}")
    with open(f'metad/{name}/uncertain_step.txt', 'w') as f:
        f.write(f"{calc.uncertain_steps}\n")
    if save_uncertain:
        sampler.generate_traj(name,"geom/uncertain_traj")

    if time:
        print(f"Total time: {end - start} seconds")
        print(f"Time per step: {(end-start)/(time_fs/stepsize)} seconds")
        with open(f'metad/{name}/info.txt','a+') as f:
            f.write(f"Total time: {end - start} seconds\n")
            f.write(f"Time per step: {(end - start) / (time_fs / stepsize)} seconds\n")

    # save_plots_std(stdev_pre, stdev_pre,
    #                    os.path.join('metad', f'std_plots/rxn{rxn}_{iteration}_{time_fs}fs_std.png'))

def sampling(result_dir, qchem_dir, metad=True,repeat=1,rxn='16',cpu=False, time_fs = 2000, stepsize=0.1):
    # Do not support double spin state reaction for now
    print(f"Running metad for rxn{rxn}")
    committee = CommitteeRegressor.from_dir(result_dir,force_cpu=cpu,iteration=-1)
    iteration = committee.get_latest_iteration()
    assert type(iteration) == int
    # run ase md simulations and sample points of uncertain
    disagreement_thresh = [2, 100]
    calc = MLCommitteeAseCalculator(committee, disagreement_thresh)
    sampler = DataSampler(calc)

    if metad:
        sampler.run_metad_simulation(
            f'/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/md/irc_geom/TS_{rxn}.xyz',
            f'metad/{committee.prefix}metad_TS{rxn}_traj_{iteration}',
            f'metad/rxn{rxn}.dat',
            repeat=repeat,
            time_fs=time_fs,
            stepsize = stepsize)
        stdev_pre = sampler.calc.stdev
        nstructure,stdev = sampler.generate_qchem_input_fragmo_and_sad(f'metad_TS{rxn}_{iteration}',
                                                    qchem_dir,
                                                    max_number='auto',
                                                    spin=get_spin_for_rxn(rxn),
                                                    report_stdev=True)
        with open(f'{result_dir}/data_and_stdev.txt','a') as f:
            f.write(f"{iteration}   {rxn}   {np.mean(stdev_pre)}   {np.mean(stdev)}   {len(stdev_pre)}   {nstructure}\n")
        save_plots_std(stdev_pre, stdev,os.path.join(result_dir,f'std_plots/rxn{rxn}_{iteration}_std.png'))
    else:
        sampler.run_simulation(
            f'/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/md/irc_geom/TS_{rxn}.xyz',
            f'md/{committee.prefix}md_TS{rxn}_traj_{iteration}',
            time_fs=time_fs,
			temp=1000)
        qchem_dir = f'geom/{committee.prefix}md_TS{rxn}_{iteration}_{time_fs}fs_qchem'
        sampler.generate_qchem_input(f'md_TS{rxn}_{iteration}', qchem_dir,spin=get_spin_for_rxn(rxn))


def retrain(result_dir,npz_name):
    #retrain
    print(f"Retraining with additional data {npz_name}")
    committee = CommitteeRegressor.from_dir(result_dir,force_cpu=True,iteration=-1)
    #committee.update_setting('training','epochs',5000)
    #committee.update_setting('training','lr_scheduler',['plateau', 15, 20, 0.7, 1.0e-6])
    #committee.update_setting('al','added',[])
    iteration = committee.get_latest_iteration()
    if type(npz_name) is list:
        for npz in npz_name:
            committee.add_training_data(npz, no_repeat=True)
    elif npz_name==None:
        pass
    else:
        committee.add_training_data(npz_name)
    committee.teach(new_submit_script=True)
    # make sure all models are running
    time.sleep(100)
    while not ((type(committee.get_latest_iteration()) == int) and (committee.get_latest_iteration() == iteration +1)):
        time.sleep(100)
        iter_list = committee.get_latest_iteration()
        if type(iter_list) != int:
            idx = np.argmin(iter_list)
            if not is_job_in_queue(f'{committee.prefix}model_{idx}'):
                print(f"model {idx} failed to retrain due to unknown reason, resubmitting...")
                os.system(f'sbatch slurm_{idx}.sh')
        if np.max(iter_list) > iteration +1:
            raise ValueError(f"Number of iteration {iter_list} larger than expected. Please double check")


def run_qchem(qchem_dir,jobs_per_file=300):
    print(f"Running qchem for {qchem_dir}")
    cwd = os.getcwd()
    os.chdir(qchem_dir)
    os.system('rm slurm_*.sh R-*.out.txt')

    threads = 16
    script_head = f"""#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH -A lr_ninjaone
#SBATCH -p csd_lr6_192
#SBATCH -q condo_ninjaone
#SBATCH --job-name=qchem_{qchem_dir.split('/')[-1]}_1
#SBATCH --output=R-%x.out.txt

source ~/.bashrc
#module load qchem/5.2

"""
    script_str = script_head
    i=0 #actual number of files that need to be run
    for file in sorted(glob.glob('*.in')):
        filename = file[:-3]
        out_file = filename + '.out'
        if os.path.isfile(out_file):
            with open(out_file, 'r') as f:
                lines = f.readlines()
            # check out file is complete, if so skip this file
            complete = False
            for l in lines[-10:]:
                if "Thank you very much for using Q-Chem.  Have a nice day." in l:
                    complete = True
            try:
                mol = load_one(out_file, fmt='qchemlog')
            except:
                complete = False
                print(f"Problem loading file {out_file} using loadone. Rerun.")
            if complete:
                # print(f"skip {out_file} because calculation already complete")
                continue
        i+=1
        cmd = f'qchem -nt {threads} {filename}.in {filename}.out\n'
        script_str += cmd
        if i % jobs_per_file == 0 and i!=0:
            with open(f'slurm_{i // jobs_per_file}.sh','w') as f:
                f.write(script_str)
                f.write("echo 'complete'")
            os.system(f'sbatch slurm_{i // jobs_per_file}.sh')
            script_str = script_head.replace(f"{qchem_dir.split('/')[-1]}_1",f"{qchem_dir.split('/')[-1]}_{i // jobs_per_file +1}")
    if i % jobs_per_file !=0:
        with open(f'slurm_{i // jobs_per_file +1}.sh', 'w') as f:
            f.write(script_str)
            f.write("echo 'complete'")
        os.system(f'sbatch slurm_{i // jobs_per_file +1}.sh')
    os.chdir(cwd)





def qchem_to_npz(qchem_dir,npz_name,rxn, in_progress = False):
    print(f"qchem {qchem_dir} to npz {npz_name}")
    if in_progress:
        # check whether qchem jobs are finished
        time.sleep(100) # leave some time for job submission
        while is_job_in_queue(f"qchem_{qchem_dir.split('/')[-1]}"):
            time.sleep(300)
        # check slurm output
        time.sleep(100)
        rerun = False
        for f in glob.glob(f"{qchem_dir}/R-*.out.txt"):
            result = subprocess.check_output(['tail', '-n', '1', f])
            line = result.decode('utf-8').strip()
            if line !='complete':
                rerun = True
                print(f"{f} quitted without completion, job resubmitted")
                print(line)
                cwd = os.getcwd()
                os.chdir(qchem_dir)
                os.system('rm R-*.out.txt')
                os.chdir(cwd)
                run_qchem(qchem_dir)
        if rerun:
            return qchem_to_npz(qchem_dir, npz_name, rxn, in_progress=True)
    data = DataSampler.parse_qchem_output(qchem_dir,rxn=rxn,fragmo_and_sad=True)
    write_data_npz(data,npz_name)










###########################
###   Helper functions  ###
###########################


def save_plots_std(std_pre,std,name):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hist(std_pre)
    ax[1].hist(std)
    plt.setp(ax[0], xlabel='Std for structures sampled from metad')
    plt.setp(ax[1], xlabel='Std for down-sampled structures')
    plt.setp(ax[:], ylabel='Count')
    plt.title(f'Std distribution {name.split("/")[-1][:-8]}')
    plt.savefig(name, dpi=300)

###############
### Parsers ###
###############

def parse_evaluate(arguments):
    parser = argparse.ArgumentParser(
        description="Evaluate performance of the model"
        )
    parser.add_argument("-r", "--by_rxn", default=True,
                        help="Write model performance for each reaction separately")
    parser.add_argument("-m", "--model", default=None, type=str,
                        help="Path to root directory of the model.")
    parser.add_argument('--cpu', dest='cpu', action='store_true', help="Run on cpu")
    parser.add_argument('--no-cpu', dest='cpu', action='store_false')
    parser.set_defaults(cpu=True)
    return parser.parse_args(arguments)

def parse_initial_train(arguments):
    parser = argparse.ArgumentParser(
        description="Initialize active learning and start first training"
        )
    parser.add_argument("-d", "--npz", nargs='*', default=[],
                        help="Path and name of additional npz files.")
    parser.add_argument("-n", "--nlearners", default=4,
                        help="Path and name of additional npz files.")
    parser.add_argument("-m", "--model", default=None, type=str,
                        help="Path to root directory of the model.")

    return parser.parse_args(arguments)

def parse_metad(arguments):
    parser = argparse.ArgumentParser(
        description="Metadynamics sampling for active learning new structures"
        )
    parser.add_argument("-d", "--qchem_dir", default=None, type=str,
                        help="Path to directory to be created that contains *.in files.")
    parser.add_argument("-e", "--ensemble", default='NVT', type=str,
                        help="Ensemble to do metadynamics. NVE or NVt ")
    parser.add_argument("-m", "--model", default=None, type=str,
                        help="Path to root directory of the model.")
    parser.add_argument("-p", "--plumed", default=None, type=str,
                        help="Plumed setting file ")
    parser.add_argument("-r", "--rxn", default=None, type=str,
                        help="Reaction number to do metadynamics. One reaction at a time ")
    parser.add_argument("-s", "--stepsize", default=0.1, type=float,
                        help="Stepsize of metadynamics in fs.")
    parser.add_argument("-t", "--time_fs", default=2000, type=int,
                        help="Length of metadynamics in fs.")
    parser.add_argument("--iter", default=-1, type=int,
                        help="Model active leaning round number")
    parser.add_argument("--suffix", default="", type=str,
                        help="suffix fo the output directory")
    parser.add_argument("--temp", default=300, type=int,
                        help="Tempreture for metadynamics in Kelvin")
    parser.add_argument("--thresh", default=10, type=float,
                        help="Thresh for switching to qchem")

    parser.set_defaults(cpu=True)
    parser.add_argument('--cpu', dest='cpu', action='store_true',help="Run metadynamics on cpu")
    parser.add_argument('--no-cpu', dest='cpu', action='store_false')
    parser.set_defaults(hybrid=False)
    parser.add_argument('--hybrid', dest='hybrid', action='store_true', help="Run metadynamics with hybrid of MLCommittee models and semiempirical/qchem methods")
    parser.set_defaults(production=True)
    parser.add_argument('--production', dest='production', action='store_true', help="In production mode will not generate qchem.in files")
    parser.add_argument('--generate_qchem', dest='production', action='store_false')
    parser.set_defaults(save_uncertain=False)
    parser.add_argument('--save_uncertain', dest='save_uncertain', action='store_true',
                        help="Save steps driven with semiempirical methods as a ase traj")
    parser.set_defaults(timeit=False)
    parser.add_argument('--timeit', dest='timeit', action='store_true',
                        help="print the total time and time per step")


    parser.set_defaults(production=False)

    return parser.parse_args(arguments)

def parse_qchem(arguments):
    parser = argparse.ArgumentParser(
        description="Run qchem for given dir that contains .in files"
        )
    parser.add_argument("qchem_dir",
                        help="Path to directory that contains *.in files.")
    parser.add_argument("npz",
                        help="Path and name of output npz files.")
    parser.add_argument("-r", "--rxn", default=None, type=str,
                        help="Reaction number to do metadynamics. One reaction at a time ")

    return parser.parse_args(arguments)

def parse_retrain(arguments):
    parser = argparse.ArgumentParser(
        description="Retraining models with additional data in active learning loops"
        )
    parser.add_argument("-d", "--npz", nargs='*', default=None,
                        help="Path and name of additional npz files.")

    parser.add_argument("-m", "--model", default=None, type=str,
                        help="Path to root directory of the model.")

    return parser.parse_args(arguments)

if __name__ =='__main__':
    args = sys.argv[1:]
    print(sys.argv)
    task = args.pop(0)
    cwd = os.getcwd()
    os.chdir('/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/local/active_learning/')

    if task == "metad":
        # still need to deal with fragmo vs sad
        parsed = parse_metad(args)
        if parsed.production:
            metad_production(parsed.model,repeat=1,rxn=parsed.rxn,cpu=parsed.cpu,temp=parsed.temp,
                 time_fs=parsed.time_fs, stepsize=parsed.stepsize, ensemble=parsed.ensemble, hybrid=parsed.hybrid,
                 plumed_in_file=parsed.plumed, save_uncertain = parsed.save_uncertain, time=parsed.timeit,
                 iteration= parsed.iter, suffix= parsed.suffix, thresh= parsed.thresh)
        else:
            sampling(parsed.model,parsed.qchem_dir,metad=True,repeat=1,rxn=parsed.rxn,cpu=parsed.cpu,
                 time_fs=parsed.time_fs, stepsize=parsed.stepsize)
    elif task == "qchem":
        parsed = parse_qchem(args)
        run_qchem(parsed.qchem_dir)
        qchem_to_npz(parsed.qchem_dir,parsed.npz, parsed.rxn, in_progress = True)
    elif task =="retrain":
        parsed = parse_retrain(args)
        retrain(parsed.model,parsed.npz)
    elif task =="initial_train":
        parsed = parse_initial_train(args)
        initial_training(parsed.model, parsed.nlearners, added = parsed.npz)
    elif task == "evaluate":
        parsed = parse_evaluate(args)
        evaluate(parsed.model,by_rxn = parsed.by_rxn, cpu = parsed.cpu)
    else:
        raise NotImplementedError("Available Tasks are: ['evaluate', 'initial_train', 'metad','qchem','retrain']")

    os.chdir(cwd)
