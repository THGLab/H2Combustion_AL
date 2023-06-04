#!/usr/bin/env python

# @Time: Oct 2021
# @Author: Nancy Xingyi Guan

import os
import glob
import shutil
import time
import numpy as np
import pandas as pd
import sys
import multiprocessing as mp
from functools import partial
import ase.io
from combust.utils import get_spin_for_rxn,run_command,CommandExecuteError, write_and_submit_to_slurm, is_job_in_queue
from md.active_learning.active_learning_tasks import retrain, evaluate_npzs
from md.active_learning.active_learner import ActiveLearner,CommitteeRegressor
from md.active_learning.data_sampler import DataSampler


task_file = "/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/md/active_learning/active_learning_tasks.py"


######################################
###  Separate steps in training  #####
######################################

def initial_training(result_dir,added=[]):
    if os.path.isdir(result_dir):
        raise ValueError(f'{result_dir} already exist, please use a new dir name to initialize')
    if len(added) == 0:
        os.system(f'python {task_file} initial_train -m {result_dir}')
    else:
        os.system(f'python {task_file} initial_train -m {result_dir} --npz {" ".join(added)}')
    while not os.path.isdir(result_dir):
        time.sleep(60)
    #allow some time to set up contents in dir
    time.sleep(100)
    with open(f'{result_dir}/status.txt', 'w') as f:
        f.write('initialize\n')


def metad_sampling(committee, result_dir, time_fs = 2000, cpu=True, stepsize = 0.1):
    print("metad_sampling")
    print(f"Iteration: {committee.get_latest_iteration()} Progress (batch number): {committee.progress_monitor()}")
    while np.mean(committee.progress_monitor()) < 1600:
        time.sleep(300) #check every 5 min
    #sampling submit new script
    iteration = committee.get_latest_iteration()
    rxns = committee.settings['al']['metad_rxns']

    for rxn in rxns:
        qchem_dir = f'geom/{committee.prefix}metad_TS{rxn}_{iteration}_{time_fs}fs_qchem'
        name = qchem_dir.split('/')[-1][:-6]
        # os.system(f'python active_learning_tasks.py metad {qchem_dir} -m {result_dir} -r {rxn} -t {time_fs} --{"" if cpu else "no-"}cpu')
        cmd = f'python {task_file} metad -d {qchem_dir} -m {result_dir} -r {rxn} -t {time_fs} -s {stepsize} --{"" if cpu else "no-"}cpu'
        write_and_submit_to_slurm(cmd,name,'geom/script',prefix="")


    with open(f'{result_dir}/status.txt', 'a') as f:
        f.write('metad\n')


def qchem_and_parse_single(iteration, time_fs, prefix, rxn):
    qchem_dir = f'geom/{prefix}metad_TS{rxn}_{iteration}_{time_fs}fs_qchem'
    npz_name = f'geom/new_data/max300/{prefix}metad_TS{rxn}_{iteration}_{time_fs}fs.npz'
    if os.path.isfile(npz_name):
        #skip qchem because npz already exist
        return npz_name
    while not os.path.isdir(qchem_dir):
        time.sleep(60)
    time.sleep(60) # give some time between mkdir and file generation
    # if no geometry sampled, skip
    if len(os.listdir(qchem_dir)) == 0:
        print(f"No geom found for {qchem_dir}, skipped")
        return ""
    cmd = f'python {task_file} qchem {qchem_dir} {npz_name} -r {rxn} '
    # write_and_submit_to_slurm(cmd,f'qchem_{iteration}_{rxn}','geom/script')
    os.system(cmd)
    while not os.path.isfile(npz_name):
        time.sleep(300)
    return npz_name

def qchem_and_parse(committee,status_file,time_fs =2000, rxns = ['09','10','13','16','17','18']):
    print("qchem_and_parse")
    iteration = committee.get_latest_iteration()
    npzs =[]
    with mp.Pool() as pool:  # use all available cores, otherwise specify the number you want as an argument
        for result in pool.map(partial(qchem_and_parse_single, iteration, time_fs,committee.prefix), rxns):
            npzs.append(result)
            print(f"Qchem and parse finished for {result}")
    with open(status_file, 'a') as f:
        f.write('qchem\n')
    print("added_npz: ", [n for n in npzs if len(n)>0])
    return [n for n in npzs if len(n)>0]

def qchem_and_parse_from_traj(traj_file,rxn,npz_name,qchem_dir):
    # this is for adding points from a ase.traj file with all uncertain points
    # skip input generation if qchem dir already exsist
    if os.path.isfile(npz_name):
        print(f"{npz_name} already exist. skip qchem")
        return True
    if not os.path.isdir(qchem_dir) or len(os.listdir(qchem_dir)) == 0:
        name = npz_name.split('/')[-1][:-4]
        ds = DataSampler(calc=None)
        traj = ase.io.read(traj_file, index=":", format="traj")
        print(f"Traj {traj_file} has {len(traj)} data points")
        ds.generate_qchem_input_fragmo_and_sad(name,qchem_dir,max_number=1000,geoms=traj,report_stdev=False,spin=get_spin_for_rxn(rxn))

    if len(os.listdir(qchem_dir)) == 0:
        shutil.rmtree(qchem_dir)
        print(f"No qchem input for {qchem_dir}")
        return False

    cmd = f'python {task_file} qchem {qchem_dir} {npz_name} -r {rxn}'
    try:
        code, out, err = run_command(cmd, timeout=100000)
    except CommandExecuteError as e:
        print(f"error in running command {cmd}: {e}")
        print("out: ", out)
        print("err: ", err)
        raise

    return True

def check_npz_complete(added_npz, restart = True):
    if len(added_npz) > 0:
        npz_available = np.array([False]*len(added_npz))
        time.sleep(100)
        i = 0
        while not npz_available.all():
            if i>2 and not is_job_in_queue("qchem_metad_TS"):
                print(added_npz)
                print(npz_available)
                print(npz_available.all())
                return False
            for i,npz in enumerate(added_npz):
                if os.path.isfile(npz):
                    npz_available[i] = True
            time.sleep(100)
            i+=1
    return True

def retraining(result_dir, added_npz):
    print("retraining")
    #waiting for npz to become available
    completed = check_npz_complete(added_npz,restart=False)
    if not completed:
        raise ValueError(f"At least one of the added npz not found: {added_npz}")
    # os.system(
    #     #     f'python active_learning_tasks.py retrain -m {result_dir} -d {added_npz}')
    retrain(result_dir,added_npz)
    with open(f'{result_dir}/status.txt', 'a') as f:
        f.write('retrain\n')



def training_loop(result_dir, initialize = True,metad_time_fs = 2000, stepsize = 0.1):
    # The idea is to run multiple loops without stopping
    #  training (first 1000 epoch) -> metad (save once every ? structure becomes avail) -> qchem (start as avail) -> retrain
    #                          \_> continue training
    if initialize:
        initial_training(result_dir)

    steps = ['metad','qchem','retrain']
    committee = CommitteeRegressor.from_dir(result_dir, force_cpu=True, iteration=-1)
    status_file = f'{result_dir}/status.txt'
    with open(status_file,'r') as f:
        status = f.readlines()[-1].strip()
    print("Status: ",status)
    if status == "initialize" or status == "retrain":
        while True:
            metad_sampling(committee, result_dir,time_fs=metad_time_fs,stepsize=stepsize)
            added_npz = qchem_and_parse(committee,status_file,time_fs=metad_time_fs)
            retraining(result_dir, added_npz)
    elif status == "metad" or status == "qchem":
        while True:
            added_npz = qchem_and_parse(committee, status_file,time_fs=metad_time_fs)
            retraining(result_dir, added_npz)
            metad_sampling(committee, result_dir,time_fs=metad_time_fs,stepsize=stepsize)
    else:
        raise NotImplementedError(f"Found unknown status {status} for {result_dir}")

###################
### other tasks ###
###################

def copy_latest_model(result_dir,new_dir, replace = False):
    if os.path.isdir(new_dir):
        if replace:
            shutil.rmtree(new_dir)
            os.mkdir(new_dir)
        else:
            if len(os.listdir(new_dir)) != 0:
                raise FileExistsError(f"{new_dir} is not empty")
    else:
        os.mkdir(new_dir)
    committee = CommitteeRegressor.from_dir(result_dir, force_cpu=True, iteration=-1)
    it = committee.get_latest_iteration()
    if np.mean(committee.progress_monitor()) < 1600:
        it -= 1
    for i in range(len(committee.learner_list)):
        os.mkdir(f"{new_dir}/model_{i + 1}")
        shutil.copytree(os.path.join(result_dir, f"model_{i}/training_{it}"),
                        os.path.join(new_dir, f"model_{i}/training_{it}"))
    for f in glob.glob(os.path.join(result_dir, "config*")):
        shutil.copy(f, new_dir)
    return True




def evaluate_aimd_nm(result_dir,fname,irc=False,cpu = False):
    assert fname.endswith("xlsx")
    committee = CommitteeRegressor.from_dir(result_dir, force_cpu=True, iteration=-1)
    n_iter = committee.get_latest_iteration()
    aimd = glob.glob(f"{committee.settings['data']['root']}/*_aimd.npz")
    nm = glob.glob(f"{committee.settings['data']['root']}/*_nm.npz")
    npzs = np.concatenate([aimd,nm])
    if irc:
        irc = glob.glob(f"{committee.settings['data']['root']}/*_irc.npz")
        npzs = np.concatenate([npzs,irc])

    dfs = []

    for i in range(n_iter):
        res_dict = evaluate_npzs(result_dir,npzs=npzs,iteration=i,cpu = cpu)
        df = pd.DataFrame(columns=['mae_E', 'mae_F', 'n_data'])
        for k,v in res_dict.items():
            df.loc[k] = v
        dfs.append(df)


    # Creating Excel Writer Object from Pandas
    with pd.ExcelWriter(fname, engine='xlsxwriter') as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet('Validation')
        writer.sheets['Validation'] = worksheet
        start = 0
        for i, df in enumerate(dfs):
            df.to_excel(writer, sheet_name='Validation', startrow=start, startcol=0)
            start += df.shape[0] + 2

def run_qchem_traj(traj):
    root = "geom/uncertain_traj"
    npz_path = "geom/new_data/uncertain_traj"
    rxn = traj.split('/')[-1].split('TS')[1][:2]
    name = traj.split('/')[-1][:-5]
    folder = f"{root}/{name}"
    print(f"Call qchem and parse for {traj}")
    qchem_and_parse_from_traj(traj, rxn, f"{npz_path}/{name}.npz", folder)

def run_uncertain_traj_prl(uncertain_trajs):
    with mp.Pool() as pool:  # use all available cores, otherwise specify the number you want as an argument
        pool.map(run_qchem_traj, uncertain_trajs)

if __name__ =='__main__':
    result_dir = sys.argv[1]  # 'model_al_1kperrxn_bwsl_1'
    training_loop(result_dir,initialize=False,metad_time_fs = int(sys.argv[2]),stepsize=0.5)
    # committee = CommitteeRegressor.from_dir(result_dir, force_cpu=True, iteration=-1)
    # committee.update_setting('al', 'metad_rxns',  ['01','09', 10,13,16,17,18])
    # committee.update_setting('al', 'added', glob.glob("geom/new_data/combined/*.npz"))
    # added_npz = glob.glob("geom/new_data/uncertain_traj/*.npz")
    # committee.update_setting('training', 'epochs', 5000)
    # committee.update_setting('training', 'lr_scheduler', ['plateau', 15, 50, 0.7, 1.0e-6])
    # retraining(result_dir, added_npz)


    ### for adding uncertain traj to training ###
    # uncertain_trajs = glob.glob("geom/uncertain_traj/*.traj")
    # run_uncertain_traj_prl(uncertain_trajs)

    # evaluate_aimd_nm(result_dir,"eval_ori.xlsx",irc=True)
    # copy_latest_model(result_dir,"newest_model", replace=False)

