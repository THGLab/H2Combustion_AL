#!/usr/bin/env python

# @Time: 10/13/21 
# @Author: Nancy Xingyi Guan
# @File: active_learner.py

import os
import numpy as np
from glob import glob
import pandas as pd
import warnings
from md.active_learning.train_newtonnet import train_newtonnet,NewtonNetModel


import yaml



class ActiveLearner():
    def __init__(self,
                 idx,
                 yml,
                 result_dir,
                 pretrained = False,
                 force_cpu = False,
                 iteration=-1,
                 prefix="",
                 lattice = None,
                 **fit_kwargs
                 ):
        self.idx = idx
        self.yml =yml
        self.path = f'{result_dir}/model_{idx}'
        self.result_dir = result_dir
        self.model = None
        self.cpu = force_cpu
        self.prefix = prefix
        self.lattice = lattice
        if pretrained == False:
            self.teach(new_submit_script=True)
        else:
            self.update_model(iteration=iteration)

    def get_latest_iteration(self):
        trainings = [os.path.join(self.path, tr) for tr in os.listdir(self.path) if
                     os.path.isdir(os.path.join(self.path, tr))]
        return len(trainings)

    def _get_model_path(self,iteration=-1):
        trainings = [os.path.join(self.path, tr) for tr in os.listdir(self.path) if
                     os.path.isdir(os.path.join(self.path, tr))]
        if len(trainings) > 0:
            # sort by index
            def getint(name):
                num = name.split('_')[-1]
                return int(num)

            trainings.sort(key=getint)
            # print(trainings)
            # print(trainings[iteration])
            if iteration > 0:
                iteration -= 1 # change from index start at 1 to index start at 0
            pretrained = os.path.join(os.path.join(os.getcwd(), trainings[iteration]),'models/best_model_state.tar')
        else:
            pretrained = False
        return pretrained

    def update_yml_pretrained(self):
        #update pretrained model for yml in parent directory
        settings = yaml.safe_load(open(self.yml, "r"))
        settings['model']['pre_trained']=self._get_model_path()
        with open(self.yml, "w") as f:
            yaml.dump(settings, f, default_flow_style=None)

    def update_setting(self,key1,key2,value):
        settings = yaml.safe_load(open(self.yml, "r"))
        settings[key1][key2]=value
        with open(self.yml, "w") as f:
            yaml.dump(settings, f, default_flow_style=None)

    def update_model(self,iteration=-1):
        self.model = NewtonNetModel(self._get_model_path(iteration),self.yml,lattice=self.lattice,force_cpu=self.cpu)


    def predict(self,npz_data):
        return self.model.predict(npz_data)



    def teach(self, new_submit_script = False, **kwargs):
        """
        Use the updated setting to retrain the predictor with the augmented dataset.

        Args:
            new_submit_script: bool
            #NOT IMPLEMENTED YET
            bootstrap: If True, training is done on a bootstrapped dataset. Useful for building Committee models
                with bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
                Useful when working with models where the .fit() method doesn't retrain the model from scratch (e. g. in
                tensorflow or keras).
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        settings = yaml.safe_load(open(self.yml, "r"))
        prefix = settings['general']['prefix']
        if new_submit_script == False:
            train_newtonnet(self.yml,**kwargs)
            self.update_model()
        elif new_submit_script == 'perlmutter':
            # submit a new job to perlmutter gpu
            slurm_script = f'slurm_{self.idx}.sh'
            with open(slurm_script, 'w') as f:
                ###perlmutter###
                f.writelines(
                    f"""#!/bin/bash
  
#SBATCH -A m4103_g
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH --time=24:00:00
#SBATCH --comment=96:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
###SBATCH -c 16
#SBATCH --gpus-per-task=4
#SBATCH -J {prefix}model_{self.idx}
#SBATCH --output={self.result_dir}/slurm_{self.idx}.out.txt

source ~/.bashrc
module load python/3.7
conda activate torch-gpu

srun python /pscratch/sd/n/nguan/hcombust/active_learning/train_newtonnet.py {self.yml} True     
conda deactivate
""")
        else:
            #submit a new job to cluster
            slurm_script=f'slurm_{self.idx}.sh'
            settings = yaml.safe_load(open(self.yml, "r"))
            ndevice = len(settings['general']['device'])
            if ndevice == 2:
                constr = 'es1_v100'
            elif ndevice == 4:
                constr = 'es1_2080ti'
            else:
                raise ValueError(f"Please check the device setting to use 2 or 4 cuda device. Current setting: {settings['general']['device']}")
            with open(slurm_script,'w') as f:
                ###lawrencium###
                f.writelines(
                    f"""#!/bin/bash

#SBATCH --job-name={prefix}model_{self.idx}
#SBATCH --output={self.result_dir}/slurm_{self.idx}.out.txt
#SBATCH --account=lr_ninjaone
#SBATCH --partition=es1
#SBATCH --qos=condo_ninjaone_es1
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --nodes=1
#SBATCH --gres=gpu:{ndevice}
#SBATCH --constraint {constr}  ##(other options: es1_v100, es1_1080ti, es1_2080ti)


source ~/.bashrc
module load python/3.7
conda activate torch-gpu
###source activate torch
export DISPLAY=""


python /global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/md/active_learning/train_newtonnet.py {self.yml} False

conda deactivate
"""
                             )
                ###nersc###
#                 f.writelines(
#                     f"""#!/bin/bash
#
# #SBATCH -A m1863_g
# #SBATCH -C gpu
# #SBATCH -q early_science
# #SBATCH -n 1
# #SBATCH --ntasks-per-node=1
# #SBATCH -c 128
# #SBATCH --gpus-per-task=1
# #SBATCH --job-name=model_{self.idx}
# #SBATCH --output=slurm_{self.idx}.out.txt
# #SBATCH --time=48:00:00
#
# source ~/.bashrc
# module load python/3.7
# conda activate torch-gpu
#
# python /global/homes/n/nguan/AIMD_H_Combustion/H2Combustion/md/active_learning/train_newtonnet.py {self.yml}
#
# conda deactivate
# """
#                 )
            os.system(f'sbatch {slurm_script}')

    def _add_training_data(self,npz, no_repeat=True):
        settings = yaml.safe_load(open(self.yml, "r"))
        settings['al']['added'].append(npz)
        if no_repeat:
            settings['al']['added'] = [*set(settings['al']['added'])]
        with open(self.yml, "w") as f:
            yaml.dump(settings, f, default_flow_style=None)

    def evaluate(self, write_df=True, data_root=None,by_rxn = True):
        n_trainings = len([os.path.join(self.path, tr) for tr in os.listdir(self.path) if
                     os.path.isdir(os.path.join(self.path, tr))])
        settings = yaml.safe_load(open(self.yml, "r"))

        if 'dialation' in settings['data']:
            dialation = settings['data']['dialation']
        else:
            dialation = False

        if data_root== None:
            data_root = settings['data']['root']
        dfs = [] # list of dfs for each iteration
        filename =os.path.join(self.path,'evaluation.xlsx')
        if os.path.isfile(filename):
            dfs = parse_excel_sheet(filename)
            print(f"{len(dfs)} precalculated in {filename}")
            if len(dfs)<=1:
                dfs =[]
            else:
                dfs = dfs[:-1] # discard last one to avoid unfinished calculations
        print(f'n_trainings: {n_trainings}, new eval: {list(range(len(dfs),n_trainings))}')
        for i in range(len(dfs),n_trainings):
            self.update_model(iteration=i)
            df = self.model.evaluate(data_root=data_root, by_rxn=by_rxn,dialation=dialation)
            print(f'Model {self.idx} training {i+1}: \n{df}')
            dfs.append(df)

        if write_df:
            # Creating Excel Writer Object from Pandas
            with pd.ExcelWriter(os.path.join(self.path,'evaluation.xlsx'), engine='xlsxwriter') as writer:
                workbook = writer.book
                worksheet = workbook.add_worksheet('Validation')
                writer.sheets['Validation'] = worksheet
                start = 0
                for i,df in enumerate(dfs):
                    df.to_excel(writer, sheet_name='Validation', startrow=start, startcol=0)
                    start += df.shape[0]+1

        return dfs

    def evaluate_npzs(self,npzs):
        """

        Parameters
        ----------
        npzs: list of npz files to be evaluated

        Returns
        -------
        res: dict {npz: np.array([e_error,f_error, n_data])}
        """
        res = {}
        for npz in npzs:
            name = os.path.basename(npz)
            res[npz] = np.array(self.model.evaluate_npz(npz))
        return res

"""
Classes for committee based algorithms
--------------------------------------
"""

class CommitteeRegressor():
    """
    This class is an abstract model of a committee-based active learning regression.

    Args:
        learner_list: A list of ActiveLearners forming the CommitteeRegressor.
        query_strategy: Query strategy function.
        on_transformed: Whether to transform samples with the pipeline defined by each learner's estimator
            when applying the query strategy.


    """
    def __init__(self, learner_list, result_dir, settings, cpu=False):
        self.learner_list = learner_list
        self.cpu=cpu
        self.path = result_dir
        self.settings = settings
        self.prefix = self.settings['general']['prefix']

    @classmethod
    def initialize_committee(cls,yaml_template,nlearners,result_dir,added=[]):
        if not os.path.isdir(result_dir):
            os.mkdir(result_dir)
        os.system(f'cp {yaml_template} {result_dir}')
        yaml_template = os.path.join(result_dir, yaml_template.split('/')[-1])
        # initializing the learner
        ymls = CommitteeRegressor.initialize_yamls(yaml_template, nlearners, result_dir,added=added)
        settings = yaml.safe_load(open(ymls[0], "r"))
        with open(f'{result_dir}/data_and_stdev.txt','a') as f:
            f.write(f"iteration   rxn   mean_std_pre  mean_std_downsampled  nstructures_pre   nstructures_downsampled\n")
        os.mkdir(os.path.join(result_dir,'std_plots'))
        learners = []
        for i, yml in enumerate(ymls):
            learner = ActiveLearner(i, yml,result_dir, pretrained=False)
            learners.append(learner)
        return cls(learners, result_dir, settings)

    @classmethod
    def from_dir(cls,result_dir,force_cpu=False,iteration=-1, lattice = None):
        if not os.path.isdir(result_dir):
            raise FileNotFoundError(f'No directory found for {result_dir}')
        ymls = glob(f'{result_dir}/*_[0-9].yml')
        settings = yaml.safe_load(open(ymls[0], "r"))
        ymls = sorted(ymls, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        learners = []
        for i, yml in enumerate(ymls):
            learner = ActiveLearner(i, yml,result_dir, pretrained=True,force_cpu=force_cpu,iteration=iteration, lattice=lattice)
            learners.append(learner)
        return cls(learners,result_dir,settings,cpu=force_cpu)


    @classmethod
    def initialize_yamls(cls,config_template, repeat, result_dir,added=[],added_split=np.array([0.8,0.1,0.1])):
        '''

        Parameters
        ----------
        config_template: path to a config.yaml file. The repeat of yml files will be generated in the same directory
        repeat: number of repeat of yml files to be generated
        result_dir:output parent directory of the ML model
        added: list of npz files for additional training
        added_split: a list of 3 floats for ratio of train,val and test

        Returns
        -------
        ymls: a list of names of yml files

        '''
        settings = yaml.safe_load(open(config_template, "r"))
        ymls = []
        for i in range(repeat):
            new_yml = f'{config_template[:-4]}_{i}.yml'
            settings['general']['me'] = new_yml
            settings['general']['output'] = [f'{result_dir}/model_{i}' , 1]
            settings['al']['added'] = added
            settings['al']['added_split'] = added_split.tolist()
            settings['data']['random_states'] = 90 + i
            ymls.append(new_yml)
            with open(new_yml, "w") as f:
                yaml.dump(settings, f ,default_flow_style=None)
        return ymls


    def q_test(self, data, q_ref=0.829):
        """

        Parameters
        ----------
        data: 1d array with shape (nlearners,)
        q_ref: float
            the default reference Q value 0.829 is for a significance level of 95% and 4 data points

        Returns
        -------
        idx: int or None
            the index to be filtered out (return only one index for now as the default is only 4 learners)
        """
        dataset = np.sort(data)
        q_stat_min = (dataset[1] - dataset[0]) / (dataset[-1] - dataset[0])
        q_stat_max = (dataset[-1] - dataset[-2]) / (dataset[-1] - dataset[0])
        idx = None
        if q_stat_min > q_ref:
            idx = np.argmin(data)
        elif q_stat_max > q_ref:
            idx = np.argmax(data)
        return idx

    def predict(self, npz_data, disagreement = False, **predict_kwargs):
        """
        Predicts the values of the samples by averaging the prediction of each regressor.
        Use Dixon's Q test to rule out outliers
        https://en.wikipedia.org/wiki/Dixon%27s_Q_test

        Args:
            npz_data: The samples to be predicted.
            disagreement: choose from ['std','minmax',False]

            **predict_kwargs: Keyword arguments to be passed to the :meth:`vote` method of the CommitteeRegressor.

        Returns:
            The predicted class labels for X.
        """
        vote = self.vote(npz_data, **predict_kwargs) # vote['E']shape (ndata,1,nlearner), vote['F'] shape (ndata,natom,3,nlearner)

        # Use Dixon's Q test to rule out outliers
        # https://www.semanticscholar.org/paper/Statistical-treatment-for-rejection-of-deviant-of-Rorabacher/f72157d3683fd5df5af65e816a211e8aef6cab23/figure/3
        q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
               0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
               0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
               0.308, 0.305, 0.301, 0.29
               ] #start from n=3
        Q95 = {n: q for n, q in zip(range(3, len(q95) + 1), q95)}
        qref = Q95[len(self.learner_list)]
        indices = np.apply_along_axis(self.q_test, 1, vote['E'].reshape(-1,len(self.learner_list)),q_ref =qref)
        # ruling out the outliers for means
        means = {'E': np.mean(vote['E'], axis=-1), #shape (ndata,1,)
                 'F': np.mean(vote['F'], axis=-1)} #shape (ndata,natom,3,)
        # modify means by ruling out the outliers
        for i, idx in enumerate(indices): # i on data index, idx is the learner idex to delete
            if idx is not None:
                means['E'][i] = np.mean(np.delete(vote['E'][i,:,:], idx, axis=-1), axis=-1)
                means['F'][i] = np.mean(np.delete(vote['F'][i,:,:,:], idx, axis=-1), axis=-1)
                # print(f"for energies {vote['E'][i,:,:]}, {idx} is an outlier, Mean E: {means['E'][i]}, Mean F:{means['F'][i]}")

        if disagreement == False:
            return means
        elif disagreement == 'std':
            all_component_force_std = np.std(vote['F'], axis=-1) #shape (ndata,natom,3)
            std = {'E': np.std(vote['E'], axis=-1), #shape (ndata,1)
                 'F': np.max(np.max(all_component_force_std,axis=2),axis=1)} # shape (ndata), force std defined as the max std among 3 * natoms force components
            # print('means:', means)
            # print('std:', std)
            return means,std
        elif disagreement == 'std_olremoved':
            all_component_force_std = np.std(vote['F'], axis=-1)  # shape (ndata,natom,3)
            std = {'E': np.std(vote['E'], axis=-1),  # shape (ndata,1)
                   'F': np.max(np.max(all_component_force_std, axis=2),
                               axis=1)}  # shape (ndata), force std defined as the max std among 3 * natoms force components
            # modify std by removing outliers
            for i, idx in enumerate(indices):# i on data index, idx is the learner idex to delete
                if idx is not None:
                    std['E'][i] = np.std(np.delete(vote['E'][i,:,:], idx, axis=-1), axis=-1)
                    all_component_force_std[i] = np.std(np.delete(vote['F'][i,:,:,:],idx, axis=-1), axis=-1)  # shape (natom,3)
                    std['F'][i] = np.max(all_component_force_std[i])

            return means, std

        elif disagreement== 'minmax':
            all_component_force_minmax = np.max(vote['F'], axis=-1)-np.min(vote['F'], axis=-1)
            minmax =  {'E':np.max(vote['E'], axis=-1)-np.min(vote['E'], axis=-1), #shape (ndata,1)
                       'F':np.max(np.max(all_component_force_minmax,axis=2),axis=1)} #shape (ndata)
            return  means,minmax
        elif disagreement == 'values':
            return means, vote
        else:
            raise ValueError(f"disagreement {disagreement} not in allowed types")

    def vote(self, npz_data, **predict_kwargs):
        """
        Predicts the values(force) for the supplied data for each regressor in the CommitteeRegressor.

        Args:
            npz_data: The samples to cast votes.
            **predict_kwargs: Keyword arguments to be passed to :meth:`predict` of the learners.

        Returns:
            The predicted value for each regressor in the CommitteeRegressor and each sample in npz_data.
        """
        prediction = {'E':np.zeros(shape=(*npz_data['E'].shape, len(self.learner_list))), #shape (ndata,1,nlearner)
                    'F':np.zeros(shape=(*npz_data['F'].shape, len(self.learner_list)))}  #shape (ndata,natom,3,nlearner)

        for learner_idx, learner in enumerate(self.learner_list):
            pred=learner.predict(npz_data)
            prediction['E'][:,:, learner_idx] = pred['E']#.detach().numpy()
            prediction['F'][:,:,:, learner_idx] = pred['F']#.detach().numpy()

        return prediction

    def add_training_data(self, new_npz, no_repeat = True):
        """
        Adds the new data and label to the known data for each learner, but does not retrain the model.

        Args:
            new_npz: The new samples for which the labels are supplied by the expert(Qchem) provided in npz(dict) format.

        Note:
            If the learners have been fitted, the features in npz have to agree with the training samples which the
            classifier has seen.
        """
        for learner in self.learner_list:
            learner._add_training_data(new_npz, no_repeat=no_repeat)

    def teach(self,new_submit_script=False,pretrained = True,**kwargs):
        """
        Retrains learners with current yaml file.

        Args:
            new_submit_script: bool. Submit separate jobs to slurm for training models in parallel
            bootstrap: If True, trains each learner on a bootstrapped set. Useful when building the ensemble by bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        # super().teach(X, y, bootstrap=bootstrap, only_new=only_new, **fit_kwargs)
        for learner in self.learner_list:
            if pretrained:
                learner.update_yml_pretrained()
            else:
                learner.update_setting('model','pre_trained',False)
            learner.teach(new_submit_script,**kwargs)

    def retrain(self,model_idx,training_idx):
        os.system(f'rm -r {self.path}/model_{model_idx}/training_{training_idx}')
        self.learner_list[model_idx].teach(new_submit_script=True)

    def update_setting(self,key1,key2,value):
        for learner in self.learner_list:
            learner.update_setting(key1,key2,value)


    def evaluate(self,write_df=True,data_root=None,by_rxn=True):
        """

        Parameters
        ----------
        write_df: bool
            whether or not to generate evaluation.xlsx file in model directory
        data_root: str or None
            used in case HCombustion data is stored at different path. Default is None for use data_root in yaml file.

        Returns
        -------
        dfs: list of pd.DataFrame for each iteration
        """

        dfs = []
        
        results = []

        for learner in self.learner_list:
            result = learner.evaluate(write_df=write_df,data_root=data_root,by_rxn=by_rxn)
            results.append(result) #array of df shape(nlearner,n_iteration)

        n_trainings = len(results[0])
        print('n_trainings: ',n_trainings)
        print('n_leaners: ', len(results))

        with pd.ExcelWriter(os.path.join(self.path, 'evaluation.xlsx'), engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet('Validation')
            writer.sheets['Validation'] = worksheet
            start =0
            for iteration in range(n_trainings):
                dfs_of_iteration = [l[iteration] for l in results]
                df = pd.concat(dfs_of_iteration).groupby(level=0).mean()
                dfs.append(df)
                print('iteration: ',iteration)
                print(df)

                df.to_excel(writer, sheet_name='Validation', startrow= start, startcol=0,
                            float_format = "%0.2f")
                start += df.shape[0]+1

        return dfs

    def evaluate_npzs(self,npzs):
        """

        Parameters
        ----------
        npzs: list of dict

        Returns
        -------
        res: dict {npz:np.array([e_error,f_error, n_data])}
        """

        dics = []
        for learner in self.learner_list:
            dics.append(learner.evaluate_npzs(npzs))
        # take average of learner predictions
        res = dics[0]
        for d in dics[1:]:
            for k in res.keys():
                res[k] += d[k]
        for k in res.keys():
            res[k] *= 1/(len(self.learner_list))
        return res


    def get_latest_iteration(self):
        latest =[]
        for learner in self.learner_list:
            latest.append(learner.get_latest_iteration())
        if len(set(latest)) !=1:
            warnings.warn(f"Active learners have different rounds of training {latest}. Please check and fix")
            return np.unique(latest)
        else:
            return int(np.unique(latest))

    def progress_monitor(self,iteration=-1):
        epochs = []
        for learner in self.learner_list:
            log_file = os.path.join(learner.path,f"training_{learner.get_latest_iteration()}","log.csv")
            if os.path.isfile(log_file):
                with open(log_file,'r') as f:
                    lines = f.readlines()
                epochs.append(len(lines)-1)
            else:
                epochs.append(0)
        return epochs



def parse_excel_sheet(file, sheet_name=0, threshold=5):
    '''parses multiple tables from an excel sheet into multiple data frame objects.
     Returns dfs, where dfs is a list of data frames '''
    # xl = pd.ExcelFile(file)
    # entire_sheet = xl.parse(sheet_name=sheet_name)
    entire_sheet = pd.read_excel(file, engine='openpyxl', sheet_name=sheet_name)

    # count the number of non-Nan cells in each row and then the change in that number between adjacent rows
    n_values = np.logical_not(entire_sheet.isnull()).sum(axis=1)
    n_values_deltas = n_values[1:] - n_values[:-1].values

    # define the beginnings and ends of tables using delta in n_values
    table_beginnings = n_values_deltas > threshold
    table_beginnings = table_beginnings[table_beginnings].index
    table_endings = n_values_deltas <- threshold
    table_endings = table_endings[table_endings].index
    table_beginnings = table_beginnings.insert(0, 0)
    table_endings = table_endings.insert(len(table_endings),len(n_values))
    if len(table_beginnings) < len(table_endings) or len(table_beginnings) > len(table_endings)+1:
        raise BaseException('Could not detect equal number of beginnings and ends')

    # make data frames
    dfs = []

    for ind in range(len(table_beginnings)):
        start = table_beginnings[ind] +1
        if start ==1:
            start =0
        if ind < len(table_endings):
            stop = table_endings[ind]
        else:
            stop = entire_sheet.shape[0]
        print(stop-start)
        df = pd.read_excel(file, engine='openpyxl',sheet_name=sheet_name, skiprows=start, nrows=stop-start,index_col=0)
        dfs.append(df)

    return dfs






