#!/usr/bin/env python

# @Time: 10/13/21 
# @Author: Nancy Xingyi Guan
# @File: train_newtonnet.py

import sys
import numpy as np
import torch
from torch.optim import Adam
import yaml
import pandas as pd
from itertools import chain

from newtonnet.layers.activations import get_activation_by_string
from newtonnet.models import NewtonNet

from newtonnet.train import Trainer
from combust.data.parse_raw import *
from combust.utils import standardize_batch
from md.ase_interface import data_loader


# torch.autograd.set_detect_anomaly(True)
torch.set_default_tensor_type(torch.DoubleTensor)

class NewtonNetModel():
    ### Constructor ###
    def __init__(self, model_path, settings_path, lattice=None,force_cpu=False, **kwargs):
        """
        Constructor for NewtonNetModel

        Parameters
        ----------
        model_path: str
            path to the model. eg. '5k/models/best_model_state.tar'
        settings_path: str
            path to the .yml setting path. eg. '5k/run_scripts/config_h2.yml'
        model_type: str
            type of model. Default to NewtonNet(MNMP) Options: ['NewtonNet','EquiNet','DRFFM','MNMP']
        lattice: array of (9,)
            lattice vector for pbc
        kwargs
        """
        self.settings = yaml.safe_load(open(settings_path, "r"))
        if force_cpu:
            self.settings['general']['device'] = 'cpu'
        if lattice is not None:
            self.lattice = np.array(lattice).reshape(9, )
        else:
            self.lattice = lattice

        # device
        if type(self.settings['general']['device']) is list:
            self.device = [torch.device(item) for item in self.settings['general']['device']]
        else:
            self.device = [torch.device(self.settings['general']['device'])]

        torch.set_default_tensor_type(torch.DoubleTensor)

        self._load_model(model_path)

    def _load_model(self, model_path):
        # load NewtonNet model
        # settings
        settings = self.settings

        # model
        # activation function
        activation = get_activation_by_string(settings['model']['activation'])

        pbc = False
        if self.lattice is not None:
            pbc = True

        model = NewtonNet(resolution=settings['model']['resolution'],
                          n_features=settings['model']['n_features'],
                          activation=activation,
                          n_interactions=settings['model']['n_interactions'],
                          dropout=settings['training']['dropout'],
                          max_z=10,
                          cutoff=settings['data']['cutoff'],  ## data cutoff
                          cutoff_network=settings['model']['cutoff_network'],
                          normalizer=(0.0, 1.0),
                          normalize_atomic=settings['model']['normalize_atomic'],
                          requires_dr=settings['model']['requires_dr'],
                          device=self.device[0],
                          create_graph=True,
                          shared_interactions=settings['model']['shared_interactions'],
                          return_latent=settings['model']['return_latent'],
                          layer_norm=settings['model']['layer_norm'],
                          pbc=pbc
                          )

        print('Loading model: ',model_path)
        model.load_state_dict(torch.load(model_path, map_location=self.device[0])['model_state_dict'], strict=False)

        self.model = model
        self.model.to(self.device[0])
        self.model.eval()
        self.model.requires_dr = True



    def predict(self, data_dict, batch_size=50):
        """
        prediction for energies and forces for provided npz file
        Parameters
        ----------
        data_dict
        batch_size

        Returns
        -------

        """
        if self.lattice is not None:
            env = PeriodicEnvironment(cutoff=self.settings['data']['cutoff'])
        else:
            env = ExtensiveEnvironment()
        data_gen = data_loader(data=data_dict,
                                         env_provider=env,
                                         batch_size=batch_size,  # settings['training']['val_batch_size'],
                                         device=self.device[0],
                                        )
        e = []
        f = []
        data_preds=dict()
        steps = int(np.ceil(len(data_dict['E']) / batch_size))
        for val_step in range(steps):
            val_batch = next(data_gen)
            val_preds = self.model(val_batch)
            if self.device[0]=='cpu':
                e.append(val_preds['E'].detach().numpy())
                f.append(val_preds['F'].detach().numpy())
            else:
                e.append(val_preds['E'].detach().cpu().numpy())
                f.append(val_preds['F'].detach().cpu().numpy())
        data_preds['E'] = np.concatenate(e, axis=0)
        data_preds['F'] = np.concatenate(f, axis=0)


        return data_preds

    def metric_ae(self, preds, data, divider=None):
        """absolute error"""
        if data.ndim < preds.ndim:
            data = data[:, None]
        ae = np.abs(preds - data)
        if divider is not None:
            if divider.ndim < ae.ndim:
                divider = divider[:, None]
            ae /= divider
        return ae

    def masked_average(self, y, atom_mask):
        # handle rotation-wise loader batch size mismatch
        if atom_mask.shape[0] > y.shape[0]:
            # assert atom_mask.shape[1] == y.shape[1]
            atom_mask = atom_mask.reshape(y.shape[0], -1, y.shape[1])  # B, n_rot, A
            atom_mask = atom_mask.mean(axis=1)

        y = y[atom_mask!=0]

        return y

    def validation(self,generator,steps):
        """A pruned version of newtonnet.train.trainer Trainer.validation() function"""
        val_error_energy = []
        val_error_force = []
        energy_pred = []
        force_pred = []
        e = []
        f = []
        AM = []

        for val_step in range(steps):
            val_batch = next(generator)
            val_preds = self.model(val_batch)

            if val_preds['E'].ndim == 3:
                E = val_batch["E"].unsqueeze(1).repeat(1,val_batch["Z"].shape[1],1)
            else:
                E = val_batch["E"]
            val_error_energy.append(self.metric_ae(
                val_preds['E'].detach().cpu().numpy(), E.detach().cpu().numpy(),
                divider=val_batch['NA'].detach().cpu().numpy() if 'NA' in val_batch else None))
            energy_pred.append(val_preds['E'].detach().cpu().numpy())
            predicted_force = val_preds['F'].detach().cpu().numpy()
            target_force = val_batch["F"].detach().cpu().numpy()
            val_error_force.append(self.metric_ae(predicted_force, target_force))
            AM.append(val_batch["AM"].detach().cpu().numpy())

        outputs = dict()
        AM = standardize_batch(list(chain(*AM)))
        outputs['E_ae'] = np.concatenate(val_error_energy, axis=0)
        F_ae = standardize_batch(list(chain(*val_error_force)))
        outputs['F_ae_masked'] = self.masked_average(F_ae, AM)
        return outputs




    def evaluate_from_gens(self, train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizer):
        """
        prediction for energies and forces error for provided generator
        Parameters
        ----------
        data_dict
        batch_size

        Returns
        -------

        """


        outputs = self.validation(train_gen, tr_steps)
        train_mae_E = np.mean(outputs['E_ae'])
        train_mae_F = np.mean(outputs['F_ae_masked'])

        outputs = self.validation(val_gen, val_steps)
        val_mae_E = np.mean(outputs['E_ae'])
        val_mae_F = np.mean(outputs['F_ae_masked'])

        outputs = self.validation(test_gen, test_steps)
        test_mae_E = np.mean(outputs['E_ae'])
        test_mae_F = np.mean(outputs['F_ae_masked'])


        return [train_mae_E,train_mae_F,val_mae_E,val_mae_F,test_mae_E,test_mae_F]

    def evaluate(self,data_root=None,by_rxn = True,dialation=False):
        if data_root != None:
            self.settings['data']['root'] = data_root
        # data
        # for fast checking of code
        # self.settings['data']['trsize_perrxn_max'] = 10
        # self.settings['data']['test_size'] = 10
        # self.settings['data']['val_size'] = 10
        # self.settings['al']['added'] = []
        df = pd.DataFrame(columns=['train_E', 'train_F', 'val_E', 'val_F', 'test_E', 'test_F'])
        print("evaluating for rxns")
        if by_rxn:
            for rxn in self.settings['data']['reaction']:
                train_gen, val_gen, irc_gen, test_gen, \
                tr_steps, val_steps, irc_steps, test_steps, normalizer = parse_h2_reaction(self.settings,
                                                                                           self.device[0],
                                                                                           rxn=rxn)

                maes = self.evaluate_from_gens(train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps,
                                               normalizer)
                df.loc[f'rxn{rxn}'] = maes
        print("evaluating for dialations")

        if dialation:
            for pre in self.settings['data']['dialation_reaction']:
                npz = os.path.join(data_root, '%s_irc_dialation.npz' % pre)
                if os.path.exists(npz):
                    train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizer = parse_train_test_npz(
                        npz, self.settings, self.device[0])
                    maes = self.evaluate_from_gens(train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps,
                                                   normalizer)
                    df.loc[npz.split('/')[-1][:-4]] = maes

        print("evaluating for all HCombustion")

        train_gen, val_gen, irc_gen, test_gen, \
        tr_steps, val_steps, irc_steps, test_steps, normalizer = parse_h2_reaction(self.settings,self.device[0])
                                                               # self.settings['general']['device'])
        maes = self.evaluate_from_gens(train_gen, val_gen,  test_gen, tr_steps, val_steps, test_steps, normalizer)

        # df.loc[len(df.index)] = ['HCombustion', *maes]
        df.loc['HCombustion'] =  maes
        print("evaluating for added files")

        # read added npz files
        for npz in self.settings['al']['added']:
            # npz_path = os.path.join(dir_path, npz)
            if os.path.exists(npz):
                train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizer = parse_train_test_npz(npz,
                                                                                            self.settings,self.device[0])
                maes = self.evaluate_from_gens(train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps,
                                               normalizer)
                df.loc[npz.split('/')[-1][:-4]] = maes
        return df

    def evaluate_npz(self,npz):
        data = dict(np.load(npz, allow_pickle=True))
        n_data = len(data['N'])
        env = ExtensiveEnvironment()
        data_gen = extensive_train_loader(data=data,
                                         env_provider=env,
                                         batch_size=self.settings['training']['val_batch_size'],
                                         n_rotations=0,
                                         freeze_rotations=False,
                                         keep_original=True,
                                         device=self.device[0],
                                         shuffle=False,
                                         drop_last=False)
        steps = int(np.ceil(len(data['N']) / self.settings['training']['val_batch_size']))
        outputs = self.validation(data_gen, steps)
        mae_E = np.mean(outputs['E_ae'])
        mae_F = np.mean(outputs['F_ae_masked'])
        return [mae_E,mae_F, n_data]






def train_newtonnet(settings_path,force_cpu=False,resume = False):
    # settings
    # settings_path = 'config_h2.yml'
    settings = yaml.safe_load(open(settings_path, "r"))
    if force_cpu:
        settings['general']['device'] = 'cpu'

    # device
    if type(settings['general']['device']) is list:
        device = [torch.device(item) for item in settings['general']['device']]
    else:
        device = [torch.device(settings['general']['device'])]



    # data
    train_gen, val_gen, irc_gen, test_gen, \
    tr_steps, val_steps, irc_steps, test_steps, normalizer, ref_e = parse_h2_reaction_with_addition(settings, device[0])
    print('normalizer: ', normalizer)

    # model
    # activation function
    activation = get_activation_by_string(settings['model']['activation'])

    model = NewtonNet(resolution=settings['model']['resolution'],
                   n_features=settings['model']['n_features'],
                   activation=activation,
                   n_interactions=settings['model']['n_interactions'],
                   dropout=settings['training']['dropout'],
                   max_z=10,
                   cutoff=settings['data']['cutoff'],  ## data cutoff
                   cutoff_network=settings['model']['cutoff_network'],
                   normalizer=normalizer,
                   normalize_atomic=settings['model']['normalize_atomic'],
                   requires_dr=settings['model']['requires_dr'],
                   device=device[0],
                   create_graph=True,
                   shared_interactions=settings['model']['shared_interactions'],
                   return_latent=settings['model']['return_latent'],
                   layer_norm=settings['model']['layer_norm']
            )

    # load pre-trained model
    if settings['model']['pre_trained']:
        model_path = settings['model']['pre_trained']
        model.load_state_dict(torch.load(model_path)['model_state_dict'])

    # optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params,
                     lr=settings['training']['lr'],
                     weight_decay=settings['training']['weight_decay'])

    # loss
    w_energy = settings['model']['w_energy']
    w_force = settings['model']['w_force']
    w_f_mag = settings['model']['w_f_mag']
    w_f_dir = settings['model']['w_f_dir']


    def custom_loss(preds, batch_data, w_e=w_energy, w_f=w_force, w_fm=w_f_mag, w_fd=w_f_dir, boltzmann_weigts=False):

        # compute the mean squared error on the energies
        diff_energy = preds['E'] - batch_data["E"]
        assert diff_energy.shape[1] == 1
        if boltzmann_weigts:
            kt = 10000 * 1.98726 * 10**(-3)# kB = 1.380658*10-23 J K-1 * 1.4393x10+20 (kcal/mol)/J
            # per atom energy difference with e_ref
            e_diff_weighted = batch_data["E"]/torch.count_nonzero(batch_data["Z"], dim=1).reshape(-1,1)-ref_e
            # e_diff_weighted = torch.linspace(1,100,50).reshape(-1,1)
            weights = torch.exp(-e_diff_weighted/kt) #shape (batchsize,1)
            weights[weights >1] = 1

            # weights[np.where(batch_data['bw'] == 0)] = 1
            ### for testing and hyperparam tuning of temp ###
            # vis = torch.cat([weights,e_diff_weighted],axis=1)
            # torch.set_printoptions(precision=2,sci_mode=False)
            # if not torch.all(weights==1):
            #     print('boltzmann weights',vis)
            # sys.exit()
        else:
            weights = torch.ones(batch_data["E"].shape)
        err_sq_energy = torch.mean(diff_energy**2 * weights)
        err_sq = w_e * err_sq_energy

        # compute the mean squared error on the forces
        # print('force shape',preds['F'].shape)
        force_weights = weights.unsqueeze(2).repeat(1, preds['F'].shape[1], 3)

        diff_forces = preds['F'] - batch_data["F"]
        err_sq_forces = torch.mean(diff_forces**2 * force_weights)
        err_sq = err_sq + w_f * err_sq_forces



        # compute the mean square error on the force magnitudes
        if w_fm > 0:
            diff_forces = torch.norm(preds['F'], p=2, dim=-1) - torch.norm(batch_data["F"], p=2, dim=-1)
            err_sq_mag_forces = torch.mean(diff_forces ** 2 * weights)
            err_sq = err_sq + w_fm * err_sq_mag_forces

        if w_fd > 0:
            cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            direction_diff = 1 - cos(preds['hs'][-1][1], batch_data["F"])
            # direction_diff = direction_diff * torch.norm(batch_data["F"], p=2, dim=-1)
            direction_loss = torch.mean(direction_diff * weights.repeat(1,preds['F'].shape[1]))
            err_sq = err_sq + w_fd * direction_loss

        if settings['checkpoint']['verbose']:
            print('\n',
                  ' '*8, 'energy loss: ', err_sq_energy.detach().cpu().numpy(), '\n')
            print(' '*8, 'force loss: ', err_sq_forces.detach().cpu().numpy(), '\n')

            if w_fm>0:
                print(' '*8, 'force mag loss: ', err_sq_mag_forces, '\n')

            if w_fd>0:
                print(' '*8, 'direction loss: ', direction_loss.detach().cpu().numpy())

        return err_sq

    # training
    trainer = Trainer(model=model,
                      loss_fn=custom_loss,
                      optimizer=optimizer,
                      requires_dr=settings['model']['requires_dr'],
                      device=device,
                      yml_path=settings['general']['me'],
                      output_path=settings['general']['output'],
                      script_name=settings['general']['driver'],
                      lr_scheduler=settings['training']['lr_scheduler'],
                      energy_loss_w= w_energy,
                      force_loss_w=w_force,
                      loss_wf_decay=settings['model']['wf_decay'],
                      checkpoint_log=settings['checkpoint']['log'],
                      checkpoint_val=settings['checkpoint']['val'],
                      checkpoint_test=settings['checkpoint']['test'],
                      checkpoint_model=settings['checkpoint']['model'],
                      verbose=settings['checkpoint']['verbose'],
                      hooks=settings['hooks'],
                      resume = resume,)


    trainer.print_layers()

    # tr_steps=1; val_steps=0; irc_steps=0; test_steps=0

    trainer.train(train_generator=train_gen,
                  epochs=settings['training']['epochs'],
                  steps=tr_steps,
                  val_generator=val_gen,
                  val_steps=val_steps,
                  irc_generator=irc_gen,
                  irc_steps=irc_steps,
                  test_generator=test_gen,
                  test_steps=test_steps,
                  clip_grad=1.5,
                  boltzmann_weigts=settings['training']['boltzmann_weights'])

    print('done!')


def h2_reaction_addition(settings, all_sets):
    """
    append new data to all_sets
    Parameters
    ----------
    settings: dict
        dict of yaml file

    all_sets: dict

    Returns
    -------

    """

    dir_path = settings['data']['root']

    # read added npz files
    for npz in settings['al']['added']:
        # npz_path = os.path.join(dir_path, npz)
        if os.path.exists(npz):
            data = dict(np.load(npz,allow_pickle=True))
            n_data =  data['R'].shape[0]
            #
            # data_select = sample_without_replacement(new_data['R'].shape[0],
            #                                        n_data,
            #                                        random_state=settings['data']['random_states'])
            if settings['data']['cgem']:
                assert data['E'].shape == data['CE'].shape
                assert data['F'].shape == data['CF'].shape
                data['E'] = data['E'] - data['CE']
                data['F'] = data['F'] - data['CF']

            if settings['training']['boltzmann_weights']:
                data['bw']= np.ones(data['E'].shape)
            val_size = max(1,int(n_data * settings['al']['added_split'][1]))
            test_size = max(1,int(n_data * settings['al']['added_split'][2]))
            train_size = int(n_data - val_size - test_size)
            dtrain, dval, dtest = split(data,
                                        train_size=train_size,
                                        test_size=test_size,
                                        val_size=val_size,
                                        random_states=settings['data']['random_states'],
                                        stratify=None)
            all_sets['train'].append(dtrain)
            all_sets['val'].append(dval)
            all_sets['test'].append(dtest)
        else:
            warnings.warn(f'added npz {npz} is missing.')

    return all_sets

def h2_reaction_single(reaction_number, settings, all_sets, boltzmann_weigts=False,balance_rxn_number=0):
    """

    Parameters
    ----------
    reaction_number: int

    settings: dict
        dict of yaml file

    all_sets: dict

    balance_rxn_number: int
        a precalculated number to balance out reactions by adding more nm points for rxns without metad

    Returns
    -------

    """

    dir_path = settings['data']['root']

    # file name prefix
    if reaction_number < 10:
        pre = '0%i'%reaction_number
    elif reaction_number >= 10:
        pre = '%i'%reaction_number
    # elif reaction_number == 6:
    #     pre = ['0%ia_irc.npz' % reaction_number, '0%ib_irc.npz' % reaction_number]
    # elif reaction_number == 12:
    #     pre = ['%ia_irc.npz' % reaction_number, '%ib_irc.npz' % reaction_number]

    # read npz files
    aimd = nm = irc = None
    aimd_path = os.path.join(dir_path, '%s_aimd.npz'%pre)
    if os.path.exists(aimd_path):
        aimd = dict(np.load(aimd_path))
    nm_path = os.path.join(dir_path, '%s_nm.npz'%pre)
    if os.path.exists(nm_path):
        nm = dict(np.load(nm_path))
    irc_path = os.path.join(dir_path, '%s_irc.npz'%pre)
    if os.path.exists(irc_path):
        irc = dict(np.load(irc_path))

    # merge aimd and normal mode data
    if settings['data']['normal_mode'] and nm is not None:
        data = dict()
        n_nm = min(settings['data']['size_nmode_max'], nm['R'].shape[0])
        nm_select = sample_without_replacement(nm['R'].shape[0],
                                               n_nm,
                                               random_state=settings['data']['random_states'])
        # add additional data from normal modes to balance number of data points for each reaction
        if settings['al']['balance_rxns'] and (reaction_number not in settings['al']['metad_rxns']):
            if n_nm < nm['R'].shape[0]:
                n_nm_added = min(balance_rxn_number, nm['R'].shape[0] - n_nm)
                added_select_idx = sample_without_replacement(nm['R'].shape[0]-len(nm_select),
                                               n_nm_added,
                                               random_state=settings['data']['random_states'])
                added_select = np.array([i for i in range(nm['R'].shape[0]) if i not in nm_select])[added_select_idx]
                nm_select = np.concatenate([np.array(nm_select),np.array(added_select)])
                n_nm = n_nm + n_nm_added

        if aimd is not None:
            for k in aimd.keys():
                data[k] = np.concatenate([aimd[k], nm[k][nm_select]], axis=0)

            assert data['R'].shape[0] == (aimd['R'].shape[0]+n_nm)
        else:
            data = None
            warnings.warn('AIMD data for reaction# %i are missing.'%reaction_number)

    elif aimd is not None:
        data = aimd

    else:
        data = None
        warnings.warn('both AIMD and normal mode data for reaction# %i are missing.'%reaction_number)

    if settings['data']['cgem']:
        assert data['E'].shape == data['CE'].shape
        assert data['F'].shape == data['CF'].shape
        data['E'] = data['E'] - data['CE']
        data['F'] = data['F'] - data['CF']
        irc['E'] = irc['E'] - irc['CE']
        irc['F'] = irc['F'] - irc['CF']

    train_size = settings['data']['trsize_perrxn_max']
    if train_size == -1:
        train_size = None  # to select all remaining data in each reaction

    if data is not None:
        if boltzmann_weigts:
            data['bw'] = np.zeros(data['E'].shape)
        dtrain, dval, dtest = split(data,
                                    train_size=train_size,
                                    test_size=settings['data']['test_size'],
                                    val_size=settings['data']['val_size'],
                                    random_states=settings['data']['random_states'],
                                    stratify=None)
    else:
        dtrain, dval, dtest = None, None, None

    # compile data sets
    all_sets['train'].append(dtrain)
    all_sets['val'].append(dval)
    all_sets['test'].append(dtest)
    all_sets['irc'].append(irc)

    return all_sets

def h2_reaction_dialation(pre, settings, all_sets):
    """
    append new data to all_sets
    Parameters
    ----------
    pre: str (reaction number)

    settings: dict
        dict of yaml file

    all_sets: dict

    Returns
    -------

    """

    dir_path = settings['data']['root']

    # read npz files
    ircd_npz = os.path.join(dir_path, '%s_irc_dialation.npz' % pre)

    if os.path.exists(ircd_npz):
        data = dict(np.load(ircd_npz,allow_pickle=True))
        n_data =  min(data['R'].shape[0],np.floor(settings['data']['dialation_trsize_perrxn']/settings['al']['added_split'][0]))

        #
        # data_select = sample_without_replacement(new_data['R'].shape[0],
        #                                        n_data,
        #                                        random_state=settings['data']['random_states'])
        if settings['data']['cgem']:
            assert data['E'].shape == data['CE'].shape
            assert data['F'].shape == data['CF'].shape
            data['E'] = data['E'] - data['CE']
            data['F'] = data['F'] - data['CF']

        if settings['training']['boltzmann_weights']:
            data['bw']= np.ones(data['E'].shape)
        test_size = max(1, int(n_data * settings['al']['added_split'][2]))
        val_size = max(1, int(n_data * settings['al']['added_split'][1]))
        train_size = int(n_data - test_size - val_size)
        dtrain, dval, dtest = split(data,
                                    train_size=train_size,
                                    test_size=test_size,
                                    val_size=val_size,
                                    random_states=settings['data']['random_states'],
                                    stratify=None)
        all_sets['train'].append(dtrain)
        all_sets['val'].append(dval)
        all_sets['test'].append(dtest)
    else:
        warnings.warn(f'added npz {ircd_npz} is missing.')

    return all_sets

def parse_h2_reaction_with_addition(settings, device):
    """
    Hydrogen Combustion Parser

    Parameters
    ----------
    settings: instance of yaml file
    device: torch devices

    Returns
    -------
    generator: train, val, irc, test generators, respectively
    int: n_steps for train, val, irc, test, respectively
    tuple: tuple of mean and standard deviation of energies in the training data

    """

    # list of reaction number(s)
    reaction_number = settings['data']['reaction']

    if isinstance(reaction_number, int):
        reaction_number = [reaction_number]

    # compile dictionary of train, test, val, and irc data
    all_sets = defaultdict(list)
    balance_number = 0
    if settings['al']['balance_rxns']:
        count =0
        for data in settings['al']['added']:
            d = dict(np.load(data))
            count+= d['R'].shape[0]
        balance_number = int(count / len(settings['al']['metad_rxns']))
        print(f"Balance rxns by adding {balance_number} NM points for non-metad rxns")
    for rxn_n in reaction_number:
        all_sets = h2_reaction_single(rxn_n, settings, all_sets,
                                      boltzmann_weigts=settings['training']['boltzmann_weights'],balance_rxn_number=balance_number)

    if ('dialation' in settings['data']) and settings['data']['dialation']:
        d_reaction_number = settings['data']['dialation_reaction']
        if isinstance(d_reaction_number, str):
            d_reaction_number = [d_reaction_number]
        for rxn_n in d_reaction_number:
            h2_reaction_dialation(rxn_n, settings, all_sets,)

    all_sets = h2_reaction_addition(settings,all_sets)

    # for reference of boltzmann weight, record the highest energy point per atom
    # Use highest in irc + 10
    ref_es = []
    for irc_dict in all_sets['irc']:
        if irc_dict["RXN"][0][0] == '08':
            continue
        ref_e = (np.max(irc_dict['E'])) / irc_dict['N'][0][0]
        print(f'Rxn:{irc_dict["RXN"][0][0]} reference E per atom:{ref_e}')
        ref_es.append(ref_e)
    ref_e = np.max(ref_es)
    print('final ref_e',ref_e)

    dtrain = concat_listofdicts(all_sets['train'], axis=0)
    dval = concat_listofdicts(all_sets['val'], axis=0)
    dtest = concat_listofdicts(all_sets['test'], axis=0)
    irc = concat_listofdicts(all_sets['irc'], axis=0)

    # final down-sampling of training data
    n_train = settings['data']['train_size']
    if n_train == -1:
        n_train = dtrain['R'].shape[0]

    n_train = min(n_train, dtrain['R'].shape[0])
    n_select = sample_without_replacement(dtrain['R'].shape[0],
                                           n_train,
                                           random_state=settings['data']['random_states'])
    for k in dtrain.keys():
        dtrain[k] = dtrain[k][n_select]

    normalizer = (dtrain['E'].mean(), dtrain['E'].std())

    n_tr_data = dtrain['R'].shape[0]
    n_val_data = dval['R'].shape[0]
    n_irc_data = irc['R'].shape[0]
    n_test_data = dtest['R'].shape[0]
    print("# data (train,val,test,irc): %i, %i, %i, %i"%(n_tr_data,n_val_data,n_test_data,n_irc_data))

    tr_batch_size = settings['training']['tr_batch_size']
    val_batch_size = settings['training']['val_batch_size']
    tr_rotations = settings['training']['tr_rotations']
    val_rotations = settings['training']['val_rotations']

    # freeze rotatios
    # Todo: it seems that we don't need separated tr and val anymore
    # Todo: consider keep_original scenario in the code
    # if settings['training']['tr_frz_rot']:
    #     if settings['training']['saved_angle_path']:
    #         tr_fra_rot = list(np.load(settings['training']['saved_angle_path']))[:tr_rotations+1]
    #     tr_frz_rot = (np.random.uniform(-np.pi, np.pi, size=3)
    #                   for _ in range(tr_rotations+1))
    #     val_frz_rot = tr_frz_rot
    # else:
    #     tr_frz_rot = settings['training']['tr_frz_rot']
    #     val_frz_rot = settings['training']['val_frz_rot']

    # generators
    project = settings['general']['driver']
    if project not in ['voxel_cart_rotwise.py']:
        # steps
        tr_steps = int(np.ceil(n_tr_data / tr_batch_size)) * (tr_rotations + 1)
        val_steps = int(np.ceil(n_val_data / val_batch_size)) * (val_rotations + 1)
        irc_steps = int(np.ceil(n_irc_data / val_batch_size)) * (val_rotations + 1)
        test_steps= int(np.ceil(n_test_data / val_batch_size)) * (val_rotations + 1)

        env = ExtensiveEnvironment()

        train_gen = extensive_train_loader(data=dtrain,
                                           env_provider=env,
                                           batch_size=tr_batch_size,
                                           n_rotations=tr_rotations,
                                           freeze_rotations=settings['training']['tr_frz_rot'],
                                           keep_original=settings['training']['tr_keep_original'],
                                           device=device,
                                           shuffle=settings['training']['shuffle'],
                                           drop_last=settings['training']['drop_last'])

        val_gen = extensive_train_loader(data=dval,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=settings['training']['shuffle'],
                                         drop_last=settings['training']['drop_last'])

        irc_gen = extensive_train_loader(data=irc,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        test_gen = extensive_train_loader(data=dtest,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        return train_gen, val_gen, irc_gen, test_gen, tr_steps, val_steps, irc_steps, test_steps, normalizer, ref_e

    else:

        tr_steps = int(np.ceil(n_tr_data / tr_batch_size))
        val_steps = int(np.ceil(n_val_data / val_batch_size))
        irc_steps = int(np.ceil(n_irc_data / val_batch_size))
        test_steps = int(np.ceil(n_test_data / val_batch_size))

        env = ExtensiveEnvironment()

        train_gen = extensive_loader_rotwise(data=dtrain,
                                           env_provider=env,
                                           batch_size=tr_batch_size,
                                           n_rotations=tr_rotations,
                                           freeze_rotations=settings['training']['tr_frz_rot'],
                                           keep_original=settings['training']['tr_keep_original'],
                                           device=device,
                                           shuffle=settings['training']['shuffle'],
                                           drop_last=settings['training']['drop_last'])

        val_gen = extensive_loader_rotwise(data=dval,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=settings['training']['shuffle'],
                                         drop_last=settings['training']['drop_last'])

        irc_gen = extensive_loader_rotwise(data=irc,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        test_gen = extensive_loader_rotwise(data=dtest,
                                         env_provider=env,
                                         batch_size=val_batch_size,
                                         n_rotations=val_rotations,
                                         freeze_rotations=settings['training']['val_frz_rot'],
                                         keep_original=settings['training']['val_keep_original'],
                                         device=device,
                                         shuffle=False,
                                         drop_last=False)

        return train_gen, val_gen, irc_gen, test_gen, tr_steps, val_steps, \
               irc_steps, test_steps, normalizer, ref_e

def parse_train_test_npz(npz, settings, device):
    """
    implementation based on train and validation size.
    we don't need the test_size in this implementaion.

    Parameters
    ----------
    settings: instance of yaml file
    device: torch.device
        list of torch devices

    Returns
    -------
    generator: train, val, test generators, respectively
    int: n_steps for train, val, test, respectively
    tuple: tuple of mean and standard deviation of energies in the training data

    """
    data = dict(np.load(npz, allow_pickle=True))
    n_data = data['R'].shape[0]

    if settings['data']['cgem']:
        assert data['E'].shape == data['CE'].shape
        assert data['F'].shape == data['CF'].shape
        data['E'] = data['E'] - data['CE']
        data['F'] = data['F'] - data['CF']

    test_size = max(1,int(n_data * settings['al']['added_split'][2]))
    val_size = max(1,int(n_data * settings['al']['added_split'][1]))
    train_size = int(n_data - test_size - val_size)
    dtrain, dval, dtest = split(data,
                                train_size=train_size,
                                test_size=test_size,
                                val_size=val_size,
                                random_states=settings['data']['random_states'],
                                stratify=None)


    # extract data stats
    normalizer = (dtrain['E'].mean(), dtrain['E'].std())

    n_tr_data = dtrain['R'].shape[0]
    n_val_data = dval['R'].shape[0]
    n_test_data = dtest['R'].shape[0]
    print("data size: (train,val,test): %i, %i, %i"%(n_tr_data,n_val_data,n_test_data))

    tr_batch_size = settings['training']['tr_batch_size']
    val_batch_size = settings['training']['val_batch_size']
    tr_rotations = settings['training']['tr_rotations']
    val_rotations = settings['training']['val_rotations']

    # generators
    me = settings['general']['driver']

    # steps
    tr_steps = int(np.ceil(n_tr_data / tr_batch_size)) * (tr_rotations + 1)
    val_steps = int(np.ceil(n_val_data / val_batch_size)) * (val_rotations + 1)
    test_steps= int(np.ceil(n_test_data / val_batch_size)) * (val_rotations + 1)

    env = ExtensiveEnvironment()

    train_gen = extensive_train_loader(data=dtrain,
                                       env_provider=env,
                                       batch_size=tr_batch_size,
                                       n_rotations=tr_rotations,
                                       freeze_rotations=settings['training']['tr_frz_rot'],
                                       keep_original=settings['training']['tr_keep_original'],
                                       device=device,
                                       shuffle=settings['training']['shuffle'],
                                       drop_last=settings['training']['drop_last'])

    val_gen = extensive_train_loader(data=dval,
                                     env_provider=env,
                                     batch_size=val_batch_size,
                                     n_rotations=val_rotations,
                                     freeze_rotations=settings['training']['val_frz_rot'],
                                     keep_original=settings['training']['val_keep_original'],
                                     device=device,
                                     shuffle=settings['training']['shuffle'],
                                     drop_last=False)

    test_gen = extensive_train_loader(data=dtest,
                                     env_provider=env,
                                     batch_size=val_batch_size,
                                     n_rotations=val_rotations,
                                     freeze_rotations=settings['training']['val_frz_rot'],
                                     keep_original=settings['training']['val_keep_original'],
                                     device=device,
                                     shuffle=False,
                                     drop_last=False)

    return train_gen, val_gen, test_gen, tr_steps, val_steps, test_steps, normalizer

if __name__ =='__main__':
    model_idx = sys.argv[1].split('.')[-2][-1] # sys.argv[1] look like models_active_learning_1kperrxn_1/config_h2_0.yml
    label = sys.argv[1].split('/')[-2].replace('models_active_learning_','')
    train_newtonnet(sys.argv[1],resume=eval(sys.argv[2]))
