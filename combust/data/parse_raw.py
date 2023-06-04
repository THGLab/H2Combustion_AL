import os
import numpy as np
import warnings
from collections import defaultdict

from sklearn.utils.random import sample_without_replacement
from sklearn.model_selection import train_test_split

from combust.data import ExtensiveEnvironment, PeriodicEnvironment
from combust.data import extensive_train_loader, extensive_loader_rotwise

from ase.io import iread
import math
import pickle

def concat_listofdicts(listofdicts, axis=0):
    """

    Parameters
    ----------
    listofdicts: list
        values must be 2d arrays
    axis: int

    Returns
    -------
    dict

    """
    data = dict()
    for k in listofdicts[0].keys():
        data[k] = np.concatenate([d[k] for d in listofdicts], axis=axis)

    return data


def split(data, train_size, test_size, val_size, random_states=90, stratify=None):
    """

    Parameters
    ----------
    data: dict
    train_size: int
    test_size
    val_size
    random_states
    stratify: None or labels

    Returns
    -------
    dict: train data
    dict: val data
    dict: test data

    """

    tr_ind, val_ind = train_test_split(list(range(data['R'].shape[0])),
                                      test_size=val_size,
                                      random_state=random_states,
                                      stratify=stratify)

    if stratify is not None:
        stratify_new = stratify[tr_ind]
    else:
        stratify_new = None

    tr_ind, te_ind = train_test_split(tr_ind,
                                       test_size=test_size,
                                       train_size=train_size,
                                       random_state=random_states,
                                       stratify=stratify_new)

    train = dict()
    val = dict()
    test = dict()
    for key in data:
        train[key] = data[key][tr_ind]
        val[key] = data[key][val_ind]
        test[key] = data[key][te_ind]

    if stratify is not None:
        train['L'] = stratify[tr_ind]
        val['L'] = stratify[val_ind]
        test['L'] = stratify[te_ind]

    return train, val, test


def h2_reaction(reaction_number, settings, all_sets, boltzmann_weigts=False):
    """

    Parameters
    ----------
    reaction_number: int

    settings: dict
        dict of yaml file

    all_sets: dict

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


def parse_h2_reaction(settings, device,rxn='all'):
    """

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
    if rxn=='all':
        reaction_number = settings['data']['reaction']
    else:
        reaction_number = rxn

    if isinstance(reaction_number, int):
        reaction_number = [reaction_number]

    # compile dictionary of train, test, val, and irc data
    all_sets = defaultdict(list)
    for rxn_n in reaction_number:
        all_sets = h2_reaction(rxn_n, settings, all_sets)

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

        return train_gen, val_gen, irc_gen, test_gen, tr_steps, val_steps, irc_steps, test_steps, normalizer

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

        return train_gen, val_gen, irc_gen, test_gen, tr_steps, val_steps, irc_steps, test_steps, normalizer



def parse_train_test(settings, device, unit='kcal'):
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
    # meta data
    train_path = settings['data']['train_path']
    test_path = settings['data']['test_path']   # can be False

    train_size = settings['data']['train_size']
    val_size = settings['data']['val_size']


    # read data
    data = np.load(train_path)
    test = None
    if test_path:
        test = dict(np.load(test_path))

    # take care of inconsistencies
    dtrain = dict()
    dtest = dict()

    for key in list(data.keys()):
        # copy Z embarrassingly Todo: make it data efficient by flexible environment module
        if key == 'z':
            dtrain['Z'] = np.tile(data['z'], (data['E'].shape[0], 1))
            if test is not None:
                dtest['Z'] = np.tile(test['z'], (test['E'].shape[0], 1))

        elif key == 'E':
            if data['E'].ndim == 1:
                dtrain['E'] = data['E'].reshape(-1,1)
            else:
                dtrain[key] = data[key]

            if test is not None:
                if test['E'].ndim == 1:
                    dtest['E'] = test['E'].reshape(-1, 1)
                else:
                    dtest[key] = test[key]

        elif key in ['R','F','Z']:
            dtrain[key] = data[key]
            if test is not None:
                dtest[key] = test[key]

    # convert unit
    if unit == 'ev':
        dtrain['E'] = dtrain['E'] * 23.061
        dtrain['F'] = dtrain['F'] * 23.061

    # split the data
    dtrain, dval, dtest_leftover = split(dtrain,
                                        train_size=train_size,
                                        test_size=None,
                                        val_size=val_size,
                                        random_states=settings['data']['random_states'])
    if test is None:
        test_size = settings['data'].get('test_size', -1)
        if test_size == -1:
            dtest = dtest_leftover
        else:
            test_size = min(test_size, dtest_leftover['R'].shape[0])
            n_select = sample_without_replacement(dtest_leftover['R'].shape[0],
                                                  test_size,
                                                  random_state=settings['data']['random_states'])
            for k in dtest_leftover.keys():
                dtest[k] = dtest_leftover[k][n_select]


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
                                     drop_last=settings['training']['drop_last'])

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


