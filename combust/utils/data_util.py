import numpy as np
from ase import Atoms
from ase.io.trajectory import TrajectoryWriter

from combust.utils.utility import check_data_consistency, combine_rxn_arrays, write_data_npz
from combust.utils.rxn_data import rxn_dict

def get_spin_for_rxn(rxn_num):
    if not isinstance(rxn_num, str) or len(rxn_num) < 2:
        raise ValueError("Argument rxn_num is expected to be a string of length >= 2!")

    rxn = 'rxn' + rxn_num
    spin = rxn_dict[rxn]['folder'].split('.')[-1]
    spin_to_int = {'s':1, 'd':2, 't':3, 'q':4}
    return  spin_to_int[spin]

def npz_to_traj(npz,filename):
    data = dict(np.load(npz, allow_pickle=True))
    coords = data['R']
    atomic_num = data['Z'][0]
    natoms = data['N'][0][0]
    writer = TrajectoryWriter(filename)
    for i,coord in enumerate(coords):
        atoms = Atoms(numbers=atomic_num[:natoms],positions=coord[:natoms])
        writer.write(atoms)
    writer.close()



def merge_npz(list_of_npz,outname):
    total = []
    for npz in list_of_npz:
        data = dict(np.load(npz, allow_pickle=True))
        data = npz_clean_up(data)
        total.append(data)
    data = combine_rxn_arrays(total, n_max=6)
    check_data_consistency(data)

    # write data dictionary
    write_data_npz(data, outname)

def npz_clean_up(data):
    """ due to fragmo and sad, an npz for the same rxn may contain data['Z'] of different order.
        i.e. For rxn13 it can be [8, 8, 8, 1, 1, 0] or [8, 8, 1, 1, 8, 0]
        This function unifies that and make data consistent
    """
    nums = np.unique(data['Z'], axis=0)
    if len(nums) == 1:
        return data
    elif len(nums) > 1:
        while len(nums) >1:
            data = swap_data_based_on_nums(data,nums)
            nums = np.unique(data['Z'], axis=0)
        return data
    else:
        raise ValueError(f"npz has following unique Z valuess: {nums}")

def swap_data_based_on_nums(data,nums):
    # only swap nums[1] to num[0]
    mapping = {}  # num[1]idx : new num[1]idx such that num[1][new_idx] == num[0]
    num1_idx = list(range(len(nums[0])))
    for i, n in enumerate(nums[0]):
        idx_of_num1_idx = 0
        while not n == nums[1][num1_idx[idx_of_num1_idx]]:
            idx_of_num1_idx += 1
        mapping[i] = num1_idx[idx_of_num1_idx]
        num1_idx.pop(idx_of_num1_idx)
    # print(f"Mapping of {nums[1]} to become equal to {nums[0]}")
    # print(mapping)
    new_data = {}
    for k, v in data.items():
        new_data[k] = v.copy()
        if k not in ['N', 'E', 'RXN']:
            # 'F','Z','R'
            for i, mol_info in enumerate(v):  # v shape(N,natom,(3))
                if np.all(data['Z'][i] == nums[1]):  # need to modify
                    for key, val in mapping.items():
                        # if k == 'Z':
                        #     print(f"{new_data[k][i][key]} : {mol_info[val]}")
                        new_data[k][i][key] = mol_info[val]

    return new_data

