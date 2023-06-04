import numpy as np


from ase.units import *
from ase.calculators.calculator import Calculator


import torch
import yaml

from newtonnet.layers import get_activation_by_string
from newtonnet.models import NewtonNet

from combust.data import ExtensiveEnvironment,PeriodicEnvironment
from combust.data import batch_dataset_converter




##-------------------------------------
##     ML model ASE interface
##--------------------------------------
class MLAseCalculator(Calculator):
    implemented_properties = ['energy', 'forces'] #, 'stress'
    # default_parameters = {'xc': 'ani'}
    # nolabel = True

    ### Constructor ###
    def __init__(self, model_path, settings_path, model_type ='NewtonNet', lattice = None, **kwargs):
        """
        Constructor for MLAseCalculator

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
        Calculator.__init__(self, **kwargs)
        self.settings = yaml.safe_load(open(settings_path, "r"))
        self.dyn = None
        if not lattice is None:
            self.lattice = np.array(lattice).reshape(9,)
        else:
            self.lattice = lattice

        # device
        if type(self.settings['general']['device']) is list:
            self.device = [torch.device(item) for item in self.settings['general']['device']]
        else:
            self.device = [torch.device(self.settings['general']['device'])]

        torch.set_default_tensor_type(torch.DoubleTensor)

        self._load_model(model_path,model_type)

    def calculate(self, atoms=None, properties=['energy','forces'],system_changes=None):
        super().calculate(atoms,properties,system_changes)
        data = data_formatter(atoms, self.lattice)
        pred = self.predict(data)
        energy = pred['E'][0][0].data.cpu().numpy()
        energy = float(energy) * kcal / mol
        forces = pred['F'][0].data.cpu().numpy()
        forces =  forces * kcal / mol / Ang
        # atomic energies (au -> eV)
        # H = -0.5004966690 * 27.211
        # O = -75.0637742413 * 27.211
        # n_H = np.count_nonzero(np.array(atoms.get_atomic_numbers()) == 1)
        # n_O = np.count_nonzero(np.array(atoms.get_atomic_numbers()) == 8)
        # atomic_energy = n_H*H +n_O*O
        # energy+=atomic_energy
        if 'energy' in properties:
            self.results['energy'] = energy
        if 'forces' in properties:
            self.results['forces'] = forces



    def _load_model(self,model_path,model_type= 'NewtonNet'):
        #load NewtonNet model
        # settings
        settings = self.settings

        # model
        # activation function
        activation = get_activation_by_string(settings['model']['activation'])

        pbc = False
        if self.lattice is not None:
            pbc = True

        if model_type == 'NewtonNet':
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

        else:
            raise ValueError("Unsupported model type. Choose from: ['NewtonNet']")



        model.load_state_dict(torch.load(model_path, map_location=self.device[0])['model_state_dict'], )
        # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'],)



        self.model = model
        self.model.to(self.device[0])
        self.model.eval()
        self.model.requires_dr = True





    def predict(self,data_dict):
        if self.lattice is not None:
            env = PeriodicEnvironment(cutoff=self.settings['data']['cutoff'])
        else:
            env = ExtensiveEnvironment()
        data_gen = data_loader(data=data_dict,
                                          env_provider=env,
                                          batch_size=1,  # settings['training']['val_batch_size'],
                                          device=self.device[0],)
        val_batch = next(data_gen)
        data_preds = self.model(val_batch)

        return data_preds

    def set_dyn(self, dyn):
        self.dyn = dyn



def data_formatter(atoms,lattice=None):
    """
    convert ase.Atoms to input format of the model

    Parameters
    ----------
    atoms: ase.Atoms

    Returns
    -------
    data: dict
        dictionary of arrays with following keys:
            - 'R':positions
            - 'Z':atomic_numbers
            - 'E':energy
            - 'F':forces
    """
    data  = {
        'R':np.array(atoms.get_positions(), dtype=np.double)[np.newaxis, ...], #shape(ndata,natoms,3)
        'Z': np.array(atoms.get_atomic_numbers())[np.newaxis, ...], #shape(ndata,natoms)
        'E': np.zeros((1,1)), #shape(ndata,1)
        'F': np.zeros((1,len(atoms.get_atomic_numbers()), 3)),#shape(ndata,natoms,3)
    }
    if lattice is not None:
        assert len(lattice) == 9, 'lattice for pbc should be an array of size 9'
        data['lattice'] = np.array(lattice,dtype='float')
    return data


def data_loader(data,
               env_provider=None,
               batch_size=32,
               device=None,
                        ):
    r"""
    The main function to load and iterate data based on the extensive environment provider.

    Parameters
    ----------
    data: dict
        dictionary of arrays with following keys:
            - 'R':positions
            - 'Z':atomic_numbers
            - 'E':energy
            - 'F':forces
            optional:
            - 'lattice': lattice vector for pbc shape(9,)

    env_provider: ShellProvider
        the instance of combust.data.ExtensiveEnvironment

    batch_size: int, optional (default: 32)
        The size of output tensors

    n_rotations: int, optional (default: 0)
        Number of times to rotate voxel boxes for data augmentation.
        If zero, the original orientation will be used.


    device: torch.device
        either cpu or gpu (cuda) device.

    Yields
    -------
    BatchDataset: instance of BatchDataset with the all batch data

    """
    n_data = data['R'].shape[0]  # D
    n_atoms = data['R'].shape[1]  # A


    # get neighbors
    if env_provider is not None:
        if 'lattice' in data.keys() and data['lattice'] is not None:
            # periodic env
            neighbors, neighbor_mask, atom_mask, _, _ = env_provider.get_environment(data['R'], data['Z'], data['lattice'])
        else:
            # extensive env
            neighbors, neighbor_mask, atom_mask,_,_ = env_provider.get_environment(data['R'], data['Z'])

    # iterate over data snapshots
    seen_all_data = 0
    while True:

        # split by batch size and yield
        data_atom_indices = list(range(n_data))


        split = 0
        while (split) * batch_size <= n_data:
            # Output a batch
            data_batch_idx = data_atom_indices[split *
                                               batch_size:(split + 1) *
                                               batch_size]

            if env_provider is None:
                N = None
                NM = None
                Z = data['Z'][data_batch_idx]
                AM = np.zeros_like(Z)
                AM[Z != 0] = 1
            else:
                N = neighbors[data_batch_idx]
                NM = neighbor_mask[data_batch_idx]
                AM = atom_mask[data_batch_idx]


            batch_dataset = {
                'R': data['R'][data_batch_idx],   # B,A,3
                'Z': data['Z'][data_batch_idx], # B,A
                'E': data['E'][data_batch_idx], # B,1
                'F': data['F'][data_batch_idx],    # B,A,3
                'N': N,     # B,A,A-1
                'NM': NM,   # B,A,A-1
                'AM': AM,   # B,A
            }
            if 'lattice' in data:
                batch_dataset['lattice'] = data['lattice']
            # batch_dataset = BatchDataset(batch_dataset, device=device)
            batch_dataset = batch_dataset_converter(batch_dataset, device)
            yield batch_dataset
            split += 1

        seen_all_data += 1


##-------------------------------------
##     ASE interface for Plumed calculator
##--------------------------------------
class PlumedCalculator(Calculator):
    implemented_properties = ['energy', 'forces']  # , 'stress'
    def __init__(self, ase_plumed, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.ase_plumed = ase_plumed
        self.counter = 0
        self.prev_force =None
        self.prev_energy = None

    def calculate(self, atoms=None, properties=['forces'],system_changes=None):
        super().calculate(atoms,properties,system_changes)
        forces = np.zeros((atoms.get_positions()).shape)
        energy = 0

        model_force = np.copy(forces)
        self.counter += 1
        # every step() call will call get_forces 2 times, only do plumed once(2nd) to make metadynamics work correctly
        # there is one call to get_forces when initialize
        # print(self.counter)
        # plumed_forces, plumed_energy = self.ase_plumed.external_forces(self.counter , new_forces=forces,
        #
        #                                                                delta_forces=True)
        if self.counter % 2 == 1:
            plumed_forces,plumed_energy = self.ase_plumed.external_forces((self.counter + 1) // 2 - 1, new_forces=forces,
                                                            new_energy=energy,delta_forces=True)
            self.prev_force = plumed_forces
            self.prev_energy = plumed_energy
            # print('force diff', np.sum(plumed_forces - model_force))
        else:
            plumed_forces = self.prev_force
            plumed_energy = self.prev_energy
            # print(self.counter)
        # if self.counter % 500 == 0:
        #     print('force diff', np.linalg.norm(plumed_forces - model_force))


        # delta energy and forces
        if 'energy' in properties:
            self.results['energy'] = plumed_energy
        if 'forces' in properties:
            self.results['forces'] = plumed_forces

if __name__ == '__main__':
    pass
