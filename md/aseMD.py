import ase
import ase.io
import ase.optimize
import ase.md.velocitydistribution
import ase.md.verlet
from ase.md.langevin import Langevin
from ase.units import *

from md.aseplumed import AsePlumed
from md.ase_interface import MLAseCalculator,data_formatter, PlumedCalculator
from ase.units import kcal,mol
from ase.calculators.mixing import SumCalculator
from ase.calculators.calculator import Calculator
from ase.calculators.qchem import QChem
from ase.io.trajectory import Trajectory
import ase.build

from combust.utils.rxn_data import rxn_dict
from combust.utils.plots import get_cn_arrays


import timeit
import numpy as np
import glob
import os
import warnings

import rmsd

###################
### Calculators ###
###################

##-------------------------------------
##     ML model ASE interface
##--------------------------------------
class MLCommitteeAseCalculator(Calculator):
    implemented_properties = ['energy', 'forces']  # , 'stress'

    # default_parameters = {'xc': 'ani'}
    # nolabel = True

    ### Constructor ###
    def __init__(self, committee, disagreement_thresh, lattice=None, **kwargs):
        """
        Constructor for MLCommitteeAseCalculator

        Parameters
        ----------
        disagreement_thresh: int/float or [int/float,int/float]
        lattice: array of (9,)
            lattice vector for pbc
        kwargs
        """
        Calculator.__init__(self, **kwargs)
        self.committee = committee
        if lattice is None:
            self.lattice = lattice
        else:
            self.lattice = np.array(lattice).reshape(9, )
        self.geom = []
        self.stdev = []  # stdev for self.geom
        self.uncertain_steps = []
        if isinstance(disagreement_thresh,int) or isinstance(disagreement_thresh,float):
            self.thresh = [disagreement_thresh,99999]
        else:
            self.thresh = disagreement_thresh
        self.count = 0
        self.dyn = None

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=None):
        super().calculate(atoms, properties, system_changes)
        data = data_formatter(atoms, self.lattice)
        prediction, deviation = self.committee.predict(data, disagreement='std')
        # prediction, votes = self.committee.predict(data, disagreement='values')
        if self.thresh is not None:
            if self.count % 2 == 0:
                std = np.max(deviation['F'])
                if std > self.thresh[0] and std < self.thresh[1]:
                    self.geom.append(atoms.copy())
                    self.stdev.append(std)
                    self.uncertain_steps.append(self.dyn.nsteps)

        # convert to ase units
        energy = prediction['E'][0][0]  # .data.cpu().numpy()
        energy = float(energy) * kcal / mol
        forces = prediction['F'][0]  # .data.cpu().numpy()
        forces = forces * kcal / mol / Ang
        # atomic energies (au -> eV)
        H = -0.5004966690 * 27.211
        O = -75.0637742413 * 27.211
        n_H = np.count_nonzero(np.array(atoms.get_atomic_numbers()) == 1)
        n_O = np.count_nonzero(np.array(atoms.get_atomic_numbers()) == 8)
        atomic_energy = n_H * H + n_O * O
        energy += atomic_energy
        if 'energy' in properties:
            self.results['energy'] = energy
        if 'forces' in properties:
            self.results['forces'] = forces
        self.count+=1
        # print(f"count: {self.count} dyn step: {self.dyn.nsteps}") #count = 2n+2

    def set_dyn(self, dyn):
        self.dyn = dyn

class MLCommittee_qchem_hybrid_AseCalculator(Calculator):
    """For production runs, when the std from MLCommittee models is larger than thresh,
    will use energy and forces from SemiEmpirical calculations instead"""
    implemented_properties = ['energy', 'forces']  # , 'stress'

    # default_parameters = {'xc': 'ani'}
    # nolabel = True

    ### Constructor ###
    def __init__(self, committee,qchem_calc, disagreement_thresh:float, lattice=None,energy_thresh=-26.74, **kwargs):
        """
        Constructor for MLAseCalculator

        Parameters
        ----------
        committee: active_learner.CommiteeRegressor
        qchem_calc: ase.calculators.qchem Qchem calculator
            eg:
            self.calc = QChem(label=os.path.join(output_dir,qchem_inname), #input file name without .in
                         method='PBEh-3c',
                         basis='def2-mSVP',
                         charge = charge,
                         multiplicity = multiplicity,
                         nt=16,np=1)
        disagreement_thresh: int or float. Stdev above this number will switch from ML to Qchem.
        lattice: array of (9,)
            lattice vector for pbc
        energy_thresh: float. per atom energy smaller than this number is considered as chemically relevant. Higher than
                    this number do not need accurate predictions
        kwargs
        """
        Calculator.__init__(self, **kwargs)
        self.committee = committee
        if lattice is None:
            self.lattice = lattice
        else:
            self.lattice = np.array(lattice).reshape(9, )
        self.qchem_calc = qchem_calc
        self.geom = []
        self.stdev = []  # stdev for self.geom
        self.uncertain_steps = []
        self.thresh = disagreement_thresh
        assert (type(self.thresh) is int) or (type(self.thresh) is float), "thresh for MLCommittee_SemiEmpirical_AseCalculator has to be int or float"
        self.energy_thresh = energy_thresh
        self.dyn = None
        self.count = 0

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=None):
        """
        This is a hybrid calculator. When energy is small/relevant and force std is large, will ignore ML result and use
        semiempricial method instead.

        Parameters
        ----------
        atoms
        properties
        system_changes

        Returns
        -------

        """
        super().calculate(atoms, properties, system_changes)
        data = data_formatter(atoms, self.lattice)
        prediction, deviation = self.committee.predict(data, disagreement='std_olremoved')
        # prediction, votes = self.committee.predict(data, disagreement='values')
        std = np.max(deviation['F'])
        use_ML = False
        if np.mean(prediction['E']) / len(atoms.get_atomic_numbers()) > self.energy_thresh:
            use_ML = True
        else:
            if std > self.thresh:
                # use semi-empirical/DFT method
                if self.count %2 ==0:
                    self.geom.append(atoms.copy())
                    self.stdev.append(std)
                    self.uncertain_steps.append(self.dyn.nsteps)
                print(self.dyn.nsteps)
                print(f"qchem for {atoms} with std = {std}")
                try:
                    forces = self.qchem_calc.get_property('forces', atoms)
                    energy = self.qchem_calc.get_property('energy', atoms)
                except Exception as e:
                    # if qchem failed, use ML model instead
                    print(e)
                    use_ML = True
            else:
                use_ML = True
        if use_ML:
            #use model prediction
            # convert to ase units
            energy = prediction['E'][0][0]  # .data.cpu().numpy()
            energy = float(energy) * kcal / mol
            forces = prediction['F'][0]  # .data.cpu().numpy()
            forces = forces * kcal / mol / Ang
            # atomic energies (au -> eV)
            H = -0.5004966690 * 27.211
            O = -75.0637742413 * 27.211
            n_H = np.count_nonzero(np.array(atoms.get_atomic_numbers()) == 1)
            n_O = np.count_nonzero(np.array(atoms.get_atomic_numbers()) == 8)
            atomic_energy = n_H * H + n_O * O
            energy += atomic_energy

        if 'energy' in properties:
            self.results['energy'] = energy
        if 'forces' in properties:
            self.results['forces'] = forces

        self.count += 1
        # print(f"count: {self.count} dyn step: {self.dyn.nsteps}") count = 2n+2

    def set_dyn(self, dyn):
        self.dyn = dyn

class ML_SemiEmpirical_AseCalculator(Calculator):
    implemented_properties = ['energy', 'forces']  # , 'stress'

    # default_parameters = {'xc': 'ani'}
    # nolabel = True

    ### Constructor ###
    def __init__(self, qchem_calc, ml_calc, weight_factor=1, lattice=None, **kwargs):
        """
        Constructor for MLAseCalculator

        Parameters
        ----------

        lattice: array of (9,)
            lattice vector for pbc
        kwargs
        """
        Calculator.__init__(self, **kwargs)
        if lattice is None:
            self.lattice = lattice
        else:
            self.lattice = np.array(lattice).reshape(9, )
        self.geom = []
        self.count = 0
        self.dyn = None
        self.qchem_calc= qchem_calc
        self.ml_calc = ml_calc
        self.weight_factor = weight_factor
        self.ratios = []


    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=None):
        # decide which result to use based on difference in forces
        se_force = self.qchem_calc.get_property('forces', atoms)
        ml_force = self.ml_calc.get_property('forces', atoms)
        weights = [1,0]
        ratio = np.linalg.norm(se_force - ml_force) / np.linalg.norm(se_force) * self.weight_factor
        self.ratios.append(ratio)
        print(f'Step: {self.dyn.nsteps}, Ratio: {ratio}')
        if ratio > self.weight_factor:
            weights=[0,1]
            if self.count % 2 == 0:
                self.geom.append(atoms.copy())

        results = {}
        for prop in properties:
            contributs = [calc.get_property(prop, atoms) for calc in [self.ml_calc,self.qchem_calc]]
            results[f'{prop}_contributions'] = contributs
            results[prop] = sum(weight * value for weight, value
                                in zip(weights, contributs))


        self.results = results
        self.count += 1


class Sum_Calc(SumCalculator):
    ### Constructor ###
    def __init__(self, ml_calc, qm_calc, atoms=None):
        """
        Constructor for MLAseCalculator

        Parameters
        ----------

        lattice: array of (9,)
            lattice vector for pbc
        kwargs
        """
        SumCalculator.__init__(self, [ml_calc,qm_calc], atoms)
        self.dyn = None
        self.ml_calc = ml_calc

    def set_dyn(self, dyn):
        self.dyn = dyn
        self.ml_calc.set_dyn(dyn)



##-------------------------------------
##     ASE MD Runner
##--------------------------------------
class AseMD():
    ### Constructor ###
    def __init__(self, atoms, calc, temp = 1000, stepsize = 0.1, time_fs = 200, **kwargs):
        self.atoms = atoms
        self._temp = temp  #Kelvin
        self._time_fs = time_fs * ase.units.fs
        self._stepsize = stepsize * ase.units.fs
        self._steps = int(time_fs / stepsize)
        self.calc = calc
        #only when using energy conservation as convergence criterion
        self._prev_energy = 0
        self._count = 0
        self.geom = []




    @property
    def temp(self):
        """Temperature of simulation in Kelvin"""
        return self._temp

    @temp.setter
    def temp(self,temp_value):
        """Setter for temperature of simulation in Kelvin"""
        self._temp = temp_value


    @property
    def stepsize(self):
        """Step size of simulation in femtoseconds"""
        return self._stepsize

    @stepsize.setter
    def stepsize(self, stepsize):
        """Setter for stepsize of simulation in femtoseconds"""
        self._stepsize = stepsize * ase.units.fs
        self._steps = int(self._time_fs / self._stepsize)

    @property
    def steps(self):
        """Number of steps of simulation"""
        return self._steps
    #
    # @steps.setter
    # def steps(self, steps):
    # 	"""Setter for number of steps of simulation"""
    # 	self._steps = steps

    @property
    def time_fs(self):
        """Length of simulation in femtoseconds"""
        return self._time_fs

    @time_fs.setter
    def time_fs(self, time_fs):
        """Setter for length of simulation in femtoseconds"""
        self._time_fs = time_fs * ase.units.fs
        self._steps = int(self._time_fs  / self._stepsize)

    def print_energy(self,dyn):
        """Function to print the potential, kinetic and total energy."""
        a = self.atoms
        print('step: ', dyn.nsteps)
        epot = a.get_potential_energy() / len(a) / (ase.units.kcal / ase.units.mol)
        ekin = a.get_kinetic_energy() / len(a) / (ase.units.kcal / ase.units.mol)
        print('Energy per atom: Epot = %.3f kcal/mol  Ekin = %.3f kcal/mol (T=%3.0fK)  '
              'Etot = %.3f kcal/mol' % (epot, ekin, ekin * (ase.units.kcal / ase.units.mol) / (1.5  * ase.units.kB), epot + ekin))

    def print_energy_update_max_step(self,dyn,output_filename,output_dir):
        """Function to print the potential, kinetic and total energy."""
        a = self.atoms
        epot = a.get_potential_energy() / len(a) / (ase.units.kcal / ase.units.mol)
        ekin = a.get_kinetic_energy() / len(a) / (ase.units.kcal / ase.units.mol)
        print('step: ',dyn.nsteps)
        print('Energy per atom: Epot = %.3f kcal/mol  Ekin = %.3f kcal/mol (T=%3.0fK)  '
              'Etot = %.3f kcal/mol' % (epot, ekin, ekin * (ase.units.kcal / ase.units.mol) / (1.5 * ase.units.kB), epot + ekin))
        #reset max_step to terminate run if committed
        res = committer_analysis(output_filename.split('_')[1],f'{output_dir}/{output_filename}.traj',True)
        if not len(res['neither'])>0 and dyn.nsteps>10:
            print(f'max_step reset from {dyn.max_steps} to {dyn.nsteps} because committed to {res}')
            dyn.max_steps = dyn.nsteps

    def print_energy_update_max_step_by_E_conservation(self,dyn,thresh, ngeom):
        """Function to print the potential, kinetic and total energy."""
        a = self.atoms
        epot = a.get_potential_energy() / len(a) / (ase.units.kcal / ase.units.mol)
        ekin = a.get_kinetic_energy() / len(a) / (ase.units.kcal / ase.units.mol)
        step = dyn.nsteps
        if step % 1000 ==0:
            print('step: ',step)
            print('Energy per atom: Epot = %.3f kcal/mol  Ekin = %.3f kcal/mol (T=%3.0fK)  '
              'Etot = %.3f kcal/mol' % (epot, ekin, ekin * (ase.units.kcal / ase.units.mol) / (1.5 * ase.units.kB), epot + ekin))
        if step > 1 and np.abs(epot+ ekin - self._prev_energy) > thresh:
            print(np.abs(epot+ ekin - self._prev_energy))
            self._count += 1
            self.geom.append(a.copy())
        self._prev_energy = epot+ ekin
        #reset max_step to terminate run if conservation fails
        if self._count > ngeom and dyn.nsteps>10:
            print(f'max_step reset from {dyn.max_steps} to {dyn.nsteps} because energy conservation ({thresh} kcal/mol) \
            has failed {ngeom} number of times')
            dyn.max_steps = dyn.nsteps


    def run_simulation(self, output_dir, interval, output_filename ='md', ensemble='NVE',convergence_criteria = None,**kwargs):
        self.atoms.set_calculator(self.calc)

        if os.path.isfile(f'{output_dir}/{output_filename}.traj'):
            os.system(f'rm -r {output_dir}/*')

        if ensemble == 'NVE':
            # NVE simulation
            ase.md.velocitydistribution.MaxwellBoltzmannDistribution(self.atoms, temperature_K = self._temp, force_temp=True)
            ase.md.velocitydistribution.Stationary(self.atoms) # Sets the center-of-mass momentum to zero.
            ase.md.velocitydistribution.ZeroRotation(self.atoms) # Sets the total angular momentum to zero by counteracting rigid rotations.
            print("Initial temperature from velocities %.2f" % self.atoms.get_temperature())

            dyn = ase.md.verlet.VelocityVerlet(
                self.atoms,
                self._stepsize,
                trajectory=f'{output_dir}/{output_filename}.traj',
                logfile=f'{output_dir}/{output_filename}.log',
            )
        elif ensemble =='NVT':
            dyn = Langevin(self.atoms,
                           self._stepsize,
                           temperature_K=self.temp,
                           friction=0.002,
                           trajectory = f'{output_dir}/{output_filename}.traj',
                           logfile = f'{output_dir}/{output_filename}.log',
                           )

        else:
            raise NotImplementedError(f'Ensemble {ensemble} not supported')

        self.calc.set_dyn(dyn)
        if convergence_criteria == 'committed':
            args = (dyn,output_filename,output_dir)
            print(interval)
            dyn.attach(self.print_energy_update_max_step, interval, *args)
        elif convergence_criteria == 'energy_conservation':
            try:
                args = (dyn, kwargs['thresh'], kwargs['ngeom'])
            except:
                warnings.warn("did not specify thresh and ngeom, use default value")
                thresh = 3  # thresh in kcal/mol
                ngeom = 10
                args = (dyn, thresh, ngeom)
            dyn.attach(self.print_energy_update_max_step_by_E_conservation, interval, *args)

        else:
            args = [dyn]
            dyn.attach(self.print_energy,interval, *args)
        start = timeit.default_timer()
        print('Number of steps: ', self._steps)
        dyn.run(self._steps)
        end = timeit.default_timer()
        print('Total time: ', end - start)



def RMSD(a,b):
    "align and calculate rmsd, input are nparrays"
    # Manipulate
    A = a - rmsd.centroid(a)
    B = b - rmsd.centroid(b)
    U = rmsd.kabsch(A, B)
    A = np.dot(A, U)
    return rmsd.rmsd(A, B)

def committer_analysis(rxn_number, trajs, silent = True, option = "CN"):
    """

    Parameters
    ----------
    rxn_number: str of length 2.
    trajs: str or list of str. Path to a .traj file
    silent: bool.
    option: str. One of ["RMSD","CN"]

    Returns
    -------

    """
    # option
    if isinstance(trajs,str):
        trajs = [trajs]
    result = {'reactant': [], 'product': [], 'neither':[]}
    split_cutoff = 1.6
    # setting for lrc runs
    data_dir = "/global/scratch/users/nancy_guan/ML/AIMD_H_combustion/H2Combustion/md/irc_geom"
    reactant = f'{data_dir}/reactant_{rxn_number}.xyz'
    product = f'{data_dir}/product_{rxn_number}.xyz'
    ts = f'{data_dir}/TS_{rxn_number}.xyz'
    if option == "RMSD":
        reactants = ase.build.separate(ase.io.read(reactant), scale = split_cutoff)
        products = ase.build.separate(ase.io.read(product), scale=split_cutoff)
        reactant_numbers = [sorted(atom.get_atomic_numbers()) for atom in reactants]
        product_numbers = [sorted(atom.get_atomic_numbers()) for atom in products]
    elif option == "CN":
        rxn_num = "rxn"+rxn_number
        cn1s = rxn_dict[rxn_num]['cn1']
        cn2s = rxn_dict[rxn_num]['cn2']
        mu = rxn_dict[rxn_num]['mu']
        re_cn1,re_cn2 = get_cn(ase.io.read(reactant),mu,cn1s,cn2s)
        pr_cn1, pr_cn2 = get_cn(ase.io.read(product), mu, cn1s, cn2s)
        ts_cn1, ts_cn2 = get_cn(ase.io.read(ts), mu, cn1s, cn2s)
        if ts_cn1 == re_cn1 and ts_cn2 == re_cn2:
            raise ValueError(f"rxn{rxn_number} not valid for committer analysis. Highest energy point is reactant ")
        elif ts_cn1 == pr_cn1 and ts_cn2 == pr_cn2:
            raise ValueError(f"rxn{rxn_number} not valid for committer analysis. Highest energy point is product ")
        cn1_thresh = [0.2 * np.abs(re_cn1 - ts_cn1),0.2 * np.abs(pr_cn1 - ts_cn1)]
        cn2_thresh = [0.2 * np.abs(re_cn2 - ts_cn2), 0.2 * np.abs(pr_cn2 - ts_cn2)]

    for idx,traj in enumerate(trajs):
        if not silent:
            print(traj)
        traj = Trajectory(traj)
        atoms = traj[-1]
        #using RMSD
        if option == "RMSD":
            components = ase.build.separate(atoms, scale = split_cutoff)
            components_numbers = [sorted(atoms.get_atomic_numbers()) for atoms in components]
            # print(components_numbers)
            matches_reactant = np.zeros(len(components))
            matches_product = np.zeros(len(components))
            for i,component_number in enumerate(components_numbers):
                for j,reactant_number in enumerate(reactant_numbers):
                    #check if one piece of component matches one piece of reactant
                    if np.array_equal(reactant_number,component_number):
                        if not silent:
                            print(f'reactant {j} RMSD: ', RMSD(components[i].get_positions(),reactants[j].get_positions()))
                        if RMSD(components[i].get_positions(),reactants[j].get_positions()) < 0.1 :
                            matches_reactant[i] = 1
                            continue
                for k, product_number in enumerate(product_numbers):
                    # check if one piece of component matches one piece of product
                    if np.array_equal(product_number, component_number):
                        if not silent:
                            print(f'product {k} RMSD: ',RMSD(components[i].get_positions(), products[k].get_positions()))
                        if RMSD(components[i].get_positions(), products[k].get_positions()) < 0.1:
                            matches_product[i] = 1
                            continue
            if np.all(matches_reactant):
                result['reactant'].append(idx)
            elif np.all(matches_product):
                result['product'].append(idx)
            else:
                result['neither'].append(idx)

        # using coordination number
        elif option == "CN":
            cn1,cn2 = get_cn(atoms,mu,cn1s,cn2s)
            if np.abs(cn1 - re_cn1) < cn1_thresh[0] and np.abs(cn2 - re_cn2) < cn2_thresh[0]:
                result['reactant'].append(idx)
            elif np.abs(cn1 - pr_cn1) < cn1_thresh[1] and np.abs(cn2 - pr_cn2) < cn2_thresh[1]:
                result['product'].append(idx)
            else:
                result['neither'].append(idx)
    if not silent:
        print(result)


    return result


def get_cn(atoms,mu,cn1s,cn2s):
    atnums = np.array(atoms.get_atomic_numbers())
    atcoords = np.tile(np.array(atoms.get_positions()), (1, 1, 1))
    cn1 = get_cn_arrays(np.tile(atnums, (1, 1)), atcoords, cn1s, mu=mu[0], sigma=3.0)
    cn2 = get_cn_arrays(np.tile(atnums, (1, 1)), atcoords, cn2s, mu=mu[1], sigma=3.0)
    return cn1,cn2


class AseMLCommiteeMetadynamics(AseMD):
    def __init__(self, plumed_in_file, atoms, model_path, lattice, model_type='NewtonNet', temp=1000, stepsize=0.1,
                 time_fs=200,
                 **kwargs):
        AseMD.__init__(self, atoms, None, temp, stepsize, time_fs, **kwargs)

        self.ase_plumed = AsePlumed(atoms, self.stepsize, plumed_in_file, plumed_in_file.replace('dat', 'out'))
        plumed_calc = PlumedCalculator(self.ase_plumed)
        yml = glob.glob(f'{model_path}/run_scripts/*.yml')
        assert len(yml) == 1, f'{len(yml)} yaml files found in {model_path}/run_scripts'
        yml = yml[0]
        ML_calc = MLCommitteeAseCalculator(f'{model_path}/models/best_model_state.tar', yml,
                                           model_type=model_type, lattice=lattice)
        self.calc = Sum_Calc(ML_calc, plumed_calc, atoms=atoms)


class AseQChemMetadynamics(AseMD):
    def __init__(self, plumed_in_file, atoms, qchem_inname , temp=1000, stepsize=0.1,multiplicity=1,charge=0,
                 time_fs=200,output_dir = './',
                 **kwargs):
        AseMD.__init__(self,atoms, None, temp, stepsize, time_fs, **kwargs)

        self.ase_plumed = AsePlumed(atoms, self.stepsize, plumed_in_file, plumed_in_file.replace('dat', 'out'))
        plumed_calc= PlumedCalculator(self.ase_plumed)
        qchem_calc = QChem(label=os.path.join(output_dir,qchem_inname), #input file name without .in
                         method='wB97X-V',
                         basis='cc-pVTZ',
                         charge = charge,
                         multiplicity = multiplicity,
                           scf_max_cycles='500',
                           incdft='0',
                           symmetry='false',
                         nt=16,np=1)
        self.calc = SumCalculator([qchem_calc,plumed_calc],atoms=atoms)

class AseQChemMD(AseMD):
    def __init__(self, atoms, qchem_inname , temp=1000, stepsize=0.1,multiplicity=1,charge=0,
                 time_fs=200,output_dir = './',
                 **kwargs):
        AseMD.__init__(self,atoms, None, temp, stepsize, time_fs, **kwargs)

        self.calc = QChem(label=os.path.join(output_dir,qchem_inname), #input file name without .in
                         method='wB97X-V',
                         basis='cc-pVTZ',
                         charge = charge,
                         multiplicity = multiplicity,
                          scf_max_cycles='500',
                          incdft='0',
                          symmetry='false',
                         nt=16,np=1)

class AseMLMetadynamics(AseMD):
    def __init__(self, plumed_in_file, atoms, model_path, lattice = None, model_type='NewtonNet', temp=1000, stepsize=0.1, time_fs=200,
                 **kwargs):
        AseMD.__init__(self,atoms, None, temp, stepsize, time_fs, **kwargs)

        self.ase_plumed = AsePlumed(atoms, self.stepsize, plumed_in_file, plumed_in_file.replace('dat', 'out'))
        plumed_calc= PlumedCalculator(self.ase_plumed)
        yml = glob.glob(f'{model_path}/run_scripts/*.yml')
        assert len(yml) == 1, f'{len(yml)} yaml files found in {model_path}/run_scripts'
        yml = yml[0]
        ML_calc = MLAseCalculator(f'{model_path}/models/best_model_state.tar', yml,
                               model_type=model_type, lattice=lattice)
        self.calc = Sum_Calc(ML_calc,plumed_calc,atoms=atoms)