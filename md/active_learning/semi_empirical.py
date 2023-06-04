#!/usr/bin/env python

# @Time:Feb 2022
# @Author: Nancy Xingyi Guan
# @File: semi_empirical.py


from md.aseMD import AseMD
from ase.calculators.qchem import QChem

import os
import numpy as np






class AseSemiEmpiricalMD(AseMD):
    def __init__(self, atoms, qchem_inname , temp=1000, stepsize=0.1,multiplicity=1,charge=0,
                 time_fs=200,output_dir = './',
                 **kwargs):
        AseMD.__init__(self,atoms, None, temp, stepsize, time_fs, **kwargs)

        self.calc = QChem(label=os.path.join(output_dir,qchem_inname), #input file name without .in
                         method='PBEh-3c',
                         basis='def2-mSVP',
                         charge = charge,
                         multiplicity = multiplicity,
                         nt=16,np=1)

