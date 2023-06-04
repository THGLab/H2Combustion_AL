"""
H2Combustion Reaction Data.
"""

#for visualization of HCombustion data paper Guan, X.; Das, A.; Stein, C. J.; Heidar-Zadeh, F.; Bertels, L.; Liu, M.; Haghighatlari, M.; Li, J.; Zhang, O.; Hao, H.; Leven, I.; Head-Gordon, M.; Head-Gordon, T. A Benchmark Dataset for Hydrogen Combustion. Sci Data 2022, 9 (1), 215. https://doi.org/10.1038/s41597-022-01330-5.

rxn_dict_paper = {
    'rxn01': {
        'title': r'RXN01: OH + O $\rightarrow$ O$_2$ + H (quartet)',
        'xtitle': 'CN2 [O1-O2]',
        'ytitle': 'CN1 [H1-O2]',
        'cn1': ['O2_H1'],
        'cn2': ['O1_O2'],
        'natoms': 3,
        'mu': [0.97,1.21],   # mu for plotting CN1 & CN2
        'folder': '01_H+O2_OH+O.q',
    },
    'rxn02': {
        'title': r'RXN02: O + H$_2$$ \rightarrow$ OH + H (triplet)',
        'xtitle': 'CN2 [O1-H1]',
        'ytitle': 'CN1 [H1-H2]',
        'cn1': ['H1_H2'],
        'cn2': ['H1_O1'],
        'natoms': 3,
        'mu': [0.741, 0.97],   # mu for plotting CN1 & CN2
        'folder': '02_O+H2_OH+H.t',
    },
    'rxn03': {
        'title': r'RXN03: H2 + OH$ \rightarrow$ H$_2$O + H (doublet)',
        'xtitle': 'CN2 [O1-H2]',
        'ytitle': 'CN1 [H1-H2]',
        'cn1': ['H1_H2'],
        'cn2': ['H2_O1'],
        'natoms': 4,
        'mu': [0.741, 0.97],   # mu for plotting CN1 & CN2
        'folder': '03_H2+OH_H2O+H.d',
    },
    'rxn04': {
        'title': r'RXN04:  2 OH $ \rightarrow$ H$_2$O + O (triplet)',
        'xtitle': 'CN2 [O1-H2]',
        'ytitle': 'CN1 [O2-H2]',
        'cn1': ['H2_O2'],
        'cn2': ['O1_H2'],
        'natoms': 4,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '04_H2O+O_2OH.t',
    },
    'rxn05': {
        'title': r'RXN05: H$_2$$ \rightarrow$ 2H (singlet)',
        'xtitle': 'CN2 [H1-H2]',
        'ytitle': 'CN1 [H1-H2]',
        'cn1': ['H1_H2'],
        'cn2': ['H1_H2'],
        'natoms': 2,
        'mu': [0.74,0.74],   # mu for plotting CN1 & CN2
        'folder': '05_H2_2H.s',
    },
    'rxn06a': {
        'title': r'RXN06: O$_2$$ \rightarrow$ 2O (singlet)',
        'xtitle': 'CN2 [O1-O2]',
        'ytitle': 'CN1 [O1-O2]',
        'cn1': ['O1_O2'],
        'cn2': ['O1_O2'],
        'natoms': 2,
        'mu': [1.21,1.21],   # mu for plotting CN1 & CN2
        'folder': '06_O2_2O.s',
    },
    'rxn06b': {
        'title': r'RXN06: O$_2$$ \rightarrow$ 2O (triplet)',
        'xtitle': 'CN2 [O1-O2]',
        'ytitle': 'CN1 [O1-O2]',
        'cn1': ['O1_O2'],
        'cn2': ['O1_O2'],
        'natoms': 2,
        'mu': [1.21,1.21],   # mu for plotting CN1 & CN2
        'folder': '06_O2_2O.t',
    },
    'rxn07': {
        'title': r'RXN07: OH$ \rightarrow$ O + H (doublet)',
        'xtitle': 'CN2 [O1-H1]',
        'ytitle': 'CN1 [O1-H1]',
        'cn1': ['O1_H1'],
        'cn2': ['O1_H1'],
        'natoms': 2,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '07_OH_O+H.d',
    },
    'rxn08': {
        'title': r'RXN08: H + OH$ \rightarrow$ H$_2$O (singlet)',
        'xtitle': 'CN2 [O1-H2]',
        'ytitle': 'CN1 [O1-(H1,H2)]',
        'cn1': ['O1_H2','O1_H1'],
        'cn2': ['O1_H2'],
        'natoms': 3,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '08_H+OH_H2O.s',
    },
    'rxn09': {
        'title': r'RXN09: HO$_2 \rightarrow$ H + O$_2$ (doublet)',
        'xtitle': 'CN2 [O1-H1]',
        'ytitle': 'CN1 [O2-H1]',
        'cn1': ['O1_H1'],
        'cn2': ['O2_H1'],
        'natoms': 3,
        'mu': [0.96,1.84],   # mu for plotting CN1 & CN2
        'folder': '09_H+O2_HO2.d',
    },
    'rxn10': {
        'title': r'RXN10: HO2 + H$ \rightarrow$ H2 + O2 (triplet)',
        'xtitle': 'CN2 [H1-H2]',
        'ytitle': 'CN1 [O2-H1]',
        'cn1': ['O2_H1'],
        'cn2': ['H1_H2'],
        'natoms': 4,
        'mu': [0.96,0.74],   # mu for plotting CN1 & CN2
        'folder': '10_HO2+H_H2+O2.t',
    },
    'rxn11': {
        'title': r'RXN11: HO$_2$ + H$ \rightarrow$ 2OH (triplet)',
        'xtitle': 'CN2 [O2-H2]',
        'ytitle': 'CN1 [O1-O2]',
        'cn1': ['O1_O2'],
        'cn2': ['O2_H2'],
        'natoms': 4,
        'mu': [1.31, 0.97],   # mu for plotting CN1 & CN2
        'folder': '11_HO2+H_2OH.t',
    },
    'rxn12a': {
        'title': r'RXN12: HO$_2$ + O$ \rightarrow$ OH + O$_2$ (doublet)',
        'xtitle': 'CN2 [H1-O3]',
        'ytitle': 'CN1 [H1-(O1,O2)]',
        'cn1': ['O1_H1','O2_H1'],
        'cn2': ['H1_O3'],
        'natoms': 4,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '12_HO2+O_OH+O2.d',
    },
    'rxn12b': {
        'title': r'RXN12: HO$_2$ + O$ \rightarrow$ OH + O$_2$ (quartet)',
        'xtitle': 'CN2 [H1-O3]',
        'ytitle': 'CN1 [H1-(O1,O2)]',
        'cn1': ['O1_H1','O2_H1'],
        'cn2': ['H1_O3'],
        'natoms': 4,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '12_HO2+O_OH+O2.q',
    },
    'rxn12': {
        'title': r'RXN12: HO$_2$ + O$ \rightarrow$ OH + O$_2$',
        'xtitle': 'CN2 [H1-O3]',
        'ytitle': 'CN1 [H1-(O1,O2)]',
        'cn1': ['O1_H1','O2_H1'],
        'cn2': ['H1_O3'],
        'natoms': 4,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '12_HO2+O_OH+O2.d',
    },
    'rxn13': {
        'title': r'RXN13: H$_2$O + O$_2 \rightarrow$ HO$_2$ + OH (triplet)',
        'xtitle': 'CN2 [H1-O1]',
        'ytitle': 'CN1 [H1-O3]',
        'cn1': ['H1_O3'],
        'cn2': ['O1_H1'],
        'natoms': 5,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '13_HO2+OH_H2O+O2.t',
    },
    'rxn14': {
        'title': r'RXN14:  H$_2$O$_2$ + O$_2 \rightarrow$ 2 HO$_2$ (triplet)',
        'xtitle': 'CN2 [H2-O3]',
        'ytitle': 'CN1 [H2-O2]',
        'cn1': ['O2_H2'],
        'cn2': ['H2_O3'],
        'natoms': 6,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '14_2HO2_H2O2+O2.t',
    },
    'rxn15': {
        'title': r'RXN15: H$_2$O$_2 \rightarrow $ 2 OH (singlet)',
        'xtitle': 'CN2 [O1-O2]',
        'ytitle': 'CN1 [O1-H2]',
        'cn1': ['O1_H2'],
        'cn2': ['O1_O2'],
        'natoms': 4,
        'mu': [0.96,1.54],   # mu for plotting CN1 & CN2
        'folder': '15_H2O2_2OH.s',
    },
    'rxn16': {
        'title': r'RXN16: H$_2$O + OH$ \rightarrow$  H$_2$O$_2$ + H (doublet)',
        'xtitle': 'CN2 [O1-O2]',
        'ytitle': 'CN1 [O2-H3]',
        'cn1': ['O2_H3'],
        'cn2': ['O1_O2'],
        'natoms': 5,
        'mu': [0.96,1.43],   # mu for plotting CN1 & CN2
        'folder': '16_H2O2+H_H2O+OH.d',
    },
    'rxn17': {
        'title': r'RXN17: HO$_2$ + H$_2 \rightarrow$ H$_2$O$_2$ + H (doublet)',
        'xtitle': 'CN2 [O2-H2]',
        'ytitle': 'CN1 [H2-H3]',
        'cn1': ['H2_H3'],
        'cn2': ['O2_H2'],
        'natoms': 5,
        'mu': [0.741, 0.97],   # mu for plotting CN1 & CN2
        'folder': '17_H2O2+H_HO2+H2.d',
    },
    'rxn18': {
        'title': r'RXN18: HO$_2$ + OH$ \rightarrow$ H$_2$O$_2$ + O (triplet)',
        'xtitle': 'CN2 [O2-H2]',
        'ytitle': 'CN1 [O3-H2]',
        'cn1': ['H2_O3'],
        'cn2': ['O2_H2'],
        'natoms': 5,
        'mu': [0.97, 0.97],   # mu for plotting CN1 & CN2
        'folder': '18_H2O2+O_HO2+OH.t',
    },
    'rxn19': {
        'title': r'RXN19: H$_2$O$_2$ + OH$ \rightarrow$ H$_2$O + HO$_2$ (doublet)',
        'xtitle': 'CN2 [O3-H2]',
        'ytitle': 'CN1 [O1-H2]',
        'cn1': ['O1_H2'],
        'cn2': ['H2_O3'],
        'natoms': 6,
        'mu': [0.97, 0.97],   # mu for plotting CN1 & CN2
        'folder': '19_H2O2+OH_H2O+HO2.d',
    },
}


#for visualization of metadynamics
rxn_dict = {
    'rxn01': {
        'title': r'RXN01: OH + O $\rightarrow$ O$_2$ + H (quartet)',
        'xtitle': 'CN2 [O1-H3]',
        'ytitle': 'CN1 [O1-O2]',
        'cn1': ['O1_O2'],
        'cn2': ['O1_H1'],
        'natoms': 3,
        'mu': [1.45,2.0],   # mu for plotting CN1 & CN2 O2 double bond 1.21
        'folder': '01_H+O2_OH+O.q',
    },
    'rxn02': {
        'title': r'RXN02: O + H$_2$$ \rightarrow$ OH + H (triplet)',
        'xtitle': 'CN2 [O1-(H1,H2)]',
        'ytitle': 'CN1 [H1-H2]',
        'cn1': ['H1_H2'],
        'cn2': ['H1_O1', 'O1_H2'],
        'natoms': 3,
        'mu': [0.741, 0.97],   # mu for plotting CN1 & CN2
        'folder': '02_O+H2_OH+H.t',
    },
    'rxn03': {
        'title': r'RXN03: H2 + OH$ \rightarrow$ H$_2$O + H (doublet)',
        'xtitle': 'CN2 [O1-(H1,H2)]',
        'ytitle': 'CN1 [H1-H2]',
        'cn1': ['H1_H2', 'H1_H3', 'H2_H3'],
        'cn2': ['H1_O1', 'H2_O1', 'H3_O1'],
        'natoms': 4,
        'mu': [0.741, 0.97],   # mu for plotting CN1 & CN2
        'folder': '03_H2+OH_H2O+H.d',
    },
    'rxn04': {
        'title': r'RXN04:  2 OH $ \rightarrow$ H$_2$O + O (triplet)',
        'xtitle': 'CN2 [O1-H2]',
        'ytitle': 'CN1 [O2-H2]',
        'cn1': ['H2_O2'],
        'cn2': ['O1_H2'],
        'natoms': 4,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '04_H2O+O_2OH.t',
    },
    'rxn05': {
        'title': r'RXN05: H$_2$$ \rightarrow$ 2H (singlet)',
        'xtitle': 'CN2 [H1-H2]',
        'ytitle': 'CN1 [H1-H2]',
        'cn1': ['H1_H2'],
        'cn2': ['H1_H2'],
        'natoms': 2,
        'mu': [0.74,0.74],   # mu for plotting CN1 & CN2
        'folder': '05_H2_2H.s',
    },
    'rxn06a': {
        'title': r'RXN06: O$_2$$ \rightarrow$ 2O (singlet)',
        'xtitle': 'CN2 [O1-O2]',
        'ytitle': 'CN1 [O1-O2]',
        'cn1': ['O1_O2'],
        'cn2': ['O1_O2'],
        'natoms': 2,
        'mu': [1.21,1.21],   # mu for plotting CN1 & CN2
        'folder': '06_O2_2O.s',
    },
    'rxn06b': {
        'title': r'RXN06: O$_2$$ \rightarrow$ 2O (triplet)',
        'xtitle': 'CN2 [O1-O2]',
        'ytitle': 'CN1 [O1-O2]',
        'cn1': ['O1_O2'],
        'cn2': ['O1_O2'],
        'natoms': 2,
        'mu': [1.21,1.21],   # mu for plotting CN1 & CN2
        'folder': '06_O2_2O.t',
    },
    'rxn07': {
        'title': r'RXN07: OH$ \rightarrow$ O + H (doublet)',
        'xtitle': 'CN2 [O1-H1]',
        'ytitle': 'CN1 [O1-H1]',
        'cn1': ['O1_H1'],
        'cn2': ['O1_H1'],
        'natoms': 2,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '07_OH_O+H.d',
    },
    'rxn08': {
        'title': r'RXN08: H + OH$ \rightarrow$ H$_2$O (singlet)',
        'xtitle': 'CN2 [O1-H2]',
        'ytitle': 'CN1 [O1-(H1,H2)]',
        'cn1': ['O1_H2','O1_H1'],
        'cn2': ['O1_H2'],
        'natoms': 3,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '08_H+OH_H2O.s',
    },
    'rxn09': {
        'title': r'RXN09: H + O$_2$$ \rightarrow$ HO$_2$ (doublet)',
        'xtitle': 'CN2 [O2-H3]',
        'ytitle': 'CN1 [O1-H3]',
        'cn1': ['O1_H1'],
        'cn2': ['O2_H1'],
        'natoms': 3,
        'mu': [0.97, 0.97],   # mu for plotting CN1 & CN2
        'folder': '09_H+O2_HO2.d',
    },
    'rxn10': {
        'title': r'RXN10: HO2 + H$ \rightarrow$ H2 + O2 (triplet)',
        'xtitle': 'CN2 [O2-H4]',
        'ytitle': 'CN1 [O2-H3]',
        'cn1': ['O2_H1'],
        'cn2': ['O2_H2'],
        'natoms': 4,
        'mu': [0.97, 1.5], # 0.741  # mu for plotting CN1 & CN2
        'folder': '10_HO2+H_H2+O2.t',
    },
    'rxn11': {
        'title': r'RXN11: HO$_2$ + H$ \rightarrow$ 2OH (triplet)',
        'xtitle': 'CN2',
        'ytitle': 'CN1',
        'cn1': ['O1_O2'],
        'cn2': ['H1_O1', 'H1_O2', 'O1_H2', 'O2_H2'],
        'natoms': 4,
        'mu': [1.45, 0.97],   # mu for plotting CN1 & CN2
        'folder': '11_HO2+H_2OH.t',
    },
    'rxn12a': {
        'title': r'RXN12: HO$_2$ + O$ \rightarrow$ OH + O$_2$ (doublet)',
        'xtitle': 'CN2 [H1-O3]',
        'ytitle': 'CN1 [H1-(O1,O2)]',
        'cn1': ['O1_H1','O2_H1'],
        'cn2': ['H1_O3'],
        'natoms': 4,
        'mu': [0.96,0.96],   # mu for plotting CN1 & CN2
        'folder': '12_HO2+O_OH+O2.d',
    },
    'rxn12b': {
        'title': r'RXN12: HO$_2$ + O$ \rightarrow$ OH + O$_2$ (quartet)',
        'xtitle': 'CN2 [O2-O4]',
        'ytitle': 'CN1 [H3-O4]',
        'cn1': ['H1_O3'],
        'cn2': ['O2_O3'],
        'natoms': 4,
        'mu': [0.96,2.0],   # mu for plotting CN1 & CN2
        'folder': '12_HO2+O_OH+O2.q',
    },
    'rxn12': {
        'title': r'RXN12: HO$_2$ + O$ \rightarrow$ OH + O$_2$',
        'xtitle': 'CN2 [H1-O3]',
        'ytitle': 'CN1 [H3-O4]',
        'cn1': ['H1_O3'],
        'cn2': ['O2_O3'],
        'natoms': 4,
        'mu': [0.96,2.0],   # mu for plotting CN1 & CN2
        'folder': '12_HO2+O_OH+O2.d',
    },
    'rxn13': {
        'title': r'RXN13: HO$_2$ + OH$ \rightarrow$ H$_2$O + O$_2$ (triplet)',
        'xtitle': 'CN2 [H3-O5]',
        'ytitle': 'CN1 [O1-O5]',
        'cn1': ['O1_O3'],
        'cn2': ['H1_O3'],
        'natoms': 5,
        'mu': [1.8, 0.97,],   # mu for plotting CN1 & CN2
        'folder': '13_HO2+OH_H2O+O2.t',
    },
    'rxn14': {
        'title': r'RXN14: 2 HO$_2$ _ H$_2$O$_2$ + O$_2$ (triplet)',
        'xtitle': 'CN2 [O3-O5]',
        'ytitle': 'CN1 [O3-H4]',
        'cn1': ['O2_H2'],
        'cn2': ['O2_O3'],
        'natoms': 6,
        'mu': [0.97, 2.0],   # mu for plotting CN1 & CN2
        'folder': '14_2HO2_H2O2+O2.t',
    },
    'rxn15': {
        'title': r'RXN15: H$_2$O$_2 \rightarrow $ 2 OH (singlet)',
        'xtitle': 'CN2 [O1-O2]',
        'ytitle': 'CN1 [O1-H2]',
        'cn1': ['O1_H2'],
        'cn2': ['O1_O2'],
        'natoms': 4,
        'mu': [1.87,1.21],   # mu for plotting CN1 & CN2
        'folder': '15_H2O2_2OH.s',
    },
    'rxn16': {
        'title': r'RXN16: H$_2$O + OH$ \rightarrow$  H$_2$O$_2$ + H (doublet)',
        'xtitle': 'CN2 [O2-H5]',
        'ytitle': 'CN1 [O3-H5]',
        'cn1': ['O2_H3'],
        'cn2': ['O1_H3'],
        'natoms': 5,
        'mu': [0.97, 2.5],  #1.43 # mu for plotting CN1 & CN2
        'folder': '16_H2O2+H_H2O+OH.d',
    },
    # 'rxn16': {
    #     'title': r'RXN16: H$_2$O + OH$ \rightarrow$  H$_2$O$_2$ + H (doublet)',
    #     'xtitle': 'CN2 [O1-O2]',
    #     'ytitle': 'CN1 [O2-H3]',
    #     'cn1': ['O2_H3'],
    #     'cn2': ['O1_O2'],
    #     'natoms': 5,
    #     'mu': [0.96, 1.43],  # mu for plotting CN1 & CN2
    #     'folder': '16_H2O2+H_H2O+OH.d',
    # },
    'rxn17': {
        'title': r'RXN17: HO$_2$ + H$_2 \rightarrow$ H$_2$O$_2$ + H (doublet)',
        'xtitle': 'CN2 [O2-H5]',
        'ytitle': 'CN1 [H4-H5]',
        'cn1': ['H2_H3'],
        'cn2': ['O2_H3'],
        'natoms': 5,
        'mu': [0.741, 1.5], #0.741  # mu for plotting CN1 & CN2
        'folder': '17_H2O2+H_HO2+H2.d',
    },
    'rxn18': {
        'title': r'RXN18: HO$_2$ + OH$ \rightarrow$ H$_2$O$_2$ + O (triplet)',
        'xtitle': 'CN2 [O5-H4]',
        'ytitle': 'CN1 [O2-O5]',
        'cn1': ['O2_O3'],
        'cn2': ['H2_O3'],
        'natoms': 5,
        'mu': [2.0, 0.97], #1.43   # mu for plotting CN1 & CN2
        'folder': '18_H2O2+O_HO2+OH.t',
    },
    'rxn19': {
        'title': r'RXN19: H$_2$O$_2$ + OH$ \rightarrow$ H$_2$O + HO$_2$ (doublet)',
        'xtitle': 'CN2 [O3-(H1,H2,H3)]',
        'ytitle': 'CN1 [O2-(H1,H2,H3)]',
        'cn1': ['O1_H1', 'O1_H2', 'O1_H3'],
        'cn2': ['H1_O3', 'H2_O3', 'O3_H3'],
        'natoms': 6,
        'mu': [0.97, 0.97],   # mu for plotting CN1 & CN2
        'folder': '19_H2O2+OH_H2O+HO2.d',
    },
}
