import numpy as np
import pandas as pd


ATOM_LIST = ['H', 'N', 'C', 'O', 'S']

ATOM_Z_DICT = {'H': 1,
               'C': 4,
               'N': 5,
               'O': 6,
               'F': 7,
               'S': 6,  # 16
               'Cl': 7} # 17

RES_LIST = ['A',
 'C',
 'D',
 'E',
 'F',
 'G',
 'H',
 'I',
 'K',
 'L',
 'M',
 'N',
 'P',
 'Q',
 'R',
 'S',
 'T',
 'V',
 'W',
 'Y']

RES_DICT = {}
for r, res in enumerate(RES_LIST):
    RES_DICT.update({res: r})


BOND_TYPE_DF = pd.DataFrame()

idx = 0
for i, ai in enumerate(ATOM_LIST):
    for j, aj in enumerate(ATOM_LIST):
        if i <= j:
            BOND_TYPE_DF.loc[ai, aj] = idx
            BOND_TYPE_DF.loc[aj, ai] = idx 
            idx += 1

BOND_TYPE_DF = BOND_TYPE_DF.astype(np.int8)