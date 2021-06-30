from processing.processor import *
from processing.utils3 import *
from processing.utils2 import *
import pandas as pd
import os
import sys
import functools
import operator
import random
from tqdm import tqdm, trange
import time
import numpy as np
import pandas
from math import degrees
import requests as r
from Bio import SeqIO, pairwise2
from io import StringIO
import gemmi
from gemmi import cif


PATH = 'data/'
STRUCTURE_PATH = PATH + 'mmcif/'


ALL_COLS = ['group_PDB' ,
'id' ,
'type_symbol' ,
'label_atom_id' ,
'label_alt_id' ,
'label_comp_id' ,
'label_asym_id' ,
'label_entity_id' ,
'label_seq_id' ,
'pdbx_PDB_ins_code' ,
'Cartn_x' ,
'Cartn_y' ,
'Cartn_z' ,
'occupancy' ,
'B_iso_or_equiv' ,
'pdbx_formal_charge' ,
'auth_seq_id' ,
'auth_comp_id' ,
'auth_asym_id' ,
'auth_atom_id' ,
'pdbx_PDB_model_num' ]


def load_cifs(pdb_id):
    path = 'data/mmcif/' + pdb_id + '.cif'
    try:
        doc = cif.read_file(path)  # copy all the data from mmCIF file
        lol = []  # list of lists
        for b, block in enumerate(doc):
            table = block.find('_atom_site.', COLS)
            for row in table:
                lol.append([pdb_id]+list(row))
    except Exception as e:
        print("Hoppla. %s" % e)
        sys.exit(1)
    cols = ['PDB']+COLS
    st = gemmi.read_structure(path)
    model = st[0]
    if len(st) > 1:
        print("There are multiple models!")
    rol = []
    for chain in model:
        for r, res in enumerate(chain.get_polymer()):
            # previous_residue() and next_residue() return previous/next
            # residue only if the residues are bonded. Otherwise -- None.
            prev_res = chain.previous_residue(res)
            next_res = chain.next_residue(res)
            try:
                phi, psi = gemmi.calculate_phi_psi(prev_res, res, next_res)
            except:
                phi, psi = np.nan, np.nan
            try:
                omega = gemmi.calculate_omega(res, next_res)
            except:
                omega = np.nan
            rol.append([res.label_seq, res.subchain, 
                        degrees(phi), degrees(omega), degrees(psi)])
    cols2 = ['label_seq_id', 'label_asym_id', 'phi', 'omega', 'psi']
    rol_df = pd.DataFrame(data=rol, columns=cols2)
    rol_df['label_seq_id'] = rol_df['label_seq_id'].astype(int)
    lol_df = pd.DataFrame(data=lol, columns=cols)
    lol_df['label_seq_id'] = lol_df.apply(lambda x: int(x.label_seq_id) if x.label_seq_id != '.' else np.nan, axis=1)
    return pd.merge(lol_df, rol_df, how='inner', on=['label_asym_id', 'label_seq_id'])


def spread_gen_pos_to_residue_atoms(structure):
    try:
        if 'gprot_pos' in list(structure.columns):
            structure['generic_position'] = structure.apply(lambda x: x.gen_pos if x.gprot_pos == '' else x.gprot_pos, axis=1)
            c_gprot = list(structure[structure['gprot_pos'] != '']['label_asym_id'])
        else:
            structure['generic_position'] = structure['gen_pos']
            c_gprot = []
        residue_seq_ids = list(set(list(structure['label_seq_id'])))
        chains = list(set(list(structure['label_asym_id'])))
        # for all chains
        for c in tqdm(chains):
            c_rec = list(structure[structure['gen_pos'] != '']['label_asym_id'])
            check_gen_pos_chains = list(set(c_rec+c_gprot))
            # check if chain has residues with a generic position number
            if c in check_gen_pos_chains:
                # for all residues
                for rsi in tqdm(residue_seq_ids):
                    block_gen_pos_ = structure[(structure['label_seq_id']==rsi) &
                                               (structure['label_asym_id']==c) &
                                               (structure['generic_position']!='')]
                    if len(block_gen_pos_)>0: 
                        block_idxs = list(structure[(structure['label_seq_id']==rsi) &
                                                    (structure['label_asym_id']==c)].index)
                        block_gen_pos = block_gen_pos_['generic_position'].iloc[0]
                        # for all indexes within that residue
                        for idx in block_idxs:
                            structure.loc[idx, 'generic_position'] = block_gen_pos
        return structure
    except:
        print("Exception found for:", structure['PDB'].iloc[0])
        
        
def write_cif_with_gen_seq_numbers(structure, replace_idx=-4):
    pdb_id = structure['PDB'].iloc[0]
    path = 'data/mmcif/' + pdb_id + '.cif'
    doc = cif.read_file(path)  # copy all the data from mmCIF file
    lol = []  # list of lists
    for b, block in enumerate(doc):
        col_overwrite = block.find('_atom_site.', ['id'])
        table = block.find('_atom_site.', ALL_COLS)
        for r, row in enumerate(table):
            if row[1] in list(structure['id']):
                gen_pos = structure[structure['id']==row[1]]['generic_position'].iloc[0]
                if gen_pos == '':
                    gen_pos = '-'
                row[replace_idx] = gen_pos
            else:
                row[replace_idx] = '-'
    return doc