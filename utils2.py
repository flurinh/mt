from utils import *
from gpcrdb_soup import *
import Bio
import yaml
import os
from functools import partial
from operator import is_not
from tqdm import tqdm
import numpy as np
# Import pairwise2 module
from Bio import pairwise2
# Import format_alignment method
from Bio.pairwise2 import format_alignment


def clean_alignment(al_df):
    for i, c in enumerate(al_df):
        if i == 0:
            al_df = al_df.rename(columns={c: 'ID'})
        elif c.find('Unnamed') != -1:
            al_df = al_df.rename(columns={c:seg})
        else:
            seg = c
    positions = al_df.iloc[0].values
    al_df['clean_id'] = al_df.ID.apply(lambda x: str(x).replace('[Human] ', '').replace('&amp', '').replace(';',''))
    al_df['TM7_combined'] = al_df.TM7.agg(''.join, axis=1).apply(lambda x: x.replace('-','').replace('_',''))
    al_df['H8_combined'] = al_df.H8.agg(''.join, axis=1).apply(lambda x: x.replace('-','').replace('_',''))
    columns = list(set(list(al_df.columns)))
    del_cols = list(filter(lambda x: (('combined' not in x) and ('clean' not in x) and ('ID' not in x)), columns))
    for cols in del_cols:
        del al_df[cols]
    al_df = al_df.drop(0)
    return al_df


def pdb_data(files, path='data/pdb/active/'):
    parser = PDBParser()
    struct_lists = []
    for i, f in enumerate(tqdm(files)):
        name = f[-8:-4]
        structure = parser.get_structure(id=name, file=f)
        for model in structure:
            for chain in model:
                c = chain
        del structure
        seq = get_seq(c)
        polypeptides = get_phi_psi_list(c)
        pp_ids = []
        pp_lens = []
        pp_seqs = []
        psi_phi = []
        # Using the C-N distance(?) we separate the AA-sequence into polypeptides (important when looking at the complexes)
        for p, pp in enumerate(polypeptides):
            pp_ids.append(p)
            pp_lens.append(pp[0])
            pp_seqs.append(pp[1])
            psi_phi.append(pp[2])        
            data = [[name, len(seq), seq, pp_ids, pp_lens, pp_seqs, psi_phi]]
        df = pd.DataFrame(data=data, columns=['prot_id', 'prot_len', 'prot_seq', 'pp_ids', 'pp_lens','pp_seqs','psi_phi'])
        struct_lists.append(df)
    return pd.concat(struct_lists, ignore_index=True)


def structure_to_full(table: pd.DataFrame, structure: pd.DataFrame, alignments: pd.DataFrame):
    # missing segment information
    full = table.merge(alignments, how='inner', left_on='pdb_lower', right_on='clean_id')
    # pdb position information
    full = full.merge(structure, how='inner', left_on='PDB', right_on='prot_id')
    full['full_prot_seq'] = full['prot_seq'].apply(lambda x: [y for y in x if y is not None]).str.join('')
    try:
        del full['Unnamed: 0']
        del full['filler']
        del full['filler2']
    except:
        pass
    # pp position information
    # return full table
    return full


def align_seg_to_seq(z):
    X = z[0]
    Y = z[1]
    pdb = z[2]
    a_ = pairwise2.align.globalms(X, Y, 3, -.5, -.1, -0.1)[0]
    score = a_.score
    score /= len(Y)  # Is this a balanced representation?
    matching = [0 if (a_.seqB[i] == '-') else 1 for i in range(len(a_.seqB))]
    # get mean
    res_id = [idx for idx, val in enumerate(matching) if val != 0]
    res_arr = np.asarray(res_id)
    mean = np.mean(res_arr)
    # get std
    std = np.std(res_arr)
    # get start
    start = matching.index(1)
    # get end
    end = len(matching) - matching[::-1].index(1)
    return start, end, mean, std, score, res_id, pdb


def get_align_dict(full: pd.DataFrame):
    l_seq = list(full['full_prot_seq'])
    l_seg = list(full['TM7_combined'])
    pdb = list(full['PDB'])
    cols = ['start', 'end', 'mean', 'std', 'score', 'res', 'PDB']
    l = []
    for z in zip(l_seq, l_seg, pdb):
        values = align_seg_to_seq(z)
        zipped = zip(cols, values)
        a_s = dict(zipped)
        l.append(a_s)
    a_df = pd.DataFrame(columns=cols)
    a_df = a_df.append(l, True)
    return full.merge(a_df, how='inner', left_on='PDB', right_on='PDB')
        
"""
def disp3(df: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        display(df.head(3))
"""