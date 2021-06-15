# from deprecated.gpcrdb_soup import *
from tqdm import tqdm
import numpy as np
import pandas as pd
# Import pairwise2 module
from Bio import pairwise2
# Import format_alignment method
import ast



def find_sections(nums, min_length):
    nums.sort()
    nums.append(1e9)
    ans=[]
    l=nums[0]
    for i in range(1,len(nums)):
        if nums[i] != nums[i-1] + 1:
            if (nums[i-1] - l) >= min_length:
                ans.append((l, nums[i-1]))
            l=nums[i]
    return ans



def clean_alignment(al_df):
    print("cleaning alignment")
    for i, c in enumerate(al_df):
        if i == 0:
            al_df = al_df.rename(columns={c: 'ID'})
        elif c.find('Unnamed') != -1:
            al_df = al_df.rename(columns={c:seg})
        else:
            seg = c
    positions = [x for x in list(al_df.iloc[0].values) if '.' in str(x)]
    al_df['clean_id'] = al_df.ID.apply(lambda x: str(x).replace('[Human] ', '').replace('&amp', '').replace(';',''))
    al_df['TM7_combined'] = al_df.TM7.agg(''.join, axis=1).apply(lambda x: x.replace('-','_'))
    al_df['TM7_clean'] = al_df.TM7_combined.apply(lambda x: x.replace('_',''))
    al_df['H8_combined'] = al_df.H8.agg(''.join, axis=1).apply(lambda x: x.replace('-','_'))
    al_df['H8_clean'] = al_df.H8_combined.apply(lambda x: x.replace('_',''))
    columns = list(set(list(al_df.columns)))
    del_cols = list(filter(lambda x: (('combined' not in x) and ('clean' not in x) and ('ID' not in x)), columns))
    for cols in del_cols:
        del al_df[cols]
    al_df = al_df.drop(0)
    al_df['roi_pos'] = al_df.apply(lambda x: [z[0] for z in list(zip(positions, x.TM7_combined+x.H8_combined)) if not '_' in z[1]], axis=1)
    al_df['roi_seq'] = al_df.apply(lambda x: x.TM7_clean+x.H8_clean, axis=1)
    return al_df, positions


def pdb_data(files, path='data/pdb/active/'):
    parser = PDBParser()
    struct_lists = []
    for i, f in enumerate(tqdm(files)):
        name = f[-8:-4]
        structure = parser.get_structure(id=name, file=f)
        for m, model in enumerate(structure):
            for c, chain in enumerate(model):
                if m + c == 0:
                    first_chain = chain
        del structure
        seq = get_seq(first_chain)
        polypeptides = get_phi_psi_list(first_chain)
        pp_ids = []
        pp_lens = []
        pp_seqs = []
        psi_phi = []
        # Using the C-N distance(?) we separate the AA-sequence into polypeptides (important when looking at the complexes)
        for p, pp in enumerate(polypeptides):
            pp_ids.append(p)
            pp_lens.append(pp[0])
            pp_seqs.append(str(pp[1]))
            psi_phi.append(np.asarray(pp[2]))        
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
    full['uniprotid'] = full.apply(lambda x: pdbtouniprot(x.PDB), axis=1)
    full['uniprot_seq'] = full.apply(lambda x: get_uniprot_seq(x.uniprotid), axis=1)
    # pp position information
    # return full table
    return full


def align_seg_to_seq(z, padding=None, padding_l=None, padding_r=5):
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
    if padding != None:
        start = max(0, start-padding)
        end = min(len(X), end+padding)
    if padding_l != None:
        start = max(0, start-padding_l)
    if padding_r != None:
        end = min(len(X), end+padding_r)
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
    return full.merge(a_df, how='inner', left_on='PDB', right_on='PDB')  # ignore_index=True


def complete_data(full: pd.DataFrame, 
                  max_std_alignment=None, elongate=True, padding_r=5, target='NPFIY', filter_bad_checks=False):
    # get alignment: https://towardsdatascience.com/pairwise-sequence-alignment-using-biopython-d1a9d0ba861f
    complete = get_align_dict(full)
    # filter by maximum alignment standard deviation (basically if it is wrong)
    if max_std_alignment!=None:
        complete = complete[complete['std'] < max_std_alignment]
    # replace the TM7 with an elongated version
    if elongate:
        complete['TM7_found'] = complete.apply(lambda x: x.full_prot_seq[x.start:x.end+padding_r], axis=1)
        if filter_bad_checks:
            max_diff = 15  # maximum difference in sequence lengths between detected and true TM7 region
            complete = complete[complete['TM7_combined'].map(len) + max_diff - complete['TM7_found'].map(len) >= 0]
    # extract target sequence from the TM7
    complete['target_wrt_tm7'] = complete.apply(lambda x: align_seg_to_seq([x.TM7_combined, target, x.PDB]), axis=1)
    return complete


def get_target_df(complete: pd.DataFrame, target='NPXXY', valid=True):
    def sum_start(a, b):
        return a + b[0]
    def sum_end(a, b):
        return a + b[1]
    def extend(ls):
        is_list = isinstance(ls[0], list)
        for i, l in enumerate(ls):
            if is_list & i == 0:
                out = l
            elif is_list:
                out.extend(l)
            else:
                return ls
        return out
    def clean_pps(x):
        return ast.literal_eval(x.replace('Seq', '').replace('(', '').replace(')', ''))
    target_df = complete[['PDB', 'uniprot(gene)', 'Resolution', 'PDB date', \
                          'TM7_found', 'score', 'prot_len', 'full_prot_seq']].copy()
    target_df['pp_seqs'] = complete.apply(lambda x: clean_pps(x.pp_seqs), axis=1).copy()
    target_df['pp_seq_lens'] = target_df.apply(lambda x: [len(y) for y in x.pp_seqs], axis=1).copy()
    target_df['start'] = complete['target_wrt_tm7'].apply(lambda x: x[0]).copy()
    target_df['end'] = complete['target_wrt_tm7'].apply(lambda x: x[1]).copy()
    # target_df.loc[:, 'target_seq'] = target
    target_df['start_absolute'] = complete.apply(lambda x: sum_start(x.start, x.target_wrt_tm7), axis=1).copy()
    target_df['end_absolute'] = complete.apply(lambda x: sum_end(x.start, x.target_wrt_tm7), axis=1).copy()
    target_df['full_aligned_seg'] = target_df.apply(lambda x: x.full_prot_seq[x.start_absolute:x.end_absolute], axis=1)
    target_df['TM7_aligned_seg'] = target_df.apply(lambda x: x.TM7_found[x.start:x.end], axis=1)
    target_df['psi_phi'] = complete.apply(lambda x: extend(ast.literal_eval(x.psi_phi)), axis=1)
    if valid:
        target_df['target_angles'] = target_df['psi_phi'][target_df['start_absolute']:target_df['end_absolute']]
        return target_df[target_df['full_aligned_seg']==target_df['TM7_aligned_seg']]
    else:
        return target_df
        