from utils import *
from gpcrdb_soup import *

import Bio
import yaml
import os
from functools import partial
from operator import is_not

# Check project file structure

# Prepare / Download GPCRdb receptor data table

uniprot = False  # specify if we want to scrape for uniprot links ~

# table = get_table(reload=True, save=True, uniprot=uniprot)

table = get_table(reload=False)

df = table

# This is the thing we might want to specify with argparse
filtered = df[df['Cl.'].str.contains('A')]
filtered = filtered[filtered['Species'].str.contains('Human')]

# 2.1) find active state (complex i.p.)
filtered_complex = filtered[filtered['Family'].str.contains('Gs')]
active = filtered_complex[filtered_complex['State'].str.contains('Active')]

# Data loading
files_a, prots_a = get_pdb_files(path='data/pdb/active')  # Get all downloaded pdb files in specified path
print("Found {} proteins: {}.".format(len(prots_a), prots_a))

# find inactive counter parts of the active proteins ==> use gene/uniprot, Family and species to match them

genes = list(set(active['uniprot(gene)'].values.tolist()))
print(genes)

inactive = filtered[(filtered['uniprot(gene)'].isin(genes))
                    & (filtered['State'].str.contains('Inactive'))
                    & (filtered['Species'].str.contains('Human'))]

# Data loading
files_i, prots_i = get_pdb_files(path='data/pdb/inactive')  # Get all downloaded pdb files in specified path
print("Found {} proteins: {}.".format(len(prots_i), prots_i))


prots = [*prots_i, *prots_a]
print(len(prots))


seg_aligns = get_alignment(prots_a)

print(yaml.dump(seg_aligns, default_flow_style=False))
"""
# Data loading
files, prots = getpdbsfiles()  # Get all downloaded pdb files in specified path
# print("Found {} pdb files: {}.".format(len(prots), prots))

seg_aligns = get_alignment(prots)
#  print(yaml.dump(seg_aligns, default_flow_style=False))  # yaml makes it more readable


def delta_phi_psi(phi_psi_1: list, phi_psi_2: list):
    # given two lists of phi and psi values calculate delta values (conformation change)
    return None


def create_datatable(files):
    parser = PDBParser()
    struct_lists = []
    for i, f in enumerate(files):
        name = f[12:23]
        print("Getting structure for", name)
        structure = parser.get_structure(id=name, file=f)
        for model in structure:
            for chain in model:
                c = chain
        del structure

        seq = get_seq(c)
        sec_structs = get_sec_structs(name, seq, seg_aligns)

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
        seq = list(filter(partial(is_not, None), seq))
        data = [[name, len(seq), seq, pp_ids, pp_lens, pp_seqs, psi_phi, sec_structs.values]]
        df = pd.DataFrame(data=data, columns=['prot_id', 'prot_len', 'prot_seq', 'pp_ids', 'pp_lens','pp_seqs','psi_phi', \
            'sec_structs'])
        struct_lists.append(df)
    return pd.concat(struct_lists, ignore_index=True)
"""
"""
table = create_datatable(files)
table.to_csv('readout.csv')
print(table['sec_structs'])
"""