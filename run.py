from utils import *
from gpcrdb_soup import *

import Bio
import yaml
import os
from functools import partial
from operator import is_not

# Check project file structure

# Prepare / Download GPCRdb receptor data table
"""
if os.path.isfile('data/structure.csv'):
    # print("Loading GPCRdb receptor data table...")
    df = pd.read_csv('data/structure.csv')
else:
    page = getpage()
    table = find_table(page)
    df = create_structure_df(table)
    df.to_csv('data/structure.csv')"""

page = getpage()
table = find_table(page)

# Data download

# 1) define filters
print(COLS)

# This is the thing we might want to specify with argparse
filtered = df[df['Cl.'].str.contains('A')]
filtered = filtered[filtered['Species'].str.contains('Human')]
"""
# 2)
# 2.1) find active state (complex i.p.)
filtered_complex = filtered[filtered['Family'].str.contains('Gs')]
active = filtered_complex[filtered_complex['State'].str.contains('Active')]
# 2.2) download

# 3.1) find inactive state
inactive = filtered[filtered['State'].str.contains('Inactive')]
# 3.2) download
"""


# download single structure ('refined')
"""
Refined structures
GPCRdb provides regularly updated refined structures where missing segments are modeled using the GPCRdb homology
modeling pipeline (Pándy-Szekeres et al. 2018). This entails modeling missing segments (helix ends, loops, H8),
reverting mutations to wild type and remodeling distorted regions based on our in-house manual structure curation.
The refined structures are available on the Structures (gpcrdb.org/structure) and Structure models pages
(gpcrdb.org/structure/homology_models).
"""

# download_refined_structure('7DDZ', 'data/sandbox')

# idea is to filter the df, use download script to download / load the selection


# Data loading
files, prots = getpdbsfiles()  # Get all downloaded pdb files in specified path
# print("Found {} pdb files: {}.".format(len(prots), prots))

seg_aligns = get_alignment(prots)
#  print(yaml.dump(seg_aligns, default_flow_style=False))  # yaml makes it more readable


def delta_phi_psi(phi_psi_1: list, phi_psi_2: list):
    # given two lists of phi and psi values calculate delta values (conformation change)
    return None


def create_datatable(files):
    """
    input: filenames of pdb files to analyse
    """
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
table = create_datatable(files)
table.to_csv('readout.csv')
print(table['sec_structs'])
"""