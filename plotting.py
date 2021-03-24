import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import time


RESIDUE_LIST = ['TRP','CYS','MET','HIS','TYR','PHE','PRO','GLN','ASN','ARG','ILE','THR','ASP','SER','VAL','GLY','LYS','ALA',\
         'GLU','LEU']
ATOM_LIST = ['CA', 'C', 'O', 'N']


def flat_l(l):
    flatten = lambda l: [item for sublist in l for item in sublist]
    return flatten(l)


def plot_len_hist(data, save=False, show=True):
    # the histogram
    plt.figure(figsize=[10,8])
    hist, bin_edges = np.histogram(np.asarray(data['n_residues']), bins=50)
    plt.bar(bin_edges[:-1], hist, width=25, color='#0504aa',alpha=0.7)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.xlabel('Sequence Lengths')
    plt.ylabel('Occurances')
    plt.title('Histogram of Protein Lengths')
    plt.grid(True)
    if show:
        plt.show()
    if save:
        plt.savefig(fname='Visualization/length_hist_'+str(time.time())+'.png')


def plot_res_hist(data, idx=None, save=False, show=True, res_list=RESIDUE_LIST):
    """
    occ: sorted list of occurances corresponding to number of times residue appeared overall in our data
    residues: sorted list of residues
    """
    # the histogram
    plt.figure(figsize=[14, 8])
    if idx is None:
        ll = flat_l(data['Res_names'].tolist())
    else:
        ll = flat_l(data.loc[idx].res)
    d=Counter(ll)
    nbins = len(res_list)
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, nbins))
    
    patches = plt.bar(d.keys(), d.values())
                          
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    plt.xticks(rotation=90)
    plt.xlabel('Residues')
                      
    plt.title('Histogram of Residue Distribution')
    if show:
        plt.show()
    if save:
        plt.savefig(fname='Visualization/residue_hist_'+str(time.time())+'.png')