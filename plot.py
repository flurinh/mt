import pandas as pd
import matplotlib.pyplot as plt
import math


RESIDUE_LIST = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'Z']


def plot_ramachandran(angles, name='', show=True, save=False):
    angles=angles*180/math.pi
    plt.figure(figsize=[14, 8])
    plt.scatter(angles[:,0], angles[:,1])
    plt.ylabel('psi')
    plt.xlabel('phi')
    plt.title('psi-phi plot: '+name)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    if show:
        plt.show()
    if save:
        path = 'plots/'+name+'.png'
        plt.savefig(path)
        
        
def make_r_plots(valid_, active='', show=True, save=False):
    for _ in range(len(valid_)):
        data = valid_.iloc[_]
        name = data['uniprot(gene)']+'_'+active
        angles = np.asarray(valid_['target_angles'].iloc[_])
        if len(angles) > 1:
            plot_ramachandran(angles, name, show=show, save=save)

            
def flat_l(l):
    flatten = lambda l: [item for sublist in l for item in sublist]
    return flatten(l)

            
def plot_res_hist(data, idx=None, save=False, show=True, res_list=RESIDUE_LIST):
    """
    occ: sorted list of occurances corresponding to number of times residue appeared overall in our data
    residues: sorted list of residues
    """
    # the histogram
    plt.figure(figsize=[14, 8])
    if idx is None:
        ll = flat_l(data['res_name'].tolist())
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