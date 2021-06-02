import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist
import seaborn as sns
from matplotlib import rcParams


def get_closest_atoms(res1: pd.DataFrame, res2: pd.DataFrame):
    xyz1 = res1[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
    xyz2 = res2[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
    dists = cdist(xyz1, xyz2)
    print(min(dists))
    idxs = np.armin(dists)
    idx1 = idx[0]
    idx2 = idx[2]
    print(dists[idx[0], idx[1]])
    return res1.iloc[idx1], res2.iloc[idx2]

def dists_to_frame(pdb_id, dists, col_x, col_y):
    df = pd.DataFrame(dists, columns = col_x)
    return df.set_index([col_y])

def get_min_dist_table(l, section='H5', poi=('G.H5.23', 3.50), start=3.40, end=3.53, eps=0.05):
    if (start == None) or (end == None):
        start = poi[1] - eps
        end = poi[1] + eps
    if not isinstance(l, list):
        l = list(l)
    list_dists_df_list = []
    list_poi_list = []

    for i in range(len(l)):
        dists_df_list = []
        poi_list = []
        for j in range(len(l[i])):
            ex = p.dfl[l[i][j]]
            if 'gprot_pos' in ex.columns:
                pdb_id = ex['PDB'].iloc[0]
                r_ids = ex[(ex['gen_pos1'] > start) & 
                           (ex['gen_pos1'] < end)]['label_seq_id'].unique().to_list()
                g_ids = ex[(ex['gprot_pos'].str.contains(section))]['label_seq_id'].unique().to_list()
                r = ex[(ex[['label_seq_id']].isin(r_ids))][['label_seq_id', 'label_atom_id', 
                                                            'Cartn_x', 'Cartn_y', 'Cartn_z']]
                g = ex[(ex[['gprot_pos']].isin(g_ids))][['label_seq_id', 'label_atom_id', 
                                                         'Cartn_x', 'Cartn_y', 'Cartn_z']]
                
                r_idxs = r[['label_seq_id', 'label_atom_id']]
                g_idxs = g[['label_seq_id', 'label_atom_id']]
                
                r_xyz = r[['Cartn_x', 'Cartn_y', 'Cartn_z']]
                g_xyz = g[['Cartn_x', 'Cartn_y', 'Cartn_z']]
                
                dists = get_closest_atoms(r, g)
                
                
        list_dists_df_list.append(dists_df_list)
        list_poi_list.append(poi_list)
    return list_poi_list, list_dists_df_list

def get_interaction_tables(l, section='H5', poi=('G.H5.23', 3.50), start=3.40, end=3.53, eps=0.05):
    if (start == None) or (end == None):
        start = poi[1] - eps
        end = poi[1] + eps
    if not isinstance(l, list):
        l = list(l)
    list_dists_df_list = []
    list_poi_list = []

    for i in range(len(l)):
        dists_df_list = []
        poi_list = []
        for j in range(len(l[i])):
            ex = p.dfl[l[i][j]]
            if 'gprot_pos' in ex.columns:
                pdb_id = ex['PDB'].iloc[0]
                col_x = ex[(ex['gen_pos1'] > start) & 
                           (ex['gen_pos1'] < end)&
                           (ex['label_atom_id'] == 'CA')]['gen_pos1'].to_list()
                ra = ex[(ex['gen_pos1'] > start) &
                       (ex['gen_pos1'] < end) &
                       (ex['label_atom_id'] == 'CA')][['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
                col_y = ex[(ex['gprot_pos'].str.contains(section)) &
                           (ex['label_atom_id'] == 'CA')]['gprot_pos'].to_list()
                ga = ex[(ex['gprot_pos'].str.contains(section)) &
                           (ex['label_atom_id'] == 'CA')][['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
                
                dists = cdist(ra, ga).T
                
                
                dist_df = dists_to_frame(pdb_id, dists, col_x, col_y)
                if (poi[1] in col_x) & (poi[0] in col_y):
                    poi_value = dist_df.loc[poi]
                else:
                    poi_value = np.nan
                if (len(col_x) > 0) & (len(col_y) > 0):
                    dists_df_list.append((pdb_id, i, dist_df))
                if poi_value != np.nan:
                    poi_list.append((pdb_id, i, poi_value))
        list_dists_df_list.append(dists_df_list)
        list_poi_list.append(poi_list)
    return list_poi_list, list_dists_df_list


def get_cell(dataframe, row_idx, col_idx):
    if (row_idx in dataframe.index.to_list()) & (col_idx in dataframe.columns.to_list()):
        return dataframe.loc[row_idx, col_idx].astype(float)
    else:
        return None
    
    
    
def make_overview_df(dists_df_list):
    cols = []
    rows = []
    for i in range(len(dists_df_list)):
        l = dists_df_list[i]
        df = l[2]
        g_ind = df.index.to_list()
        r_ind = df.columns.to_list()
        cols = list(set(cols + r_ind))
        rows = list(set(rows + g_ind))
    occ_df = pd.DataFrame(index=rows, columns=cols)
    mean_df = pd.DataFrame(index=rows, columns=cols)
    std_df = pd.DataFrame(index=rows, columns=cols)
    for i in range(len(rows)):
        for j in range(len(cols)):
            val_list = []
            row_idx = rows[i]
            col_idx = cols[j]
            for k in range(len(dists_df_list)):
                df = dists_df_list[k][2]
                val = get_cell(df, row_idx, col_idx)
                if type(val) == pd.core.series.Series:
                    val = val.iloc[0]
                if val != None:
                    val_list.append(val)
            occ_df.loc[row_idx, col_idx] = len(val_list)
            mean_df.loc[row_idx, col_idx] = np.mean(val_list)
            std_df.loc[row_idx, col_idx] = np.std(val_list)
    occ_df = occ_df.sort_index().reindex(sorted(occ_df.columns), axis=1).astype(int)
    mean_df = mean_df.sort_index().reindex(sorted(mean_df.columns), axis=1).astype(float)
    std_df = std_df.sort_index().reindex(sorted(std_df.columns), axis=1).astype(float)
    return occ_df, mean_df, std_df


def make_overview_plots(df, title='Occurances', cl='A', gprot='Gs', figsize=(20, 15), path='plots/', show=True, save=False):
    name = title + '_' + cl + '_' + gprot
    rcParams['figure.figsize'] = 20, 15
    ax = sns.heatmap(df, cmap='RdYlGn_r', linewidths=.1, annot=True)
    ax.set_title(title + ' ' + cl + ' ' + gprot)
    if show:
        ax.plot()
    if save:
        ax.figure.savefig(path+name+'.png')
        
        
def get_overview_diff(std_df1, std_df2, mean_df1, mean_df2, ab=False, cutoff_mean=10):
    col1 = std_df1.columns.to_list()
    col2 = std_df2.columns.to_list()
    ind1 = std_df1.index.to_list()
    ind2 = std_df2.index.to_list()
    if col1 != col2:
        print("receptor gen numbers do not match")
    if ind1 != ind2:
        print("gprotein gen numbers do not match")
    col = sorted(list(set(col1+col2)))
    ind = sorted(list(set(ind1+ind2)))
    val1 = std_df1.to_numpy().astype(float)
    val2 = std_df2.to_numpy().astype(float)
    mask_m1 = mean_df1.to_numpy().astype(float) > cutoff_mean
    mask_m2 = mean_df2.to_numpy().astype(float) > cutoff_mean
    data = val1-val2
    if ab:
        data = np.abs(data)
    data[mask_m1] = np.nan
    data[mask_m2] = np.nan
    return pd.DataFrame(data=data, index=ind, columns=col)