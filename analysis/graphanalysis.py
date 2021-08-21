from processing.utils import *
from processing.utils2 import *
from processing.utils3 import *
from processing.gpcrdb_soup import *
from processing.download import *
from processing.processor import *
from processing.df_to_cif import *
from processing.bondtypes import *
from processing.affinities import *
from analysis.analysis import *
from neuralnet.delta_net import *
from neuralnet.egnn_sparse import *
from neuralnet.net_utils import *
from neuralnet.h5_net import *


from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from scipy.spatial.distance import cdist, pdist
from tqdm import trange
import matplotlib
from matplotlib import cm


cmap = cm.get_cmap('plasma', 21) # set how many colors you want in color map
# modify colormap
alpha = 1.5

colors = []
for ind in range(cmap.N):
    c = []
    new_colors = [max(0.5, min(.9, x * alpha)) for x in cmap(ind)]
    colors.append(new_colors)
my_cmap = matplotlib.colors.ListedColormap(colors, name = 'bright')

# initiate full atom list
FULL_ATOM_LIST = ['N',
 'CA',
 'C',
 'O',
 'CB',
 'CG',
 'SD',
 'CE',
 'OD1',
 'ND2',
 'OG1',
 'CG2',
 'CD',
 'OE1',
 'OE2',
 'CD1',
 'CD2',
 'CE1',
 'CE2',
 'CZ',
 'OH',
 'CG1',
 'OG',
 'NZ',
 'NE',
 'NH1',
 'NH2',
 'NE2',
 'NE1',
 'CE3',
 'CZ2',
 'CZ3',
 'CH2',
 'ND1',
 'OD2',
 'SG',
 'OXT']


class GraphProcessor(Dataset):
    def __init__(self,
                 d: list,
                 p: CifProcessor,
                 label_type = 'Log(Emax/EC50)',
                 verbose=0):
        self.d = d
        self.p = p
        self.allow_exception=False
        self.verbose = verbose
        
        self.simplified = False
        if 'generic_position' in list(p.dfl[0].columns):
            self.simplified=True
        
        self.atom_list = FULL_ATOM_LIST
        self.edge_criteria = ['self', 'distance']
        self.edge_features = ['unitary', 'distance', 'EM']
        self.node_criteria = ['H5']
        self.assign_labels(label_type=label_type)
    
    # ================================================================================================================
    
    def simplify(self):
        if not self.simplified:
            for i in trange(len(self.p.dfl)):
                if self.allow_exception:
                    try:
                        df = self.p.dfl[i]
                        self.p.dfl[i] = pd.DataFrame()
                        self.p.dfl[i] = self._simplify_gen_pos(df).reset_index()
                    except:
                        pass
                else:
                    df = self.p.dfl[i]
                    self.p.dfl[i] = pd.DataFrame()
                    self.p.dfl[i] = self._simplify_gen_pos(df).reset_index()
                self.simplified = True
        else:
            pass
        
    def _simplify_gen_pos(self, df):
        def sgp(gen_pos='', gprot_pos=''):
            if gprot_pos != '':
                return gprot_pos
            else:
                return gen_pos.split('x')[0]
        cols = list(df.columns)
        if ('gen_pos' in cols) and ('gprot_pos' in cols):
            df['generic_position'] = df.apply(lambda x: sgp(x.gen_pos, x.gprot_pos), axis=1)
            df.drop(['group_PDB', 'label_seq_id', 'label_asym_id',
                    'auth_seq_id', 'id',
                    'phi', 'omega', 'psi', 'label_comp_id',
                    'label_2_uni', 'gen_pos', 'gen_pos1', 'gen_pos2', 'gprot_pos',
                    'uniprot_comp_id', 'fam_comp_id', 'uniprot_comp_sid'], axis=1, inplace=True)
            return df
        elif 'gen_pos' in cols:
            df['generic_position'] = df.apply(lambda x: sgp(x.gen_pos), axis=1)
            df.drop(['group_PDB', 'label_seq_id','label_asym_id',
                   'auth_seq_id', 'id',
                   'phi', 'omega', 'psi','label_comp_id',
                   'label_2_uni', 'gen_pos', 'gen_pos1', 'gen_pos2',
                   'uniprot_comp_sid'], axis=1, inplace=True)
            return df
        else:
            df['generic_position'] = ''
            df.drop(['group_PDB', 'label_seq_id','label_asym_id',
                   'auth_seq_id', 'id','label_comp_id',
                   'phi', 'omega', 'psi'], axis=1, inplace=True)   
            return df
        
    # ================================================================================================================
    
    def assign_labels(self, label_type = 'Log(Emax/EC50)'):
        A = AffinityProcessor(setting='families')
        A.set_label_type(label_type)
        A.set_group()
        prev_len = len(self.p.table)
        if self.verbose > 0:
            print("Filtering out samples with no associated affinity value...")
        self.p = filter_valid_pdbs_with_affinities(self.p, A)
        if self.verbose > 0:
            print("Retaining {} / {} samples!".format(len(self.p.table), prev_len))
        label_df = make_label_df(self.p, A, label_type)
        self.p.table = pd.merge(self.p.table, label_df, on='PDB')
    
    # ================================================================================================================
    
    def set_atom_list(self, filtered_atom_list=['CA']):
        self.atom_list = filtered_atom_list
        if self.verbose > 0:
            print("Set atom_list to {}.".format(self.atom_list))
    
    def apply_atom_list_filter(self):
        for i in range(len(self.p.dfl)):
            self.p.dfl[i] = self.p.dfl[i][self.p.dfl[i]['label_atom_id'].isin(self.atom_list)]
        if self.verbose > 0:
            print("Filtered p by atom_list.")
        
    # ================================================================================================================
    
    def create_graph(self, 
                     filter_by_chain=True,
                     gpcr=True,
                     gprotein=True,
                     auxilary=True,
                     node_criteria='H5', 
                     edge_criteria='radius',
                     h5start=13,
                     cons_r_res=['1.50', '3.50', '7.53'],
                     radius=12,
                     max_edge_dist=7):
        if node_criteria!=None:
            self.node_criteria = node_criteria
        self.selection = []
        self.edges = []
        self.edge_attrs = []
        self.d = []
        self.dl = []
        for i in trange(len(self.p.dfl)):
            # INITIALIZATION
            df = self.p.dfl[i]
            self._select_edge_criterion()
            self._select_edge_features()
            self._select_node_criterion()
            # NODE SELECTION
            df_idxs = self._get_selection(df,
                                          filter_by_chain=filter_by_chain,
                                          gpcr=gpcr,
                                          gprotein=gprotein,
                                          auxilary=auxilary,
                                          h5start=h5start,
                                          cons_r_res=cons_r_res,
                                          radius=radius)
            df_sele = self._filter_by_index_list(df, df_idxs)
            if len(df_sele) > 0:
                grn = df_sele['generic_position']
                self.selection.append(df_sele)
                # EDGES: LIST
                edges = self._create_edge_list(df=df_sele, max_edge_dist=max_edge_dist)
                self.edges.append(edges)
                # EDGES: ATTRIBUTES
                edge_attr = self._get_edge_features(df_sele, edges, max_edge_dist)
                self.edge_attrs.append(edge_attr)
                y = self._get_label(df['PDB'].iloc[0])
                # Todo: get the node features (Z or atom-label (or both))
                x = df_sele.apply(lambda x: RES_DICT[x.label_comp_sid], axis=1).to_numpy().astype(np.int8)
                pos = df_sele[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
                self.d.append(self._get_data_object(x, edges, edge_attr, y, pos, grn))
                self.dl.append(df['PDB'].iloc[0])
                
                if self.verbose > 0:
                    print('\n\n\n')
            
            
    # ----------------------------------------------------------------------------------------------------------------
    
    def _select_edge_criterion(self, edge_criteria=[], edge_crit_dist=5):
        if not isinstance(edge_criteria, list):
            edge_criteria = [edge_criteria]
        if len(edge_criteria) > 0:
            self.edge_criteria = edge_criteria
        if self.verbose > 0:
            print("Selected edge criterion:", self.edge_criteria)
    
    def _select_edge_features(self, edge_features=[]):
        if not isinstance(edge_features, list):
            edge_features = [edge_features]
        if len(edge_features) > 0:
            self.edge_features = edge_features
        
        if self.verbose > 0:
            print("Selected edge features:", self.edge_features)
        
    def _select_node_criterion(self, node_criteria=[]):
        if not isinstance(node_criteria, list):
            node_criteria = [node_criteria]
        if len(node_criteria) > 0:
            self.node_criteria = node_criteria
        if self.verbose > 0:
            print("Selected criterion for node selection:", self.node_criteria)
              
    # ----------------------------------------------------------------------------------------------------------------
    
    def _get_selection(self, 
                       df: pd.DataFrame,
                       filter_by_chain,
                       gpcr,
                       gprotein,
                       auxilary,
                       h5start,
                       cons_r_res,
                       radius):
        # get a list of nodes (you can have multiple criterions met at the same time...)
        df.reset_index(inplace=True, drop=True)
        valid = []  # indices filtered by chain
        idxs = []  # list of selected nodes
        df_idxs = []  # list of selected nodes filtered by chain
        
        if filter_by_chain:
            
            if self.verbose > 0:
                print("Filtering by chain!")
            if gpcr:
                if self.verbose > 0:
                    print("Adding gpcr to selection!")
                a = len(valid)
                valid += self._get_gpcr(df)
                if self.verbose > 0:
                    print("Number valid gpcr atoms:", len(valid)-a)
            if gprotein:
                if self.verbose > 0:
                    print("Adding gprotein to selection!")
                a = len(valid)
                valid += self._get_gprotein(df)
                if self.verbose > 0:
                    print("Number valid gprotein atoms:", len(valid)-a)
            if auxilary:
                if self.verbose > 0:
                    print("Adding auxilary chains to selection!")
                a = len(valid)
                valid += self._get_auxilary(df)
                if self.verbose > 0:
                    print("Number valid auxilary atoms:", len(valid)-a)
        else:
            valid = list(df.index)
        valid = list(set(valid))
        if self.verbose > 0:
            print("In total using {} valid atoms!".format(len(valid)))
        
        if len(valid) >= 0:
            if 'H5' in self.node_criteria:
                if self.verbose > 0:
                    print("Searching nodes meeting H5 criterion...")
                idxs += self._get_h5(df, h5start, radius)
            if 'Interaction Surface' in self.node_criteria:
                if self.verbose > 0:
                    print("Searching nodes in the interaction surface...")
                idxs += self._get_interaction_surface(df)
            if 'Interaction Site' in self.node_criteria:
                if self.verbose > 0:
                    print("Searching nodes in the interaction site...")
                idxs += self._get_interaction_site(df, cons_r_res=['1.50', '3.50', '7.53'], radius=radius)
            for idx in idxs:
                if idx in valid:
                    df_idxs.append(idx)
            return df_idxs
        else:
            if self.verbose > 0:
                print("No valid nodes found!")
            return []
    
    def _get_gpcr(self, df):
        # list(df.index)
        some_gen_pos = ['3.50', '6.50', '7.50']
        df_sgp = df[df['generic_position'].isin(some_gen_pos)]
        if len(df_sgp) > 0:
            gpcr_chain = df_sgp.iloc[0]['auth_asym_id']
            return list(df[df['auth_asym_id'].str.contains(gpcr_chain)].index)
        else:
            return []
    
    def _get_gprotein(self, df):
        # any chain with labels that mark it as a gprotein ~ i.e. generic residue number of said chain contains 'G.'
        df_gp = df[df['generic_position'].str.contains('G.')]
        if len(df_gp) > 0:
            gprot_chain = df_gp.iloc[0]['auth_asym_id']
            return list(df[df['auth_asym_id'].str.contains(gpcr_chain)].index)
        else:
            return []
        
    def _get_auxilary(self, df):
        # any chain with labels that mark it as a gprotein ~ i.e. generic residue number of said chain contains 'G.'
        chain_df = df.groupby('auth_asym_id')['generic_position'].nunique()
        chains = list(chain_df[chain_df==1].index)
        if len(chains) > 0:
            auxilary_df = df[df['auth_asym_id'].isin(chains)]
            return list(auxilary_df.index)
        else:
            return []
    
    def _get_h5(self, df, h5start, h5radius):
        # any chain with 
        if self.verbose > 0:
            print("Checking for nodes within H5 radius:", h5radius)
        strs = ['G.H5.' + str(x) for x in range(27) if ((x >= h5start) & (x <= 26))]
        df_h5 = df[df['generic_position'].isin(strs)]
        xyz_h5 = df_h5[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
        if len(xyz_h5) > 0:
            xyz_full = df[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
            D = cdist(xyz_full, xyz_h5).T
            h5_list = []
            for p in range(D.shape[1]):
                if np.any(D[:, p] < h5radius):
                    h5_list.append(p)
            return h5_list
        else:
            return []
    
    def _get_interaction_surface(self, df, max_dist=7, n_neighbor_res=2):
        pass
    
    def _get_interaction_site(self, df, cons_r_res, radius):
        if self.verbose > 0:
            print("getting interaction site")
        int_site_corners = df[df['generic_position'].isin(cons_r_res)]
        if len(int_site_corners) == len(cons_r_res):
            xyz_corners = int_site_corners[['Cartn_x', 'Cartn_y', 'Cartn_z']].astype(float).to_numpy().mean(axis=1)
            xyz_full = df[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
            D = cdist(xyz_full, xyz_corners[None,:]).T
            h5_list = []
            for p in range(D.shape[1]):
                if np.any(D[:, p] < radius):
                    h5_list.append(p)
            return h5_list
        else:
            return []
    
    def _filter_by_index_list(self, df, df_idxs):
        if len(df_idxs) > 0:
            filtered = df.loc[df.index[df_idxs]].reset_index(drop=True)
            return filtered
        else:
            return pd.DataFrame()
    
    # ----------------------------------------------------------------------------------------------------------------
    
    def _create_edge_list(self, 
                          df: pd.DataFrame,
                          max_edge_dist=7,
                          ):
        """
        Return a list of shape (2, number_of_edges)
        """
        edge_list = []
        if 'residual' in self.edge_criteria:
            # add connection between all atoms within the same residue
            pass
        if 'EM' in self.edge_criteria:
            pass
        if 'distance' in self.edge_criteria:
            xyz = df[['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
            D = cdist(xyz, xyz, 'euclidean')
            edge_dist_list = []
            for pi in range(D.shape[0]):
                if 'self' not in self.edge_criteria:
                    for pj in range(D.shape[1]):
                        D[i, i] = max_edge_dist + 1  # this removes all self interactions
                edges = np.where(D[:, pi] <= max_edge_dist)[0].tolist()
                for e in edges:
                    edge_dist_list.append((pi, e))
            edge_list += edge_dist_list
        elif 'self' in self.edge_criteria:
            # each node interacts with itself
            idxs = list(df.index)
            self_interactions = [(x, x) for x in idxs]
            edge_list += self_interactions
        return edge_list
        
        
    def _get_em_dist(self,
                     xyz1,
                     z1,
                     xyz2,
                     z2
                    ):
        em_dists = []
        # z1 * z2 / (dist^6)
        return em_dists
        
    # ----------------------------------------------------------------------------------------------------------------
        
    def _get_edge_features(self, df, edge_list, max_edge_dist):
        # unitary? --> 1 (for every edge)
        edge_features = []
        for _, (i, j) in enumerate(edge_list):
            features = []
            xyzi = df.iloc[i][['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
            xyzj = df.iloc[j][['Cartn_x', 'Cartn_y', 'Cartn_z']].to_numpy().astype(float)
            if 'unitary' in self.edge_features:
                features += [1]
            # distance? --> (euclidean distance)
            if 'distance' in self.edge_features:
                dist = np.linalg.norm(xyzi-xyzj)
                features += [dist / max_edge_dist]
            # EM?  --> z1 * z2 / r^6
            if 'em' in self.edge_features:
                # xyzi = df.iloc[i][]
                # xyzj = df.iloc[j][]
                features +=  [0]
            # atom-atom-interaction as a class (that we can embed)?
            if 'bond_embedding' in self.edge_features:
                features += [0]
            edge_features.append(features)
        return edge_features
              
    # ----------------------------------------------------------------------------------------------------------------

    def _get_label(self, pdb_id):
        return self.p.table[self.p.table['PDB']==pdb_id].iloc[0][['Gs','Gi/o','Gq/11','G12/13']]
              
    # ----------------------------------------------------------------------------------------------------------------
    
    def __getitem__(self, idx):
        return self.d[idx]
    
    def _get_data_object(self, x, edge_index, edge_attr, y, pos, grn):
        """
        Input:
        data.x: Node feature matrix with shape [num_nodes, num_node_features]
        data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
        data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
        data.pos: Node position matrix with shape [num_nodes, num_dimensions]
        """
        x = torch.tensor(x, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        pos = torch.tensor(pos, dtype=torch.float)
        return Data(x=x, edge_index=edge_index.T, edge_attr=edge_attr, y=y, pos=pos, grn=grn)
    
    
    def __getitem__(self, idx):
        return self.d[idx]
    
    def __len__(self):
        return len(self.d)
    
    # ================================================================================================================
    
    
    
def get_edge_vecs(pos, edges):
    Xe=[]
    Ye=[]
    Ze=[]
    for e in edges:
        if e[0] != e[1]:
            Xe += [pos[e[0]][0], pos[e[1]][0], None]
            Ye += [pos[e[0]][1], pos[e[1]][1], None]
            Ze += [pos[e[0]][2], pos[e[1]][2], None]
    return Xe, Ye, Ze


def plot_graph(gp, idx, cmap, weights=None, showgrid=False, title='graph'):
    data = gp.d[idx]
    selection = gp.selection[idx]
    points = data.pos.numpy()
    edges = data.edge_index.numpy().T
    Xe, Ye, Ze = get_edge_vecs(points, edges)
    if weights==None:
        weights=selection['generic_position'].astype(float)
        
    E = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        line=dict(color='rgb(125,125,125)', width=3),
        hoverinfo='none',
        name='Edges',
        )
    
    P = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        text=selection['generic_position'].astype(float),
        hoverinfo='text',
        mode='markers',
        name=selection['PDB'].iloc[0],
        marker={
            'size': 7,
            'opacity': 0.8,
            'color': weights,
        }
    )
    
    
    
    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    )
    data = [E, P]
    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.update_traces(textposition='top center')
    plot_figure.update_layout(scene = dict(
                                xaxis = dict(
                                     backgroundcolor="rgb(255, 255, 255)",
                                     gridcolor="white",
                                     showbackground=showgrid,
                                     zerolinecolor="white",),
                                yaxis = dict(
                                    backgroundcolor="rgb(255, 255, 255)",
                                    gridcolor="white",
                                    showbackground=showgrid,
                                    zerolinecolor="white"),
                                zaxis = dict(
                                    backgroundcolor="rgb(255, 255, 255)",
                                    gridcolor="white",
                                    showbackground=showgrid,
                                    zerolinecolor="white",),),
                              )
    plot_figure.show()
    # plot_figure.write_image("plots/mt/"+title+".png")