import torch
import torch.nn.functional as f
import math
import numpy as np
import pandas as pd
import json
import csv
from random import randint, seed
import plotly.graph_objects as go

from utils import *
from Processing.loader import *
from Processing.distribution import *
# from rendering import *


torch.pi = torch.Tensor([3.14159265358979323846])
# https://docs.databricks.com/_static/notebooks/deep-learning/pytorch-images.html


# add jit to make this faster --> many if statements !
# @torch.jit.script
class ChemEnv(torch.nn.Module):  # gym.Env
    """
    Custom Environment -> Serves functionality to manipulate 3D protein structure.
    We provide both functionality to serve folded or unfolded states to allow for both optimization
    of folding and unfolding. (Unfolding the protein might serve as a methodology to generate training data.)
    """
    metadata = {'render.modes': ['static', 'rotating'],
                'render.representation': ['aa', 'het', 'amidplanes', 'sticks']}

    def __init__(self,
                 mode='fold',
                 path='data/',
                 n_actions=3,
                 n_players=2,
                 limit = None,
                 setting=3,
                 resolution=(1280, 720),
                 device='cuda',
                 verbose=0,
                 input_list=None,
                 ):
        super(ChemEnv, self).__init__()
        self.mode = mode
        self.verbose = verbose
        
        # Filter settings
        self.setting_explained = {0: "Only C-alphas - labelled according to their residue",
                         1: "Hetero-representation - Backbone atoms are labelled according to their element,"\
                             " except C-alpha, which is labelled according to its AA",
                         2: "Hetero-representation - Backbone atoms are labelled according to their element,"\
                             "C-beta is labelled according to its AA",
                         3: "Hetero-representation 3, we include H atoms",
                         4: "Hetero-representation 4, we include H atoms",
                         5: "Full Protein is represented as element-wise pointcloud "\
                             "(drastically reduces embedding space)"}
        
        self.settings = {0: ['pad'+'CA'],
                         1: ['pad']+ATOM_LIST,
                         2: ['pad']+ATOM_LIST+['CB'],
                         3: ['pad']+ATOM_LIST+['HN'],
                         4: ['pad']+ATOM_LIST+['CB']+['HN'],
                         5: None}
        self.setting = setting
        self.label_dict = None
        self.rev_label_dict = None
        self.load_label_dict()
        self.atom_list=list(set(self.settings[setting]))
        self.residue_list = RESIDUE_LIST
        self.atoms_per_plane = len(self.atom_list) - 1
        
        # fix mode --> observation either is target, to pretrain / see if it can predict angles
        # if mode is fixed we give it the folded protein and tell it to predict correct angles
        # i.e. it should predict current angles
        
        # define Input --> we can use residues only, or go by atom_list etc (based on setting)
        
        # define action and observation space
        self.n_players = n_players
        self.n_actions = n_actions
        
        # indices
        self.limit = limit
        self.prot_idx = 0
        self.t = 0  # t is the current centroid residue index
        self.ts = None
        self.res_indices = None
        self.atom_indices = None  # how do we rotate if we can only index by 
        
        # loader
        self.max_natoms = None
        self.max_nres = None
        self.dataset = None
        self.init_dataset()
        self.loader = None
        self.init_loader()
        
        # constants
        self.dcaca = 3.793  # distance between two c-alphas in common trans-amid-plane
        self.psi = 20.819  # angle between Ca-CO-bond and amid-plane diagonal
        self.phi = 14.56 + 180  # + 180 because it is the angle between Ca and N (thus reverse direction)
        self.omega = 0  # 180
        
        self.dcaca_cis = 2.907
        self.phicis = 360 + (-30.35)
        self.thetacis = 180 + 31.754
        
        self.ca = [0.000,0.000,0.000]
        self.cb = [0.314, -1.477, 0.000]
        self.c = [1.411, 0.537, 0.000]
        self.o = [1.620, 1.759, 0.000]
        self.n = [self.dcaca-1.413, -0.407, 0.000]
        self.h = [self.dcaca-1.533, -1.399, 0.000]
        
        # Masking and Mapping
        self.atom_mask = None
        self.res_mask = None
        self.atom_dict = None
        
        
        # unfolded state
        self.pos = None  # ---> corresponds to our observable
        self.rotvecs = None  # ---> rotation axes for all residues in form of their chemical-bond-xyz-direction
        self.dists = None
        
        # folded state
        self.tpos = None  # ---> corresponds to our observable for unfolding or our 
        self.trotvecs = None  # ---> rotation axes for all residues in form of their chemical-bond-xyz-direction
        self.tdists = None
        
        # Storing data for use in plotting or as memory
        csvfile_data = path + 'data_' + str(limit) + '.csv'
        csvfile_meta = path + 'meta_' + str(limit) + '.csv'
        
        # plotting
        self.rgb = np.zeros(resolution)
        self.fig = None
        
        # device
        self.device = device
        self.to(torch.device(self.device))
    
    def mode(self, mode):
        """
        mode: specify environment state, either 'fold' or 'unfold'
        """
        self.mode = mode
        print("ChemEnv set to {}ing mode. Now serving {} to model."
              .format(mode, 'self.tpos' if mode == 'unfold' else 'self.pos'))
        
    def init_dataset(self):
        self.dataset = DataSet(limit=self.limit, setting=self.setting)
        self.max_natoms = self.dataset.max_natoms
        self.max_nres = self.dataset.max_nres

    def init_loader(self):
        self.loader = iter(DataLoader(self.dataset, drop_last=True, batch_size = self.n_players, shuffle=True))
        
    # =======================================================================================================
    
    def load_data(self, filename):
        # csv saves lists as strings
        from ast import literal_eval as eval
        # fix our numpy arrays
        def from_np_array(array_string):
            try:
                array_string = ','.join(array_string.replace('[ ', '[').split())
                array_string = array_string.replace('[,', '[').replace(',]', ']')
                return np.array(eval(array_string))
            except:
                return np.NaN  # this detects proteins with incorrect format (no cristal structure)
        assert '.csv' in filename, print("Expected input data format is CSV (.csv)")
        self.dataset = pd.read_csv(filename, converters={'pos': from_np_array, 'res': eval, 'y':from_np_array},\
                                   index_col=0).loc[:self.limit]
        if self.verbose > 0:
            print("Loaded data from {}. Total number of proteins: {}.".format(filename, self.dataset.count().len))
    
    def load_label_dict(self):
        with open('data/label_dict_'+str(self.setting)+'.json', 'r') as fp:
            self.label_dict = json.load(fp)
        self.label_dict.update({'pad':-1})
        self.label_dict = {k: v+1 for k, v in sorted(self.label_dict.items(), key=lambda item: item[1])}
        self.rev_label_dict = dict((v,k) for k,v in self.label_dict.items())
                
    # =======================================================================================================
    
    def reset(self):
        """
        Initialize new environments.
        """
        try:
            batch = next(self.loader)
        except StopIteration:
            self.loader = iter(DataLoader(self.dataset, drop_last=True, batch_size = self.n_players, shuffle=True))
            batch = next(self.loader)
        
        
        # Indices
        self.s = 0  # residue index
        self.t = 0  # atom index
        self.ts = torch.zeros(self.n_players)  # pivot indices  ==>  goal but not yet supported
        self.atom_indices = batch[6][:,0]
        self.res_indices = batch[6][:,1]
        self.max_atom_len = max(self.atom_indices)
        self.max_res_len = max(self.res_indices) + 1  # incorrect values (according to count by atom_indices)
        
        self.names = batch[0]
        if self.verbose > 0:
            print("Folding {} proteins.".format(len(self.names)))
            print("Maximum number residues in batch is:", self.max_res_len)
            print("Maximum number atoms in batch is:", self.max_atom_len)
        
        # create masks
        self.atom_mask = self.create_mask(self.atom_indices, (self.n_players, self.max_atom_len))
        self.res_mask = self.create_mask(self.res_indices, (self.n_players, self.max_res_len))
        
        # Atoms
        self.ya = batch[1][:,:self.max_atom_len]
        self.tpos = batch[2][:,:self.max_atom_len]
        
        # Residues
        self.yr = None
        self._init_res_labels()
        self.trotangles = [batch[3][:, :self.max_res_len],
                           batch[4][:, :self.max_res_len],
                           batch[5][:, :self.max_res_len]]
        
        # Initialize labels (proteins and corresponding label list)
        self.atom_label_list = None
        self.res_label_list = None
        self._init_label_lists()
        
        # target
        # Why not directly use distance matrix between pos and tpos? Because relative distances matter
        self.tdists = torch.cdist(self.tpos, self.tpos, p=2)
        
        # initialization
        self._init_aminplane()
        self._init_pos()
        self._init_rotvecs()
        
        assert self.pos.shape == self.tpos.shape, print("Current position and target tensor shape" \
            "should be the same, but are ==> current state:{} and target:{}".format(self.pos.shape,
                                                                                    self.tpos.shape))
        
        self.dists = torch.cdist(self.pos, self.pos, p=2)
        self.center = self.pos[:,self.t]

    def step(self, next_state):
        # lets define our action space as [phi, theta] where phi in [-1, 1] and theta in [-1, 1]
        self._take_action(next_state)
        self.dists = torch.cdist(self.pos[0], self.pos[0], p=2)
        obs = self.pos, self.ya
        reward = torch.einsum('pij->p',self.tdists-self.dists)
        done = False  # sum over dists(masked /w )
        info = "Current data:\npos     --> shape:{} | type:{}".format(self.pos.shape, self.pos.dtype)+\
            "\nrotvecs --> shape:{} | type:{}".format(self.rotvecs.shape, self.rotvecs.dtype)+\
            "\nya       --> shape:{} | type:{}".format(self.ya.shape, self.ya.dtype)+\
            "\nyr       --> shape:{} | type:{}".format(self.yr.shape, self.yr.dtype)
        self.s += 1
        self.t += self.atoms_per_plane
        self.ts += self.atoms_per_plane
        if self.s == self.max_res_len:
            self.s = 0
        if self.t >= self.max_atom_len:
            self.t = 0
        self.ts[self.ts >= self.max_atom_len] = 0
        return obs, reward, done, info
        
    # =======================================================================================================
    
    def masking(self, a, mask):
        return torch.einsum('ij,ij->ij', a, mask.to(torch.bool))
    
    def create_mask(self, indices, mask_shape):
        mask = torch.zeros(mask_shape, dtype=torch.uint8)
        for i in range(indices.shape[0]):
            mask[i,:indices[i]] +=1
        return mask
           
    # =======================================================================================================
    
    def _init_res_labels(self):
        indices = torch.LongTensor([1 if ((i - 1) % 5 == 0) else 0 \
                                    for i in range(self.max_atom_len)])\
                                    .repeat(self.n_players, 1).bool()
        self.yr = self.ya[indices].reshape(self.n_players, -1)
    
    def _init_label_lists(self):
        # y is a double/nested list of atoms
        l = {}
        for _, l1 in enumerate(ce.ya.data.tolist()):
            l_ = []
            for l2 in l1:
                l_.append(self.rev_label_dict[l2])
            l.update({self.names[_]:l_})
        self.atom_label_list = l
        # y is a double/nested list of residues
        l = {}
        for _, l1 in enumerate(ce.yr.data.tolist()):
            l_ = []
            for l2 in l1:
                l_.append(self.rev_label_dict[l2])
            l.update({self.names[_]:l_})
        self.res_label_list = l
        
    # =======================================================================================================
    
    def _init_aminplane(self):
        plane = []
        # we initiate the protein on the x-axis (all acs spaced equally with self.dcaca)
        if 'CA' in self.atom_list:
            plane.append(self.ca)
        if 'CB' in self.atom_list:
            plane.append(self.cb)
        if 'C' in self.atom_list:
            plane.append(self.c)
        if 'O' in self.atom_list:
            plane.append(self.o)
        if 'N' in self.atom_list:
            plane.append(self.n)
        if 'HN' in self.atom_list:
            plane.append(self.h)
        self.plane = torch.Tensor(plane)
    
    def _init_pos(self):
        """
        Our convention is to initialize the end of our linear sequence as centroid (0., 0., 0.) and then to
        extend it on the x-axis by a distance we calculated as 3.793 $\mathrm {\AA}$ (literature says its 3.7842)
        for each residue.
        """
        planes = self.plane.unsqueeze(0).repeat(self.n_players, self.max_res_len-1, 1)
        n_terminus = torch.Tensor([-1.413, -0.407, 0.000]).unsqueeze(dim=0).\
            repeat(self.n_players, 1, 1)
        c_terminus = torch.Tensor([1.411 + (self.max_res_len) * self.dcaca, 0.537, 0.000]).unsqueeze(dim=0).\
            repeat(self.n_players, 1, 1)
        o_c_terminus = torch.Tensor([1.620 + (self.max_res_len) * self.dcaca, 1.759, 0.000]).unsqueeze(dim=0).\
            repeat(self.n_players, 1, 1)
        ca_end = torch.Tensor([self.dcaca * self.max_res_len, 0.000, 0.000]).unsqueeze(dim=0).\
            repeat(self.n_players, 1, 1)
        self.pos = torch.stack(
            [
                torch.linspace(0.000, (self.max_res_len - 1) * self.dcaca, steps=self.max_res_len-1), 
                torch.zeros(self.max_res_len-1),
                torch.zeros(self.max_res_len-1)
            ], dim = 1).unsqueeze(dim=0).repeat(self.n_players, self.atoms_per_plane, 1)
        self.pos += planes
        self.pos = torch.cat([self.pos, ca_end], dim=1) 
        if 'N' in self.atom_list:
            self.pos = torch.cat([n_terminus, self.pos], dim=1)
        if 'C' in self.atom_list:
            self.pos = torch.cat([c_terminus, self.pos], dim=1)
        if 'O' in self.atom_list:
            self.pos = torch.cat([o_c_terminus, self.pos], dim=1)
        if self.verbose > 0:
            print("Initialized starting position!")
        
    def _init_rotvecs(self):
        """
        Rotation vectors are only given for phi φ , and theta ϑ
        Note we have not, but definitely should implement omega angle to differentiate trans- and 
        cis-peptide-groups.
        Using crude trigonometry calculus we came up with delta phi and delta theta, representing the angular 
        offset of the rotational axes representing the direction of the rotating bonds within the amidplane from
        the amid-plane diagonal (imagine a straight line from ca_1 to ca_2).
        We here initialze them as rotation vectors lying in the xy plane. The x axis corresponds to the 
        direction of the amid plane diagonal.
        """
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        vec_psi = torch.cat([
            torch.cos(torch.Tensor(self.n_players, self.max_res_len, 1).fill_(self.psi / 180 * torch.pi)),
            torch.sin(torch.Tensor(self.n_players, self.max_res_len, 1).fill_(self.psi / 180 * torch.pi)),
            torch.zeros(self.n_players, self.max_res_len, 1)
        ], dim=2)
        vec_phi = torch.cat([
            torch.cos(torch.Tensor(self.n_players, self.max_res_len, 1).fill_(self.phi / 180 * torch.pi)),
            torch.sin(torch.Tensor(self.n_players, self.max_res_len, 1).fill_(self.phi / 180 * torch.pi)),
            torch.zeros(self.n_players, self.max_res_len, 1)
        ], dim=2)
        vec_omega = torch.cat([
            torch.cos(torch.Tensor(self.n_players, self.max_res_len, 1).fill_(self.omega / 180 * torch.pi)),
            torch.sin(torch.Tensor(self.n_players, self.max_res_len, 1).fill_(self.omega / 180 * torch.pi)),
            torch.zeros(self.n_players, self.max_res_len, 1)
        ], dim=2)
        """
        We should implement rotation vector omega (and just use a cutoff function to only rotate / change 
        position if the output of this thrid variable is extremely close to -1 or 1 -> we have to define more complex
        operation in that case.) The rotation probably would become quite complex, since we also have to change
        positions (i.e. it would be a Rotation + Translation... compare _update_state.
        """
        self.rotvecs = torch.cat([vec_psi.unsqueeze(2), vec_phi.unsqueeze(2), vec_omega.unsqueeze(2)], dim=2)
        self.rotangles = torch.cat([torch.Tensor([self.psi for _ in range(self.max_res_len)]).unsqueeze(1), 
                                    torch.Tensor([self.phi for _ in range(self.max_res_len)]).unsqueeze(1), 
                                    torch.Tensor([self.omega for _ in range(self.max_res_len)]).unsqueeze(1)], 
                                   dim=1)
        self.rotangles = self.rotangles.unsqueeze(0).repeat(self.n_players, 1, 1)
    
    # =======================================================================================================
    
    def _take_action(self, next_state):  # our action space atm is just 2 angles (phi and theta)
        """
        We want to get T_omega and R_omega, or Ti_omega and Ri_omega (the inverse) for cis to trans.        
        """
        self._center()
        action = (self.rotangles[:, self.s] - next_state)
        self.rotangles[:, self.s] = next_state
        R_psi = self._action2rot(vec=self.rotvecs[:, :, 0], action=action[:,0])
        R_phi = self._action2rot(vec=self.rotvecs[:, :, 1], action=action[:,1])
        R_omega = self._action2rot(vec=self.rotvecs[:, :, 2], action=action[:,2])
        self._update_states(R_psi=R_psi, R_phi=R_phi, R_omega=None)
        
    def _update_states(self, R_psi=None, R_phi=None, R_omega=None):
        """
        Note: numpy einsum is ~4-8x faster than torch einsum (EVEN WITH CUDA)
        Later we would like to have the model specify what peptide it wants to have as center to act on.
        this would lead to using a masking operation on self.pos[:, (:)centroid-depending-on-player(:), :].
        """
        self._update_vecs(R_psi, R_phi)
        self._update_pos(R_psi, R_phi)
        if R_omega != None:
            """
            Todo: - execute Translation (according to change of amid-plane) T_omega
                  - execute Rotation R_omega
                  - define reverse operation (since we have to come back from rotated position)
            """
            pass
    
    def _action2rot(self, vec, action):
        """
        Note: the rotation direction given by angle does not matter (since the network learns anyway),
        just keep it the same!
        We scale phi and theta to in-between [-180, 180] degrees,
        then calculate the Rotation matrix according to the Rodriguez formula: 
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        
        Input
        vec: vector of rotation axis, either phi_vec or theta_vec
        action: scalar between -1 and 1, we scale ithttps://stackoverflow.com/questions/6802577/rotation-of-3d-vector by 180 to get the rotation angle in degrees
        
        Output
        R: Rotation matrix of shape (n_players, 3, 3)
        """
        action *= torch.pi
        I = torch.eye(3).unsqueeze(0).repeat(self.n_players, 1, 1)
        kx = vec[:, self.s, 0] # * action
        ky = vec[:, self.s, 1] # * action
        kz = vec[:, self.s, 2] # * action
        k0 = torch.zeros(kx.shape)
        K = torch.Tensor([[k0.tolist(), (-kz).tolist(), ky.tolist()],
                         [(kz).tolist(), k0.tolist(), (-kx).tolist()],
                         [(-ky).tolist(), kx.tolist(), k0.tolist()]])
        K = K.permute(2,0,1).numpy()
        K1 = np.einsum('b,bij->bij', np.sin(action.numpy()), K)
        K2 = np.einsum('b,bij->bij', (1 - np.cos(action)).numpy(), np.einsum('bik,bkj->bij',K,K))
        # https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
        return torch.tensor(I + K1 + K2)

    def _update_vecs(self, R_psi, R_phi):
        """
        There is still a problem: I currently dont address the centroid indices (basically for r_phi rotation vecs
        I should be using self.t+1 (so it rotates the r_theta of the current centroid as well) and oon the right side )
        r_phi should be rotated.
        
        We want to multiply vectors and positions before a certain index with a rotation matrix.
        We calculate the entire rotation for now -> rotvec(vecs,R) and then only update the state of vecs at the non-masked postions
        """
        prev = self.rotvecs
        self.rotvecs[:,:self.s] = torch.cat([
            torch.bmm(self.rotvecs[:,:self.s,0,:], R_psi)[:,:,None,:],
            torch.bmm(self.rotvecs[:,:self.s,1,:], R_psi)[:,:,None,:],
            torch.bmm(self.rotvecs[:,:self.s,2,:], R_psi)[:,:,None,:]], 
            dim=2)
        self.rotvecs[:,self.s+1:] = torch.cat([
            torch.bmm(self.rotvecs[:,self.s+1:,0,:], R_phi)[:,:,None,:],
            torch.bmm(self.rotvecs[:,self.s+1:,1,:], R_phi)[:,:,None,:],
            torch.bmm(self.rotvecs[:,self.s+1:,2,:], R_phi)[:,:,None,:]], 
                dim=2)
        self.rotvecs = f.normalize(self.rotvecs, p=2, dim=3)
        
    def _update_pos(self, R_psi, R_phi):
        assert torch.sum(self.pos[0,self.t]) == 0, \
            print("When updating our position the centroid is not the origin:", self.pos[0, self.t])
        self.pos[:,self.t+1:] = torch.bmm(self.pos[:,self.t+1:,:], R_phi)
        self.pos[:,:self.t] = torch.bmm(self.pos[:,:self.t,:], R_psi)
        
    def _center(self):
        """
        We center all atoms onto our current centroid (the amino acid at step self.t)
        """
        self.center = self.pos[:, self.t]
        self.pos = self.pos.sub(self.center[:,None])
        if self.verbose > 0:
            print("New centroid --> residue {} translated from {} to {}."\
                  .format(self.t, self.center[0], self.pos[0, self.t]))
    
    def get_mem_use(self):
        """
        According to some quick tests a randn (1000, 1000, 1000) tensor has size of 2 Gb,
        the memorysize is 4 billion --> divide output by 2~ size in gb.
        """
        def memsize(a):
            if a == None:
                return 0
            else:
                return a.element_size() * a.nelement()
        ms = 0
        ms += memsize(self.pos)
        ms += memsize(self.tpos)
        ms += memsize(self.rotvecs)
        ms += memsize(self.trotvecs)
        ms += memsize(self.dists)
        ms += memsize(self.tdists)
        return ms
    
    # =======================================================================================================
        
    def render(self, state=True, target=False):
        """
        here we update self.rgb (containing our video/image data)
        :: Does not work well: render(bxyz, y, radius=0.5, res=20, scale=0.08, trans=0.0, record=True)
        :: Maybe use at a later time to get a sick thing going..
        """
        if state & target:
            self.render(state=True, target=False)
            self.render(state=False, target=True)
            return
        elif state:
            print("rendering state is too slow using pygame.. but you can import from rendering")
            
        else:
            pass
    
    def plot_plane(self, point_size=3, bond_width=5, show_points=True, show_bonds=True):
        e1 = torch.Tensor(self.ca).unsqueeze(0)
        e2 = torch.Tensor(self.c).unsqueeze(0)
        e3 = torch.Tensor(self.o).unsqueeze(0)
        e4 = torch.Tensor(self.n).unsqueeze(0)
        e6 = (torch.Tensor(self.ca) + torch.Tensor([self.dcaca, 0, 0])).unsqueeze(0)
        e5 = torch.Tensor(self.h).unsqueeze(0)
        xyz = torch.cat([e1,e2,e3,e4,e5,e6], dim=0).numpy()
        pairs = [[xyz[0],xyz[1]],[xyz[1],xyz[2]],[xyz[1],xyz[3]],[xyz[3],xyz[4]],[xyz[3],xyz[5]]]
        
        # plot coloring
        atoms = ['CA','C','O','N','HN','CA']
        atom_ids = [0, 1, 2, 3, 4, 0]
        bond_ids = [0, 1, 0, 2, 0]
        
        # traces
        points = []
        if show_points:
            for i in range(xyz.shape[0]):
                points.append(go.Scatter3d(x=(xyz[i, 0],), 
                                           y=(xyz[i, 1],), 
                                           z=(xyz[i, 2],),
                                           text=atoms[i],
                                           marker=dict(
                                               color=atom_ids[i],
                                               colorscale='Viridis',
                                               opacity=0.7,
                                               size=point_size,
                                           ))
                             )
        vectors = []
        if show_bonds:
            for i, c in enumerate(pairs):
                X1, Y1, Z1 = zip(c[0])
                X2, Y2, Z2 = zip(c[1])
                vector = go.Scatter3d(x = [X1[0],X2[0]],
                                      y = [Y1[0],Y2[0]],
                                      z = [Z1[0],Z2[0]],
                                      marker = dict(
                                          color = bond_ids[i],
                                          line=dict(width=5,)
                                                   )
                                     )
                vectors.append(vector)
        fig = go.Figure(data = vectors + points)
        fig.update_layout(showlegend = False)
        fig.show()
    
    def plot(self, 
             player=0, 
             res=500,
             target=False,
             grid=False,
             plot_points=True,
             show_atomtypes=False,
             plot_bonds=False,
             point_size=3,
             bond_width=5,
             show=False, 
             save=True,
             return_=False):
        """
        plotting function - this is not a function viable to call during training due to its speed-predicament.
        We can consider calling it once an epoch ends to inspect the result.
        """
        point_labels = self.atom_label_list[self.names[player]]
        #print(point_labels)
        if target:
            xyz = self.tpos[player].numpy()
        else:
            xyz = self.pos[player].numpy()
        max_ = max(xyz[0])
        y = self.ya[player].numpy()
        axis = dict(
            range=[-max_, max_],
            showbackground=False,
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            title='',
            nticks=3,
        )
        layout = dict(
            width=res,
            height=res,
            scene=dict(
                xaxis=axis,
                yaxis=axis,
                zaxis=axis,
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(
                    up=dict(x=0, y=1, z=0),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=0, z=2),
                    projection=dict(type='perspective'),
                ),
            ),
            paper_bgcolor='rgba(255,255,255,255)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0)
        )
        points = []
        if plot_points:
            if show_atomtypes:
                for i in range(xyz.shape[0]):
                    points.append(go.Scatter3d(x=(xyz[i, 0],), 
                                               y=(xyz[i, 1],), 
                                               z=(xyz[i, 2],),
                                               text=point_labels[i],
                                               marker=dict(
                                                   size=point_size,
                                                   color=y[i],
                                                   colorscale='Viridis',
                                                   opacity=0.7,
                                               )
                                           ))
            else:
                for i in range(xyz.shape[0]):
                    points.append(go.Scatter3d(x=(xyz[i, 0],), 
                                               y=(xyz[i, 1],), 
                                               z=(xyz[i, 2],),
                                               text=point_labels[i],
                                               marker=dict(
                                                   size=point_size,
                                                   colorscale='Viridis',
                                                   opacity=0.7,
                                               )
                                           ))
        sticks = []
        if plot_bonds:
            for s in range(self.max_res_len-1):
                # psi
                vector = go.Scatter3d(x = self.rotvecs.numpy()[:,s,0,0],
                                      y = self.rotvecs.numpy()[:,s,0,1],
                                      z = self.rotvecs.numpy()[:,s,0,2],
                                      marker = dict(size = [0,5],
                                                    color = 0,
                                                    line=dict(width=5,
                                                              color='DarkSlateGrey')),
                                      )
                sticks.append(vector)
                # phi
                vector = go.Scatter3d(x = self.rotvecs.numpy()[:,s,1,0],
                                      y = self.rotvecs.numpy()[:,s,1,1],
                                      z = self.rotvecs.numpy()[:,s,1,2],
                                      marker = dict(size = [0,5],
                                                    color = 1,
                                                    line=dict(width=5,
                                                              color='DarkSlateGrey')),
                                      )
                sticks.append(vector)
        data = points + sticks
        self.fig = go.Figure(data = data,
                        layout=layout)
        self.fig.update_layout(showlegend=False)
        if show:
            self.fig.show()
        if save:
            self.fig.write_image("Visualization/Snaps/plot_{}_step_{}.png".format(self.names[player], self.t))
        if return_:
            return data
    
    def plot_comparsion(self, data_state, data_target):
        # plot them side by side
        return