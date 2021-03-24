import csv
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
 
from IPython.display import display, HTML

    
class DataSet(Dataset):
    def __init__(self, 
                 folder='data/training/',
                 setting=3,
                 limit=None,
                 norm=True,
                 get_distr=False,
                 ):
        self.data_path=folder
        
        assert setting != None, print("No setting specified.")
            
        self.setting = setting
        self.data = pd.DataFrame()
        self.limit = limit
        self.load()
        
        self.data['na'] = self.data.apply(lambda x: len(x.y), axis=1)
        self.data['nr'] = self.data.apply(lambda x: (len(x.psi)+len(x.phi)+len(x.omega))//3, axis=1)  # give 1 too little
        self.max_natoms = self.data.na.max()
        self.max_nres = self.data.nr.max()
        
        if norm:
            # update xyz
            pass
        
        if get_distr:
            self.distr = self.get_distr()
        else:
            self.distr = None
            
    
    # ======================================================================================================
    
    def get_distr(self):
        # use self.data to get distr
        return np.asarray([0])

    # ======================================================================================================
    
    def load(self):
        # csv saves lists as strings (and numpy arrays get messed up as well)
        # fix our residue-list
        filename = self.data_path+'data_{}.csv'.format(self.setting)
        print("Loading data from", filename)
        from ast import literal_eval as eval
        # fix our numpy arrays
        def xyz_from_csv(array_string):
            array_string = ','.join(array_string.replace('[ ', '[').split())
            array_string = array_string.replace('[,', '[').replace(',]', ']').replace(',,', ',')
            return np.asarray(eval(array_string))
        def angles_from_csv(array_string):
            array_string = ','.join(array_string.replace('[ ', '[').split())
            array_string = array_string.replace('[,', '[').replace(',]', ']').replace(',,', ',').\
                replace('None', '181')  # this only works on angles and is a hack
            return np.asarray(eval(array_string))
            #except:
            #    return np.NaN  # this detects proteins with incorrect format (no cristal structure)
        # Check Input file
        assert '.csv' in filename, print("Expected input data format is CSV (.csv).")
        self.data = pd.read_csv(filename, converters={'y': eval,
                                                      'het':eval,
                                                      'xyz':xyz_from_csv,
                                                      'psi':angles_from_csv,
                                                      'phi':angles_from_csv,
                                                      'omega':angles_from_csv,})
        if self.limit != None:
            self.data = self.data.head(self.limit) 
    
    def save(self, filename):
        print("Saving data to", filename)
        self.data.to_csv(filename, index=False)
    
    # ======================================================================================================
    
    def display(self):
        display(HTML(self.data.to_html()))
    
    # ======================================================================================================
    
    def __len__(self):
        return self.data.count()[0]
    
    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        name = data['prot_id']
        n_atoms = data['na']
        n_res = data['nr']
        xyz = np.asarray(data['xyz'])
        # xyz = xyz - mean of xyz
        y = np.asarray(data['y'])+1  # add one to differentiate from padding
        psi = np.asarray(data['psi'])
        phi = np.asarray(data['phi'])
        omega = np.asarray(data['omega'])
        xyz_padded = np.pad(xyz, ((0, self.max_natoms-xyz.shape[0]), (0, 0)), mode='constant')
        y_padded = np.pad(y, (0, self.max_natoms-y.shape[0]), mode='constant')
        psi = np.pad(psi, (0, self.max_nres-psi.shape[0]), mode='constant')
        phi = np.pad(phi, (0, self.max_nres-phi.shape[0]), mode='constant')
        omega = np.pad(omega, (0, self.max_nres-omega.shape[0]), mode='constant')
        if self.distr != None:
            return name,\
                torch.from_numpy(y_padded),\
                torch.from_numpy(xyz_padded),\
                torch.from_numpy(psi), torch.from_numpy(phi), torch.from_numpy(omega),\
                torch.LongTensor([n_atoms, n_res]),\
                torch.from_numpy(self.distr)
        else:
            return name,\
                torch.from_numpy(y_padded),\
                torch.from_numpy(xyz_padded),\
                torch.from_numpy(psi), torch.from_numpy(phi), torch.from_numpy(omega),\
                torch.LongTensor([n_atoms, n_res])
    
    # ======================================================================================================
    