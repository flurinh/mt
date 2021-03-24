from utils import *

from os import listdir, rename
from os.path import isfile, join

from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import *

import pandas as pd
import numpy as np
from tqdm import tqdm, trange


PARSER = PDBParser()

class Processing():
    def __init__(self,
                 data_path='data/',
                 file_name='data_new.csv',
                 limit=None,
                 parse=False,
                 mode='residue',
                 verbose=1):
        self.verbose = verbose
        self.data_path = data_path
        self.limit = limit
        self.mode = mode
        self.data = pd.DataFrame()
        if parse:
            print("Parsing raw data to dataframe...")
            self.files = self.get_files()
            if self.limit != None:
                self.files = self.files[:self.limit]
            self.parse_pdb()
            self.res_list = self.get_res_list()

    def get_files(self):
        files = [join(self.data_path+'raw', f) for f in listdir(self.data_path+'/raw') \
                 if (isfile(join(self.data_path+'raw', f)) and 'pdb' in f)]
        if self.limit != None:
            if self.verbose==1:
                print("Found {} files. Using {} files.".format(len(files), self.limit))
            return files[:self.limit]
        else:
            if self.verbose==1:
                print("Found {} files.".format(len(files)))
            return files
    
    def parse_pdb(self, first=True):
        """
        mode: specify whether output should be positional data of C-alphas of all residues (mode='residue')
              or to use all atoms (mode='atomic') # or at least Ca-O-H-Ca
        """
        # currently always uses Ca mode (atom-mode is not implemented)
        idx = 0
        for f in tqdm(self.files):
            # currently we always select the first model
            model = self.get_structure(f, first)
            # Here we extract xyz values of C-alphas and their residue's name (3 letter convention, 'GLU' etc)
            pos, res_ids = self.get_data(model)
            if pos is None:
                continue
            self.data = self.data.append({'idx': int(idx), 'name':f[-8:-4], 'pos':pos, 'res':res_ids, 'len':len(res_ids)}, \
                               ignore_index=True)
            idx+=1
        return

    def save(self, filename=None):
        if filename is None:
            filename=self.data_path+'data_{}.csv'.format(self.data.count().len)
        self.data.to_csv(filename)
        return

    def load(self, filename='data_1930.csv', limit=None):
        # csv saves lists as strings
        if limit!= None:
            self.limit=limit-1
        filename = self.data_path + filename
        # fix our residue-list
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
        self.data = pd.read_csv(filename, converters={'pos': from_np_array, 'res': eval}).loc[:self.limit]
        print("Loaded data from {}. Total number of proteins: {}.".format(filename, self.data.count().len))
        return

    def plot_distr(self, mode='len', idx=None, save=False):
        """
        plot different data visualizations
        modes: - 'len': plot distribution of amino sequence lengths
               - 'residues': plot occurances of residues overall
        """
        assert mode in ['len', 'residues'], print("make sure mode is either 'len' or 'residues'")
        if mode is 'len':
            plot_len_hist(self.data, save)
        else:
            plot_res_hist(self.data, idx, save)

    def get_data(self, model):
        data = []
        res_ids = []
        for i, residue in enumerate(model.get_residues()):
            try:
                data.append(residue['CA'].get_coord())
                res_ids.append(residue.get_resname())
            except:
                print("Couldnt find residues coordinates.")
                return None, None
        return np.asarray(data), res_ids

    def get_structure(self, input, first=True):
        """
        first: specify whether to only use the first model of a pdb file, default True
        """
        print("Loading structure of", input)
        if first:
            return PARSER.get_structure(input[-8:-4], input)
        else:
            out = []
            for i, f in PARSER.get_structure(input[-8:-4], input):
                out.append(f[i])
            return out

    def get_res_list(self):
        """
        Run this on your dataset, and copypaste the list to utils: "RES_LIST"
        """
        flatten = lambda l: [item for sublist in l for item in sublist]
        l = self.data['res'].tolist()
        return sorted(list(set(flatten(l))))
    
    def rl_remove_variants(self, n_aa=20):
        """
        {'LEU': 13034, 'GLU': 9023, ' Ca': 5, ..} --> Remove proteins where rare variants occur,
        use a cutoff of n_aa = 20 to remove all rare variants by only selecting (the normal) 20 aminoacids.
        """
        d = Counter(flat_l(self.data['res'].tolist()))
        v, k = zip(*sorted(zip(d.values(), d.keys())))
        self.res_list = dict(zip(k[-n_aa:], v[-n_aa:]))
        return k[:-n_aa], k[-n_aa:]  # return k[-n_aa] just so I can copy it to utils.py as an initialization array
    
    def remove_variants(self):
        c_tot = self.data.count().len
        k, _ = self.rl_remove_variants()
        new = self.data
        for key in k:
            new = new[new['res'].apply(lambda x: key not in x)]
        self.data = new.reset_index()
        c_red = self.data.count().len
        print("After removing {} variants our data contains {} proteins.".format(c_tot-c_red, c_red))
        self.res_list = _
        return
    
    def remove_outliers(self, min_len=100, max_len=6000):
        c_tot = self.data.count().len
        new = self.data[(min_len < self.data['len']) & (self.data['len'] < max_len)]
        self.data = new.reset_index()
        c_red = self.data.count().len
        print("After removing {} outliers with {} > length > {} our data contains {} proteins."\
              .format(c_tot-c_red, min_len, max_len, c_red))
        del new
        return

    def get_common_aas(self):
        return RED_RL
    
    def label_ca(self):
        def mkdict(rl):
            d = {}
            for i, l in enumerate(reversed(rl)):
                d.update({l:i})
            return d
        def mapaa(aas, d):
            y = []
            for aa in aas:
                y.append(d[aa])
            return np.asarray(y)
        ltodict = mkdict(self.get_common_aas())
        temp = self.data.copy()
        self.data['y'] = temp[['res']].applymap(lambda x: mapaa(x, ltodict))
        del temp
        return
    
    def clean(self):
        try:
            self.data = self.data.filter(['name', 'len', 'pos', 'y', 'res'])
        except:
            print("Couldnt clean up this mess.. check your df!")
            
    def fix_pos(self):
        # this only needs to be called if we loaded everything from a csv file -> transformed the np array into a string list
        c_tot = self.data.count().len
        self.data = self.data[~self.data['pos'].isna()]
        c_red = self.data.count().len
        print("After removing {} proteins without structure (xyz data) our dataset contains {} proteins."\
              .format(c_tot-c_red, c_red))
        return
    
    def create_dataset(self):
        # store name
        # store pos  --> numpy!
        # store y
        # store residues
        pass