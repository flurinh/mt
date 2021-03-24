from utils import *

import json
from itertools import compress
from os import listdir, rename
from os.path import isfile, join, isdir

from Bio.PDB.PDBList import PDBList
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import *


import pandas as pd
import numpy as np
from tqdm import tqdm, trange

from IPython.display import display, HTML


PARSER = PDBParser(QUIET=True)


class PDBProcessor():
    def __init__(self,
                 data_path='data/',
                 database='pdb', # Or SWISS_MODEL
                 file_name='data_new.csv',
                 start=0,
                 limit=None,
                 parse=False,
                 setting=3,  # setting 5 means we parse everything, 4 is retain C_B = AA + Backbone etc
                 verbose=1):
        """
        DATAFRAME FORMAT
        
        --------------------------------------------------------------------------------------------------------
        | atom_type | atom_pos | pa_res_ids | pa_res_names | res_ids | psi | phi | omega |
        --------------------------------------------------------------------------------------------------------
        prot_id refers to the pdb name
        res refers to the  residue an atom belongs to
        xyz is the position [x, y, z]
        calpha is a bool
        phi & psi are floats in [-1, 1] (corresponding to [-180°, 180°] rotation angles)
        omega is a boolean stating wheter the amid plane configuration is trans (False) or cis (True)
        """
        self.data = pd.DataFrame()
        
        # Verbose
        self.verbose = verbose
        
        # Data strutcture
        self.sparse_df = False
        self.data_path = data_path
        self.database = database
        self.modded = True
        
        # Maximum number of aggregated samples
        self.limit = limit
        # Starting index of file
        self.start = start
        
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
        
        self.settings = {0: ['CA'],
                         1: ATOM_LIST,
                         2: ATOM_LIST+['CB'],
                         3: ATOM_LIST+['HN'],
                         4: ATOM_LIST+['CB']+['HN'],
                         5: None}
        
        self.setting = setting
        self.atom_list=list(set(self.settings[setting]))
        self.residue_list = RESIDUE_LIST
        self.segment_list = SEGMENT_LIST
                
        if parse:
            self.parse()

    # =======================================================================================================
    
    def getListOfFiles(self, dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                if '.pdb' in fullPath:
                    allFiles.append(fullPath)
        return allFiles
    
    def get_num_files(self):
        return len(getListOfFiles(join(self.data_path+self.database)))
    
    def get_files(self):
        """
        todo: search in all subdirectories... of self.data_path + self.database
        """
        files = self.getListOfFiles(join(self.data_path+self.database))
        if self.limit != None:
            if self.verbose==1:
                print("Found {} files. Using {} files.".format(len(files), self.limit))
            return files[self.start:self.start+self.limit]
        else:
            if self.verbose==1:
                print("Found {} files.".format(len(files)))
            return files
    
    # =======================================================================================================
    
    def parse(self):
        self.modded = False
        self.files = self.get_files()
        if self.verbose > 0:
            print("Parsing raw data to dataframe...")
            print("Total number of files found:", len(self.files))
        self.parse_pdb()
    
    def parse_pdb(self, first=True):
        """
        mode: specify whether output should be positional data of C-alphas of all residues (mode='residue')
              or to use all atoms (mode='atomic') # or at least Ca-O-H-Ca
        """
        # currently always uses Ca mode (atom-mode is not implemented)
        idx = 0
        for _, f in tqdm(enumerate(self.files)):
            # currently we always select the first model
            # print(f)
            with open(f) as f_open:
                first_line = f_open.readline()
            f_open.close()
            if not (('DNA' in first_line) or ('RNA' in first_line)):
                model = self.get_structure(f, first)
                # Here we extract xyz values of all atoms, their types
                # and their residue's name (3 letter convention, 'GLU' etc)
                # Note that the raw atom type (eg. 'C' for 'CA') is just the first letter!
                struct = self.get_atoms(model)
                # struct: atom_type, atom_pos, pa_res_ids, pa_res_names, res_ids, res_names, segment
                #              0         1           2           3         4           5         6
                if struct != None:
                    psi, phi, omega = self.get_angles(model)
                    n_chains = len(struct)
                    for chain in struct:
                        if len(chain[1]) > 0:
                            n_atoms = len(chain[1])

                            # get number of residues
                            #n_res = chain[0].count('CA')
                            # atoms
                            atom_pos = np.asarray(chain[1])  # we use this to get rid of the dataformat of pdb ("Vector")

                            # residues
                            psi, phi, omega = np.asarray(psi), np.asarray(phi), np.asarray(omega)
                            chain_start = min(chain[2])
                            chain_stop = max(chain[2])

                            # convert numpy arrays to list in order to save to csv file (parsing of 2d np arr bugs)
                            self.data = self.data.append({'prot_id':f[-8:-4], 
                                                          'n_atoms':n_atoms,
                                                          'n_residues':chain_stop-chain_start,
                                                          'res_id':chain[4],
                                                          'res_name':chain[5],
                                                          'phi':phi.tolist()[chain_start:chain_stop],
                                                          'psi':psi.tolist()[chain_start:chain_stop],
                                                          'omega':omega.tolist()[chain_start:chain_stop],
                                                          'segment':chain[6],
                                                          'chain_id':chain[7],
                                                          'atom':chain[0], 
                                                          'xyz':atom_pos.tolist(),
                                                          'pa_res_id':chain[2],
                                                          'pa_res_name':chain[3]},
                                                         ignore_index=True)
    
    # =======================================================================================================
    
    def get_angles(self, model):
        model.atom_to_internal_coordinates()
        psi = []
        phi = []
        omega = []
        for _, res in enumerate(model.get_residues()):
            # print("psi",res.internal_coord.get_angle("psi"))
            if res.get_resname() != 'HOH':
                try:
                    psi.append(res.internal_coord.get_angle("psi"))
                except:
                    # print("psi exc", _)
                    psi.append(181)
                # print("phi",res.internal_coord.get_angle("phi"))
                try:
                    phi.append(res.internal_coord.get_angle("phi"))
                except:
                    # print("phi exc", _)
                    phi.append(181)
                # print("omega",res.internal_coord.get_angle("omega"))
                try:
                    omega.append(res.internal_coord.get_angle("omega"))
                except:
                    # print("omega exc", _)
                    omega.append(181)
                # print("")
        return psi, phi, omega
    
    def reset_data(self):
        return [], [], [], [], [], [], [], []
    
    def get_atoms(self, model):
        """
        model is our primary structure model that we extract atoms, angles, xyz etc from
        """
        # Per protein
        output = []
        # iterators
        res_id = 0
        calc_h = True
        ti = 0
        triplet = []
        atom_type, atom_pos, pa_res_ids, pa_res_names, res_ids, res_names, segment, chain_ids = self.reset_data()
        first = True
        try:
            for chain in model:
                chain_id = chain.get_id()
                if first:
                    old_chain_id = chain_id
                    first = False
                for res in chain:
                    res_name = res.get_resname()
                    if (res.get_id()[0][0] != 'H') and (res_name!='HOH'):  # ignore hetero atoms (water, dms etc)
                        tokens = ['C', 'N', 'CA']
                        segment.append(res.get_segid().replace(' ', '').replace(',', ''))
                        for atom in res.get_atoms():
                            name = atom.get_name()
                            if name == 'HN':  # maybe overkill
                                #print("H position information included in structure! Skipping H position calculation.")
                                calc_h = False
                            pos = np.asarray(atom.get_coord())
                            # add Hydrogen into the mix here (to avoid messing up indexing)
                            if calc_h:
                                if name == tokens[ti]:
                                    triplet.append(pos)
                                    # We get atoms in order: [N, CA, O, CB, N, CA, ..., C, O]  --> we get triplets
                                    if ti == 2:
                                        h_pos = self.get_h_pos(triplet)
                                        atom_pos.append(h_pos)
                                        atom_type.append('HN')
                                        pa_res_names.append(res_name)
                                        pa_res_ids.append(res_id)
                                        triplet = []
                                    ti = (ti + 1) % 3
                            atom_type.append(name)  # use the first letter if you want raw atom type
                            atom_pos.append(pos)
                            pa_res_names.append(res_name)
                            pa_res_ids.append(res_id)
                        chain_ids.append(chain_id)
                        res_names.append(res_name)
                        res_ids.append(res_id)
                        res_id += 1
                    # We split data by chains (focus on secondary and tertiary structure not quaternary)
                    if chain_id != old_chain_id:
                        output.append([atom_type, atom_pos, pa_res_ids, pa_res_names, res_ids, res_names, 
                                      segment, chain_ids])
                        atom_type, atom_pos, pa_res_ids, pa_res_names, res_ids, res_names, segment, chain_ids = self.reset_data()
                        ti = 0
                        triplet = []
                    old_chain_id = chain_id
            output.append([atom_type, atom_pos, pa_res_ids, pa_res_names, res_ids, res_names, segment, chain_ids])
            return output
        except:
            return None
    
    def get_structure(self, input, first=True):
        """
        first: Specifies whether to only use the first model of a pdb file.
        """
        if self.verbose > 0:
            print("Loading structure of", input)
        if first:
            return PARSER.get_structure(input[-8:-4], input)[0]
        else:
            out = []
            for i, f in PARSER.get_structure(input[-8:-4], input):
                out.append(f[i])
            return out
    
    def get_h_pos(self, triplet):
        """
        input: List [cb, n, ca2]
        Function to infer the Hydrogen position of the backbone amine. We assume an pure sp2 configuration
        and angles of 120° between cb-n|n-ca, cb-n|n-h and h-n|n-ca! This comes very close to the true position
        in the trans-state of the amid-plate. The distance ||h-n|| = 1.
        This is easily done by adding the two vectors n->cb and n->ca and normalzing the vector to length 1 and
        reversing the direction
        """
        c, n, ca2 = triplet
        vec = n - c + n - ca2
        h_pos = n + (- 1 * (vec) / np.linalg.norm(vec))
        return h_pos
    
    # =======================================================================================================
    
    # DATAFRAME FILTERING
    
    def filter_data(self, setting=None, atom_list=None):
        """
        This function lets us modify our data to produce a training-dataset in form of a new dataframe
        with filtered and smaller datasize footprint.
        """
        assert (setting == None) ^ (atom_list == None), print("setting and atom_list are mutually exclusive!")
        if atom_list!=None:
            self.atom_list=atom_list
        if setting!=None:
            self.setting = setting
            self.atom_list = list(set(self.settings[setting]))
        if self.verbose > 0:
            print("Filter settings:", self.setting_explained[self.setting])
            # if folder for processed data in subdirectory data/mode does not exist create it & set new path
            # filter data either atom or residue wise
            # save data
            # define aggregate function (from limit to limit to allow continuous processing of our data)
            print("Filtering by list of atoms:", self.atom_list)
        if self.atom_list!=None:
            self.filter_by_atom_list()
        self.modded = True
        # Not implemented
        if False:
            # remove proteins containing variants outside specs --> currently done by "clean()"
            print("Filtering by list of residues:", self.residue_list)
            self.filter_by_residue_list()
            self.modded = True
        # Not implemented
        if False:
            print("Filtering by list of segments:", self.segment_list)
            self.filter_by_segment()
            self.modded = True
        
    def filter_atoms_(self, atom, xyz, pa_res_id, pa_res_name):
        atom_filter = [(a in self.atom_list) for a in atom]
        n_atoms = sum(atom_filter)
        xyz = [np.asarray(xyz_).tolist() for xyz_ in xyz]
        xyz = list(compress(xyz, atom_filter))
        atom = list(compress(atom, atom_filter))
        pa_res_id = list(compress(pa_res_id, atom_filter))
        pa_res_name = list(compress(pa_res_name, atom_filter))
        return atom, xyz, pa_res_id, pa_res_name, n_atoms
        
    def filter_by_atom_list(self, atom_list=None):
        if atom_list!=None:
            self.atom_list=atom_list
        # removing everything except 'CA' is NOT ADVISED 
        # ---> then we have no information about the rotational direction (i.e. confusion between psi <-> phi)
        # Columns update
        self.data['atom'], self.data['xyz'], \
            self.data['pa_res_id'], self.data['pa_res_name'], self.data['n_atoms'] = \
            zip(*self.data.apply(lambda x: self.filter_atoms_(
                x.atom, x.xyz, x.pa_res_id, x.pa_res_name), axis=1))
    
    def filter_by_residue_list(self, valid_aas=None):
        """
        Rare variants are NOT "filtered out" of the atom-list. I filter out proteins containing rare variants!
        """
        # Speedy version: check if length of unique 
        # Figure out how to get a mask where each False entry corresponds to a protein that contains
        # nonvalid AAs (or some other trash we do not allow)
        pass
    
    def filter_by_segment(self):
        """
        Remove all proteins not containing segment information.
        Split remaining proteins into segments?
        --> segments are targets
        """
        pass
    
    # =======================================================================================================
        
    def set_atom_list(self, atom_list):
        if atom_list != None:
            self.atom_list = atom_list
        elif self.atom_list == None:
            self.atom_list = ATOM_LIST
        else:
            print("No atom_list specified.")
    
    def set_residue_dict(self, residue_list):
        if residue_list != None:
            self.residue_list = residue_list
        elif self.residue_list == None:
            self.residue_list = RESIDUE_LIST
        else:
            print("No residue_list specified.")
    
    def set_segment_dict(self, segment_list):
        if segment_list != None:
            self.segment_list = segment_list
        elif self.segment_list == None:
            self.segment_list = SEGMENT_LIST
        else:
            print("No segment_list specified.")
    
    # =======================================================================================================
            
    def get_residue_variants(self, n_aa=20):
        """
        {'LEU': 13034, 'GLU': 9023, ' Ca': 5, ..} --> Remove proteins where rare variants occur,
        use a cutoff of n_aa = 20 to remove all rare variants by only selecting (the normal) 20 aminoacids.
        """
        d = Counter(flat_l(self.data['res_name'].tolist()))
        v, k = zip(*sorted(zip(d.values(), d.keys())))
        n = min(n_aa, len(d))
        self.res_list = dict(zip(k[-n:], v[-n:]))
        return k[:-n], k[-n:]  # return k[-n_aa] just so I can copy it to utils.py as an initialization array
    
    def remove_variants(self, n_aa=20):
        c_tot = self.data.prot_id.count()
        k, _ = self.get_residue_variants(n_aa)
        for key in k:
            self.data = self.data[self.data['res_name'].apply(lambda x: key not in x)]
        c_red = self.data.prot_id.count()
        if self.verbose > 0:
            print("After removing {} variants our data contains {} polypeptide chains.".format(c_tot-c_red, c_red))
        self.res_list = _
    
    def remove_outliers(self, min_len=30, max_len=1000):
        c_tot = self.data.prot_id.count()
        self.data = self.data[(min_len < self.data['n_residues']) & (self.data['n_residues'] < max_len)]
        c_red = self.data.prot_id.count()
        if self.verbose > 0:
            print("After removing {} outliers with {} > length > {} our data contains {} polypeptide chains."\
                  .format(c_tot-c_red, min_len, max_len, c_red))
        
    def clean(self, min_len=30, max_len=1000, n_aa=20):
        self.remove_outliers(min_len=min_len, max_len=max_len)
        self.remove_variants(n_aa)
    
    # =======================================================================================================

    def label(self, custom_label_dict=None):
        """
        Create label column --> go from seperate atom and residue to hetero-representation (except case 6)
        to combined dict (rl = reduced list of atoms/residues)
        """
        if self.setting < 5:
            rl = self.atom_list + self.residue_list
            label_dict = self.create_label_dict(rl)
            self.heterogen_col()
            self.label_(label_dict)
        else:
            label_dict = self.create_label_dict(self.atom_list)
            self.data['het'] = self.data['atom'].reset_index(drop=True)
            self.label_(label_dict)
        self.save_label_dict(label_dict)
    
    def save_label_dict(self, label_dict):
        with open('data/label_dict_'+str(self.setting)+'.json', 'w') as fp:
            json.dump(label_dict, fp)
            
    def load_label_dict(self):
        with open('data/label_dict_'+str(self.setting)+'.json', 'r') as fp:
            label_dict = json.load(fp)
        return label_dict
    
    def heterogen_col(self):
        def comp(res, atom):
            return [(r if a == 'CA' else a) for a, r in zip(atom, res)]
        self.data['het'] = self.data.apply(lambda x: comp(x.pa_res_name, x.atom), axis=1).reset_index(drop=True)
    
    def create_label_dict(self, rl):
            d = {}
            for i, l in enumerate(reversed(rl)):
                d.update({l:i})
            return d
    
    def label_(self, label_dict):
        def map_(l, d):
            try:
                return [d.get(_) for _ in l]
            except:
                return None
        self.data['y'] = self.data.apply(lambda x: map_(x.het, label_dict), axis=1)
        self.data = self.data[self.data['y'] != None]
    
    # deprecated
    def get_res_list(self):
        """
        Run this on your dataset, and copypaste the list to utils: "RES_LIST"
        """
        flatten = lambda l: [item for sublist in l for item in sublist]
        l = self.data['res_name'].tolist()
        return sorted(list(set(flatten(l))))
    
    # =======================================================================================================
    
    def create_training_df(self, 
                          setting=None,
                          cols=['prot_id', 'y', 'het', 'xyz','psi','phi','omega']):
        """
        Target can also be 'segment'. We only retain essential columns and store the complete dictionary, 
        ready to be loaded for training. --> 2 columns: Input (point/atomwise), Target (angles)
        
        assert not ((setting == None) and (self.setting == None)), print("No training setting specified.")
        """
        if setting != None:
            self.setting = setting
        if self.verbose > 0:
            print("Generating training dataframe... setting ({}): {}".format(self.setting, 
                                                                             self.settings[self.setting]))
        if self.limit==None:
            limit=self.data.count()[0]
        if self.setting != None:
            if setting != None:
                if setting > self.setting:
                    self.setting = setting
                    if self.verbose > 0:
                        print("Overwriting higher complexity setting.")
                else:
                    print("Current data has lower complexity than requested training data setting.")
                    return False
        # clean data (only keep requested columns and remove items with uncomplete samples
        self.clean()
        # Filter according to setting
        self.filter_data(setting=self.setting)
        # Create label column
        self.label()
        # select columns
        training_df = self.data[cols]
        # save training data
        filename = self.data_path+'training/data_{}_{}_{}.csv'\
            .format(self.start, self.limit, self.setting)
        training_df.to_csv(filename, index=False)
    
    # =======================================================================================================
    
    def save(self, filename=None):
        """
        Save a single file.
        """
        if filename is None:
            filename=self.data_path+'processed/data_{}_{}.csv'.format(self.start, self.limit)
            if self.modded:
                filename=self.data_path+'processed/data_{}_{}_{}.csv'.\
                format(self.start, self.limit, '_'.join(self.atom_list))
        if self.verbose > 0:
            print("Saving data to", filename)
        self.data.to_csv(filename, index=False)

    def load(self, start=0, limit=None, atom_list=None):
        """
        Load a single file.
        """
        if start!=0:
            self.start=start
        if limit!= None:
            self.limit=limit
        filename='data_{}_{}.csv'.format(self.start, self.limit)
        if atom_list!=None:
            self.atom_list = atom_list
            filename='data_{}_{}_{}.csv'.format(self.start, self.limit, '_'.join(self.atom_list))
        filename = self.data_path+'processed/' + filename
        if self.verbose > 0:
            print("Trying to load data from", filename)
        self.data = self.load_(filename)
        if self.verbose > 0:
            print("Loaded data from {}. Total number of proteins: {}.".format(filename, self.data.count()[0]))
        
    def load_(self, filename):
        # csv saves lists as strings (and numpy arrays get messed up as well)
        # fix our residue-list
        from ast import literal_eval as eval
        # fix our numpy arrays
        def xyz_from_csv(array_string):
            array_string = ','.join(array_string.replace('[ ', '[').split())
            array_string = array_string.replace('[,', '[').replace(',]', ']').replace(',,', ',')
            return eval(array_string)
        def angles_from_csv(array_string):
            array_string = ','.join(array_string.replace('[ ', '[').split())
            array_string = array_string.replace('[,', '[').replace(',]', ']').replace(',,', ',').\
                replace('None', '181')  # this only works on angles and is a hack
            return eval(array_string)
            #except:
            #    return np.NaN  # this detects proteins with incorrect format (no cristal structure)
        # Check Input file
        assert '.csv' in filename, print("Expected input data format is CSV (.csv)")
        return pd.read_csv(filename, converters={'res_id':eval,
                                                 'res_name':eval,
                                                 'phi':angles_from_csv,
                                                 'psi':angles_from_csv,
                                                 'omega':angles_from_csv,
                                                 'segment':eval,
                                                 'chain_id':eval,
                                                 'atom':eval, 
                                                 'xyz':xyz_from_csv,
                                                 'pa_res_id':eval,
                                                 'pa_res_name':eval})
       
    # =======================================================================================================
    
    def create_complete_trainingset(self, step, limit, setting=None):
        """
        We load data from multiple files inside training-folder corresponding to correct data format.
        """
        # get all files ending in training/data_*_{}_{}.csv (step, atom_list)  where * is a multiple of step
        assert not((setting == None) and (self.setting == None)), print("Can't load data for unspecified settings.")
        if setting != None:
            self.setting = setting
        max_files = 100000 // step
        all_files = [self.data_path+'training/'+'data_{}_{}_{}.csv'.\
                     format(s*step, step, self.setting) for s in range(max_files)][:limit]
        dfs = []
        assert len(all_files) > 0, print("Did not find data for specified settings!")
        for filename in all_files:
            try:
                df = self.load_training_files_(filename)
                dfs.append(df)
            except:
                pass
        self.data = pd.concat(dfs, axis=0, ignore_index=True)
        if self.verbose > 0:
            print("Loaded data from {}. Total number of polypeptide chains: {}.".\
                  format(filename, self.data.count()[0]))
        self.data.to_csv(self.data_path+'training/'+'data_{}.csv'.format(self.setting), index=False)
    
    def load_training_files_(self, filename):
        # csv saves lists as strings (and numpy arrays get messed up as well)
        # fix our residue-list
        from ast import literal_eval as eval
        # fix our numpy arrays
        def xyz_from_csv(array_string):
            array_string = ','.join(array_string.replace('[ ', '[').split())
            array_string = array_string.replace('[,', '[').replace(',]', ']').replace(',,', ',')
            return eval(array_string)
        def angles_from_csv(array_string):
            array_string = ','.join(array_string.replace('[ ', '[').split())
            array_string = array_string.replace('[,', '[').replace(',]', ']').replace(',,', ',').\
                replace('None', '181')  # this only works on angles and is a hack
            return eval(array_string)
            #except:
            #    return np.NaN  # this detects proteins with incorrect format (no cristal structure)
        # Check Input file
        assert '.csv' in filename, print("Expected input data format is CSV (.csv).")
        return pd.read_csv(filename, converters={'y': eval,
                                                 'het':eval,
                                                 'xyz':xyz_from_csv,
                                                 'psi':angles_from_csv,
                                                 'phi':angles_from_csv,
                                                 'omega':angles_from_csv,})
    
    # =======================================================================================================
    
    def plot_distr(self, mode='len', idx=None, save=False, show=True):
        """
        plot different data visualizations
        modes: - 'len': plot distribution of amino sequence lengths
               - 'residues': plot occurances of residues overall
        """
        assert mode in ['len', 'residues'], print("make sure mode is either 'len' or 'residues'")
        if mode is 'len':
            plot_len_hist(self.data, save, show)
        else:
            plot_res_hist(self.data, idx, save, show)
            
    # Color should be based on either hydrophobic, -philic, -charged, -aromatic
    # or sorted that way and color based on
    
    # =======================================================================================================
    
    def display(self, idx=None):
        if idx != None:
            display(HTML(self.data.iloc[idx].to_html()))
        else:
            display(HTML(self.data.to_html()))
    
    # =======================================================================================================