from processing.utils3 import *
from processing.utils2 import *
import pandas as pd
import os
import sys
import functools
import operator
import random
from tqdm import tqdm, trange
import time
import numpy as np
import pandas
from math import degrees
import requests as r
from Bio import SeqIO, pairwise2
from io import StringIO
import gemmi
from gemmi import cif


class CifProcessor():
    def __init__(self, 
                 path = 'data/',
                 structure = 'mmcif/',
                 starting_idx=0,
                 limit=None,
                 shuffle = False,
                 reload=True,
                 remove_hetatm=True,
                 allow_exception=False):
        self.path = path
        self.structure_path = self.path + structure
        self.raw_folder = path + 'raw/'
        self.path_table = path + 'gpcrdb/' + 'table.pkl'
        
        self.shuffle = shuffle
        self.limit = limit
        self.reload = reload
        self.remove_hetatm = remove_hetatm
        self.allow_exception = allow_exception
        
        self.filenames, self.pdb_ids = self.get_pdb_files()
        self.filenames = self.filenames[starting_idx:]
        self.pdb_ids = self.pdb_ids[starting_idx:]
        if self.limit == None:
            self.limit = len(self.pdb_ids)
        if len(self.filenames) > self.limit:
            self.filenames = self.filenames[:self.limit]
            self.pdb_ids = self.pdb_ids[:self.limit]
            
        # Columns for structure dataframe
        self.cols = ['group_PDB', 'auth_asym_id', 'label_asym_id', 'label_seq_id', 'auth_seq_id', 
                     'label_comp_id', 'id', 'label_atom_id', 
                     'type_symbol', 'Cartn_x', 'Cartn_y', 'Cartn_z']
        self.numbering = pd.DataFrame()
        self.dfl_list = []
        self.dfl = []

        self.g_dict1 = {
            'Gs':'αs',
            'Gia1':'αi1',
            'Gia2':'αi2',
            'Go':'αo',
            'Gta1':'αt1',
            'Gq':'αq',
            'G11':'α11',
        }

        self.g_dict2 = dict((v,k) for k,v in self.g_dict1.items())
        # print('\nDict:', g_dict2)

        self.uniprot_dict = {
            'αi1': ['P63096', 'P63097', 'P10824'],
            'αi2': ['P04899'],
            'αs': ['Q5JWF2', 'P04896', 'P63092'],
            'αo': ['P09471'],
            'α11': ['K7EL62', 'P29992'],
            'αq': ['P50148'],
            'αt1': ['P50148', 'P04695']
        }

        
    # ==============================================================================================================
    
    def dfl_to_list(self):
        self.dfl_list = []
        for i in range(len(self.dfl)):
            self.dfl_list.append(self.dfl[i]['PDB'].iloc[0].upper())
    
    
    def get_dfl_indices(self, filtered_list):
        return [self.dfl_list.index(filtered_list[i]) if (filtered_list[i] in self.dfl_list)\
                else None for i in range(len(filtered_list))]
    
    
    def get_pdb_files(self):
        # just a helper function that returns all pdb files in specified path
        (_, _, filenames) = next(os.walk(self.structure_path))
        if self.shuffle:
            random.shuffle(filenames)
        files = [self.structure_path + x for x in filenames]
        pdb_ids = list(set([x[-8:-4] for x in files]))
        return files, pdb_ids
    
    
    def make_metainfo(self, reload_numbering=False, reload_mapping=False, overwrite=True):
        if reload_mapping:
            del self.mappings
            self.mappings = pd.DataFrame()
        if reload_numbering:
            del self.numbering
            self.numbering = pd.DataFrame()
        for i, pdb_id in tqdm(enumerate(self.pdb_ids)):
            if i == 0:
                if reload_mapping:
                    print("reload mapping i = 0")
                    self.mappings = self.get_mapping(pdb_id)
                elif pdb_id not in list(self.mappings['PDB'].unique()):
                    print("reload mapping i = 0, adding new data for", pdb_id)
                    self.mappings = self.mappings.append(self.get_mapping(pdb_id), ignore_index=True)
                if reload_numbering:
                    protein, family = self.get_prot_info(pdb_id)
                    numbering = self.get_res_nums(protein)
                    numb = pd.DataFrame([pdb_id, protein, family, numbering]).T
                    # numb = [pdb_id, protein, self.entry_to_ac(protein), family, numbering]
                    numb.columns = ['PDB', 'identifier', 'family', 'numbering']
                    self.numbering = self.numbering.append(numb)
                elif pdb_id not in list(self.numbering['PDB'].unique()):
                    protein, family = self.get_prot_info(pdb_id)
                    numbering = self.get_res_nums(protein)
                    numb = pd.DataFrame(data=[pdb_id, protein, family, numbering]).T
                    numb.columns = ['PDB', 'identifier', 'family', 'numbering']
                    self.numbering = self.numbering.append(numb, ignore_index=True)
            else:
                if reload_mapping or (pdb_id not in list(self.mappings['PDB'].unique())):
                    print("loading mapping for", pdb_id)
                    self.mappings = self.mappings.append(self.get_mapping(pdb_id), ignore_index=True)
                if reload_numbering or (pdb_id not in list(self.numbering['PDB'].unique())):
                    protein, family = self.get_prot_info(pdb_id)
                    numbering = self.get_res_nums(protein)
                    numb = pd.DataFrame(data=[pdb_id, protein, family, numbering]).T
                    numb.columns = ['PDB', 'identifier', 'family', 'numbering']
                    self.numbering = self.numbering.append(numb, ignore_index=True)
        if overwrite:
            self.to_pkl_metainfo()

            
    def make_raws(self, overwrite=False):
        self.dfl = []
        for i, pdb_id in tqdm(enumerate(self.pdb_ids)):
            if i < self.limit:
                # only process if the file has not already been generated
                # if not self.reload & 
                if (not os.path.isfile(self.raw_folder + pdb_id + '.pkl')) or overwrite:
                    protein, family = self.get_prot_info(pdb_id)
                    if protein != None:
                        structure = self.load_cifs(pdb_id)
                        structure['identifier'] = protein.upper()
                        if self.remove_hetatm:
                            structure = structure[structure['group_PDB']!='HETATM']
                        structure['label_seq_id'] = structure['label_seq_id'].astype(np.int64)
                        structure['label_comp_sid'] = structure.apply(lambda x:
                                                            gemmi.find_tabulated_residue(x.label_comp_id).one_letter_code, 
                                                            axis=1)
                        self.dfl.append(structure)
        self.to_pkl(folder=self.raw_folder, overwrite=overwrite)
        self.dfl_to_list()
                        
    # ==============================================================================================================
        
    def entry_to_ac(self, entry: str):
        query = 'https://www.uniprot.org/uniprot/' + entry + '.txt'
        response = requests.get(query)
        return response.text.split('\n')[1].split('AC   ')[1][:6]
    
    def get_prot_info(self, pdb_id):
        # query structure
        query = 'https://gpcrdb.org/services/structure/'+pdb_id.upper()+'/'
        response = requests.get(query)
        if len(response.json()) > 0:
            protein = response.json()['protein']
            family = response.json()['family']
            return protein, family
        else:
            return None, None
    
    def get_res_nums(self, protein):
        # query uniprot -> res num
        query = 'https://gpcrdb.org/services/residues/extended/'+protein+'/'
        response = requests.get(query)
        # select res num
        # assign res_num to structure data
        return response.json()
    
    def get_mapping(self, pdb_id):
        maps = get_mappings_data(pdb_id)[pdb_id.lower()]['UniProt']
        uniprots = maps.keys()
        full_table=pd.DataFrame()
        for i, uniprot in enumerate(uniprots):
            table = pd.DataFrame.from_dict(maps[uniprot])
            table['PDB'] = pdb_id
            table['uniprot'] = uniprot
            if i == 0:
                full_table = table
            else:
                full_table = full_table.append(table, ignore_index=True)
        return full_table
    
    
    def save_gprotein_df(self, gprot_df, path='data/'):
        gprot_df.to_pickle(path+'gprotein_df.pkl')
    
    
    def load_gprotein_df(self, path='data/'):
        return pd.read_pickle(path+'gprotein_df.pkl')
    
    # ==============================================================================================================
    
    def load_cifs(self, pdb_id):
        path = 'data/mmcif/' + pdb_id + '.cif'
        try:
            doc = cif.read_file(path)  # copy all the data from mmCIF file
            lol = []  # list of lists
            for b, block in enumerate(doc):
                table = block.find('_atom_site.', self.cols)
                for row in table:
                    lol.append([pdb_id]+list(row))
        
        except Exception as e:
            print("Hoppla. %s" % e)
            sys.exit(1)
        cols = ['PDB']+self.cols
        st = gemmi.read_structure(path)
        model = st[0]
        if len(st) > 1:
            print("There are multiple models!")
        rol = []
        for chain in model:
            for r, res in enumerate(chain.get_polymer()):
                # previous_residue() and next_residue() return previous/next
                # residue only if the residues are bonded. Otherwise -- None.
                prev_res = chain.previous_residue(res)
                next_res = chain.next_residue(res)
                try:
                    phi, psi = gemmi.calculate_phi_psi(prev_res, res, next_res)
                except:
                    phi, psi = np.nan, np.nan
                try:
                    omega = gemmi.calculate_omega(res, next_res)
                except:
                    omega = np.nan
                rol.append([res.label_seq, res.subchain, 
                            degrees(phi), degrees(omega), degrees(psi)])
        cols2 = ['label_seq_id', 'label_asym_id', 'phi', 'omega', 'psi']
        rol_df = pd.DataFrame(data=rol, columns=cols2)
        rol_df['label_seq_id'] = rol_df['label_seq_id'].astype(int)
        lol_df = pd.DataFrame(data=lol, columns=cols)
        lol_df['label_seq_id'] = lol_df.apply(lambda x: int(x.label_seq_id) if x.label_seq_id != '.' else np.nan, axis=1)
        return pd.merge(lol_df, rol_df, how='inner', on=['label_asym_id', 'label_seq_id'])
    
    # ==============================================================================================================   
            
    def to_pkl_metainfo(self):
        self.numbering.to_pickle(self.path + 'data_numbering.pkl')
        self.table.to_pickle(self.path + 'data_table.pkl')
        self.mappings.to_pickle(self.path + 'data_mappings.pkl')
    
    
    def to_pkl(self, mode='', folder='data/processed/', overwrite=False):
        for d, df in enumerate(self.dfl):
            if len(df) <=0:
                print('No data to write!', self.dfl_list[d])
            else:
                pdb_id = df['PDB'].unique()[0]
                if mode=='':
                    if d == 0:
                        print("Writing files without generic numbers...")
                    filename = folder + pdb_id + '.pkl'
                elif mode=='r':
                    if d == 0:
                        print("Writing files with generic numbers on receptors.")
                    filename = folder + pdb_id + '_r.pkl'
                elif mode=='g':
                    if d == 0:
                        print("Writing files with generic numbers on gproteins.")
                    filename = folder + pdb_id + '_g.pkl'
                elif mode=='rg':
                    if d == 0:
                        print("Writing files with generic numbers on receptors and gproteins.")
                    filename = folder + pdb_id + '_rg.pkl'
                else:
                    print("Mode {} not implemented!".format(mode))
                
                if (not os.path.isfile(filename)) or overwrite:
                    df.to_pickle(filename)
    
    # ==============================================================================================================   
    
    def del_pkl(self, folder='data/raw/'):
        files = [f for f in os.listdir(folder) if '.pkl' in f]
        for file in files:
            os.remove(folder + file)
    
    def del_pkl_metainfo(self):
        os.remove(self.path + 'data_numbering.pkl')
        os.remove(self.path + 'data_table.pkl')
        os.remove(self.path + 'data_mappings.pkl')
            
    # ==============================================================================================================
    
    def read_pkl(self, mode='', folder='data/processed/'):
        files = [f for f in os.listdir(folder) if '.pkl' in f]
        
        if 'rg' in mode:
            print("Reading files with generic numbers on receptors and gproteins.")
            files = [f for f in files if ('rg' in f)]
        elif 'g' in mode:
            print("Reading files with generic numbers on gproteins.")
            files = [f for f in files if ('g' in f) and ('r' not in f)]
        elif 'r' in mode:
            print("Reading files with generic numbers on receptors.")
            files = [f for f in files if ('r' in f) and ('g' not in f)]
        else:
            print("Reading files without generic numbers...")
            files = [f for f in files if (not 'r' in f) and (not 'g' in f)]
        
        self.dfl = []
        for f in tqdm(files):
            self.dfl.append(pd.read_pickle(folder+f).reset_index().drop('index', axis=1))
            
            if len(f) >= 8:
                self.dfl_list.append(f[:-(len(f)-4)])
            else:
                self.dfl_list.append('')
        self.dfl_to_list()
    
    def read_pkl_metainfo(self):
        self.numbering = pd.read_pickle(self.path + 'data_numbering.pkl')
        self.table = pd.read_pickle(self.path + 'data_table.pkl')
        self.mappings = pd.read_pickle(self.path + 'data_mappings.pkl')
    
    # ==============================================================================================================    
    
    def get_stacked_maps(self, pdb):
        # add gene to mapping
        mappings_ = self.mappings[self.mappings['PDB']==pdb]
        pref_chain = self.table[self.table['PDB']==pdb.upper()]['Preferred Chain'].iloc[0]
        map_df_list = []
        if len(mappings_)==0:
            print("Did not find {}'s mapping! Have to look it up now.. Please run make_metainfo() before assigning gen. nums!"\
                  .format(pdb))
            mappings_ = self.get_mapping(pdb)
        for j in range(len(mappings_)):
            chain = pd.DataFrame.from_dict(mappings_.iloc[j]['mappings'])['chain_id'].iloc[0]
            identifier = mappings_.iloc[j]['name']
            uniprot = mappings_.iloc[j]['uniprot']
            dict_ = pd.DataFrame.from_dict(mappings_.iloc[j]['mappings'])
            dict_['identifier'] = identifier
            dict_['uniprot'] = uniprot
            map_df_list.append(pd.DataFrame.from_dict(dict_))
        stacked_maps = pd.concat(map_df_list)
        stacked_maps = stacked_maps[stacked_maps['chain_id']==pref_chain]
        stacked_maps['PDB'] = pdb
        return stacked_maps


    def get_generic_nums(self, pdb_id):
        sequence_numbers = []
        amino_acids = []
        generic_numbers = []
        for i in self.numbering[self.numbering['PDB']==pdb_id].iloc[0]['numbering']:
            if i['alternative_generic_numbers'] != []:
                sequence_numbers.append(i['sequence_number'])
                amino_acids.append(i['amino_acid'])
                generic_numbers.append(i['display_generic_number'])
        return list(zip(sequence_numbers, amino_acids, generic_numbers))


    def get_generic_number(self, zpd0, zipped_pos_dict, l2u, comp_sid):
        if l2u >= 0:
            if l2u in zpd0:
                idx = zpd0.index(l2u)
                row = zipped_pos_dict[idx]
                f0 = float(row[2].split('x')[0])
                f1 = float(row[2].split('x')[1])
                if row[1] == comp_sid:
                    return row[2], row[1], f0, f1
                else:
                    return row[2]+'?', row[1], f0, f1
            else:
                return ['', '', 0, 0]
        else:
            return ['', '', 0, 0]
        
        
    def get_uniprot_seq(self, code):
        print("Looking up uniprot fasta sequence for", code)
        baseUrl="http://www.uniprot.org/uniprot/"
        try:
            currentUrl=baseUrl + code + ".fasta"
            response = r.post(currentUrl)
            cData=''.join(response.text)
            Seq=StringIO(cData)
            pSeq=list(SeqIO.parse(Seq,'fasta'))
            pSeq0 = pSeq[0]
            # print("Full Seq:", str(pSeq0.seq))
            return str(pSeq0.seq)
        except:
            return ''

        
    def _get_res_nums(self, roi_idxs, a, b):
        al = list(a)
        bl = list(b)
        ai_ = 0  # note this is the position in the list of pdb seq id numbers (roi_idxs)
        bj_ = 1  # note this is the uniprot seq id
        ab = []  # contains indices of b where a <> b
        abij = []
        for _ in range(len(al)):
            ai = al[_]
            bj = bl[_]
            if ai != '-':
                if bj != '-':
                    # ai: '?', bj: '?'
                    abij.append(roi_idxs[ai_])
                    ab.append(bj_)
                    ai_+=1
                    bj_+=1
                else:
                    # ai: '?', bj: '-'
                    abij.append(roi_idxs[ai_])
                    ab.append(0)
                    ai_+=1
            else:
                # ai: '-', bj: '?'
                # there is no case where ai: '-' and  bi: '-'
                bj_+=1
        return abij, ab

    
    def _label_2_uni_via_uniprot(self, data,
                                pdb_id, uniprot_id, uniprot_identifier, 
                                start_label_seq_id, end_label_seq_id,
                                start_uniprot, end_uniprot,
                                pref_chain):
        print("{}: Using uniprot (looking up {}) as a reference, due to missmatch in SIFTS!"\
              .format(pdb_id, uniprot_id))
        # ---> I need a uniprot number corresponding to each of the label_seq_ids!!!!
        uni_seq = self.get_uniprot_seq(uniprot_identifier)
        # TODO: 1) make a dict for each pdb-residue- and -label_seq_id-pair
        roi = data[(data['label_seq_id'] >= start_label_seq_id) &
                        (data['label_seq_id'] <= end_label_seq_id) &
                        (data['label_atom_id'] == 'CA') &
                        (data['auth_asym_id'] == pref_chain)][['label_seq_id', 'label_comp_sid']]
        roi_idxs = list(roi['label_seq_id'])
        pdb_res_list = ''.join(list(roi['label_comp_sid']))
        #       2) make a dict for each uniprot-residue- and -uniprot_seq_id-pair
        uni_seq_list = list(uni_seq)[start_uniprot:end_uniprot]
        idxs = [x+1 for x in range(len(uni_seq_list))]
        uni_dict = dict(zip(idxs, uni_seq_list))
        # DO A SEQ ALIGNMENT
        a, _, b, _ = self._get_pairwise_indices(pdb_res_list, uni_seq, open_gap_pen=-10, ext_gap_pen=-0.05)
        abij, ab = self._get_res_nums(roi_idxs, a, b)
        label_dict = dict(zip(abij, ab))
        # I need a dict of pdb_indices -> uniprot_indices
        lines = data[(data['label_seq_id'] >= start_label_seq_id) &
                     (data['label_seq_id'] <= end_label_seq_id) &
                     (data['label_atom_id'] == 'CA') &
                     (data['auth_asym_id'] == pref_chain)]
        print("start pdb:", start_label_seq_id)
        print("start uni:", start_uniprot)
        
        for k in range(len(lines)):
            line = lines.iloc[k]
            idx = list(lines.index)[k]
            pdb_seq_label = line['label_seq_id']
            keys = [int(x) for x in list(label_dict.keys())]
            if int(pdb_seq_label) in list(keys):
                data.at[idx, 'label_2_uni'] = label_dict[int(pdb_seq_label)]
            else:
                data.at[idx, 'label_2_uni'] = 0
        return data
    
    
    def _check_label_2_uniprot(self, data, max_frac=0.1, max_len=8):
        l2uni = list(data['gen_pos'])
        l2uni_qm = [x for x in l2uni if '?' in x]
        l2uni_qm_idx = [y for y, x in enumerate(l2uni) if '?' in x]
        if len(l2uni) * max_frac < len(l2uni_qm):
            return True
        intervals = find_sections(l2uni_qm_idx, min_length = 3)
        interval_lengths = [x[1]- x[0] for x in intervals]
        print("intervals :", interval_lengths)
        error = False
        for intlen in interval_lengths:
            if intlen > max_len:
                error = True
        return error
    
    
    # def _assign_res_nums_g(self, pdb_id, mappings, structure, uniprot_gprot_list, gprot_df, res_table):
    def _assign_res_nums_r(self, structure, ref_uniprot=True):
        def make_r_columns(data):
            data['label_2_uni'] = 0
            data['gen_pos'] = ''
            data['gen_pos1'] = 0
            data['gen_pos2'] = 0
            data['uniprot_comp_sid'] = ''
            return data
        pdb_id = structure['PDB'].iloc[0]
        data = structure.reset_index().drop('index', axis=1)
        cols = data.columns
        columns = ['gen_pos', 'gen_pos1', 'gen_pos2', 'uniprot_comp_sid']
        maps_stacked = self.get_stacked_maps(pdb_id)
        
        if 'residue_number' in maps_stacked.index:
            pass
        else:
            print("Found no mapping -> not assigning any residue numbers")
            return data
        if type(maps_stacked[maps_stacked['PDB']==pdb_id].\
                loc['residue_number'][['chain_id', 'start','end','unp_start','unp_end', 'identifier', 'PDB']])\
                    == pandas.core.series.Series:
            pref_mapping = maps_stacked[maps_stacked['PDB']==pdb_id].loc['residue_number']\
                [['chain_id', 'start','end','unp_start','unp_end', 'identifier', 'PDB', 'uniprot']].to_frame().T
        else:
            pref_mapping = maps_stacked[maps_stacked['PDB']==pdb_id].\
                loc['residue_number'][['chain_id', 'start','end','unp_start','unp_end', 'identifier', 'PDB', 'uniprot']]
        pref_chain = pref_mapping['chain_id'].iloc[0]
        pref_mapping = pref_mapping.sort_values('start')
        uniprot_identifier_ = data[data['PDB']==pdb_id]['identifier'].unique()
        uniprot_identifier = uniprot_identifier_[0]
        natoms = len(data[data['PDB']==pdb_id])
        data = make_r_columns(data)
        zipped_pos_dict = self.get_generic_nums(pdb_id)
        zpd0 = list(zip(*zipped_pos_dict))[0]
        
        for j in range(len(pref_mapping)):
            row = pref_mapping.iloc[j].to_dict()
            map_identifier = row['identifier']
            map_pdb = row['PDB']
            start_label_seq_id = row['start']
            end_label_seq_id = row['end']
            start_uniprot = row['unp_start']
            end_uniprot = row['unp_end']
            uniprot_id = row['uniprot']
            if map_identifier == uniprot_identifier:
                print()
                print(map_pdb, uniprot_id)
                error = (end_label_seq_id-start_label_seq_id) != (end_uniprot-start_uniprot)
                if not error:
                    print("Trying to assign error free uniprotlabels based on SIFTS!")
                    idxs = [x for x in range(natoms+1) \
                            if ((x <= end_label_seq_id) & (x >= start_label_seq_id))]
                    vals = [x + start_uniprot - start_label_seq_id for x in range(natoms+1) \
                            if ((x <= end_label_seq_id) & (x >= start_label_seq_id))]
                    for k, idx in enumerate(idxs):
                        line = data[(data['PDB'] == pdb_id) &
                                    (data['label_seq_id'] == idx) &
                                    (data['label_atom_id'] == 'CA') &
                                    (data['auth_asym_id'] == pref_chain)]
                        lines = len(line)
                        if len(line) > 0:
                            data.at[line.index[0], 'label_2_uni'] = int(vals[k])
                        
                    data[['gen_pos', 'uniprot_comp_sid', 'gen_pos1', 'gen_pos2']] = data.apply(
                        lambda x: self.get_generic_number(zpd0, zipped_pos_dict, x.label_2_uni, x.label_comp_sid) 
                        if x.PDB == pdb_id 
                        else [x.gen_pos, x.uniprot_comp_sid, x.gen_pos1, x.gen_pos2], axis=1, result_type='expand')
                    # TODO: CHECK FOR ERRORS!
                    error = self._check_label_2_uniprot(data, max_frac=0.1, max_len=8)
                    print("Error Status:", error)
                if error and ref_uniprot:
                    print("Found error!")
                    data = self._label_2_uni_via_uniprot(data,
                                                         pdb_id, uniprot_id, uniprot_identifier, 
                                                         start_label_seq_id, end_label_seq_id,
                                                         start_uniprot, end_uniprot,
                                                         pref_chain)
                    
                    data[['gen_pos', 'uniprot_comp_sid', 'gen_pos1', 'gen_pos2']] = data.apply(
                        lambda x: self.get_generic_number(zpd0, zipped_pos_dict, x.label_2_uni, x.label_comp_sid) 
                        if x.PDB == pdb_id 
                        else [x.gen_pos, x.uniprot_comp_sid, x.gen_pos1, x.gen_pos2], axis=1, result_type='expand')
            else:
                # Didnt find correct uniprotmap (not a gpcr) ==> map_identifier
                pass
        
        # Generate generic numbers
        if type(data) == pandas.core.series.Series:
            data = data.to_frame().T
        
        # THIS APPLY CALL IS VERY SLOW!!!
        data[['gen_pos', 'uniprot_comp_sid', 'gen_pos1', 'gen_pos2']] = data.\
            apply(lambda x: self.get_generic_number(zpd0, zipped_pos_dict, x.label_2_uni, x.label_comp_sid) if x.PDB==pdb_id\
                  else [x.gen_pos, x.uniprot_comp_sid, x.gen_pos1, x.gen_pos2], axis=1, result_type='expand')
        
        # 4 replace slow apply call
        # 5 assign uniprot labels in general!?
        print("Final Error Check...")
        error = self._check_label_2_uniprot(data, max_frac=0.05, max_len=8)
        print("Error Status:", error)
        return data
    
    def assign_generic_numbers_r(self, f, pdb_ids=[], overwrite=True, folder='data/processed/', ref_uniprot=True):
        if isinstance(pdb_ids, str):
            pdb_ids = [pdb_ids]
        if len(pdb_ids) > 0:
            dfl_pdbs = pdb_ids
        else:
            dfl_pdbs=list(set(list(f['PDB'])))
        dfl_indices=self.get_dfl_indices(dfl_pdbs)
        for i in trange(len(dfl_indices)):
            if self.allow_exception:
                pdb_id = self.dfl_list[dfl_indices[i]]
                if (os.path.isfile(folder + pdb_id + '.pkl')) or (overwrite==False):
                    # load processed file instead of processing it anew
                    pass
                try:
                    structure = self.dfl[dfl_indices[i]]
                    s = self._assign_res_nums_r(structure, ref_uniprot=ref_uniprot)
                    self.dfl[dfl_indices[i]] = s
                    print("assigned dfl[{}] generic residue numbers for the receptor...".format(dfl_indices[i]))
                except:
                    print("Error parsing", pdb_id)
            else:
                pdb_id = self.dfl_list[dfl_indices[i]]
                structure = self.dfl[dfl_indices[i]]
                s = self._assign_res_nums_r(structure, ref_uniprot=ref_uniprot)
                self.dfl[dfl_indices[i]] = s
                print("assigned dfl[{}] generic residue numbers for the receptor...".format(dfl_indices[i]))
        
        self.to_pkl(mode='r', folder=folder, overwrite=overwrite)
        self.dfl_to_list()
    
    # ==============================================================================================================        
    
    def apply_filter(self, f):
        self.table = f
        pdb_ids = list(self.table['PDB'])
        dfl = []
        for df in self.dfl:
            if df['PDB'].iloc[0] in pdb_ids:
                dfl.append(df)
        self.dfl = dfl
        self.dfl_to_list()

    
    def make_filter(self,
                    pdb_ids=[],
                    Species=None,
                    State=None,
                    Cl=None,
                    Family=None,
                    Subtype=None,
                    Resolution=None,
                    Function=None,
                    gprotein=False):
        data = self.table
        if isinstance(pdb_ids, str):
            pdb_ids = [pdb_ids]
        if len(pdb_ids) > 0:
            data = data[data['PDB'].isin(pdb_ids)]
        if Species != None:
            data = data[data['Species']==Species]
        if State != None:
            data = data[data['State']==State]
        if Cl != None:
            data = data[data['Cl.'].str.contains(Cl)]
        if Family != None:
            data = data[data['Family']==Family]
        if Subtype != None:
            data = data[data['Subtype']==Subtype]
        if Resolution != None:
            data = data[data['Resolution']<=Resolution]
        if Function != None:
            data = data[data['Species']==Function]
        if gprotein:
            data = self._filter_table_via_gprotein(data)
        return data
        
        
    def _filter_table_via_gprotein(self, table):
        allowed_Fam_ = ['Gi/o', 'Gs', 'Gq/11']
        allowed_Fam = '|'.join(allowed_Fam_)
        allowed_Sub_ = ['αi1', 'αi2', 'αs', 'αo', 'α11', 'αq', 'αt1']
        allowed_Sub = '|'.join(allowed_Sub_)
        table = table[table['Family'].str.contains(allowed_Fam) &
                      table['Subtype'].str.contains(allowed_Sub)].reset_index(drop=True)
        return table
    
    
    # ==============================================================================================================   
    
    
    def load_gen_res_nums_gprot(self, path='data/alignments/residue_table.xlsx'):
        res_table = pd.read_excel(path, engine='openpyxl')
        keep_cols = ['CGN',
                 'G(s) subunit alpha isoforms short Human', 
                 'G(t) subunit alpha-1 Human', 
                 'G(i) subunit alpha-1 Human', 'G(i) subunit alpha-2 Human', 
                 'G(o) subunit alpha Human',
                 'G(q) subunit alpha Human',
                 'subunit alpha-11 Human']
        drop_cols = [x for x in res_table.columns if x not in keep_cols]
        res_table=res_table.drop(drop_cols, axis=1)
        res_table.columns = ['CGN', 'Gs', 'Gta1', 'Gia1', 'Gia2', 'Go', 'Gq', 'G11']
        # These are the rows corresponding to "beginnings/ends" of new G protein sections (nan rows)
        res_table = res_table.drop(res_table[res_table.iloc[:, 1:].isnull().all(1)].index).reset_index(drop=True)
        domain_list = res_table[res_table.iloc[:, 1:].isnull().all(1)]['CGN'].to_list()
        return res_table, domain_list
    
    
    def _get_res_num_df(self, res_table, subtype):
        res_num_dict = {}
        for i in range(len(res_table)):
            if res_table[subtype].iloc[i] != '-':
                idx = res_table['CGN'].iloc[i]
                _ = res_table[subtype].iloc[i]
                res = _[0]
                pos = _[1:]
                res_num_dict[int(pos)] = (res, idx)
        df = pd.DataFrame(res_num_dict).T
        df.columns = ['res', 'idx']
        return df
    
    
    def _get_uniprot_gprot_seqs(self, uniprot_gprot_list):
        baseUrl="http://www.uniprot.org/uniprot/"
        gprot_uniprot_seq_dict = {}
        print("Downloading uniprot gprotein sequences...")
        for code in tqdm(uniprot_gprot_list):
            # get uniprot sequence
            time.sleep(1)
            try:
                currentUrl=baseUrl + code + ".fasta"
                response = r.post(currentUrl)
                cData=''.join(response.text)
                Seq=StringIO(cData)
                pSeq=list(SeqIO.parse(Seq,'fasta'))
                pSeq0 = pSeq[0]
                seq = str(pSeq0.seq)
                gprot_uniprot_seq_dict[code] = seq
            except:
                gprot_uniprot_seq_dict[code] = ''
        return gprot_uniprot_seq_dict
    
    
    def get_gproteins_in_complex_df(self, res_table):
        uniprot_gprot_list = functools.reduce(operator.iconcat, list(self.uniprot_dict.values()), [])
        def _get_uniprot_types(uniprot, uniprot_dict, g_dict):
            for i, u in enumerate(list(uniprot_dict.values())):
                if uniprot in u:
                    idx = u.index(uniprot)
                    subtype = list(uniprot_dict.keys())[i]
                    return subtype, g_dict[subtype]
        uniprot_gprot_seq_dict = self._get_uniprot_gprot_seqs(uniprot_gprot_list)
        gprot_df = pd.DataFrame(uniprot_gprot_seq_dict.items(), columns=['Uniprot', 'Sequence'])
        gprot_df[['Subtype', 'Family']] = gprot_df\
            .apply(lambda x: _get_uniprot_types(x.Uniprot, self.uniprot_dict, self.g_dict2), axis=1, result_type='expand')
        gprot_df['FamilySequence'] = gprot_df.apply(lambda x: ''.join(list(self._get_res_num_df(res_table, x.Family)['res'])), axis=1)
        return gprot_df
    
    
    def annotate_gproteins_in_complex(self, dfl, dfl_list, mapping, table, res_table, uniprot_dict, g_dict2):
        for i in range(len(table)):
            pdb_id = table.iloc[i]['PDB']
            if pdb_id in dfl_list:
                dfl_idx = dfl_list.index(pdb_id)
                subtype = complex_info['Subtype']
                # print("Corr. uniprot id:", uniprot_dict[subtype])
                pdb_map = mapping[(mapping['PDB'] == pdb_id)]
                pdb_map = pdb_map[pdb_map['uniprot'].isin(uniprot_dict[subtype])]
                if len(pdb_map) > 0:
                    print('\n\n\n',pdb_id)
                    print("Looking up subtype", 'G'+subtype)
                    print(pdb_map['uniprot'])
                    res_num_df = self._get_res_num_df(res_table, subtype)
    
    
    def get_stacked_maps_g(self, pdb_id, mappings, uniprot_gprot_list):
        # add gene to mapping
        map_df_list = []
        mappings_ = mappings[(mappings['PDB']==pdb_id) & 
                             (mappings['uniprot'].isin(uniprot_gprot_list))]
        uniprots = []
        for j in range(len(mappings_)):
            u = mappings_['uniprot'].iloc[j]
            for k in range(3):
                uniprots.append(u)
            map_df_list.append(pd.DataFrame.from_dict(mappings_['mappings'].iloc[j]))
        _ = pd.concat(map_df_list)
        _['PDB'] = pdb_id
        _['uniprot'] = uniprots
        return _
    
    def _get_res_num_df(self, res_table, subtype):
        res_num_dict = {}
        for i in range(len(res_table)):
            if res_table[subtype].iloc[i] != '-':
                idx = res_table['CGN'].iloc[i]
                _ = res_table[subtype].iloc[i]
                res = _[0]
                pos = int(_[1:]) + 1   # this + 1 is important?
                res_num_dict[pos] = (res, idx)
        df = pd.DataFrame(res_num_dict).T
        df.columns = ['res', 'idx']
        return df


    def _get_pairwise_indices(self, a, b, open_gap_pen=-3, ext_gap_pen=-1):
        alignment = pairwise2.align.globalms(a, b, 3, -1, open_gap_pen, ext_gap_pen)[0]
        indices_a = []
        indices_b = []
        ia = 0
        ib = 0
        assert len(alignment[0][0]) == len(alignment[0][1]), print("?? Alignment length missmatch! ??")
        for i in range(len(alignment[0])):
            ia+=1
            ib+=1
            if alignment[0][i] != '-':
                indices_a.append(ia)
            if alignment[1][i] != '-':
                indices_b.append(ib)
        return alignment[0], indices_a, alignment[1], indices_b


    def _make_cgn_df(self, a, b, res_num_df):
        indices_dict = {}
        j = 0  # index of pointer in a
        k = 0  # index of pointer in b
        for i in range(len(a)):
            if (a[i] != '-') & (b[i] != '-'):
                indices_dict[j] = (a[i], b[i], k, res_num_df.iloc[k]['idx'])
                j+=1
                k+=1
            elif (a[i] == '-') & (b[i] != '-'):
                # we do not update our dict (since we dont care about anything not within our sequence/structure)
                k+=1
            elif (a[i] != '-') & (b[i] == '-'):
                indices_dict[j] = (a[i], '-', -1, '-')
                j+=1
            else:
                # this should not exist
                pass
        cgn_df = pd.DataFrame.from_dict(indices_dict).T
        cgn_df.columns = ['seq_res', 'fam_seq_res', 'fam_seq_pos', 'cgn']
        return cgn_df


    def _get_cgn_df(self, gprot_df, idx, res_table):
        f = gprot_df.iloc[idx]['Family']
        a = gprot_df.iloc[idx]['Sequence']
        b = gprot_df.iloc[idx]['FamilySequence']
        a, _, b, _ = self._get_pairwise_indices(a, b)
        res_num_df = self._get_res_num_df(res_table, f)
        return self._make_cgn_df(a, b, res_num_df)

    
    def _assign_res_nums_mini_g(self, pdb_id, structure, uniprot_gprot_list, gprot_df, res_table):
        print("_assign_res_nums_mini_g")
        uniprot_gprot_list = [x for x in uniprot_gprot_list if x != 'K7EL62']
        # 1) split complex into chains
        chains = list(structure['auth_asym_id'].unique())
        best = 10
        for c in chains:
            for gprot in uniprot_gprot_list:
                r_flag = len(structure[(structure['auth_asym_id']==c) &
                                       (structure['gen_pos'].str.contains('x'))]) > 1
                # 2) allign each part to each gprot-seq
                chain_seq = structure[(structure['auth_asym_id']==c) &
                                      (structure['label_atom_id']=='CA')]['label_comp_sid'].to_list()
                chain_seq = ''.join(chain_seq)
                gprot_seq = gprot_df[gprot_df['Uniprot']==gprot]['Sequence'].iloc[0]
                a, _, b, _ = self._get_pairwise_indices(chain_seq, gprot_seq, open_gap_pen=-10, ext_gap_pen=-.05)
                # perfect would be: a gprotein fits 100% ===> len(a) - len(gprot_seq) = len(chain) - len(gprot_seq)
                # 100% gprot identity: len(a) = len(chain)
                # each residue of gprot that doesnt match the chain then leads to more error
                score_gprot = (len(a) - len(chain_seq)) / len(a)
                score_chain = (len(a) - len(gprot_seq)) / len(a)
                # print("Chain {} <> Gprot {}\nG-Score (Gprot seq identity): {}\nR-Score (Receptor seq identity): {}"\
                #       .format(c, gprot, score_gprot, score_chain))
                mul_score = max(score_gprot, 0.01) * max(score_chain, 0.01)  # This score is pretty cool!
                if mul_score < best:
                    best_match = (c, gprot)
                    best_align = (a, b)
                    best = mul_score
                # Problem:  high sequence identities between gproteins!
                # Solution: 
                # 3.1) select gprot that aligns BEST to ANY chain  (DONE)
                # 3.2) check if for regions I could NOT align with said gprotein, there is a better mapping  (TBD)
                #      ===> this would require a local score to be calculated telling me if part of the gprotein fits well (e.g. the gaps)
        # 4) I need to create a maps_stacked df and continue as if nothing happened  ==> bring it into a format of "maps_stacked"
        #    I need starting and end-points of the aligned sequences
        #    map1 = {start_uniprot: 1, end_uniprot:  13, start_pdb: 23, end_pdb}
        #    list_of_maps = [map1, map2, map3]
        list_of_maps = self._get_maps_from_alignment(best_align[0], best_align[1], structure, best_match[0])
        return self._create_maps_stacked_from_align(list_of_maps, chain_id=best_match[0], pdb_id=pdb_id, uniprot=best_match[1])
    
    
    def _get_roi_from_padded_seq(self, l, min_size):
        list_of_conseq_num_list = []
        num_list = []
        for i, li in enumerate(list(l)):
            if li != '-':
                num_list.append(i)
            else:
                if len(num_list) >= min_size:
                    list_of_conseq_num_list.append((min(num_list), max(num_list)))
                    num_list = []
        if len(num_list) >= min_size:
            list_of_conseq_num_list.append((min(num_list), max(num_list)))
        return list_of_conseq_num_list
    
        
    def _get_maps_from_alignment(self, a, b, structure, chain_id, min_size=8):
        list_of_maps = []
        # 1 get pdb_chain_structure
        chain = structure[(structure['auth_asym_id']==chain_id) &
                          (structure['label_atom_id']=='CA')]
        # i need to look at chain>label_seq_id and chain>label_comp_sid
        # get starting and end-points of all regions of size >= min_size (in the pdb chain/structure)
        label_2_uni = 0
        indices_dict = {}
        pdb_labels = list(chain['label_seq_id'])
        pdb_seq = list(chain['label_comp_sid'])
        ai_ = 0
        bi_ = 0
        # get regions of interest from alignment (all regions above a minimum length)
        locnl_a = self._get_roi_from_padded_seq(a, min_size)
        locnl_b = self._get_roi_from_padded_seq(b, min_size)
        # find overlap of rois
        lroi = []
        for la in locnl_a:
            for lb in locnl_b:
                range_a = range(la[0], la[1]+1)
                range_b = range(lb[0], lb[1]+1)
                overlap = list(set(range_a).intersection(set(range_b)))
                if len(overlap)>0:
                    start = min(overlap)
                    end = max(overlap)
                    lroi.append((start, end))
        df_roi = pd.DataFrame(dict(lroi).items(), columns =  ['start', 'end'])
        
        ls_a = list(zip(range(0, len(list(a))), list(a)))         
        ls_a = [x for x in ls_a if x[1]!='-']
        ls_a1, ls_a2 = list(zip(*ls_a))
        ls_a = list(zip(*list(zip(ls_a1, ls_a2, pdb_labels))))
        
        df_a = pd.DataFrame({'align_idx': ls_a[0], 
                             'pdb_res': ls_a[1], 
                             'pdb_seq_id': ls_a[2]})
        
        func_a = lambda x: (df_a.align_idx == x.start) | (df_a.align_idx == x.end)
        df_a = df_a[df_roi.apply(func_a, axis=1).any()]
        
        ls_b = list(zip(range(0, len(list(b))), list(b)))
        ls_b1, ls_b2 = list(zip(*ls_b))
        ls_b = list(zip(*list(zip(ls_b1, ls_b2, list(range(1, len(ls_b1)+1))))))
        
        df_b = pd.DataFrame({'align_idx': ls_b[0], 
                             'uniprot_res': ls_b[1], 
                             'uniprot_seq_id': ls_b[2]})
        
        func_b = lambda x: (df_b.align_idx == x.start) | (df_b.align_idx == x.end)
        df_b = df_b[df_roi.apply(func_b, axis=1).any()]
        map_df = pd.merge(df_a, df_b, on='align_idx').reset_index(drop=True)
                
        if len(map_df)%2 != 0:
            print("Did not find equal number of start and end points in region of interests!\n", map_df)
            return []
        
        
        list_of_maps = []
        for m in range(int(len(map_df)/2)):
            start = map_df.iloc[m*2]
            end = map_df.iloc[m*2+1]
            start_uniprot = start['uniprot_seq_id']
            end_uniprot = end['uniprot_seq_id']
            start_pdb = start['pdb_seq_id']
            end_pdb = end['pdb_seq_id']
            list_of_maps.append({'start_uniprot': start_uniprot, 
                                 'end_uniprot': end_uniprot, 
                                 'start_pdb': start_pdb, 
                                 'end_pdb': end_pdb})
            
        return list_of_maps  # map:= {start_uniprot: 1, end_uniprot:  13, start_pdb: 23, end_pdb: 35}
    
    
    def _create_maps_stacked_from_align(self, list_of_maps, chain_id, pdb_id, uniprot):
        cols = ['entity_id', 'chain_id', 'start', 'end', 'unp_start', 'unp_end', 'struct_asym_id', 'PDB', 'uniprot']
        rows = ['author_residue_number', 'author_insertion_code', 'residue_number']        
        map_df_list = []
        # i guess struct_asym_id == chain_id for now...
        for m, map_ in enumerate(list_of_maps):
            df = pd.DataFrame(columns = cols)
            df['index'] = rows
            df.set_index('index', inplace=True)
            df['entity_id'] = [1 for _ in range(3)]
            df['chain_id'] = [chain_id for _ in range(3)]
            df['struct_asym_id'] = [chain_id for _ in range(3)]   # I CONFUSE THESE TWO ALL THE TIME..
            df['PDB'] = [pdb_id for _ in range(3)]
            df['uniprot'] = [uniprot for _ in range(3)]
            df['start'] = [map_['start_pdb'] for _ in range(3)]
            df['end'] = [map_['end_pdb'] for _ in range(3)]
            df['unp_start'] = [map_['start_uniprot'] for _ in range(3)]
            df['unp_end'] = [map_['end_uniprot'] for _ in range(3)]
            map_df_list.append(df)
        stacked_maps = pd.concat(map_df_list)
        return stacked_maps
        
    
    def _assign_res_nums_g(self, pdb_id, mappings, structure, uniprot_gprot_list, gprot_df, res_table, fill_H5=False):
        
        if self.table[self.table['PDB']==pdb_id]['State'].iloc[0] == 'Active':
            print("assinging res_nums to gprot of", pdb_id)
            try:
                maps_stacked = self.get_stacked_maps_g(pdb_id, mappings, uniprot_gprot_list)
                no_mapping=False
                print("Mapping found!")
            except:
                print("No mapping (no uniprot-seq) information found! ====> Assigning labels ourself!")
                no_mapping=True
                maps_stacked = self._assign_res_nums_mini_g(pdb_id, structure, uniprot_gprot_list, gprot_df, res_table)
            if 'residue_number' in maps_stacked.index:
                pass
            else:
                print("Empty mapping found!")
                return structure
        else:
            return structure

        selection = maps_stacked[maps_stacked['PDB']==pdb_id]
        if type(selection.loc['residue_number']) == pandas.core.series.Series:
            mapping = selection.loc['residue_number'].to_frame().T
        else:
            mapping = selection.loc['residue_number']
        mapping = mapping.sort_values('start')
        label_seq_ids = list(structure['label_seq_id'])
        if len(label_seq_ids) == 0:
            print("Did not find label_seq_ids:", pdb_id)
            return structure
        nres = max(label_seq_ids)
        structure['gprot_pos'] = ''
        structure['uniprot_comp_id'] = ''
        structure['fam_comp_id'] = ''
        for j in range(len(mapping)):
            row = mapping.iloc[j].to_dict()
            pref_chain = row['chain_id']
            uniprot = row['uniprot']
            gprot_idx = list(gprot_df['Uniprot']).index(uniprot)
            if gprot_idx >= 0:
                cgn_df = self._get_cgn_df(gprot_df, gprot_idx, res_table)
                start_label_seq_id = row['start']
                start_uniprot = row['unp_start']
                end_label_seq_id = row['end']
                end_uniprot = row['unp_end']
                
                
                print("start_uni {} > end_uni {}".format(start_uniprot, end_uniprot))
                print("start_pdb {} > end_pdb {}".format(start_label_seq_id, end_label_seq_id))
                # the start -> end should be a shift (if not we have to do an alignment eg. 6FUF)
                # iterate from start_label_seq_id to end_label_seq_id
                
                # THESE ARE OUTLIERS!! (SELF-ALIGNED)
                if no_mapping:
                    print("mini g setting index += 1")
                    cgn_df.index += 1
                    # print("looking for idx_uni in cgn_df {}".format(cgn_df))
                    seq_ids = structure[(structure['label_seq_id'] >= start_label_seq_id) &
                                        (structure['label_seq_id'] <= end_label_seq_id) &
                                        (structure['label_atom_id'] == 'CA') &
                                        (structure['auth_asym_id'] == pref_chain)]['label_seq_id']
                    seq_ids = list(seq_ids)
                    for k, idx_seq in enumerate(seq_ids):
                        idx_uni = k + start_uniprot
                        if idx_uni < 1000:
                            line = structure[(structure['label_seq_id'] == idx_seq) &
                                             (structure['label_atom_id'] == 'CA') &
                                             (structure['auth_asym_id'] == pref_chain)]
                            if len(line) > 0:
                                structure.at[line.index[0], 'label_2_uni'] = idx_uni
                                if idx_uni in list(cgn_df.index):
                                    idx_uni -= 1
                                    structure.at[line.index[0], 'gprot_pos'] = cgn_df.iloc[idx_uni]['cgn']  # this is the error
                                    structure.at[line.index[0], 'uniprot_comp_id'] = cgn_df.iloc[idx_uni]['seq_res']
                                    structure.at[line.index[0], 'fam_comp_id'] = cgn_df.iloc[idx_uni]['fam_seq_res']
                                    print("Found corr. gen num! idx_uni: {} <> idx_seq: {} <> gen pos {}"\
                                          .format(idx_uni, idx_seq, structure.at[line.index[0], 'gprot_pos']))
                # THESE ARE THE GPROTS WITH SIFTS
                else:
                    seq_len = end_label_seq_id - start_label_seq_id
                    for k in range(seq_len):
                        idx_seq = k + start_label_seq_id + 1
                        idx_uni = k + start_uniprot
                        line = structure[(structure['label_seq_id'] == idx_seq) &
                                         (structure['label_atom_id'] == 'CA') &
                                         (structure['auth_asym_id'] == pref_chain)]
                        if len(line) > 0:
                            structure.at[line.index[0], 'label_2_uni'] = idx_uni
                            if line['label_2_uni'].iloc[0] in list(cgn_df.index):
                                structure.at[line.index[0], 'gprot_pos'] = cgn_df.iloc[idx_uni]['cgn']
                                structure.at[line.index[0], 'uniprot_comp_id'] = cgn_df.iloc[idx_uni]['seq_res']
                                structure.at[line.index[0], 'fam_comp_id'] = cgn_df.iloc[idx_uni]['fam_seq_res']
                                print("Found corr. gen num! idx_uni: {} <> idx_seq: {} <> gen pos {}"\
                                      .format(idx_uni, idx_seq, structure.at[line.index[0], 'gprot_pos']))
            else:
                return structure
            if fill_H5:
                structure = self._assign_generic_numbers_h5(structure=structure)
        # TODO: if the best alignment to the full gprotein is not of the specified type (gi/o, gs etc)
        # ====> then first assign labels based on the best alignment of correct tpye
        # ====> then complete the labels with anything else (best fit)
        return structure
    
    
    def _assign_generic_numbers_h5(self, structure, max_h5_error=20, h5_tail_discrep=5):
        # 1. check if any part of the complex is labelled as a gprot
        # 2. if yes, find end of that section and if it is not labelled until H5.25 label them backwards!
        if len(list(structure['gprot_pos'].unique())) > 1:
            h5_region = structure[structure['gprot_pos'].str.contains('H5')]
            if len(h5_region) > 0:
                h5_region_nums = [int(x.split('H5.')[-1]) for x in list(h5_region['gprot_pos'].unique())]
                if max(h5_region_nums) < 26:
                    h5_chain = h5_region['label_asym_id'].iloc[0]
                    h5_chain_region = structure[structure['label_asym_id'] == h5_chain]
                    # check if helix 5 region is about thesize of the final part of the chain containing H5 residues
                    end_h5 = max(h5_chain_region['label_seq_id'])
                    end_labelled_h5 = h5_region.iloc[len(h5_region)-1]['label_seq_id']
                    unlabelled_h5 = end_h5 - end_labelled_h5
                    if (unlabelled_h5 < max_h5_error) & (unlabelled_h5 > 0):
                        final_h5_label = h5_region_nums[-1]
                        h5_tail_discrepancy = abs((unlabelled_h5 + final_h5_label) - 26)
                        # label from the end backwards
                        print(structure['PDB'].iloc[0], "unlabelled:", unlabelled_h5)
                        for i in range(unlabelled_h5):
                            label_seq_id_ = end_h5 + i
                            line = structure[(structure['label_seq_id']==label_seq_id_) &
                                             (structure['label_atom_id'] == 'CA') &
                                             (structure['label_asym_id']==h5_chain)]
                            print("label_seq_id", label_seq_id_)
                            if len(line) > 0:
                                print("assigning {} of gprotein gen. number: G.H5.{}".format(label_seq_id_, 
                                                                                             26-(unlabelled_h5-i-1)))
                                structure.at[line.index[0], 'gprot_pos'] = 'G.H5.'+str(26-(unlabelled_h5-i-1))
            return structure
        else:
            return structure
    
    
    def assign_generic_numbers_g(self, f, pdb_ids=[], overwrite=True, folder='data/processed/', fill_H5=False):
        # is the filter from "filter_dfl_via_table" -> if none provided give f = self.table
        if not isinstance(pdb_ids, list):
            pdb_ids = [pdb_ids]
        dfl_list_f=list(set(list(f['PDB'])))
        if len(pdb_ids) > 0:
            dfl_list_f = pdb_ids
        
        dfl_list_f_indices = self.get_dfl_indices(dfl_list_f)
        dfl_list_f_indices = [x for x in dfl_list_f_indices if x != None]
        
           
        res_table = self.load_gen_res_nums_gprot()[0]
        if os.path.isfile('data/'+'gprotein_df.pkl'):
            gprot_df = self.load_gprotein_df()
        else:
            gprot_df = self.get_gproteins_in_complex_df(res_table)
            self.save_gprotein_df(gprot_df)
        uniprot_gprot_list = functools.reduce(operator.iconcat, list(self.uniprot_dict.values()), [])
        
        for i in trange(len(dfl_list_f_indices)):
            if self.allow_exception:
                try:
                    pdb_id = self.dfl_list[dfl_list_f_indices[i]]
                    structure = self.dfl[dfl_list_f_indices[i]]
                    s = self._assign_res_nums_g(pdb_id,
                                                self.mappings[self.mappings['PDB']==pdb_id],
                                                structure, 
                                                uniprot_gprot_list, 
                                                gprot_df,
                                                res_table,
                                                fill_H5)
                    self.dfl[dfl_list_f_indices[i]] = s
                except:
                    print("Error parsing", pdb_id)
            else:
                pdb_id = self.dfl_list[dfl_list_f_indices[i]]
                structure = self.dfl[dfl_list_f_indices[i]]
                s = self._assign_res_nums_g(pdb_id,
                                            self.mappings[self.mappings['PDB']==pdb_id],
                                            structure, 
                                            uniprot_gprot_list, 
                                            gprot_df,
                                            res_table,
                                            fill_H5)
                self.dfl[dfl_list_f_indices[i]] = s
        self.to_pkl(mode='rg', folder=folder, overwrite=overwrite)
        self.dfl_to_list()

