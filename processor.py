from utils3 import *
from deprecated.plotting import *
from deprecated.gpcrdb_soup import *
import sys
import functools
import operator
import random
from tqdm import tqdm, trange
import time
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
    
    def get_pdb_files(self):
        # just a helper function that returns all pdb files in specified path
        (_, _, filenames) = next(os.walk(self.structure_path))
        if self.shuffle:
            random.shuffle(filenames)
        files = [self.structure_path + x for x in filenames]
        pdb_ids = list(set([x[-8:-4] for x in files]))
        return files, pdb_ids
    
    def get_metatable(self):        
        self.table = pd.read_pickle(self.path_table)
        self.table.to_pickle(self.path + 'data_table.pkl')
    
    def make_metainfo(self):
        for i, pdb_id in tqdm(enumerate(self.pdb_ids)):
            if i < self.limit:
                protein, family = self.get_prot_info(pdb_id)
                if protein == None:
                    pass
                else:
                    numbering = self.get_res_nums(protein)
                    if i == 0:
                        self.mappings = self.get_mapping(pdb_id)
                        numb = pd.DataFrame([pdb_id, protein, family, numbering]).T
                        # numb = [pdb_id, protein, self.entry_to_ac(protein), family, numbering]
                        numb.columns = ['PDB', 'identifier', 'family', 'numbering']
                        self.numbering = self.numbering.append(numb)
                    else:
                        self.mappings = self.mappings.append(self.get_mapping(pdb_id), ignore_index=True)
                        numb = pd.DataFrame(data=[pdb_id, protein, family, numbering]).T
                        numb.columns = ['PDB', 'identifier', 'family', 'numbering']
                        self.numbering = self.numbering.append(numb, ignore_index=True)

    def make_raws(self):
        for i, pdb_id in tqdm(enumerate(self.pdb_ids)):
            if i < self.limit:
                # only process if the file has not already been generated
                # if not self.reload & 
                protein, family = self.get_prot_info(pdb_id)
                if protein != None:
                    if i == 0:
                        self.structure = self.load_cifs(pdb_id)
                        self.structure['identifier'] = protein.upper()
                        if self.remove_hetatm:
                            self.structure = self.structure[self.structure['group_PDB']!='HETATM']
                            self.structure['label_seq_id'] = self.structure['label_seq_id'].astype(np.int64)
                        self.structure['label_comp_sid'] = self.structure.apply(lambda x:
                                                            gemmi.find_tabulated_residue(x.label_comp_id).one_letter_code, 
                                                            axis=1)
                    else:
                        structure = self.load_cifs(pdb_id)
                        structure['identifier'] = protein.upper()
                        if self.remove_hetatm:
                            structure = structure[structure['group_PDB']!='HETATM']
                            structure['label_seq_id'] = structure['label_seq_id'].astype(np.int64)
                        structure['label_comp_sid'] = structure.apply(lambda x:
                                                            gemmi.find_tabulated_residue(x.label_comp_id).one_letter_code, 
                                                            axis=1)
                        self.structure = self.structure.append(structure, ignore_index=True)
        
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
    
    def to_pkl_raw(self, folder='data/raw/', overwrite=False):
        for pdb_id in self.pdb_ids:
            structure = self.structure[self.structure['PDB']==pdb_id]
            if len(structure) >= 1:
                if (not os.path.isfile(folder + pdb_id + '.pkl')) or overwrite:
                    structure.to_pickle(folder + pdb_id + '.pkl')
                    print("writing to file:", folder + pdb_id + '.pkl')
    
    def to_pkl_processed(self, folder='data/processed/', overwrite=False):
        for df in self.dfl:
            pdb_id = df['PDB'].unique()[0]
            if (not os.path.isfile(folder + pdb_id + '.pkl')) or overwrite:
                df.to_pickle(folder + pdb_id + '.pkl')
                print("writing to file:", folder + pdb_id + '.pkl')
    
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
    
    def read_pkl_raw(self):
        # not needed atm
        pass
    
    def read_pkl_processed(self, mode='', folder='data/processed/'):
        files = [f for f in os.listdir(folder) if '.pkl' in f]
        
        if 'g' in mode:
            # remove all files with "g-ending" -> refers to already processed data
            files = [f for f in files if 'g.' not in f]
            
        if 'r' in mode:
            files = [f for f in files if '_r' not in f]
        
        self.dfl = []
        for f in files:
            self.dfl.append(pd.read_pickle(folder+f).reset_index().drop('index', axis=1))
            
            if len(f) >= 8:
                self.dfl_list.append(f[:-(len(f)-4)])
            else:
                self.dfl_list.append('')
    
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
        for j in range(len(mappings_)):
            chain = pd.DataFrame.from_dict(mappings_.iloc[j]['mappings'])['chain_id'].iloc[0]
            identifier = mappings_.iloc[j]['name']
            dict_ = pd.DataFrame.from_dict(mappings_.iloc[j]['mappings'])
            dict_['identifier'] = identifier
            map_df_list.append(pd.DataFrame.from_dict(dict_))
        _ = pd.concat(map_df_list)
        _ = _[_['chain_id']==pref_chain]
        _['PDB'] = pdb
        return _


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


    def get_generic_number(self, zipped_pos_dict, l2u, comp_sid):
        if l2u >= 0:
            if l2u in list(zip(*zipped_pos_dict))[0]:
                idx = list(zip(*zipped_pos_dict))[0].index(l2u)
                row = zipped_pos_dict[idx]
                if row[1] == comp_sid:
                    return row[2], row[1], float(row[2].split('x')[0]), int(row[2].split('x')[1])
                else:
                    return row[2]+'?', row[1], float(row[2].split('x')[0]), int(row[2].split('x')[1])
            else:
                return ['', '', 0, 0]
        else:
            return ['', '', 0, 0]


    def assign_generic_numbers_(self, pdb_id, overwrite, folder):
        data = pd.read_pickle(folder + pdb_id + '.pkl').reset_index().drop('index', axis=1)
        print("loaded data to assign gen. numbers from", folder + pdb_id + '.pkl')
        cols = data.columns
        columns = ['gen_pos', 'gen_pos1', 'gen_pos2', 'uniprot_comp_sid']
        _ = [i for i in columns if i in cols]
        if len(_) > 0:
            if overwrite:
                data.drop(_, axis=1, inplace=True)
                data['label_2_uni'] = 0
                data[columns[0]] = ''
                data[columns[1]] = 0
                data[columns[2]] = 0
                data[columns[3]] = ''
            else:
                return data
        else:
            data['label_2_uni'] = 0
            data[columns[0]] = ''
            data[columns[1]] = 0
            data[columns[2]] = 0
            data[columns[3]] = ''
        maps_stacked = self.get_stacked_maps(pdb_id)
        if 'residue_number' in maps_stacked.index:
            pass
        else:
            return data
        if type(maps_stacked[maps_stacked['PDB']==pdb_id].\
                loc['residue_number'][['chain_id', 'start','end','unp_start','unp_end', 'identifier', 'PDB']])\
                    == pandas.core.series.Series:
            pref_mapping = maps_stacked[maps_stacked['PDB']==pdb_id].loc['residue_number']\
                [['chain_id', 'start','end','unp_start','unp_end', 'identifier', 'PDB']].to_frame().T
        else:
            pref_mapping = maps_stacked[maps_stacked['PDB']==pdb_id].\
                loc['residue_number'][['chain_id', 'start','end','unp_start','unp_end', 'identifier', 'PDB']]
        pref_chain = pref_mapping['chain_id'].iloc[0]
        pref_mapping = pref_mapping.sort_values('start')
        uniprot_identifier_ = data[data['PDB']==pdb_id]['identifier'].unique()
        uniprot_identifier = uniprot_identifier_[0]
        natoms = len(data[data['PDB']==pdb_id])
        
        for j in range(len(pref_mapping)):
            row = pref_mapping.iloc[j].to_dict()
            map_identifier = row['identifier']
            map_pdb = row['PDB']
            start_label_seq_id = row['start']
            start_uniprot = row['unp_start']
            end_label_seq_id = row['end']
            end_uniprot = row['unp_end']
            if map_identifier == uniprot_identifier:
                idxs = [x for x in range(natoms+1) \
                        if ((x <= end_label_seq_id) & (x >= start_label_seq_id))]
                vals = [x + start_uniprot - start_label_seq_id for x in range(natoms+1) \
                        if ((x <= end_label_seq_id) & (x >= start_label_seq_id))]
                for k, idx in enumerate(idxs):
                    line = data[(data['PDB'] == pdb_id) &
                                (data['label_seq_id'] == idx) &
                                (data['label_atom_id'] == 'CA')]
                    lines = len(line)
                    if len(line) > 1:
                        line = line[line['auth_asym_id'] == pref_chain]
                    if len(line) > 0:
                        data.at[line.index[0], 'label_2_uni'] = int(vals[k])
            else:
                # Didnt find correct uniprotmap (not a gpcr) ==> map_identifier
                pass
        # Generate generic numbers
        zipped_pos_dict = self.get_generic_nums(pdb_id)
        if type(data) == pandas.core.series.Series:
            data = data.to_frame().T
        
        data[['gen_pos', 'uniprot_comp_sid', 'gen_pos1', 'gen_pos2']] = data.\
            apply(lambda x: self.get_generic_number(zipped_pos_dict, x.label_2_uni, x.label_comp_sid) if x.PDB==pdb_id\
                  else [x.gen_pos, x.uniprot_comp_sid, x.gen_pos1, x.gen_pos2], axis=1, result_type='expand')
        return data
    
    def assign_generic_numbers_r(self, pdb_ids=None, overwrite=True, folder='data/raw/'):
        dfl_ = []
        if pdb_ids != None:
            self.pdb_ids = pdb_ids
        if not isinstance(self.pdb_ids, list):
            self.pdb_ids = [self.pdb_ids]
        for pdb_id in self.pdb_ids:
            if self.allow_exception:
                print("trying to assign generic nubmers to", pdb_id)
                try:
                    dfl_.append(self.assign_generic_numbers_(pdb_id, overwrite=overwrite, folder=folder))
                    print("assigned generic numbers to", pdb_id, "\n\n\n")
                except:
                    print("assigning failed for", pdb_id)
            else:
                print("trying to assign generic nubmers to", pdb_id)
                dfl_.append(self.assign_generic_numbers_(pdb_id, overwrite=overwrite, folder=folder))
                print("assigned generic numbers to", pdb_id, "\n\n\n")
        self.dfl = dfl_
        del dfl_
    
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

    
    def filter_dfl_via_table(self,
                             Species=None,
                             State=None,
                             Family=None,
                             Subtype=None,
                             Resolution=None,
                             Function=None,
                             gprotein=False):
        data = self.table
        if Species != None:
            data = data[data['Species']==Species]
        if State != None:
            data = data[data['State']==State]
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
    
    
    def get_dfl_indices(self, filtered_list):
        return [self.dfl_list.index(filtered_list[i]) if (filtered_list[i] in self.dfl_list)\
                else None for i in range(len(filtered_list))]
    
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


    def _get_pairwise_indices(self, a, b):
        alignment = pairwise2.align.globalms(a, b, 3, -1, -3, -1)[0]
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

    
    def _assign_res_nums_g(self, pdb_id, mappings, structure, uniprot_gprot_list, gprot_df, res_table):
        try:
            maps_stacked = self.get_stacked_maps_g(pdb_id, mappings, uniprot_gprot_list)
        except:
            return structure
        if 'residue_number' in maps_stacked.index:
            pass
        else:
            return structure

        selection = maps_stacked[maps_stacked['PDB']==pdb_id]

        if type(selection.loc['residue_number']) == pandas.core.series.Series:
            mapping = selection.loc['residue_number'].to_frame().T
        else:
            mapping = selection.loc['residue_number']

        mapping = mapping.sort_values('start')
        label_seq_ids = list(structure[structure['PDB']==pdb_id]['label_seq_id'])
        if len(label_seq_ids) == 0:
            print("Did not find label_seq_ids")
            return structure
        nres = max(label_seq_ids)
        structure['gprot_pos'] = ''

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

                # the start -> end should be a shift (if not we have to do an alignment eg. 6FUF)
                # iterate from start_label_seq_id to end_label_seq_id

                seq_len = end_label_seq_id - start_label_seq_id

                for k in range(seq_len):
                    idx_seq = k + start_label_seq_id
                    idx_uni = k + start_uniprot - 1
                    line = structure[(structure['PDB'] == pdb_id) &
                                     (structure['label_seq_id'] == idx_seq) &
                                     (structure['label_atom_id'] == 'CA') &
                                     (structure['auth_asym_id'] == pref_chain)]

                    if len(line) > 0:
                        structure.at[line.index[0], 'label_2_uni'] = idx_uni + 1
                        if line['label_seq_id'].iloc[0] in list(cgn_df.index):
                            structure.at[line.index[0], 'gprot_pos'] = cgn_df.iloc[idx_uni]['cgn']
                            structure.at[line.index[0], 'uniprot_comp_id'] = cgn_df.iloc[idx_uni]['seq_res']
                            structure.at[line.index[0], 'fam_comp_id'] = cgn_df.iloc[idx_uni]['fam_seq_res']
                        else:
                            structure.at[line.index[0], 'gprot_pos'] = ''
            else:
                return structure
        return structure
    
    
    def assign_generic_numbers_g(self, f, pdb_ids=None, overwrite=True, folder='data/raw/'):
        # is the filter from "filter_dfl_via_table" -> if none provided give f = self.table
        dfl_list_f = list(f['PDB'])
        dfl_list_f_indices = self.get_dfl_indices(dfl_list_f)
        dfl_list_f_indices = [x for x in dfl_list_f_indices if x != None]
           
        res_table = self.load_gen_res_nums_gprot()[0]
        gprot_df = self.get_gproteins_in_complex_df(res_table)
        uniprot_gprot_list = functools.reduce(operator.iconcat, list(self.uniprot_dict.values()), [])
        
        for i in trange(len(dfl_list_f_indices)):
            try:
                pdb_id = self.dfl_list[dfl_list_f_indices[i]]
                structure = self.dfl[dfl_list_f_indices[i]]
                s = self._assign_res_nums_g(pdb_id,
                                            self.mappings[self.mappings['PDB']==pdb_id],
                                            structure, 
                                            uniprot_gprot_list, 
                                            gprot_df,
                                            res_table)
                self.dfl[i] = s
            except:
                print("Error parsing", pdb_id)

