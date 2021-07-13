import pandas as pd
import lxml
from processing.processor import *


class AffinityProcessor:
    def __init__(self,
                 path='data/couplings/',
                 setting='subtypes'):
        self.source = path + setting + '_coupling.xls'
        self.setting = setting
        self.groups = ['GPCRdb', 'Inoue', 'Bouvier']
        self.label_types = ['Guide to Pharmacology', 'Log(Emax/EC50)', 'pEC50', 'Emax']
        print("Reading data from {}!".format(self.source))
        self.read_data()
        print("Initialized Affinity Processor!")
        print("Please set a group --------------  {}.".format(self.groups))
        print("please set label type -----------  {}.".format(self.label_types))
    
    def read_data(self):
        if self.setting == 'families':
            self.data = pd.read_html(self.source)[0].drop('Unnamed: 0_level_0', axis=1)
        else:
            self.data = pd.read_html(self.source)[0].drop('Unnamed: 0_level_0', axis=1, level=0)
    
    def set_group(self, group='GPCRdb'):
        assert group in self.groups, print("'{}' is not a valid group name, valid are {}.".format(group, self.groups))
        print("\nSelected data of group '{}'.\n".format(group))
        if self.setting == 'subtypes':
            self.data = self.data[self.data['Source', 'Group', 'Unnamed: 1_level_2']==group]
        else:
            self.data = self.data[self.data['Source', 'Group']==group]
    
    def set_label_type(self, label_type='Guide to Pharmacology'):
        assert label_type in self.label_types, print("'{}' is not a valid label type, valid are {}."\
                                                     .format(label_type, self.label_types))
        print("\nSelected label type '{}'.\n".format(label_type))
        to_drop = [x for x in self.label_types if x != label_type]
        if self.setting == 'subtypes':
            self.data = self.data.drop(to_drop, axis=1)
        else:
            self.data = self.data.drop(to_drop, axis=1)
    
    def make_label_dict(self):
        label_dict = {}
        return label_dict
    
    
def filter_valid_pdbs_with_affinities(p: CifProcessor, A: pd.DataFrame):
    genes = []
    for i in range(len(p.dfl)):
        idf = p.dfl[i]['identifier'].iloc[0]
        if idf not in genes:
            genes.append(idf)
    has_missing_affinity = []
    for g in genes:
        g_, _g = g.split('_')
        if _g == 'HUMAN':
            if g_ not in A.data['Receptor', 'Uniprot'].to_list():
                has_missing_affinity.append(g_)
    invalid_pdb = []
    for i in range(len(p.dfl)):
        idf = p.dfl[i]['identifier'].iloc[0]
        pdb = p.dfl[i]['PDB'].iloc[0]
        g_, _g = idf.split('_')
        if (g_ in has_missing_affinity) or (_g != 'HUMAN'):
            invalid_pdb.append(pdb)
    filtered_dfl_list = [x for x in p.dfl_list if x not in invalid_pdb]
    f_labelled = p.table[p.table['PDB'].isin(filtered_dfl_list)]
    p.apply_filter(f_labelled)
    return p


def get_label(df, A, label_type):
    """
    This definitely needs some looking into!
    """
    keys = ['Gs', 'Gi/o', 'Gq/11', 'G12/13']
    idf = df['identifier'].iloc[0]
    gene, _ = idf.split('_')
    row = A.data[A.data['Receptor', 'Uniprot'] == gene]
    values = list(row[label_type].iloc[0])
    values = [float(x) if (not "'" in x) and (not '-' in x) else 0 for x in values]
    return dict((zip(keys, values)))


def make_label_df(p, A, label_type):
    p.labels = []
    for i in range(len(p.dfl)):
        d = get_label(p.dfl[i], A, label_type)
        p.labels.append(d)
    label_df = pd.DataFrame(p.labels)
    label_df['PDB'] = p.dfl_list
    return label_df







