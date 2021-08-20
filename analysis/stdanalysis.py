from processing.processor import *
from analysis.analysis import *
import numpy as np
from scipy.spatial.distance import cdist, pdist
import seaborn as sns


class StdAnalysisComplexes:
    def __init__(self,
                 P: CifProcessor
                 ):
        self.P = P
        self.queries = pd.DataFrame(columns=['mode', 'query_tag', 'pdb_id', 'poi', 'dist_poi'])
        self.dist_df_dict = {}
        self.angles_df_dict = {}
        self.helical_angles_mean = None
        self.helical_angles_std = None
        self.gs_count_df = None
        
    
    def run_dist_analysis(self, l, query_tag='', poi=('G.H5.25', 7.51), start=None, end=None, eps=0.05):
        self.dist_df_dict = {}
        if query_tag in list(self.queries['query_tag'].unique()):
            query_tag += str()
        list_poi_list, list_dists_df_list = get_interaction_tables(p=self.P, l=l, section='H5', poi=poi, 
                                                                   start=start, end=end, eps=eps)
        starting_point = 0
        for i, poi_list in enumerate(list_poi_list):
            data = ['dist', query_tag, poi_list[i][0], poi, poi_list[0][2]]
            df = pd.DataFrame(data).T
            df = df.set_axis(['mode', 'query_tag', 'pdb_id', 'poi', 'dist_poi'], axis=1)
            self.queries = self.queries.append(df, ignore_index=True)
        
        flat_lddl = []
        for l in list_dists_df_list:
            for ll in l:
                flat_lddl.append(ll)
                
        for j, ldd in enumerate(flat_lddl):
            pdb_id = ldd[0]
            table = ldd[2]
            self.dist_df_dict.update({pdb_id: table})
    
    
    def make_overview_df(self):
        dists_df_list = []
        for d in self.dist_df_dict:
            dists_df_list.append(self.dist_df_dict[d])
        cols = []
        rows = []
        for df in dists_df_list:
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
                for df in dists_df_list:
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
        
        
    def run_helical_analysis(self):
        section_helices = calculate_section_helices(self.P.dfl)
        rois = list(section_helices.columns)
        list_helical_angles = []
        for idx in range(len(self.P.dfl)):
            pdb_id = self.P.dfl[idx].iloc[0]['PDB']
            helical_angles = calc_angles_between_helices(section_helices, idx)            
            self.angles_df_dict.update({pdb_id: helical_angles})
            list_helical_angles.append(helical_angles)
        std_df = pd.DataFrame(columns=rois)
        mean_df = pd.DataFrame(columns=rois)
        for r1, roi1 in enumerate(rois):
            for r2, roi2 in enumerate(rois):
                l_r1r2 = []
                for df in list_helical_angles:
                    cell = df.loc[roi1, roi2]
                    if cell > 0:
                        l_r1r2.append(cell)
                if len(l_r1r2) > 0:
                    mean_r1r2 = np.mean(l_r1r2)
                    std_r1r2 = np.std(l_r1r2)
                    std_df.loc[roi1, roi2] = std_r1r2
                    mean_df.loc[roi1, roi2] = mean_r1r2
        self.helical_angles_mean = mean_df
        self.helical_angles_std = std_df
        self.gs_count_df = count_g_positions(self.P.dfl)
        
        
    def query(self, mode, query_tag, poi, pdb_id=None, helices=None):
        if mode == 'dist':
            target = self.queries[(self.queries['mode'] == 'dist') &
                                  (self.queries['poi'] == poi) &
                                  (self.queries['query_tag'] == query_tag)]
            return target
        elif mode == 'helices':
            if helices == None:
                return self.helical_angles_mean, self.helical_angles_std, self.gs_count_df
            elif pdb_id!=None:
                return self.angles_df_dict[pdb_id].loc[helices[0], helices[1]]
        else:
            print("Valid query modes are 'dist' and 'helices'!")