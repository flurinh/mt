from processing.utils import *
from bs4 import BeautifulSoup
from tqdm import tqdm, trange
import requests
import re
import urllib
import time
import pandas as pd



COLS = ['cross',  'filler', 'uniprot(gene)', 'filler2', 'receptor family', 'Cl.', 'Species', 'Method', 'PDB',  \
        'Refined Structure', 'Resolution', 'Preferred Chain', 'State', 'Degree active %', '% of Seq1', 'Family', \
        'Subtype', 'Note', '% of Seq2', 'Fusion', 'Antibodies', 'Name1', 'Type1', 'Function', 'Name2', 'Type2', \
        'D2x50 - S3x39', 'Sodium in structure', 'Authors', 'Reference', 'PDB date', 'Annotated']



class Download():
    def __init__(self, 
                 path='data/',
                 fileformat='pdb'):
        if fileformat=='pdb':
            self.path_pdb = path + 'pdb/'
        else:
            self.path_pdb = path + 'mmcif/'
        self.path_alignment = path + 'alignments/'
        self.path_table = path
        
        self.fileformat = fileformat
        
        self.table = None
        self.filenames, self.pdb_ids = get_pdb_files(path=self.path_pdb)
    
    # ======================================================================================================================
        
    def download_alignment(self):
        print("Not Implemented! TBD manually.")
    
    def download_pdbs(self, reload=False, update=False):
        pdb_table_ids = self.table['PDB'].tolist()
        missing = [x for x in pdb_table_ids if x not in self.pdb_ids]
        if reload or (len(missing)>0):
            print("Reloading pdb files...")
            print("Missing pdbs:", missing)
            for pdb in tqdm(missing):
                url = get_rcsb_download(pdb, self.fileformat)
                download_pdb(url, folder=self.path_pdb, fileformat=self.fileformat)
        elif update:
            self.update_pdbs()
        self.filenames, self.pdb_ids = get_pdb_files(path=self.path_pdb)
                
    def download_table(self, reload=True, filename='data_table.pkl'):
        table_path = self.path_table + filename
        if reload or (not os.path.isfile(self.path_table+filename)):
            self.table = get_table(reload=True, uniprot=False)
            self.table = self.table.drop(columns=['filler', 'filler2', 
                                                  'Refined Structure',  
                                                  '% of Seq1', 'Name2', 'Fusion', 'Note', 
                                                  '% of Seq2', 'Antibodies', 'Name1', 'Type1', 'Type2', 
                                                  'D2x50 - S3x39', 'Sodium in structure', 'Authors', 'Reference', 
                                                  'PDB date', 'Annotated', 'pdb_link'])
        else:
            self.table = pd.read_pickle(self.path_table)
        print("writing gpcrdb table to file:", table_path)
        self.table.to_pickle(table_path)        
        
    def add_row_table(self, 
                      uniprot='',
                      receptor_family='',
                      cl='',
                      species='',
                      method='',
                      pdb='',
                      resolution='',
                      pref_chain='A',
                      state='Inactive',
                      deg_act='-',
                      family='-',
                      subtype='-',
                      function=''):
        cols = self.table.columns
        print(cols)
        row = [uniprot, receptor_family, cl, species, method, pdb, resolution, pref_chain, state, deg_act, family, subtype, function]
        print(len(cols))
        print(len(row))
        row_df = pd.DataFrame([row], columns=cols)
        return row_df

    # ======================================================================================================================
    
    def update_pdbs(self):
        updatepdbs(self.path_pdb)
    
    # ======================================================================================================================
    
    
def get_page(url='https://gpcrdb.org/structure/#'):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup


def parse_html_table(table):
    n_columns = 0
    n_rows=0
    column_names = []

    # Find number of rows and columns
    # we also find the column titles if we can
    for row in table.find_all('tr'):

        # Determine the number of rows in the table
        td_tags = row.find_all('td')
        if len(td_tags) > 0:
            n_rows+=1
            if n_columns == 0:
                # Set the number of columns for our table
                n_columns = len(td_tags)

        # Handle column names if we find them
        th_tags = row.find_all('th') 
        if len(th_tags) > 0 and len(column_names) == 0:
            for th in th_tags:
                column_names.append(th.get_text())

    # Safeguard on Column Titles
    if len(column_names) > 0 and len(column_names) != n_columns:
        raise Exception("Column titles do not match the number of columns")

    columns = column_names if len(column_names) > 0 else COLS[1:]
    df = pd.DataFrame(columns = columns,
                      index= range(0,n_rows))
    row_marker = 0
    for row in table.find_all('tr'):
        column_marker = 0
        columns = row.find_all('td')
        for column in columns:
            df.iat[row_marker,column_marker] = column.get_text().replace(' ', '').replace('\n', '')
            column_marker += 1
        if len(columns) > 0:
            row_marker += 1
    # Convert to float if possible
    for col in df:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            pass
    return df


def get_table(reload=False, uniprot=False, save=True, path='data/gpcrdb/structure.pkl'):
    if reload:
        soup = get_page()
        table = soup.find("tbody")
        table = parse_html_table(table)
        # uniprot and pdb link
        # to make the dataload smaller..
        table['pdb_link'] = table.PDB.apply(get_rcsb_link)
        if uniprot:
            table['uniprot_id'] = table.PDB.apply(pdbtouniprot)
            table['uniprot_link'] = table.uniprot_id.apply(get_uniprot_link)
        if save:
            table.to_pickle(path)
    else:
        table = pd.read_pickle(path)
    return table


def create_structure_df(table):
    nrows = len(table) // len(COLS)
    holder = []
    for i in range(nrows):
        holder.append(table[i*len(COLS):(i+1)*len(COLS)])
    return pd.DataFrame(holder, columns=COLS)


def get_rcsb_link(pdb_id: str, fileformat='pdb'):
    if len(pdb_id) == 4:
        try:
            return 'https://files.rcsb.org/download/'+pdb_id+'.'+fileformat
        except:
            return None
    else:
        return None

    
def get_uniprot_link(uniprot_id: str):
    if uniprot_id == None:
        return None
    elif 4 <= len(uniprot_id) <= 6:
        try:
            return 'www.uniprot.org/uniprot/'+uniprot_id
        except:
            return None
    else:
        return None

    
def downloadzip(url: str, folder: str):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    try:
        r = requests.get(url)
        zipfname = folder + '/' + prot_id + '.zip'
        with open(zipfname, 'wb') as f:
            f.write(r.content)
        import zipfile
        with zipfile.ZipFile(zipfname, "r") as zip_ref:
            zip_ref.extractall(folder)
        os.remove(zipfname)
        return True
    except Exception:
        print("Url invalid:", url)
        return False

    
def download(url: str, folder: str, fileformat: str):
    if not os.path.isdir(folder):
        os.mkdir(folder)
    try:
        r = requests.get(url)
        loc = len(fileformat)+1
        fname = folder + '/' + url[-(loc+4):-loc] + '.' + fileformat
        with open(fname, 'wb') as f:
            f.write(r.content)
    except Exception:
        print("Url invalid:", url)

    
def download_pdb(url, folder, fileformat):
    download(url, folder, fileformat)


def download_uniprot(url, folder, fileformat):
    download(url, folder, fileformat)


def download_refined_structure(prot_id: str):
    url = 'https://gpcrdb.org/structure/homology_models/' + prot_id + '_refined_full/download_pdb'
    print("Downloading refined structure for {}.".format(prot_id))
    downloadzip(url, 'refined')


def getpdbfile(protid: str):
    pdbl = PDBList()
    return pdbl.retrieve_pdb_file(protid)


def updatepdbs(path="/data/pdb"):
    pl = PDBList(path)
    return pl.update_pdb()
