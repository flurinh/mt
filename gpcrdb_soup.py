from utils import *
from bs4 import BeautifulSoup
import requests
import re
import urllib
import time
import pandas as pd



COLS = ['cross',  'filler', 'uniprot(gene)', 'filler2', 'receptor family', 'Cl.', 'Species', 'Method', 'PDB',  \
        'Refined Structure', 'Resolution', 'Preferred Chain', 'State', 'Degree active %', '% of Seq1', 'Family', \
        'Subtype', 'Note', '% of Seq2', 'Fusion', 'Antibodies', 'Name1', 'Type1', 'Function', 'Name2', 'Type2', \
        'D2x50 - S3x39', 'Sodium in structure', 'Authors', 'Reference', 'PDB date', 'Annotated']


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


def get_table(reload=False, uniprot=False, save=True, path='data/structure.csv'):
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
            table.to_csv(path)
    else:
        table = pd.read_csv(path)
    return table


def create_structure_df(table):
    nrows = len(table) // len(COLS)
    holder = []
    for i in range(nrows):
        holder.append(table[i*len(COLS):(i+1)*len(COLS)])
    return pd.DataFrame(holder, columns=COLS)


def get_rcsb_link(pdb_id: str):
    if len(pdb_id) == 4:
        try:
            return 'https://files.rcsb.org/download/'+pdb_id+'.pdb'
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
    if not os.path.isdir('data/'+folder):
        os.mkdir('data/'+folder)
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

    
def download(url: str, folder: str):
    if not os.path.isdir('data/'+folder):
        os.mkdir('data/'+folder)
    try:
        r = requests.get(url)
        fname = 'data/'+folder + '/' + url[-8:-4] + '.pdb'
        with open(fname, 'wb') as f:
            f.write(r.content)
    except Exception:
        print("Url invalid:", url)

    
def download_pdb(url, folder=''):
    download(url, 'pdb/'+folder)


def download_uniprot(url):
    download(url, 'uniprot')


def download_refined_structure(prot_id: str):
    url = 'https://gpcrdb.org/structure/homology_models/' + prot_id + '_refined_full/download_pdb'
    print("Downloading refined structure for {}.".format(prot_id))
    downloadzip(url, 'refined')


def getpdbfile(protid: str):
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(protid)

    
def loadpdb(protid: str):
    return None


def updatepdbs(path="/data/pdb"):
    pl = PDBList(path)
    pl.update_pdb()




