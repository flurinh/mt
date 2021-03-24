"""
We use the ftp server to download everything (almost). To check the data use this link: ftp://ftp.wwpdb.org/pub/pdb/data/structures/. To download all structures divided into ?XX? id folders use this link: https://www.wwpdb.org/ftp/pdb-ftp-sites and select the corresponding command in the terminal. We used RCBS' Download coordinate files in PDB Format:

rsync -rlpt -v -z --delete --port=33444 \
rsync.rcsb.org::ftp_data/structures/divided/pdb/ ./pdb
"""


from tqdm import tqdm
import gzip
import shutil
import os
from os import listdir, rename
from os.path import isfile, join, isdir


def unzip_pdb(data_path = 'data/', database = 'pdb'):
    dirName = data_path + database
    def getListOfFiles(dirName):
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
                allFiles = allFiles + getListOfFiles(fullPath)
            else:
                if '.ent.gz' in fullPath:
                    allFiles.append(fullPath)
        return allFiles
    files = getListOfFiles(dirName)
    print(len(files))
    for f in tqdm(files):
        try:
            os.remove(f[:-6] + 'pdb')
        except:
            pass
        try:
            with gzip.open(f, 'rb') as f_in:
                with open(f[:-6] + 'pdb', 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    # os.remove(f)
        except:
            pass

unzip_pdb()  # this thing breaks your computer... pray before running :D (ca 2-3 mins)