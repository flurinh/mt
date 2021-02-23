import os
from os import walk
import pandas as pd

from urllib.request import urlopen
import json
from Bio.PDB import *


def getpdbsfiles(path='data/active/'):
    # just a helper function that returns all pdb files in specified path
    (_, _, filenames) = next(os.walk(path))
    files = [path + x for x in filenames]
    prots = list(set([x[-20:-9] for x in files]))
    return files, prots


def getuniprot(path='data/uniprot/'):
    path += 'uniprot_7tmrlist.txt'
    with open(path) as f:
        ul = f.readlines()
    human_gpcr = []
    for _ in ul:
        if '- Human' in _:
            human_gpcr.append(_)
    human_gpcr_d = {}
    for i, p in enumerate(human_gpcr):
        y = p[22:].split(' - Human')
        x = [p[:11], p[13:19], y[0], 'Human']
        human_gpcr_d.update({i: {'gene': x[0], 'id': x[1], 'name': x[2]}})
    return pd.DataFrame.from_dict(human_gpcr_d).T


# fetch a protein
def get_seq(prot):
    url = 'https://gpcrdb.org/services/protein/' + prot + '/'
    response = urlopen(url)
    protein_data = json.loads(response.read().decode('utf-8'))
    print(protein_data)
    return (protein_data['sequence'])


def getalignment(proteins: list, segment: str):
    # fetch an alignment
    id1 = ','.join(proteins)
    url = 'https://gpcrdb.org/services/alignment/protein/' + id1 + '/' + segment + '/'
    print("Searching for alignment:", url)
    response = urlopen(url)
    alignment_data = json.loads(response.read().decode('utf-8'))
    return alignment_data


def getpdbfile(protid: str):
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(protid)
    return


def pdbtouniprot(protid: str):
    return "uniprotid"


def uniprottopdb(protid: str):
    return "pdbid"