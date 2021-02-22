import os
from os import walk
import pandas as pd


def getpdbs(path='data/structures/pdb_structures/'):
    # just a helper function that returns all pdb files in specified path
    (_, _, filenames) = next(os.walk(path))
    return [path + x for x in filenames]


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

