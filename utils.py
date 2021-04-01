import os
from os import walk
import pandas as pd

import urllib
import requests
import re
from urllib.request import urlopen
from bs4 import BeautifulSoup
import json

from Bio import SeqIO
from io import StringIO

import warnings
from Bio.PDB import *
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
# pdb files contain an 'END' tag, which throws (unnecessary?) warnings


def get_pdb_files(path='data/pdb/active/'):
    # just a helper function that returns all pdb files in specified path
    (_, _, filenames) = next(os.walk(path))
    files = [path + x for x in filenames]
    pdb_ids = list(set([x[-8:-4] for x in files]))
    return files, pdb_ids


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

    
def uniprottopdb(uniprot_id: str):
    url = 'https://www.uniprot.org/uniprot/'
    params = {
        'format': 'tab',
        'query': 'ID:{}'.format(uniprot_id),
        'columns': 'id,database(PDB)'
    }
    contact = ""  # "hidberf@student.ethz.ch"
    headers = {'User-Agent': 'Python {}'.format(contact)}
    r = requests.get(url, params=params, headers=headers)
    return str(r.text).splitlines()[-1].split('\t')[-1].split(';')[:-1]


def pdbtouniprot(pdb_id: str):
    # time.sleep(.01)
    url_template = "http://www.rcsb.org/pdb/files/{}.pdb"
    url = url_template.format(pdb_id)
    response = urllib.request.urlopen(url)
    pdb = response.read().decode('utf-8')
    response.close()
    m = re.search('UNP\ +(\w+)', pdb)
    if m != None:
        return m.group(1)
    else:
        return None
    

def get_uniprot(query='',query_type='PDB_ID'):
    #query_type must be: "PDB_ID" or "ACC"
    url = 'https://www.uniprot.org/' #This is the webser to retrieve the Uniprot data
    params = {
    'from':query_type,
    'to':'ACC',
    'format':'txt',
    'query':query
    }

    data = urllib.parse.urlencode(params)
    data = data.encode('ascii')
    request = urllib.request.Request(url, data)
    with urllib.request.urlopen(request) as response:
        res = response.read()
        page=BeautifulSoup(res, features="lxml").get_text()
        page=page.splitlines()
    return page
    
def get_sequence_name(cID):
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl=baseUrl+cID+".fasta"
    response = requests.post(currentUrl)
    cData=''.join(response.text)
    Seq=StringIO(cData)
    pSeq=list(SeqIO.parse(Seq,'fasta'))
    return str(pSeq[0].seq), pSeq[0].name
    
# ======================================================================================================================


# fetch a protein
def get_seq(prot_id: str):
    url = 'https://gpcrdb.org/services/protein/' + prot_id + '/'
    response = urlopen(url)
    protein_data = json.loads(response.read().decode('utf-8'))
    return (protein_data['sequence'])


def get_uniprot_seq(uniprotid: str):
    if uniprotid is None:
        return None
    baseUrl="http://www.uniprot.org/uniprot/"
    currentUrl = baseUrl + uniprotid + ".fasta"
    response = requests.post(currentUrl)
    cData = ''.join(response.text)
    Seq = StringIO(cData)
    return list(SeqIO.parse(Seq,'fasta'))


def load_alignment_(path = 'data/alignments/GPCRdb_alignment_groupA_Gs.csv'):
    return pd.read_csv(path)


def getalignment_(proteins: list, segment: str, verbose=1):
    # fetch an alignment
    id1 = ','.join(proteins)
    url = 'https://gpcrdb.org/services/alignment/protein/' + id1 + '/' + segment + '/'
    if verbose > 0:
        print("Searching for alignment:", url)
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    r = requests.get(url, headers=headers)
    print(r)"""
    response = urlopen(url)
    alignment_data = json.loads(response.read().decode('utf-8'))
    return alignment_data


def alignmenttoindex(seq: str, seg: str, min_len=0, verbose=0):
    if len(seg) <= min_len:
        return -1
    # if we notice that the alignment with the original sequence for some reason does not work out, do not use recursion
    seg.replace('-', '')
    idx = seq.find(seg)
    if len(seg) <= 1:
        return -1
    elif idx >= 0:
        if verbose > 0:
            print("found (longest - greedy!) seg {} at position {}.".format(seg, idx))
        return idx
    else:
        return alignmenttoindex(seq, seg[:-1])

# ======================================================================================================================


def get_alignment(prots, segments=['TM7', 'ICL4', 'H8']):
    # Define secondary structure elements of interest
    seg_aligns = pd.DataFrame()
    seg_aligns['name'] = prots
    for _, s in enumerate(segments):
        alignment_data = getalignment_(prots, s)
        dp = []
        for p in prots:
            dp.append(alignment_data[p])
        seg_aligns[s] = dp
    return seg_aligns


def get_rcsb_download(id, fileformat='pdb'):
    return 'https://files.rcsb.org/download/'+id+'.'+fileformat


def get_seq(chain):
    # chain to sequence
    l = []
    for residue in chain:
        try:
            l.append(Polypeptide.three_to_one(residue.get_resname()))
        except:
            l.append(None)
    return l


def get_phi_psi_list(chain, verbose=0):
    # Use C–N distance to find polypeptides (comp. pdb docs)
    polypeptides = PPBuilder().build_peptides(chain)
    polys = []
    for poly_index, poly in enumerate(polypeptides):
        if verbose > 0:
            print("(part %i of %i)" % (poly_index+1, len(polypeptides)))
            print("length %i" % (len(poly)))
            print("from %s%i" % (poly[0].resname, poly[0].id[1]))
            print("to %s%i" % (poly[-1].resname, poly[-1].id[1]))
        phi_psi = poly.get_phi_psi_list()
        seq = poly.get_sequence()
        if verbose > 0:
            for res_index, residue in enumerate(poly):
                res_name = "%s%i" % (residue.resname, residue.id[1])
                print(res_name, phi_psi[res_index])
        polys.append([len(seq), seq, phi_psi])
    return polys


def get_sec_structs(prot_id: str, seq: str, seg_aligns: pd.DataFrame):
    # return start of secondary
    segments = seg_aligns.keys()[1:]  # ignore name-column
    sec_seq = seg_aligns[seg_aligns['name'] == prot_id].copy()
    # this creates a copy of a slice ==> gives a warning when value is set
    for s in segments:
        seg = sec_seq[s].values[0]
        seq = ''.join([x for x in seq if x is not None])
        col = s + '_idx'
        sec_seq.loc[sec_seq.index[0], col] = alignmenttoindex(seq=seq, seg=seg)  # this is the line that sets a value...
    return sec_seq
