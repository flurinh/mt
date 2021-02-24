import os
from os import walk
import pandas as pd
import requests
from urllib.request import urlopen
import json
from Bio.PDB import *
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
# pdb files contain an 'END' tag, which throws (unnecessary?) warnings


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


def download_refined_structure(prot_id: str, folder: str):
    url = 'https://gpcrdb.org/structure/homology_models/' + prot_id + '_refined_full/download_pdb'
    print("Downloading refined structure for {}.".format(prot_id))
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
        print("Did not fined refined structure for {}!".format(prot_id))
        return False


def getpdbfile(protid: str):
    pdbl = PDBList()
    pdbl.retrieve_pdb_file(protid)


def pdbtouniprot(protid: str):
    return None


def uniprottopdb(protid: str):
    return None


def loadpdb(protid: str):
    return None


def updatepdbs(path="/data/pdb"):
    pl = PDBList(path)
    pl.update_pdb()

# ======================================================================================================================


# fetch a protein
def get_seq(prot_id: str):
    url = 'https://gpcrdb.org/services/protein/' + prot_id + '/'
    response = urlopen(url)
    protein_data = json.loads(response.read().decode('utf-8'))
    return (protein_data['sequence'])


def getalignment_(proteins: list, segment: str, verbose=0):
    # fetch an alignment
    id1 = ','.join(proteins)
    url = 'https://gpcrdb.org/services/alignment/protein/' + id1 + '/' + segment + '/'
    if verbose > 0:
        print("Searching for alignment:", url)
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


def get_alignment(prots, segments = ['TM7', 'ICL4', 'H8']):
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
    for poly_index, poly in enumerate(polypeptides) :
        if verbose > 0:
            print("(part %i of %i)" % (poly_index+1, len(polypeptides)))
            print("length %i" % (len(poly)))
            print("from %s%i" % (poly[0].resname, poly[0].id[1]))
            print("to %s%i" % (poly[-1].resname, poly[-1].id[1]))
        phi_psi = poly.get_phi_psi_list()
        seq = poly.get_sequence()
        if verbose > 0:
            for res_index, residue in enumerate(poly) :
                res_name = "%s%i" % (residue.resname, residue.id[1])
                print(res_name, phi_psi[res_index])
        polys.append([len(seq), seq, phi_psi])
    return polys


def get_sec_structs(prot_id: str, seq: str, seg_aligns: pd.DataFrame):
    # return start of secondary
    segments = seg_aligns.keys()[1:]  # ignore name-column
    sec_seq = seg_aligns[seg_aligns['name'] == prot_id]
    # this creates a copy of a slice ==> gives a warning when value is set
    for s in segments:
        seg = sec_seq[s].values[0]
        seq = ''.join([x for x in seq if x is not None])
        col = s + '_idx'
        sec_seq.loc[sec_seq.index[0], col] = alignmenttoindex(seq=seq, seg=seg)  # this is the line that sets a value...
    return sec_seq
