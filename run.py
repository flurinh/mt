from utils import *
import Bio
import yaml

files, prots = getpdbsfiles()
print(prots)

segments = ['TM7', 'ICL4', 'H8']

seg_aligns = {}

for s in segments:
    alignment_data = getalignment(prots, s)
    seg_aligns.update({s: alignment_data})

print(yaml.dump(seg_aligns, default_flow_style=False))



p = PDBParser()
structure = p.get_structure("X", "pdb1fat.ent")
for model in structure:
    for chain in model:
        for residue in chain:
            for atom in residue:
                print(atom)


# Get all residues from a structure
res_list = Selection.unfold_entities(structure, "R")
# Get all atoms from a chain
atom_list = Selection.unfold_entities(chain, "A")


vector1 = atom1.get_vector()
vector2 = atom2.get_vector()
vector3 = atom3.get_vector()
vector4 = atom4.get_vector()
angle = calc_dihedral(vector1, vector2, vector3, vector4)