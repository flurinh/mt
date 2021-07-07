# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python [conda env:mtenv]
#     language: python
#     name: mtenv
# ---

# %% [markdown]
# # SETTING UP ENVIRONMENT
#
# 1) **fork github** (add on github and use git pull):
#
#     git clone git@github.com:flurinh/mt.git
#     
# 2) **create conda environment**
#
#     conda env create -f environment.yml --name gpcr
#     conda activate gpcr
#     
# 3) **create (missing) folders**

# %% [markdown]
#     unzip /data/project/bio/schertler/Flurin/data.zip

# %% [markdown]
# 4) **check if everything works** (import)

# %%
# %load_ext autoreload
# %autoreload 2
from processing.utils import *
from processing.utils2 import *
from processing.utils3 import *
from processing.gpcrdb_soup import *
from processing.download import *
from processing.processor import *
from processing.df_to_cif import *
from analysis.analysis import *

# %% [markdown]
# # DATA PROCESSING

# %% [markdown]
# The data (and file structure) is provided in a zipfile, so you do not have to download and process everything.
# This walkthrough shows how to (theoretically) build from scratch and/or update your outdated data.

# %% [markdown]
# ## DOWNLOADING / UPDATING METADATA

# %% [markdown]
# With metadata we primarily refer to the [gpcrdb](https://gpcrdb.org) [GPCR structure table](https://gpcrdb.org/structure/) containing a curated list of gpcrs (their name, uniprot id etc) - new entries in this list may automatically be updated!

# %% [markdown]
# In our code we use a dedicated downloading class which handles both the <b>table download</b> and the <b>cif file download</b> of all missing structures. The table is saved to the subdirectory "*data/gpcrdb/*", the cif (or pdb) files are put into *data/mmcif/*.

# %%
D = Download(fileformat='cif')
D.download_table(reload=True)  # without reload-flag set to True (i.e. file exists) we do not re-download
# since these are unprocessed files, existing files are NOT RE-DOWNLOADED
# however we "reference" the gpcrdb gpcr structure table to see if we have missing structures, and download those
D.download_pdbs()

# %% [markdown]
# ## DOWNLOADING MORE (META)DATA

# %% [markdown]
# Next we initialize our processing class, creatively named ***CifProcessor***, which includes all the functions used to gather more metainformation; Such as download data from **sifts mapping** (aka the mapping of the sequence of the pdb structure against the related uniprot gene sequence), the **(generic residue) numbering** from gpcrdb, as well as does the calculation of consecutive **psi-** and **phi-angles** and **assigns the generic residue numbers** for both the gpcr or (in complexes) the gprotein. (This of course includes alignment functions and many more). Last and not least the common data format in this processor class, as we will see, are python **DataFrames**.

# %% [markdown]
# Let's start slow by initializing the processor:

# %%
p = CifProcessor()

# %% [markdown]
# Next we load 3 datatables (referred to as metainfo): sifts mapping ('p.mappings'), table with uniprot ids and corresponding generic residue numbers ('p.numbering') and our previously downloaded data structure table ('p.table'). 

# %%
p.read_pkl_metainfo()

# %%
p.mappings.head(3)

# %%
p.numbering.head(3)

# %%
p.table.head(3)

# %% [markdown]
# This next step takes a very long time (**~30 minutes**), depending on if we make calls to download generic residue numbers (for uniprot ids) from [gpcrdb](https://docs.gpcrdb.org/web_services.html). (Of course if you have new samples, setting *reload_numbering* to *False* will still mean you are going to update your data to include it.) If you have an existing table, **NOT RELOADING IS ADVISED**!

# %%
p.make_metainfo(reload_numbering=False, reload_mapping=False, overwrite=True)

# %% [markdown]
# ## CREATING RAW DATAFRAMES

# %% [markdown]
# To start processing files from scratch you may run this next line, it will **DELETE** all pickle files (*.pkl*, the format of the dataframes) in your specified folder:

# %% [markdown]
# *p.del_pkl(folder='data/raw/')*

# %% [markdown]
# So far we only have a folder with unprocessed cif files - data is in text format. Very slow to process etc, so we first create the base structure of our dataset: a list of dataframes, **one dataframe per structure**. (During the loading process we also calculate the psi, phi and omega angles for all residues in the structure.) These are then saved (without any further processing to the subdirectory *data/raw/*.

# %%
p.make_raws(overwrite=False)

# %% [markdown]
# Setting overwrite to False will skip all processed samples! If it is set to True, you process all your samples and overwrite the existing data.

# %% [markdown]
# # GENERIC RESIDUE NUMBERS

# %% [markdown]
# As mentioned, the dataformat we use (after loading the cif files) are DataFrames (implements a variety of SQL methods and provides easy pythonic access to big datatables). Now each of our structures corresponds to a single DataFrame in the main data container in our processor class: p.dfl (dataframe list), for easier access and search it is paired with a list containing only the respective pdb name, p.dfl_list.

# %%
p.dfl

# %% [markdown]
# What happened? Well currently we have no new structures (since a few days ago when I last updated the data), thus no new structures were processed and added to the data list. To load our data:

# %%
p.read_pkl(mode='', folder='data/raw/')

# %% [markdown]
# Let's have a look at the first entry:

# %%
p.dfl[10].head(3)

# %%
p.dfl_list[10]

# %%
p.table

# %% [markdown] {"heading_collapsed": true}
# ## FILTERING, REDUCED TABLES AND PDB LISTS

# %% [markdown] {"hidden": true}
# Creating filters is easy with the processor and can primarily be done in two ways. The first is by applying the class function *p.make_filter()*:

# %% {"hidden": true}
f_act = p.make_filter(State='Active')

# %% [markdown] {"hidden": true}
# The implemented filter options (such as filtering by states etc) with examples in brackets are:
# - Species  (Human, Bovine)
# - State  (Active, Inactive, Intermediate)
# - Cl  (Rhodopsin, A, ...)
# - Family  (Gi/o, Gs, ...)
# - Subtype  (αi1, ...)
# - Resolution  (<=3.5)
# - Function  (Agonist, Antagonist)
# - gprotein  (filters out all complexes where not both Family and Subtype are within a predifined set of regular gproteins)
# - pdb_ids  (eg. \['7AD3', '4DKL'\] -> creates a filter only containing the listed pdbs if present in our metainfo table) 

# %% [markdown] {"hidden": true}
# Filters may be "applied", where we effectively remove all data except the ones present in the filter (simultaneously being removed from both *p.dfl* and *p.dfl_list*.

# %% {"hidden": true}
p.apply_filter(f_act)

# %% [markdown] {"hidden": true}
# The second option and this is what happens behind the scene of the *p.make_filter()* function: **QUERIES** (SQL)

# %% {"hidden": true}
f_act = f_act[f_act['Method']=='cryo-EM']

# %% {"hidden": true}
f_act.head(3)

# %% [markdown] {"hidden": true}
# So filters really are just filtered metainfotables and since they are easily interchangable / by extension also pdb lists. 

# %% [markdown] {"heading_collapsed": true}
# ## GPCR GENERIC RESIDUE NUMBERS

# %% [markdown] {"hidden": true}
# The tricky thing about assigning generic residue numbers to structures is that many times the residue sequence of the structure is not exactly the same as that of the *correct, corresponding gene*, further a structure may include many different chains, include fused parts, be in a complex with gproteins etc

# %% [markdown] {"hidden": true}
# We got the **generic residue number** from [gpcrdb](https://docs.gpcrdb.org/web_services.html), now we want to allign the uniprot sequence to the structure's sequence - this is why we got the **sifts mapping** (it's a crude alignment where we have pairs of starting and end points of intervals that align of both sequences). Unfortunately the sifts is often not correct - there are often shifts or other miss-alignments withing a given sifts interval, sometimes the sifts interval do not even have equal length... In cases where this happens we do sequence alignment ourselves.

# %% [markdown] {"hidden": true}
# The **DATAFLOW** is as follows:
# - p.table contains the "preferred chain" information that tells us which chain in the pdb is the gpcr.
# - p.mapping contains the sifts mapping (we check if the interval length is correct).
# - We get the uniprot sequence.
# - We use SQL to quickly produce new columns in our dataframe - very much like *label_seq_id* (which corresponds to the autogenerated sequence id of each atom (of its residue), but instead gives the position (if said atom/residue is present in the gene) within the uniprot sequence...
# - .. as well as the corresponding uniprot residue name.
# - We check, if there are (too) many non-matching residues of the alignment between structure and uniprot sequence and..
# - .. should there be too many too many non-matching residues, we, instead of using the sifts data to align structure to uniprot, do a sequence alignment.
# - Lastly, given the uniprot sequence number we can look up the generic residue numbers in p.numbering

# %% {"hidden": true}
p.allow_exception=False  # default

# %% [markdown] {"hidden": true}
# If set to True, this flag makes the processor skip any structures that throw errors, may be useful if you have "dirty data", but don't want to go through the pain of sorting out anything that won't get processed.

# %% [markdown] {"hidden": true}
# To start from scratch let's delete the processor (and all data loaded into memory) ... 

# %% {"hidden": true}
del p

# %% [markdown] {"hidden": true}
# ... and reinitialize it, quickly, in 3 lines:

# %% {"hidden": true}
p = CifProcessor()
p.read_pkl_metainfo()
p.read_pkl(mode='', folder='data/raw/')

# %% [markdown] {"hidden": true}
# Before (finally) running the generic residue assignment a quick explanation of the function inputs:
# - f: filter, this is a dataframe gained from filtering the p.table
# - pdb_ids: a list of pdbs to process (optional; an empty list means we use the filter)
# - overwrite: if set to True, after processing all structures get saved (overwriting existing files)
# - folder: location where the processed files get stored

# %% [markdown] {"hidden": true}
# You need to provide a filter-like input, so the processor knows which data it should assign generic residue numbers to. Imagine loading all the data from the raw folder and being interested in *Class A (Rhodopsin)* of *humans* structures only... Notably, providing a list of ***pdb_ids*** **overwrites whatever** ***filter*** you give it. If you want **NO FILTER** (i.e. process all your raw structures) just use ***f=p.table*** (not applying any filter at all).

# %% {"hidden": true}
p.assign_generic_numbers_r(f=p.table, pdb_ids=[], overwrite=True, folder='data/processed/')

# %% [markdown] {"hidden": true}
# The processed dataframes are safed to the specified folder (*data/processed/*, and the file is named '*\<pdb id>_r.pkl*'.

# %% [markdown] {"heading_collapsed": true}
# ## GPROTEIN GENERIC RESIDUE NUMBERS

# %% [markdown] {"hidden": true}
# Very similar to how assigning generic residue numbers to receptors work, however gproteins are (EVEN) more difficult to number due to a few reasons:
# - First, gpcrdb does not provide a webservice to query generic position numbers for gproteins.
# - Second, we do not know which, if any at all, chain of the pdb structure is a gprotein.
# - Third, there're gproteins in the form of helix-5-like-peptides fused to the gpcr, chimeras, mini-Gproteins, etc
# - And of course all of the problems we faced with gpcrs in terms of aligning them to the uniprot sequence. (Just worse since often there are no sifts mappings provided at all.)

# %% [markdown] {"hidden": true}
# How did we solve these issues?
# - First issue: This could be easily solved, since the number of gproteins is far less than that of receptors, and gpcrdb does provide an old excel sheet with generic residue numbers for all of them (the file was downloaded and manually updated from .xls to .xlsx). The file is stored in "*data/alignments/residue_table.xlsx*". 
# - Second issue: since there is a limited number of gproteins we do sequence alignments of ALL chains to ALL gprotein (uniprot) sequences and, using a scoring function as well as a threshold, we detect the BEST gprotein fit if it is present.
# - Third issue: This is not solved per-se, but generally two facts somewhat make our solution work: using alignments to find gproteins does make the type of gprotein we're looking at irrelevant, since IF we manage to find a good enough alignment we simply use that; Also many gproteins are similar, so even if not the correct gprotein is picked from the alignment (often happens in chimeras) due to their overall similarity the results seem consistently fine.
# - And... Pain.

# %% [markdown] {"hidden": true}
# Quick note: The function expects as an input files that have already been processed by assigning generic residue numbers to the receptor (this is useful to check if what we're trying to label is the gpcr chain). Thus you should always either load data from the processed folder with the flag *mode='r'* or if you start by processing raws first run "*assign_generic_numbers_r*".

# %% [markdown] {"hidden": true}
# To make sure everything you want to assign gprotein-generic-residue-numbers to, let's reinitialize/load our data from the correct folder. Note that the flag 'mode' in the *p.read_pkl()* function can be:
# - '' / an empty string: This loads the raw dataframes from the raw folder without gen. position numbers (requires *folder='data/raw/'*)!
# - 'r': This loads the dataframes where generic position numbers have been assigned for the receptors (but not the gproteins)!
# - 'rg': This loads the dataframes where generic position numbers have been assigned for both gpcr and gprotein!

# %% {"hidden": true}
del p
p = CifProcessor()
p.read_pkl_metainfo()
p.read_pkl(mode='r', folder='data/processed/')
f_act = p.make_filter(State='Active', Cl='Rhodopsin', gprotein=True)
f_act = f_act[f_act['Method']=='cryo-EM']

# %% {"hidden": true}
p.assign_generic_numbers_g(f=f_act, pdb_ids=[], overwrite=False, folder='data/processed/', fill_H5=True)

# %% {"hidden": true}
len(p.dfl)

# %% {"hidden": true}
p.apply_filter(f_act)

# %% {"hidden": true}
len(p.dfl)

# %% {"hidden": true}
df = p.dfl[10]

# %% {"hidden": true}
df[df['gprot_pos']!=''].head(10)

# %% [markdown] {"hidden": true}
# Assigning generic residue numbers to anything but **active complexes** is a waste of computational time and potentially errenous, since looking for the *BEST* alignment of any gprotein to *anything* in your structure **may result in things getting labelled that are NOT the gprotein** (there is a threshold on the alignment score, but I've not checked what happens... so I take no responsibility ^^). 

# %% [markdown] {"hidden": true}
# The fill_H5 flag was used since many of the H5-helices did not get labelled - especially in weird active complexes where not the full H5 is present.

# %% [markdown] {"heading_collapsed": true}
# ## Modifying *.cif* files

# %% {"hidden": true}
structure = p.dfl[10]
df = spread_gen_pos_to_residue_atoms(structure)
doc = write_cif_with_gen_seq_numbers(df, replace_idx = -7)
doc.write_file('data/modified/'+df['PDB'].iloc[0]+'_modified.cif')

# %% [markdown] {"hidden": true}
# The only thing to mention here is that cif (or pdb files in general) and their "visualization" in pymol are fixed, and we merely choose a column to overwrite (e.g. 'b-factors') with the generic position numbers which can be labelled in pymol.
# This works reasonably well for gpcr generic position numbers (since they can be represented as a float, however for gproteins the type of *string* makes this crude method fail (assigns a 0.0 instead of 'G.H5.26'), another column could be the 'auth_seq_id'...

# %% [markdown] {"heading_collapsed": true}
# # ANALSYSIS

# %% [markdown] {"hidden": true}
#

# %% {"hidden": true}

# %% [markdown] {"hidden": true}
# ## DISTANCE ANALYSIS

# %% [markdown] {"hidden": true}
# This section is deprecated (I need to look into thresholds etc again

# %% {"hidden": true}
from analysis.analysis import *

# %% {"hidden": true}
# del p
p = CifProcessor()
p.read_pkl_metainfo()
p.read_pkl(mode='rg', folder='data/processed/')

# %% {"hidden": true}
f_gio = p.make_filter(State='Active', Cl='Rhodopsin', Family='Gi/o', gprotein=True)
fuf = f_gio[f_gio['PDB']=='6FUF']
f_gio = f_gio[f_gio['Method']!='X-ray']
# Combine with 6FUF
f_gio = f_gio.append(fuf).reset_index(drop=True)
f_gs = p.make_filter(State='Active', Cl='Rhodopsin', Family='Gs', gprotein=True)
f_q11 = p.make_filter(State='Active', Cl='Rhodopsin', Family='Gq/11', gprotein=True)

# %% {"hidden": true}
len(f_gio) + len(f_gs) + len(f_q11)

# %% {"hidden": true}
filtered_indices_gio = [x for x in p.get_dfl_indices(list(f_gio['PDB'])) if x != None]
filtered_indices_gs = [x for x in p.get_dfl_indices(list(f_gs['PDB'])) if x != None]
filtered_indices_q11 = [x for x in p.get_dfl_indices(list(f_q11['PDB'])) if x != None]

# %% {"hidden": true}
section = 'H5'
poi =  'G.H5.23', 7.50
start = 7.40
end = 8.53
l = [filtered_indices_gio, filtered_indices_gs, filtered_indices_q11]

# %% {"hidden": true}
(poi)

# %% {"hidden": true}
list_poi_list, list_dists_df_list = get_interaction_tables(p, l, section=section, poi=(poi), start=start, end=end, eps=0.05)

# %% {"hidden": true}
poi_gio = [x for x in list_poi_list[0] if ((x[1] == 0) & (x[2] > 0))]
poi_gs = [x for x in list_poi_list[1] if ((x[1] == 1) & (x[2] > 0))]
poi_q11 = [x for x in list_poi_list[2] if ((x[1] == 2) & (x[2] > 0))]

# %% {"hidden": true}
len(poi_gio) + len(poi_gs) + len(poi_q11)

# %% {"hidden": true}
"""with open('gq11_350_h523.txt', 'w') as f:
    f.write(str(poi_q11))
with open('gs_350_h523.txt', 'w') as f:
    f.write(str(poi_gs))
with open('gio_350_h523.txt', 'w') as f:
    f.write(str(poi_gio))"""

# %% {"hidden": true}
list_poi_list, list_dists_df_list = get_interaction_tables(p, l, poi=(poi), start=start, end=end, eps=0.05)

# %% {"hidden": true}
occ_df, mean_df, std_df = make_overview_df(list_dists_df_list[0])  # Gio
# occ_df, mean_df, std_df = make_overview_df(list_dists_df_list[1])  # Gs
# occ_df, mean_df, std_df = make_overview_df(list_dists_df_list[2])  # Gq11

# %% {"hidden": true}
# make_overview_plots(occ_df, title='Occurances', cl='A', gprot='Gio', save=False)

# %% {"hidden": true}
# make_overview_plots(mean_df, title='Mean', cl='A', gprot='Gio', save=False)

# %% {"hidden": true}
# make_overview_plots(std_df, title='Std', cl='A', gprot='Gio', save=False)

# %% {"hidden": true}

# %% {"hidden": true}
occ_df1, mean_df1, std_df1 = make_overview_df(list_dists_df_list[0])
occ_df2, mean_df2, std_df2 = make_overview_df(list_dists_df_list[1])
diff_df = get_overview_diff(std_df1, std_df2, mean_df1, mean_df2, ab=False, cutoff_mean=12)

# %% {"hidden": true}
make_overview_plots(diff_df, title='Difference in STD', cl='Family A', gprot='- Gio vs Go - 7_50', save=True)

# %% {"hidden": true}

# %% {"hidden": true}
section = 'H5'
poi =  'G.H5.23', 3.5
start = 3.42
end = 3.53
l = [filtered_indices_gio, filtered_indices_gs, filtered_indices_q11]
list_poi_list, list_dists_df_list = get_interaction_tables(p, l, poi=poi, start=start, end=end, eps=.05)
occ_df1, mean_df1, std_df1 = make_overview_df(list_dists_df_list[0])
occ_df2, mean_df2, std_df2 = make_overview_df(list_dists_df_list[1])
diff_df = get_overview_diff(std_df1, std_df2, mean_df1, mean_df2, ab=False, cutoff_mean=12)

# %% {"hidden": true}
poi_str = list(str(poi[1]))[0] + '_' + ''.join(list(str(poi[1]))[2:])
make_overview_plots(diff_df, title='Difference in STD', cl='Family A', gprot='- Gio vs Go - ' + poi_str, save=True)

# %% {"hidden": true}
section = 'H5'
poi =  'G.H5.23', 7.5
start = 7.48
end = 7.56
l = [filtered_indices_gio, filtered_indices_gs, filtered_indices_q11]
list_poi_list, list_dists_df_list = get_interaction_tables(p, l, poi=poi, start=start, end=end, eps=.05)
occ_df1, mean_df1, std_df1 = make_overview_df(list_dists_df_list[0])
occ_df2, mean_df2, std_df2 = make_overview_df(list_dists_df_list[1])
diff_df = get_overview_diff(std_df1, std_df2, mean_df1, mean_df2, ab=False, cutoff_mean=16)

# %% {"hidden": true}
poi_str = list(str(poi[1]))[0] + '_' + ''.join(list(str(poi[1]))[2:])
make_overview_plots(diff_df, title='Difference in STD', cl='Family A', gprot='- Gio vs Go - ' + poi_str, save=True)

# %% [markdown] {"hidden": true}
# ## HELICES - ANGLE ANALYSIS

# %% [markdown] {"hidden": true}
# To do helices analysis (both helices in the gprotein and gpcrs) we need to initialize a processor in 'rg' mode:

# %% {"hidden": true}
del p

# %% {"hidden": true}
p = CifProcessor()
p.read_pkl_metainfo()
p.read_pkl(mode='rg', folder='data/processed/')

# %% [markdown] {"hidden": true}
# Filter the data...

# %% {"hidden": true}
f_act = p.make_filter(State='Active', Cl='Rhodopsin', gprotein=True)
f_act = f_act[f_act['Method']=='cryo-EM']
p.apply_filter(f_act)

# %% {"hidden": true}
len(p.dfl)

# %% {"hidden": true}
"""gs_count_df = count_g_positions(p.dfl)
cont_sec_g = find_cont_sections_g(gs_count_df, min_count=30, min_length=6)
G_SECTION_DICT = make_cont_section_dict_g(cont_sec_g)
G_SECTION_DICT['H5_1'] = (13, 23)
G_SECTION_DICT['H4_0'] = (3, 15)
R_SECTION_DICT['TM6'] = (6.35)"""

# %% {"hidden": true}
G_SECTION_DICT

# %% {"hidden": true}
R_SECTION_DICT

# %% {"hidden": true}

# %% [markdown] {"heading_collapsed": true}
# # AFFINITY/SELECTIVITY PREDCITION

# %% [markdown] {"hidden": true}
# ## AFFINITIES

# %% {"hidden": true}

# %% [markdown] {"hidden": true}
# ## DATALOADER / DATASET

# %% {"hidden": true}

# %% [markdown] {"hidden": true}
# ## NEURAL NETWORK

# %% {"hidden": true}

# %% [markdown] {"hidden": true}
# ## REVERSE NN ANALYSIS

# %% {"hidden": true}

# %% {"hidden": true}
