from bs4 import BeautifulSoup
import requests
import pandas as pd


COLS = ['cross',  'filler', 'uniprotid', 'filler2', 'receptor family', 'Cl.', 'Species', 'Method', 'PDB',  \
        'Refined Structure', 'Resolution', 'Preferred Chain', 'State', 'Degree active %', '% of Seq1', 'Family', \
        'Subtype', 'Note', '% of Seq2', 'Fusion', 'Antibodies', 'Name1', 'Type1', 'Function', 'Name2', 'Type2', \
        'D2x50 - S3x39', 'Sodium in structure', 'Authors', 'Reference', 'PDB date', 'Annotated']


def getpage(url='https://gpcrdb.org/structure/#'):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    return soup


def find_table(soup):
    table = soup.find("tbody")
    chs = table.findChildren(['tr', 'td'])
    pdb = soup.findAll("a", {"target": "_blank"}, href=True)
    pdb_links = []
    refined_links = []
    uniprot_links = []
    signprot_links = []
    protein_links = []
    pubchem = []
    for link in pdb:
        href = link['href']
        # if href.find('pubchem'):
        #     pubchem.append(href)
        # elif href.find('uniprot'):
        #     uniprot_links.append(href)

        # elif href.find('/protein/'):
        #     protein_links.append(href)
        # elif href.find('/signprot/'):
        #     signprot_links.append(href)

        if link.string and link.string.isalnum():
            s = link.string
            if len(s) == 4:
                pdb_links.append(href)
            elif s.find('refined'):
                refined_links.append(href)
    print(pdb_links)
    print(len(pdb_links))
    # print(refined_links)
    print(len(refined_links))
    print("\n\n\n")
    output = []
    for c in chs:
        if 'href' in c:
            print(c)
        if c.string != None:
            output.append(c.string.replace(' ', '').replace('\n', ''))
        else:
            output.append(None)
    return output


def create_structure_df(table):
    nrows = len(table) // len(COLS)
    holder = []
    for i in range(nrows):
        holder.append(table[i*len(COLS):(i+1)*len(COLS)])
    return pd.DataFrame(holder, columns=COLS)
