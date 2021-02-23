from bs4 import BeautifulSoup
import requests
import pandas as pd


cols = ['cross',  'filler', 'uniprotid', 'filler2', 'receptor family', 'Cl.', 'Species', 'Method', 'PDB',  \
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
    output = []
    for c in chs:
        if 'href' in c:
            print(c)
        if c.string!=None:
            output.append(c.string.replace(' ', '').replace('\n', ''))
        else:
            output.append(None)
    return output


def create_structure_df(table):
    nrows = len(table) // len(cols)
    holder = []
    for i in range(nrows):
        holder.append(table[i*len(cols):(i+1)*len(cols)])
    return pd.DataFrame(holder, columns=cols)


page = getpage()
table = find_table(page)
df = create_structure_df(table)
df.to_csv('data/structure.csv')
