import requests
import re


base_url = "https://www.ebi.ac.uk/pdbe/"
api_base = base_url + "api/"
uniprot_mapping_url = api_base + 'mappings/uniprot/'


def make_request(url, mode, pdb_id):
    """
    This function can make GET and POST requests to
    the PDBe API
    
    :param url: String,
    :param mode: String,
    :param pdb_id: String
    :return: JSON or None
    """
    if mode == "get":
        response = requests.get(url=url+pdb_id)
    elif mode == "post":
        response = requests.post(url, data=pdb_id)

    if response.status_code == 200:
        return response.json()
    else:
        print("[No data retrieved - %s] %s" % (response.status_code, response.text))
    return None


def get_mappings_data(pdb_id):
    """
    This function will GET the mappings data from
    the PDBe API using the make_request() function
    
    :param pdb_id: String
    :return: JSON
    """
    # Check if the provided PDB id is valid
    # There is no point in making an API call
    # with bad PDB ids
    if not re.match("[0-9][A-Za-z][A-Za-z0-9]{2}", pdb_id):
        print("Invalid PDB id")
        return None
    
    # GET the mappings data
    mappings_data = make_request(uniprot_mapping_url, "get", pdb_id)
    
    # Check if there is data
    if not mappings_data:
        print("No data found")
        return None
    return mappings_data


def list_uniprot_pdb_mappings(pdb_id):
    """
    This function retrieves PDB > UniProt
    mappings using the get_mappings_data() function
    
    :param pdb_id: String,
    :return: None
    """
    
    # Getting the mappings data
    mappings_data = get_mappings_data(pdb_id)
    
    # If there is no data, return None
    if not mappings_data:
        return None
    return mappings_data


def get_uniprot_pdb_residue_mapping(pdb_id, chain_id, residue_number):
    """
    This function uses get_mappings_data() function
    to retrieve mappings between UniProt and PDB
    for a PDB entry, and then maps one specific
    residue of one specific chain
    
    :param pdb_id: String,
    :param chain_id: String,
    :param residue_number: String,
    :return: Integer
    """
    
    mappings_data = get_mappings_data(pdb_id)
    
    if not mappings_data:
        return None
    
    uniprot = mappings_data[pdb_id]["UniProt"]
    for uniprot_id in uniprot:
        for i in range(len(uniprot[uniprot_id]["mappings"])):
            mapping = uniprot[uniprot_id]["mappings"][i]
            if not mapping["chain_id"] == chain_id:
                continue
            pdb_start = mapping["start"]["residue_number"]
            pdb_end = mapping["end"]["residue_number"]
            uniprot_start = mapping["unp_start"]
            uniprot_end = mapping["unp_end"]
            if residue_number >= pdb_start and residue_number <= pdb_end:
                offset = uniprot_start - pdb_start
                return residue_number + offset
    return uniprot
