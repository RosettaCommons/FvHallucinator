import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import requests
import argparse
from os.path import splitext, basename
from Bio import SeqIO


class RawTextArgumentDefaultsHelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    """CLI help formatter that includes the default value in the help dialog
    and formats as raw text i.e. can use escape characters."""
    pass


_aa_dict = {
    'A': '0',
    'C': '1',
    'D': '2',
    'E': '3',
    'F': '4',
    'G': '5',
    'H': '6',
    'I': '7',
    'K': '8',
    'L': '9',
    'M': '10',
    'N': '11',
    'P': '12',
    'Q': '13',
    'R': '14',
    'S': '15',
    'T': '16',
    'V': '17',
    'W': '18',
    'Y': '19'
}

_aa_1_3_dict = {
    'A': 'ALA',
    'C': 'CYS',
    'D': 'ASP',
    'E': 'GLU',
    'F': 'PHE',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'K': 'LYS',
    'L': 'LEU',
    'M': 'MET',
    'N': 'ASN',
    'P': 'PRO',
    'Q': 'GLN',
    'R': 'ARG',
    'S': 'SER',
    'T': 'THR',
    'V': 'VAL',
    'W': 'TRP',
    'Y': 'TYR',
    '-': 'GAP'
}


def letter_to_num(string, dict_):
    """Function taken from ProteinNet (https://github.com/aqlaboratory/proteinnet/blob/master/code/text_parser.py).
    Convert string of letters to list of ints"""
    patt = re.compile('[' + ''.join(dict_.keys()) + ']')
    num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
    num = [int(i) for i in num_string.split()]
    return num


def num_to_letter(array, dict_=_aa_dict):
    dict_rev = {}
    for key in dict_:
        dict_rev[int(dict_[key])] = key
    seq_array = [dict_rev[t] for t in array]
    return ''.join(seq_array)


def time_diff(start_time, end_time):
    """Returns the difference in time in HH:MM:SS format"""
    secs = int((end_time - start_time) % 60)
    mins = int(((end_time - start_time) // 60) % 60)
    hrs = int(((end_time - start_time) // (60 * 60)) % 60)
    return '{}:{:02}:{:02} (hrs:min:secs)'.format(hrs, mins, secs)


def one_hot_seq(seq):
    """Gets a one-hot encoded version of a protein sequence"""
    return F.one_hot(torch.LongTensor(letter_to_num(seq, _aa_dict)),
                     num_classes=20)


def load_full_seq(fasta_file):
    """Concatenates the sequences of all the chains in a fasta file"""
    with open(fasta_file, 'r') as f:
        return ''.join(
            [seq.rstrip() for seq in f.readlines() if seq[0] != '>'])


def get_fasta_chain_seq(fasta_file, chain_id):
    for chain in SeqIO.parse(fasta_file, 'fasta'):
        if ":{}".format(chain_id) in chain.id:
            return str(chain.seq)


def get_heavy_seq_len(fasta_file):
    h_len = len(get_fasta_chain_seq(fasta_file, "H"))

    return h_len


def get_fasta_basename(fasta_file):
    base = basename(fasta_file)  # extract filename w/o path
    if splitext(base)[1] == '.fasta':
        base = splitext(base)[0]  # remove .fasta if present
    return base


def get_redundant_structures(fasta_file,
                             num_structures=100,
                             seq_id=99,
                             check_cdrs=False,
                             h3_only=False):
    # Gets sequence redundant PDB IDs from SAbDab given a fasta file

    h_seq = get_fasta_chain_seq(fasta_file, "H")
    l_seq = get_fasta_chain_seq(fasta_file, "L")

    response = requests.get(
        'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/sequencesearch',
        params={
            "hchain": h_seq,
            "lchain": l_seq,
            "nostructures": num_structures,
            "minseqid": seq_id,
            "region": "Full variable region"
        })

    redundant_pdbs = [
        sele[-4:] for sele in re.findall(r"pdb=.{4}", response.text)
    ]
    redundant_pdbs = list(set(redundant_pdbs))

    if check_cdrs:
        cdr_regions = ["CDRH3"] if h3_only else [
            "CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3"
        ]
        for r in cdr_regions:
            response = requests.get(
                'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/sequencesearch',
                params={
                    "hchain": h_seq,
                    "lchain": l_seq,
                    "nostructures": num_structures,
                    "minseqid": seq_id,
                    "region": r
                })

            redundant_cdrs = [
                sele[-4:] for sele in re.findall(r"pdb=.{4}", response.text)
            ]
            redundant_pdbs += list(set(redundant_cdrs))

    return redundant_pdbs


def get_sequence_similarities(fasta_file,
                              num_sequences=100,
                              seq_id=70,
                              cdr=None):
    # Gets sequence similarities to PDB IDs from SAbDab given a fasta file
    cdr_names = {
        "h1": "CDRH1",
        "h2": "CDRH2",
        "h3": "CDRH3",
        "l1": "CDRL1",
        "l2": "CDRL2",
        "l3": "CDRL3"
    }

    h_seq = get_fasta_chain_seq(fasta_file, "H")
    l_seq = get_fasta_chain_seq(fasta_file, "L")

    if cdr == None:
        response = requests.get(
            'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/sequencesearch',
            params={
                "hchain": h_seq,
                "lchain": l_seq,
                "nostructures": num_sequences,
                "minseqid": seq_id,
                "region": "Full variable region"
            })

        redundant_pdbs = {
            sele[4:8]: float(sele[-10:-5])
            for sele in re.findall(r"pdb=.{4}.*?center;\">.{5}</td>",
                                   response.text)
        }
    elif cdr.lower() in cdr_names:
        response = requests.get(
            'http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/sequencesearch',
            params={
                "hchain": h_seq,
                "lchain": l_seq,
                "nostructures": num_sequences,
                "minseqid": seq_id,
                "region": cdr_names[cdr.lower()]
            })

        redundant_pdbs = {
            sele[4:8]: float(sele[-10:-5])
            if sele[-10:-5] != "00.00" else float(sele[-11:-5])
            for sele in re.findall(r"pdb=.{4}.*?center;\">.{5,6}</td>",
                                   response.text)
        }
    else:
        print("Unrecognized CDR name:\t{}".format(cdr))
        return None

    return redundant_pdbs
