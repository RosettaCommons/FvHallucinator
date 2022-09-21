import os
import sys
import glob
import matplotlib.pyplot as plt
import pandas as pd
from hallucination.utils.sequence_utils import biopython_developability_dataframes
from util.pdb import get_pdb_chain_seq

from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1

exe='netMHCIIpan'
print('Please edit file to specify path to netMHCIIpan')

biopython_developability_keys = ['Charge at pH7', 'Gravy', 'Instability Index']
# IgLM and Mason et al. alleles
NET_MHCII_PAN_ALLELES = \
    ['DRB1_0101', 'DRB1_0301', 'DRB1_0401', 'DRB1_0405', 'DRB1_0701',
     'DRB1_0802', 'DRB1_0901', 'DRB1_1101', 'DRB1_1201', 'DRB1_1302', 'DRB1_1501',
     'DRB3_0101', 'DRB3_0202', 'DRB4_0101', 'DRB5_0101',
      'HLA-DQA10501-DQB10201', 'HLA-DQA10501-DQB10301', 'HLA-DQA10301-DQB10302',
      'HLA-DQA10401-DQB10402', 'HLA-DQA10101-DQB10501', 'HLA-DQA10102-DQB10602',
      'HLA-DPA10103-DPB10101', 'HLA-DPA10103-DPB10201', 'HLA-DPA10103-DPB10401', 
      'HLA-DPA10103-DPB10402', 'HLA-DPA10103-DPB10501', 'HLA-DPA10103-DPB11401'
      ]

# Get SAP scores
def get_sap_scores_for_pdbs(pdb_path='',
                            pdb_files=[], 
                            path_results='./', 
                            indices_hal=None, 
                            out_path='./',
                            wt_pdb=''):
    from pyrosetta_utils import get_sapscores
    if not pdb_path=='':
        pdb_files = glob.glob(pdb_path+'*.pdb')
    else:
        assert len(pdb_files) >= 1

    rosetta_indices = indices_hal
    if not indices_hal is None:
        rosetta_indices = [t+1 for t in indices_hal]
    sap_scores = ['{}\t{}\n'.format(pdbfile, get_sapscores(pdbfile, rosetta_indices))
                    for pdbfile in pdb_files]
    outfile_sap_scores = os.path.join(path_results, 'sapscores.csv')
    open(outfile_sap_scores, 'w').write(''.join(sap_scores))
    param = 'SAP score'
    clean_param_name = param.replace(' ', '')

    outfile = '{}/develop_{}_dist.png'.format(out_path, clean_param_name)
    df = pd.DataFrame()
    df[param] = [float(t.split('\t')[1]) for t in sap_scores]
    if wt_pdb != '' and os.path.exists(wt_pdb):
        sap_wt = get_sapscores(wt_pdb, rosetta_indices)
        df_wt = pd.DataFrame()
        df_wt[param] = [sap_wt]
    plot_developability_param(df, param, df_wt, outfile)


def get_developability(sequences_file,
                        wt_pdb,
                        indices_hal,
                        wt_seq='',
                        out_path='.'):
    sequences_fasta = open(sequences_file, 'r').readlines()
    sequences_fasta = [t for t in sequences_fasta if t != '\n']
    sequences_fasta_hl = [
        ''.join(sequences_fasta[i+1: i + 4: 2])
        for i in range(0, len(sequences_fasta), 4)
    ]
    len_heavy_seqfile = len(sequences_fasta[1].rstrip())
    ids = [int(t.split('_')[1]) for t in sequences_fasta if (t.find('>') !=-1) and (t.find(':H') !=-1)]
    dsequences = {}
    assert len(ids) == len(sequences_fasta_hl)
    for id, seq in zip(ids, sequences_fasta_hl):
        dsequences[id] = seq
    traj_ids = [t for t in dsequences]
    traj_ids.sort()
    
    lines = open(sequences_file, 'r').readlines()
    sequences = [t.rstrip() for t in lines if t.find('>') == -1]
    try:
        ids = [int(t.split('_')[1]) for t in lines if (t.find('>') !=-1)]
        assert len(ids) == len(sequences)
        ids_sequences_tuples = [(id, seq) for id, seq in zip(ids, sequences)]
        dsequences = {}
        for (id, seq) in ids_sequences_tuples:
            dsequences[id] = seq
    except:
        dsequences = {}

    wt_heavy_seq = get_pdb_chain_seq(wt_pdb, 'H')
    wt_light_seq = get_pdb_chain_seq(wt_pdb, 'L')
    len_heavy = len(wt_heavy_seq)
    print(len_heavy_seqfile, len_heavy)
    assert len_heavy == len_heavy_seqfile
    df_developability = biopython_developability_dataframes(sequences_fasta_hl, len_heavy, indices_hal)
    outfile = os.path.join(out_path, 'df_developability_biopython.csv')
    df_developability.to_csv(outfile, index=False)
    df_developability_Wt = biopython_developability_dataframes([wt_seq], len_heavy, indices_hal)
    outfile = os.path.join(out_path, 'df_developability_biopython_Wt.csv')
    df_developability_Wt.to_csv(outfile, index=False)

    
    unique_chains = set(list(df_developability['Chain']))
    for chain in unique_chains:
        df_chain = df_developability[df_developability['Chain']==chain]
        df_chain_wt = df_developability_Wt[df_developability_Wt['Chain']==chain]
        for param in biopython_developability_keys:
            clean_param_name = param.replace(' ','')
            outfile = '{}/biodevelop_{}_dist_chain{}.png'.format(out_path, clean_param_name, chain)
            plot_developability_param(df_chain, param, df_chain_wt, outfile)


def get_fixed_length_substrings(input_string_dict, length=15):
    key_0 = list(input_string_dict.keys())[0]
    in_length = len(input_string_dict[key_0])
    out_strs = {'{}_{}'.format(key, i):input_string_dict[key][i:i+length]
                 for key in input_string_dict 
                 for i in range(0, in_length - length)}
    return out_strs


def get_padded_fasta(sequences_file,
                    design_indices=[t for t in range(98,108)],
                    pad_length=10, 
                    b_write=True,
                    include_substrings=False):
    sequences_fasta = open(sequences_file, 'r').readlines()
    sequences_fasta = [t for t in sequences_fasta if t != '\n']
    sequences_fasta_hl = [
        sequences_fasta[i+1].rstrip()+sequences_fasta[i + 3].rstrip()
        for i in range(0, len(sequences_fasta), 4)
    ]
    len_heavy_seqfile = len(sequences_fasta[1].rstrip())
    try:
        ids = [int(t.split('_')[1]) for t in sequences_fasta if (t.find('>') !=-1) and (t.find(':H') !=-1)]
    except:
        ids = [t for t in range(len(sequences_fasta_hl))]
    if len(ids)==0:
        ids = [t for t in range(len(sequences_fasta_hl))]
    
    print(len(ids), len(sequences_fasta_hl))
    dsequences = {}
    assert len(ids) == len(sequences_fasta_hl)
    for id, seq in zip(ids, sequences_fasta_hl):
        dsequences[id] = seq
    traj_ids = [t for t in dsequences]
    traj_ids.sort()
    
    start = design_indices[0] - pad_length
    end = design_indices[-1] + pad_length
    print(start, end)
    assert (end - start) > 10
    padded_sequences = {key:dsequences[key][start:end] for key in dsequences}
    #print(padded_sequences)
    if include_substrings:
        padded_sequences = get_fixed_length_substrings(padded_sequences)
    if b_write:
        fasta_outstr = ['>design_{}\n{}\n'.format(key, padded_sequences[key])
                    for key in padded_sequences]
                
        output_dir = os.path.dirname(sequences_file)
        tmp_fasta = '{}/paded_design_sequences.fasta'.format(output_dir)
        open(tmp_fasta,'w').write(''.join(fasta_outstr))
        
    return padded_sequences
    
def plot_developability_param(df, param, 
                            df_ref_wt=None,
                            outfile='param.png'):
    import matplotlib
    theme = {'axes.grid': True,
            'grid.linestyle': '',
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            "font.weight": 'regular',
            'xtick.color': 'black',
            'ytick.color': 'black',
            "axes.titlesize": 20,
            "axes.labelsize": 20
        }
    
    matplotlib.rcParams.update(theme)
    import seaborn as sns
    fig = plt.figure(figsize=(6,5))
    ax = plt.gca()
    ax = sns.histplot(data=df, x=param, ax=ax, stat='probability', color='blue', element="step", fill=False,
                    legend=False)
    if not df_ref_wt is None:
        print('Wt ', param, list(df_ref_wt[param])[0])
        ax.axvline(list(df_ref_wt[param])[0], ls='--', lw=2.0, c='black', zorder=1)
    plt.xticks(rotation=45)
    ax.set_xlabel(param)
    ax.set_ylabel('P({})'.format(param))
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, transparent=True)
    plt.close()


def plot_developability_param_with_baseline(df, param, 
                            df_ref_wt=None,
                            outfile='param.png',
                            hue='Method',
                            common_norm=False,
                            palette={},
                            legend=True):

    def get_color_palette():
        colors_select = ['grey', 'cornflowerblue', 'slateblue', 'violet', 'pink']
        methods = list(set(list(df[hue])))
        i=0
        for meth in methods:
            if meth=='Hallucination':
                palette[meth] = 'blue'
            elif meth in ['Wt', 'Wildtype']:
                palette[meth] = 'black'
            elif meth=='Therapeutic':
                palette[meth] = 'orange'
            else:
                palette[meth] = colors_select[i]
                i+=1
        return palette
    
    import matplotlib
    theme = {'axes.grid': True,
            'grid.linestyle': '',
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            "font.weight": 'regular',
            'xtick.color': 'black',
            'ytick.color': 'black',
            "axes.titlesize": 20,
            "axes.labelsize": 20
        }
    
    matplotlib.rcParams.update(theme)
    import seaborn as sns
    fig = plt.figure(figsize=(6,5))
    ax = plt.gca()
    if palette == {}:
        palette = get_color_palette()
    if len(palette.keys()) > 2:
        ax = sns.kdeplot(data=df, x=param, ax=ax, hue=hue,
                            fill=False,palette=palette,
                            common_norm=common_norm,
                            legend=legend)
    else:
        ax = sns.histplot(data=df, x=param, ax=ax, stat='probability', hue=hue,
                        element="step", fill=False,
                        palette=palette, legend=False, common_norm=common_norm)
    if not df_ref_wt is None:
        print('Wt ', param, list(df_ref_wt[param])[0])
        ax.axvline(list(df_ref_wt[param])[0], ls='--', lw=2.0, c='black', zorder=1)
    plt.xticks(rotation=45)
    ax.set_xlabel(param)
    ax.set_ylabel('P({})'.format(param))
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, transparent=True)
    plt.close()


def compute_immunogenicities(sequences_file, design_indices,
                             wt_pdb='', pad_length=10,
                             max_count=500):
    """
    pad length same as Mason et al.
    """
    #from mhctools import NetMHCIIpan4
    # Run netMHCIIpan with padded sequence around target_loop
    sequences_fasta = open(sequences_file, 'r').readlines()
    sequences_fasta = [t for t in sequences_fasta if t != '\n']
    sequences_fasta_hl = [
        sequences_fasta[i+1].rstrip()+sequences_fasta[i + 3].rstrip()
        for i in range(0, len(sequences_fasta), 4)
    ]
    len_heavy_seqfile = len(sequences_fasta[1].rstrip())
    try:
        ids = [int(t.split('_')[1]) for t in sequences_fasta if (t.find('>') !=-1) and (t.find(':H') !=-1)]
    except:
        ids = [t for t in range(len(sequences_fasta_hl))]
    if len(ids)==0:
        ids = [t for t in range(len(sequences_fasta_hl))]
    
    print(len(ids), len(sequences_fasta_hl))
    
    dsequences = {}
    assert len(ids) == len(sequences_fasta_hl)
    count = 0
    for id, seq in zip(ids, sequences_fasta_hl):
        dsequences[id] = seq
        if count>max_count:
            break
        count+=1
    traj_ids = [t for t in dsequences]
    traj_ids.sort()
    
    
    start = design_indices[0] - pad_length
    end = design_indices[-1] + pad_length
    print(start, end)
    assert (end - start) > 10
    padded_sequences = {key:dsequences[key][start:end] for key in dsequences}
    padded_sequences = get_fixed_length_substrings(padded_sequences)
    fasta_outstr = ['>design_{}\n{}\n'.format(key, padded_sequences[key])
                    for i,key in enumerate(padded_sequences)]
    output_dir = os.path.dirname(sequences_file)
    tmp_fasta = '{}/tmp_mhcpeptides.fasta'.format(output_dir)
    open(tmp_fasta,'w').write(''.join(fasta_outstr))
    out_xls = '{}/develop_netMHCIIpan.csv'.format(output_dir)
    if not os.path.exists(out_xls):
        command='{} -a {} -f {} -xls -xlsfile {}'.format(exe,
                ','.join(NET_MHCII_PAN_ALLELES), tmp_fasta, out_xls)
        os.system(command)
    df = pd.read_csv(out_xls, sep='\t', header=1)
    rank_cols = [col for col in df.columns if col.find('Rank.')!=-1]
    df_sel = df[rank_cols+['ID']]
    df_sel['design'] = [''.join(t.split('_')[:-1]) for t in list(df_sel['ID'])]
    df_sel['min NetMHCII Rank'] = df_sel[rank_cols].min(axis=1)
    df_sel['mean NetMHCII Rank'] = df_sel[rank_cols].mean(axis=1)
    df_sel['Set'] = ['Hallucination' for _ in list(df_sel['mean NetMHCII Rank'])]
    out_csv = '{}/develop_netMHCIIpan_ranksonly.csv'.format(output_dir)
    df_sel.to_csv(out_csv)

    df_sel_grouped = df_sel.groupby('design').min().reset_index()
    
    if wt_pdb != '':
        parser = PDBParser()
        structure = parser.get_structure('id', wt_pdb)
        wt_seq = ''
        for chain in structure.get_chains():
            wt_seq += seq1(''.join([residue.resname for residue in chain]))
        
        pad_seq_wt = {'wt':wt_seq[start:end]}
        pad_seq_wt = get_fixed_length_substrings(pad_seq_wt)
        fasta_outstr = ['>{}\n{}\n'.format(key, pad_seq_wt[key])
                    for key in pad_seq_wt]
        tmp_fasta = '{}/tmp_wt_mhcpeptides.fasta'.format(output_dir)
        open(tmp_fasta,'w').write(''.join(fasta_outstr))
        out_xls = '{}/develop_wt_netMHCIIpan.csv'.format(output_dir)
        if not os.path.exists(out_xls):
            command='{} -a {} -f {} -xls -xlsfile {}'.format(exe,
                    ','.join(NET_MHCII_PAN_ALLELES), tmp_fasta, out_xls)
            os.system(command)
        
        df_wt = pd.read_csv(out_xls, sep='\t', header=1)
        rank_cols = [col for col in df_wt.columns if col.find('Rank.')!=-1]
        df_sel_wt = df_wt[rank_cols+['ID']]
        df_sel_wt['design'] = [''.join(t.split('_')[:-1]) for t in list(df_sel_wt['ID'])]
        df_sel_wt['min NetMHCII Rank'] = df_sel_wt[rank_cols].min(axis=1)
        df_sel_wt['mean NetMHCII Rank'] = df_sel_wt[rank_cols].mean(axis=1)
        df_sel_wt['Set'] = ['Wt' for _ in list(df_sel_wt['mean NetMHCII Rank'])]
        df_sel_wt_grouped = df_sel_wt.groupby('design').min().reset_index()
        df_sel_wt_grouped['Set'] = ['Wt' for _ in list(df_sel_wt_grouped['mean NetMHCII Rank'])]
        out_csv = '{}/develop_wt_netMHCIIpan_ranksonly.csv'.format(output_dir)
        df_sel_wt.to_csv(out_csv)

    param = 'min NetMHCII Rank'
    clean_name = param.replace(' ','')
    outfile = '{}/develop_{}_netMHCIIpan_dist.png'.format(output_dir, clean_name)
    outfile_csv = '{}/develop_{}_netMHCIIpan_dist.csv'.format(output_dir, clean_name)
    df_sel_grouped.to_csv(outfile_csv, index=False)
    df_sel_grouped_filtered = df_sel_grouped[(df_sel_grouped['min NetMHCII Rank']>4)]
    print(len(df_sel_grouped_filtered['min NetMHCII Rank']))
    outfile_csv = '{}/develop_{}_netMHCIIpan_dist_filtered.csv'.format(output_dir, clean_name)
    df_sel_grouped_filtered.to_csv(outfile_csv, index=False)
    if wt_pdb != '':
        plot_developability_param(df_sel_grouped, param, df_ref_wt=df_sel_wt_grouped, outfile=outfile)
    else:
        plot_developability_param(df_sel_grouped, param, outfile=outfile)

    param = 'mean NetMHCII Rank'
    clean_name = param.replace(' ','')
    outfile = '{}/develop_{}_netMHCIIpan_dist.png'.format(output_dir, clean_name)
    if wt_pdb != '':
        df_concat = pd.concat([df_sel, df_sel_wt])
        plot_developability_param_with_baseline(df_concat, param, outfile=outfile, hue='Set')
    else:
        plot_developability_param(df_sel, param, outfile=outfile)
