from src.util.pdb import get_indices_for_interface_residues, get_pdb_chain_seq
import numpy as np
import pandas as pd
import os, glob
from src.hallucination.params import germline_data_path
from src.util.util import _aa_dict, get_fasta_chain_seq, letter_to_num
import torch

def get_imgt_and_germline(heavy_seq, light_seq):
    germline_h_gene = ''
    num_imgt = []
    cmd='ANARCI -s imgt --assign_germline -i {} --use_species human > tmp.out'
    try:
        os.system(cmd.format(heavy_seq+light_seq))
        print(cmd.format(heavy_seq+light_seq))
        anarci_output = open('tmp.out', 'r').readlines()
        clean_lines = [t.strip() for t in anarci_output]
        print(clean_lines)
        i_gene = [i for i,t in enumerate(clean_lines) if t.find('v_gene') != -1][0]
        germline_h_gene = clean_lines[i_gene+1].split('|')[2]
        nohash_lines = [t for t in clean_lines if t.find('#')==-1]
        num_imgt = [''.join(t.split()[0:2]) for t in nohash_lines 
                        if len(t.split())==3 and t.split()[2] != '-']
        print(germline_h_gene)
        print(len(num_imgt), len(heavy_seq+light_seq))
        #assert len(num_imgt) == len(cur_chain)
    except:
        print('ANARCI not found.')
    return num_imgt, germline_h_gene

def get_germline_datafile(id):
    files = glob.glob('{}/pssm*revised.csv'.format(germline_data_path))
    print(files)
    for filename in files:
        if filename.find(id) != -1:
            return filename
    return None

def is_enrichment_data_available(germline_id):
    id = germline_id.split('*')[0][4:]
    germline_file = get_germline_datafile(id)
    return (not (germline_file is None))

def get_enrichment_for_germline(germline_id):
    id = germline_id.split('*')[0][4:]
    germline_file = get_germline_datafile(id)
    if germline_file is None:
        return germline_file
    data = pd.read_csv(germline_file)
    imgt_positions = [int(t) for t in list(data.columns) if t.isnumeric()]
    aa_order = data['Unnamed: 0'].tolist()
    data.drop(columns=['Unnamed: 0'], inplace=True)
    enrichment_array_data = data.replace('#WT', '0').astype('float').to_numpy()
    aa_order_base = list(_aa_dict.keys())
    indices_rearrange = [aa_order.index(t) for t in aa_order_base]
    enrichment_array = enrichment_array_data[indices_rearrange, :]
    assert enrichment_array.shape[1] == len(imgt_positions)
    return enrichment_array, imgt_positions

def get_enrichment_from_germline_at_matching_imgt_positions(germline_id, query_imgt_pos):
    result = get_enrichment_for_germline(germline_id)
    if result is None:
        return None
    else:
        enrichment_array_data, data_imgt_pos = result
    query_imgt_pos_clean = [int(t.replace('H', '')) for t in query_imgt_pos]
    matching_indices_data =  [i for i, t in enumerate(data_imgt_pos) if t in query_imgt_pos_clean]
    matching_query_pos = [t for i,t in enumerate(query_imgt_pos) if query_imgt_pos_clean[i] in data_imgt_pos]
    #print(matching_query_pos)   
    #print(matching_indices_data)
    if len(matching_indices_data) > 0:
        enrichment_array_query = enrichment_array_data[:, matching_indices_data]
        #print(enrichment_array_query.shape)
    return enrichment_array_query, matching_query_pos


def get_fr_score_for_seq(seq, enrichment_array):
    des_array = torch.nn.functional.one_hot(torch.tensor(letter_to_num(seq, _aa_dict)),
                                            num_classes=20).numpy().transpose()
    return np.sum(des_array * enrichment_array, axis=0)

def calculate_fr_score(seq, germline_id, design_imgt_pos):
    result = \
        get_enrichment_from_germline_at_matching_imgt_positions(germline_id,
                                                                design_imgt_pos)
    if result is None:
        return None
    else:
        enrichment_array, matching_query_positions = result
    matching_pos_indices = [design_imgt_pos.index(t) for t in matching_query_positions]
    
    matching_seq = ''.join([seq[i] for i in matching_pos_indices])
    fr_score_per_pos = get_fr_score_for_seq(matching_seq, enrichment_array)
    fr_score = np.sum(fr_score_per_pos)
    return fr_score_per_pos, fr_score

def calculate_fr_scores_for_designs(germline_id, designs, design_imgt_pos,
                                    outdir='./',
                                    suffix='',
                                    wt_seq='',
                                    design_ids=[]):
    result = \
        get_enrichment_from_germline_at_matching_imgt_positions(germline_id,
                                                                design_imgt_pos)
    if result is None:
        return None
    else:
        enrichment_array, matching_query_positions = result
    
    matching_pos_indices = [design_imgt_pos.index(t) for t in matching_query_positions]
    design_seq_matching = []
    for des in designs:
        design_seq_matching.append(''.join([des[i] for i in matching_pos_indices]))

    fr_scores_per_pos = np.zeros((len(design_seq_matching), enrichment_array.shape[1]))
    for i, des in enumerate(design_seq_matching):
        fr_scores_per_pos[i, :] = get_fr_score_for_seq(des, enrichment_array)

    fr_scores_per_design = np.sum(fr_scores_per_pos, axis=1)

    if wt_seq != '':
        wt_matching_seq = ''.join([wt_seq[i] for i in matching_pos_indices])
        wt_fr_score_per_pos = get_fr_score_for_seq(wt_matching_seq, enrichment_array)
        wt_fr_score = np.sum(wt_fr_score_per_pos)
        #print('wt ', wt_fr_score)
    

    germline_id_clean = germline_id.split('*')[0]
    
    outfile_pattern = '{}/{{}}_{}{}.png'.format(outdir, germline_id_clean, suffix)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib
    theme = {'axes.grid': True,
            'grid.linestyle': '',
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            "font.weight": 'regular',
            'xtick.color': 'black',
            'ytick.color': 'black',
            "axes.titlesize": 20,
            "axes.labelsize": 18
        }
    
    matplotlib.rcParams.update(theme)
    
    outfile = outfile_pattern.format('FRScore_dist_per_design-pos')
    plt.figure(dpi=300)
    df_dict = {'FR Score': list(fr_scores_per_pos.flatten()) 
                            + list(wt_fr_score_per_pos.flatten())
                }
    ax = sns.histplot(data=df_dict,
                      x='FR Score',
                      color="royalblue",
                      legend=False,
                      stat='probability',
                      binwidth=1
                      )
    ax.set_xlabel('FR Score per mutation')
    ax.set_ylabel('Probability(FR Score)')
    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()

    outfile = outfile_pattern.format('FRScore_dist_per_design-pos_withwt')
    plt.figure(dpi=300)
    #print(fr_scores_per_pos.shape)
    positions_times_designs = []
    design_ids_times_positions = []
    for i in range(fr_scores_per_pos.shape[0]):
        positions_times_designs += matching_query_positions
        design_ids_times_positions += [design_ids[i] for _ in range(len(matching_query_positions))]
    #print(len(positions_times_designs), fr_scores_per_pos.flatten().shape[0])
    
    df_dict = {'FR Score': list(fr_scores_per_pos.flatten()) 
                            + list(wt_fr_score_per_pos.flatten()),
                'Sequence': ['Design' for _ in range(fr_scores_per_pos.flatten().shape[0])] +
                            ['Wt' for _ in range(wt_fr_score_per_pos.flatten().shape[0])],
                'Position': positions_times_designs + matching_query_positions,
                'design_id': design_ids_times_positions + \
                    [-1 for _ in range(wt_fr_score_per_pos.flatten().shape[0])] 
                }
    df_tmp = pd.DataFrame.from_dict(df_dict)
    outfile_df_csv = '{}/FRScore_per_design-pos_withwt_{}{}.csv'.format(outdir, germline_id_clean, suffix)
    df_tmp.to_csv(outfile_df_csv)
    ax = sns.histplot(data=df_dict,
                      x='FR Score',
                      hue='Sequence',
                      color="royalblue",
                      #legend=False,
                      stat='density',
                      common_norm=False,
                      multiple='dodge'#,
                      #binwidth=1
                      )
    ax.set_xlabel('FR Score per mutation')
    ax.set_ylabel('Density(FR Score)')
    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()

    outfile = outfile_pattern.format('FRScore_dist_per_design')
    outfile_array = '{}/FRScore_per_design_{}{}.npy'.format(outdir, germline_id_clean, suffix)
    np.save(outfile_array, fr_scores_per_design.flatten(), allow_pickle=False)

    outfile_array = '{}/FRScore_wt_{}{}.npy'.format(outdir, germline_id_clean, suffix)
    np.save(outfile_array, wt_fr_score, allow_pickle=False)
    if design_ids != []:
        df_fr_per_design = pd.DataFrame(data=fr_scores_per_design.flatten(),
                                        columns=['FR Score'])
        df_fr_per_design['design_id'] = [t.split('_')[1] for t in design_ids]
        outfile_csv = '{}/FRScore_per_design_{}{}.csv'.format(outdir, germline_id_clean, suffix)
        df_fr_per_design.to_csv(outfile_csv)

    plt.figure(dpi=300)
    theme = {'axes.grid': True,
            'grid.linestyle': '',
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            "font.weight": 'regular',
            'xtick.color': 'black',
            'ytick.color': 'black',
            "axes.titlesize": 20,
            "axes.labelsize": 18
        }
    
    matplotlib.rcParams.update(theme)
    fig = plt.figure(figsize=(5,4))
    ax = sns.histplot(fr_scores_per_design.flatten(),
                 color="royalblue",
                 legend=False,
                 stat='count',
                 binwidth=1)
    ax.axvline(wt_fr_score, color='black', ls='--', lw=2)
    ax = plt.gca()
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
    ax.set_xlabel('FR Score per design')
    ax.set_ylabel('count(FR Score)')
    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()

    outfile = outfile_pattern.format('FRScore_violinplot_per_design')
    plt.figure(dpi=300)
    plt.violinplot(fr_scores_per_design.flatten(), showmeans=True,
                   showmedians=True, vert=False)
    #plt.xlabel('FR Score per design')
    #plt.ylabel('Probability(FR)')
    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()


    outfile = outfile_pattern.format('FRScore_violinplot_per_design-pos')
    plt.figure(dpi=300)
    plt.violinplot(fr_scores_per_pos.flatten(), showmeans=True,
                    showmedians=True, vert=False)
    #plt.ylabel('FR Score per design')
    #plt.xlabel('Probability(FR)')
    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()

    return enrichment_array


def get_fr_scores_at_interface_for_pdb(target_pdb, fasta_file=None):

    indices = get_indices_for_interface_residues(target_pdb)

    if not fasta_file is None:
        heavy_seq = get_fasta_chain_seq(fasta_file, 'H')
        light_seq = get_fasta_chain_seq(fasta_file, 'L')
    else:
        heavy_seq = get_pdb_chain_seq(target_pdb, 'H')
        light_seq = get_pdb_chain_seq(target_pdb, 'L')
    seq = heavy_seq+light_seq
    num_imgt, germline_h_gene = get_imgt_and_germline(heavy_seq, light_seq)
    print(germline_h_gene)
    if not num_imgt == []:
        des_imgt_positions = [num_imgt[i] for i in indices]
        des_imgt_positions_h = [t for t in des_imgt_positions if t[0]=='H']
        print(des_imgt_positions)
        print(des_imgt_positions_h)
        fr_score_pos, fr_score = calculate_fr_score(seq, germline_h_gene, des_imgt_positions_h)
        print('Score ', fr_score)
        print('Score pos', fr_score_pos)
        outfile = target_pdb.split('.pdb')[0]+'_hl_interface_FRscore.txt'
        open(outfile, 'w').write('per_position{}\ntotal{}\n'.format(fr_score_pos, fr_score))


def get_germline_information(pdb_path):
    pdb_files = glob.glob(pdb_path + '/*.pdb')
    germline_h_genes = []
    enrichment_ids = []
    for pdbf in pdb_files:
        pdb_id = os.path.basename(pdbf)[:4]
        h_seq = get_pdb_chain_seq(pdbf, 'H')
        l_seq = get_pdb_chain_seq(pdbf, 'L')
        num_imgt, germline_h_gene = get_imgt_and_germline(h_seq, l_seq)
        #except:
        #    germline_h_gene = ''
        germline_h_genes.append(germline_h_gene)
        if is_enrichment_data_available(germline_h_gene):
            enrichment_ids.append(pdb_id)
    outstr = ['{}\t{}'.format(os.path.basename(p)[:4], g) for p, g in zip(pdb_files, germline_h_genes)]
    with open('{}/h_germline.txt'.format(pdb_path), 'w') as f:
        f.write('\n'.join(outstr)+'\n')
    
    with open('{}/pdbids_enrichment_data.txt'.format(pdb_path), 'w') as f:
        f.write('\n'.join(enrichment_ids)+'\n')
        
