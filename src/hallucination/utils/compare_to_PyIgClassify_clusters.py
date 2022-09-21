import pandas as pd
import os, sys
import pickle, argparse
import itertools
import numpy as np
from numpy import inf 
import seaborn as sns
import math

from src.util.pdb import get_cluster_for_cdrs
sns.set_theme(context='notebook', font='sans-serif')
sns.set_style("whitegrid", {'axes.grid' : False})

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from src.hallucination.utils.sequence_utils import sequences_to_probabilities, sequences_to_logo_without_weblogo
from src.hallucination.utils.trajectoryreader import HallucinationDataReader
from src.hallucination.utils.util import get_indices_from_different_methods

# This database was downloaded from PyIgClassify (current version is from September 2021)

cdr_list = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']


def trim_pyig_classify_cdr_to_chothia(sequence, cdr):
    cdr = cdr.lower()
    out_seq = ''
    if cdr == 'h1':
        out_seq = sequence[3:-3]
    elif cdr == 'h2':
        out_seq = sequence[2:-2]    
    elif cdr == 'h3':
        out_seq = sequence[2:]
    elif cdr == 'l2':
        out_seq = sequence[1:]
    else:
        out_seq = sequence
    return out_seq

def read_PyIgClassify_database(path_to_db_or_pickle, out_path="."):

    try:
        pssm_dict = pickle.load( open(path_to_db_or_pickle, 'rb' ))
        return pssm_dict
    except pickle.UnpicklingError:
        assert os.path.isfile(path_to_db_or_pickle)
        df = pd.DataFrame()

        try:
            df = pd.read_csv(path_to_db_or_pickle, delimiter='\t',skiprows=1)
        except:
            sys.exit("Cannot read the database file. Make sure it is a text file containing PyIgClassify cdr_data as downloaded from their website")
        sequence_dict = {}
        for cdr in cdr_list:
            sequence_dict[cdr] = {}
            exisiting_clusters_for_loop = set(df[df['CDR']==cdr]['fullcluster'])
            for cluster in exisiting_clusters_for_loop:
                sequences = list(df[df['CDR']==cdr][df['fullcluster']==cluster]['seq'])
                sequences = [trim_pyig_classify_cdr_to_chothia(s, cdr) for s in sequences]
                sequences = [s for s in sequences if not 'X' in s]
                sequence_dict[cdr][cluster] = sequences
        
        os.makedirs(os.path.join(out_path, 'logos'), exist_ok=True)
        pssm_dict = {}
        for cdr in cdr_list:
            pssm_dict[cdr] = {}
            for cluster in sequence_dict[cdr].keys():
                seq_list = sequence_dict[cdr][cluster]
                length = len(seq_list[0])
                number_of_seq = len(seq_list)
                residues = list(range(1, length + 1))
                residue_dict = {
                    'labellist' : residues,
                    'reslist': residues
                }
                logo_png = os.path.join(out_path, 'logos/{}.png'.format(cluster))

                pssm_dict[cdr][cluster] = sequences_to_probabilities(seq_list)
                if not os.path.isfile(logo_png):
                    sequences_to_logo_without_weblogo(seq_list, residue_dict, logo_png, text="Number of sequences in cluster: {}".format(number_of_seq) )

        pickle.dump( pssm_dict, open(os.path.join(out_path, "cdr_clusters_pssm_dict.pkl"), "wb" ) )
        return pssm_dict


def removed_starred_clusters(pyig_pssm_dict, cdr_name):
    remove_list = [t for t in pyig_pssm_dict[cdr_name] if t.find('*')!=-1]
    for key in remove_list:
        del pyig_pssm_dict[cdr_name][key]
    return pyig_pssm_dict

    
def bhattacharyya_distance(list_of_designed_sequences, cdr_loop, pyig_pssm_dict, outdir, target_cdr_cluster='', set_inf_to=5,
                            positions=[]):
    cdr_loop = cdr_loop.upper()
    query_pssm = sequences_to_probabilities(list_of_designed_sequences)
    loop_length = len(list_of_designed_sequences[0])
    if positions != []:
        assert len(positions) == loop_length
    else:
        positions = [i+1 for i in range(loop_length)]
    
    #make sure all sequences have the same length 
    assert [len(seq) for seq in list_of_designed_sequences] == len(list_of_designed_sequences) * [loop_length]
    # remove starred clusters
    pyig_pssm_dict = removed_starred_clusters(pyig_pssm_dict, cdr_loop)

    cluster_pssms = []
    for cluster in pyig_pssm_dict[cdr_loop].keys():
        if pyig_pssm_dict[cdr_loop][cluster].shape[0] == len(list_of_designed_sequences[0]):
            cluster_pssm = pyig_pssm_dict[cdr_loop][cluster]         
            assert query_pssm.shape == cluster_pssm.shape
            cluster_pssms.append((cluster, cluster_pssm))
            
    bhattacharyya_dict = {}
    for test_cluster in cluster_pssms:
        cluster_pssm = test_cluster[1]
        cluster_name = test_cluster[0]


        bhattacharyya_dict[cluster_name] = calc_bhattacharyya_dist(query_pssm, cluster_pssm)
    outpng = os.path.join(outdir, "bhattacharyya_distance_{cdr}.png".format(cdr=cdr_loop))
    sorted_dist_dist_dict = {}
    try:
        sorted_dist_dist_dict = plot_probability_distance_from_clusters(bhattacharyya_dict, outpng=outpng, set_inf_to=set_inf_to)
    except:
        print('Falied to plot probabilities')

    if target_cdr_cluster != '':
        print('Target cluster specified as: ', target_cdr_cluster)
        cluster_names = [cp[0].lower() for cp in cluster_pssms]
        if target_cdr_cluster.rstrip().lower() in cluster_names:
            i_cluster = cluster_names.index(target_cdr_cluster.rstrip().lower())
            target_cluster_pssm = cluster_pssms[i_cluster]
            target_bhat_dist = bhattacharyya_dict[target_cluster_pssm[0]]
            plt.plot([i for i in range(len(target_bhat_dist))], target_bhat_dist, '-o')
            plt.xticks([i for i in range(len(target_bhat_dist))], positions)
            plt.xlabel('AA position in loop')
            plt.ylabel('Bhattacharyya distance')
            plt.savefig('{}/bhattacharyya_dist_to_target_cluster.png'.format(outdir))
            plt.close()
            remaining_cluster_indices = [t for t in range(len(cluster_names)) if t !=i_cluster ]
            bhat_dist_rem_clusters = {}
            bd_array = np.zeros((len(remaining_cluster_indices), target_cluster_pssm[1].shape[0]))
            print(len(remaining_cluster_indices))
            if len(remaining_cluster_indices) > 0:
                for i, i_rem_cl in enumerate(remaining_cluster_indices):
                    cur_cluster_pssm = cluster_pssms[i_rem_cl]
                    bhat_dist_rem_clusters[cur_cluster_pssm[0]] = calc_bhattacharyya_dist(cur_cluster_pssm[1], target_cluster_pssm[1])
                    bd_array[i, :] = bhat_dist_rem_clusters[cur_cluster_pssm[0]]
                outpng = os.path.join(outdir, "bhattacharyya_distance_all_to_{}.png".format(target_cluster_pssm[0]))    
                try:
                    plot_probability_distance_from_clusters(bhat_dist_rem_clusters, outpng=outpng)
                except:
                    print('Failed to plot probabilities')
                
                bd_array[bd_array==np.Infinity] = set_inf_to
                plt.plot([i+1 for i in range(len(target_bhat_dist))], target_bhat_dist, '-o', color='black')
                plt.violinplot(bd_array, showmeans=True, showmedians=True)
                plt.xticks([i+1 for i in range(len(target_bhat_dist))], positions)
                plt.xlabel('AA position in loop')
                plt.ylabel('Bhattacharyya distance')
                plt.tight_layout()
                plt.savefig('{}/bhattacharya_dist_to_target_cluster_distalltotarget.png'.format(outdir))
                plt.close()
                
                bd_array_to_all = np.array(list(bhattacharyya_dict.values()))
                bd_array_to_all[bd_array_to_all == np.Infinity] = set_inf_to
                plt.plot([i+1 for i in range(len(target_bhat_dist))], target_bhat_dist, '-o', color='black')
                plt.violinplot(bd_array_to_all, showmeans=True, showmedians=True)
                plt.xticks([i+1 for i in range(len(target_bhat_dist))], positions)
                plt.xlabel('AA position in loop')
                plt.ylabel('Bhattacharyya distance')
                plt.tight_layout()
                plt.savefig('{}/bhattacharya_dist_to_target_cluster_disttoall.png'.format(outdir))
                plt.close()
                
    return bhattacharyya_dict, sorted_dist_dist_dict

def get_same_length_clusters(pyig_pssm_dict_all, cdr_loop, loop_length):
    pyig_pssm_dict = removed_starred_clusters(pyig_pssm_dict_all, cdr_loop)
    cluster_pssms = []
    for cluster in pyig_pssm_dict[cdr_loop].keys():
        if pyig_pssm_dict[cdr_loop][cluster].shape[0] == loop_length:
            cluster_pssm = pyig_pssm_dict[cdr_loop][cluster]         
            cluster_pssms.append((cluster, cluster_pssm))
    return cluster_pssms


def filter_same_length_clusters(pyig_pssm_dict_all, cdr_loop, loop_length):
    pyig_pssm_dict = removed_starred_clusters(pyig_pssm_dict_all, cdr_loop)
    pssm_dict_len = {}
    for cluster in pyig_pssm_dict[cdr_loop].keys():
        if pyig_pssm_dict[cdr_loop][cluster].shape[0] == loop_length:
            pssm_dict_len[cluster] = pyig_pssm_dict[cdr_loop][cluster]

    return pssm_dict_len

def bhattacharyya_coefficient(list_of_designed_sequences, cdr_loop, pyig_pssm_dict, outdir,
                              target_cdr_cluster='', set_inf_to=5,
                              positions=[]):
    cdr_loop = cdr_loop.upper()
    query_pssm = sequences_to_probabilities(list_of_designed_sequences)
    loop_length = len(list_of_designed_sequences[0])
    
    if positions != []:
        assert len(positions) == loop_length
    else:
        positions = [i+1 for i in range(loop_length)]
    

    #make sure all sequences have the same length 
    assert [len(seq) for seq in list_of_designed_sequences] == len(list_of_designed_sequences) * [loop_length]
    pyig_pssm_dict = removed_starred_clusters(pyig_pssm_dict, cdr_loop)

    cluster_pssms = get_same_length_clusters(pyig_pssm_dict,
                                             cdr_loop,
                                             len(list_of_designed_sequences[0])
                                             )
            
    bhattacharyya_coef_dict = {}
    for test_cluster in cluster_pssms:
        cluster_pssm = test_cluster[1]
        cluster_name = test_cluster[0]
        bhattacharyya_coef_dict[cluster_name] = \
            calc_bhattacharyya_dist(query_pssm, cluster_pssm, only_coefficient=True)
    #outpng = os.path.join(outdir, "bhattacharyya_distance_{cdr}.png".format(cdr=cdr_loop))
    #sorted_dist_dist_dict = plot_probability_distance_from_clusters(bhattacharyya_dict, outpng=outpng, set_inf_to=set_inf_to)

    if target_cdr_cluster != '':
        #print('Target cluster specified as: ', target_cdr_cluster)
        cluster_names = [cp[0].lower() for cp in cluster_pssms]
        if target_cdr_cluster.rstrip().lower() in cluster_names:
            i_cluster = cluster_names.index(target_cdr_cluster.rstrip().lower())
            target_cluster_pssm = cluster_pssms[i_cluster]
            target_bhat_coef = bhattacharyya_coef_dict[target_cluster_pssm[0]]
            #plt.plot([i for i in range(len(target_bhat_coef))], target_bhat_coef, '-o')
            #plt.xticks([i+1 for i in range(len(target_bhat_coef))], positions)
            #plt.xlabel('AA position in loop')
            #plt.ylabel('Bhattacharyya Coefficient')
            #plt.savefig('{}/bhattacharyya_coef_to_target_cluster.png'.format(outdir))
            #plt.close()
            remaining_cluster_indices = [t for t in range(len(cluster_names)) if t !=i_cluster ]
            bhat_coef_rem_clusters = {}
            bd_array = np.zeros((len(remaining_cluster_indices), target_cluster_pssm[1].shape[0]))
            for i, i_rem_cl in enumerate(remaining_cluster_indices):
                cur_cluster_pssm = cluster_pssms[i_rem_cl]

                bhat_coef_rem_clusters[cur_cluster_pssm[0]] = \
                    calc_bhattacharyya_dist(cur_cluster_pssm[1], target_cluster_pssm[1], only_coefficient=True)
                bd_array[i, :] = bhat_coef_rem_clusters[cur_cluster_pssm[0]]
            #outpng = os.path.join(outdir, "bhattacharyya_coef_all_to_{}.png".format(target_cluster_pssm[0]))    
            #plot_probability_distance_from_clusters(bhat_coef_rem_clusters, outpng=outpng)
            
            #plt.plot([i+1 for i in range(len(target_bhat_coef))], target_bhat_coef, '-o', color='black')
            #plt.violinplot(bd_array, showmeans=True, showmedians=True)
            #plt.xticks(positions)
            #plt.xlabel('AA position in loop')
            #plt.ylabel('Bhattacharyya Coefficient')
            #plt.tight_layout()
            #plt.savefig('{}/bhattacharya_coef_to_target_cluster_coefalltotarget.png'.format(outdir))
            #plt.close()
            
            bd_array_to_all = np.array(list(bhattacharyya_coef_dict.values()))
            bd_array_to_all[bd_array_to_all == np.Infinity] = set_inf_to
            plt.plot([i+1 for i in range(len(target_bhat_coef))], target_bhat_coef, '-o', color='black')
            plt.violinplot(bd_array_to_all, showmeans=True, showmedians=True)
            plt.xticks([i+1 for i in range(len(target_bhat_coef))], positions, rotation=45)
            plt.xlabel('AA position in loop')
            plt.ylabel('Bhattacharyya Coefficient')
            plt.tight_layout()
            plt.savefig('{}/bhattacharya_coef_to_target_cluster_disttoall.png'.format(outdir))
            plt.close()
            
    return bhattacharyya_coef_dict  #, sorted_dist_dist_dict


def calc_bhattacharyya_dist(pssm1, pssm2, only_coefficient=False):
    """ Calculate for each column. Columns are positions. Rows are the 20 aa. """
    
    assert pssm1.shape == pssm2.shape
    pssm1 = pd.DataFrame(pssm1.transpose())
    pssm2 = pd.DataFrame(pssm2.transpose())
    
    all_b_distances = []
    for i in pssm1.columns:
        dist1 = pssm1[i]
        dist2 = pssm2[i]
        b_coefficient = 0
        assert len(dist1) == len(dist2)
        for i in range(len(dist1)):
            b_coefficient += np.sqrt(dist1[i] * dist2[i])
        if not only_coefficient:
            b_distance = -np.log(b_coefficient)
        else:
            b_distance = b_coefficient
        all_b_distances.append(b_distance)
    return all_b_distances


def plot_probability_distance_from_clusters(distance_dict, outpng = 'bhattacharyya.png', set_inf_to=5):
    
    df = pd.DataFrame.from_dict(distance_dict)
    
    #set inf values
    df.replace(inf, set_inf_to, inplace=True)
    df_color_dict = {}
    
    for n in df.columns:
        df_color_dict[n] = sum(df[n])

    df['position'] = range(1,df.shape[0]+1)
    df = df.set_index('position')

    totals_dict = {}
    for name in df.columns:
        total = sum(df[name])
        totals_dict[name] = total
    sorted_totals_dict = {k: v for k, v in sorted(totals_dict.items(), key=lambda item: item[1])}

    min_total = min(totals_dict.values())
    max_total = max(totals_dict.values())
        
    color_dict = {}
    cmap = matplotlib.cm.get_cmap('coolwarm')
    
    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    max_bd = 0.0
    for name in sorted_totals_dict.keys():
        rel_total = (totals_dict[name] - min_total)/(max_total - min_total)
        color_dict[name] = cmap(rel_total)
        ax = sns.lineplot(data=df[name], color=color_dict[name])
        max_new_bd = max(list(df[name]))
        if max_new_bd > max_bd and (max_new_bd != np.Infinity):
            max_bd = max_new_bd
    lgd = fig.legend(sorted_totals_dict.keys(), bbox_to_anchor=(1.35, 1))

    plt.xlabel('Amino acid position in loop')
    plt.ylabel('distance')
    plt.ylim(0, min(math.ceil(max_bd + 0.1), 5.5))
    plt.tight_layout()
    
    fig.savefig(outpng, dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    #rainbow
    cmap = matplotlib.cm.get_cmap('rainbow')
    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    max_bd = 0.0
    for name in sorted_totals_dict.keys():
        rel_total = (totals_dict[name] - min_total)/(max_total - min_total)
        color_dict[name] = cmap(rel_total)
        ax = sns.lineplot(data=df[name], color=color_dict[name])
        max_new_bd = max(list(df[name]))
        if max_new_bd > max_bd and (max_new_bd != np.Infinity):
            max_bd = max_new_bd
    lgd = fig.legend(sorted_totals_dict.keys(), bbox_to_anchor=(1.35, 1))

    plt.xlabel('Amino acid position in loop')
    plt.ylabel('distance')
    plt.ylim(0, min(math.ceil(max_bd + 0.1), 5.5))
    plt.tight_layout()
    
    fig.savefig(outpng.split('.png')[0]+'_rainbow.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

    #top3 only
    cmap = matplotlib.cm.get_cmap('coolwarm')
    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    max_bd = 0.0
    top_3_dict = {k: sorted_totals_dict[k] for k in list(sorted_totals_dict.keys())[:3]}
    min_total = min(top_3_dict.values())
    max_total = max(top_3_dict.values())
    for name in top_3_dict:
        rel_total = (totals_dict[name] - min_total)/(max_total - min_total)
        color_dict[name] = cmap(rel_total)
        ax = sns.lineplot(data=df[name], color=color_dict[name])
        max_new_bd = max(list(df[name]))
        if max_new_bd > max_bd and (max_new_bd != np.Infinity):
            max_bd = max_new_bd
    lgd = fig.legend(top_3_dict, bbox_to_anchor=(1.35, 1))

    plt.xlabel('Amino acid position in loop')
    plt.ylabel('distance')
    plt.ylim(0,math.ceil(max_bd))
    plt.tight_layout()
    
    fig.savefig(outpng.split('.png')[0]+'_top3.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()
    
    return sorted_totals_dict


def get_sequencelist_for_indices(list_of_sequences, indices_hal):
    mask = [False] * len(list_of_sequences[0])
    for i in indices_hal:
        mask[i] = True
    list_of_cdr_sequences = ["".join(list(itertools.compress(s, mask))) for s in list_of_sequences]
    return list_of_cdr_sequences


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument( 'database', type=str, help='Current database downloaded from PyIgClassify website September 2021 or pickled file.' )
    parser.add_argument( 'target_pdb', type=str, help='path to target structure' )
    parser.add_argument( '--target_cdr_cluster', type=str, default="", help='option to specify native cdr cluster' )
    parser.add_argument( '--hallucination_dir', type=str, default="", help='path to hallucination dir containing final.fastas' )
    parser.add_argument('--outdir', type=str, default='.', help='Path to directory for output')
    
    parser.add_argument( '--cdr', type=str, default='h1', help='single cdr (e.g. "l1")' )
    parser.add_argument( '--indices', type=str, default='', help='comma separated list of chothia numbered residues to design: h:12,20,31A/l:56,57' )
    parser.add_argument( '--exclude', type=str, default='', help='comma separated list of chothia numbered residues to exclude from design: h:31A,52,53/l:97,99' )
    parser.add_argument( '--hl_interface', action='store_true', default=False, help='Not implemented! hallucinate hl interface')    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    os.makedirs(args.outdir, exist_ok=True)
    pssm_dict = read_PyIgClassify_database(args.database)
    indices_hal = get_indices_from_different_methods(args.target_pdb, args.cdr)
    list_of_final_sequences = HallucinationDataReader(indices_hal, args.target_pdb, 
                                hallucination_dir=args.hallucination_dir).list_of_final_sequences
    cdr_sequences = get_sequencelist_for_indices(list_of_final_sequences, indices_hal)
    if args.target_cdr_cluster == '':
        cdr_clusters = get_cluster_for_cdrs(args.target_pdb)
        pdb_cdr_cluster = cdr_clusters[args.cdr.rstrip().lower()]
    else:
        pdb_cdr_cluster = args.target_cdr_cluster
    bhattacharyya_distance(cdr_sequences, args.cdr, pssm_dict, args.outdir, pdb_cdr_cluster)