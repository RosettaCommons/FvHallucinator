import os
import sys
import glob
import matplotlib
from hallucination.utils.trajectoryreader import HallucinationDataReader
matplotlib.use('Agg')
import numpy as np
import argparse
import glob
import json

from util.pdb import get_pdb_chain_seq,\
    get_pdb_numbering_from_residue_indices, get_cluster_for_cdrs
from hallucination.utils.sequence_utils import *
from hallucination.utils.util import get_indices_from_different_methods, \
                                      comma_separated_chain_indices_to_dict
from hallucination.utils.compare_to_PyIgClassify_clusters import \
    bhattacharyya_distance, read_PyIgClassify_database
from hallucination.utils.h_germline_enrichment_utils import \
    calculate_fr_scores_for_designs, get_imgt_and_germline

def calculate_bhattacharya_distance(cdr_sequences,
                                    db_path, 
                                    target_pdb,
                                    cdr_name,
                                    outdir='./',
                                    target_cdr_cluster=''):
    pssm_dict = read_PyIgClassify_database(db_path)
    if target_cdr_cluster == '':
        cdr_clusters = get_cluster_for_cdrs(target_pdb)
        pdb_cdr_cluster = cdr_clusters[cdr_name.rstrip().lower()].rstrip()
    else:
        pdb_cdr_cluster = target_cdr_cluster
    
    bhattacharyya_distance_dict, sorted_total_dist_dict = bhattacharyya_distance(cdr_sequences, cdr_name, pssm_dict, outdir, pdb_cdr_cluster)
    return pdb_cdr_cluster, bhattacharyya_distance_dict, sorted_total_dist_dict, pssm_dict[cdr_name.upper()][pdb_cdr_cluster]


def process_hallucination_output(target_pdb,
                                 hal_path,
                                 out_path='./',
                                 outfile_indices='sequences_selected.fasta',
                                 cdr_name='',
                                 framework = False,
                                 hl_interface=False,
                                 include_indices={},
                                 exclude_indices={},
                                 make_plots=True,
                                 make_trajectory_movie=False,
                                 db_path=''):

    print('target_pdb: ', target_pdb)
    trajfiles_path = os.path.join(hal_path, 'trajectories')
    
    # WT sequence
    wt_heavy_seq, wt_light_seq = get_pdb_chain_seq(target_pdb,
                                                   'H'), get_pdb_chain_seq(
                                                       target_pdb, 'L')
    wt_seq = wt_heavy_seq + wt_light_seq
    print("Wildtype sequence: ", wt_seq)
    indices_hal = get_indices_from_different_methods(
        target_pdb,
        cdr_list=cdr_name,
        framework=framework,
        hl_interface=hl_interface,
        include_indices=include_indices,
        exclude_indices=exclude_indices)
    print("Indices hallucinated: ", indices_hal)

    dict_residues = {
        "reslist":
        indices_hal,
        "labellist":
        get_pdb_numbering_from_residue_indices(target_pdb, indices_hal)
    }
    print("Indices <-> Chothia: ", dict_residues)
    
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    
    # TODO: NEW CODE; to be tested
    data_reader = HallucinationDataReader(indices_hal, target_pdb, trajfiles_path)
    outfile = os.path.join(out_path, 'sequences.fasta')
    data_reader.write_final_sequences_to_fasta(outfile)
    outfile_indices = os.path.join(out_path, 'sequences_indices.fasta')
    data_reader.write_final_subsequences_to_fasta(outfile_indices)

    all_sequences = data_reader.list_of_final_des_subsequences
    all_sequences_full = data_reader.list_of_final_sequences
    all_trajectories = data_reader.dict_of_all_des_subsequence_dicts
    # select 3 trajectories for visualization
    random_selection = np.random.choice(list(all_trajectories.keys()), max(3, len(all_trajectories.keys())))
    seq_slice_lists_for_visualization = [all_trajectories[rs] for rs in random_selection]

    print("# Total number of sequences sampled: {}".format(len(all_sequences)))
    unique_sequences = list(set(all_sequences))
    print('# of Unique sequences sampled: ', len(unique_sequences))
    if len(unique_sequences) < 1:
       print('No sequences found in trajectory')
       sys.exit()

    if hl_interface:
        num_imgt, germline_h_gene = get_imgt_and_germline(wt_heavy_seq, wt_light_seq)
        if not num_imgt == []:
            print(len(num_imgt))
            print(indices_hal)
            des_imgt_positions = [num_imgt[i] for i in indices_hal if i < len(num_imgt)]
            des_imgt_positions_h = [t for t in des_imgt_positions if t[0]=='H']
            seq_indices_ref = ''.join([wt_seq[t] for t in indices_hal])
            design_ids = [key for key in data_reader.dict_of_final_des_subsequences]
            calculate_fr_scores_for_designs(germline_h_gene,
                                          all_sequences,
                                          des_imgt_positions_h,
                                          outdir=out_path,
                                          wt_seq=seq_indices_ref,
                                          design_ids=design_ids)
        
    if not make_plots:
        return unique_sequences, dict_residues

    #calculate bhattacharya dist from cluster
    pssm_target = None
    if cdr_name != '' and db_path != '':
        outdir = os.path.join(out_path, 'clusters_final')
        os.makedirs(outdir, exist_ok=True)
        cdr_cluster, distribution_distance_dict, total_distance_dict, pssm_target = \
             calculate_bhattacharya_distance(all_sequences, db_path, target_pdb, cdr_name, outdir)
        if cdr_cluster in total_distance_dict:
            bd_log = dict(cdr_name=cdr_name,
                            canonical_cluster=cdr_cluster,
                            total_bd_cluster_final=total_distance_dict[cdr_cluster],
                            min_bd_cluster_final=min(total_distance_dict.values())
                            )
        else:
            bd_log = dict(cdr_name=cdr_name, canonical_cluster=cdr_cluster)
        # write json file with details
        outfile_json_bd = os.path.join(out_path, 'pygclassify_clusters.json')
        bd_log.update(dict(bd_dict_final=distribution_distance_dict))
        open(outfile_json_bd, 'w').write(json.dumps(bd_log))

    # get positional entropy
    pe = calculate_positional_entropy(all_sequences)
    pe_dict = {'entropy_final':list(pe)}
    outfile_json_pe = os.path.join(out_path, 'positional_entropy.json')
    open(outfile_json_pe, 'w').write(json.dumps(pe_dict))

    #developability
    write_and_plot_biopython_developability(all_sequences_full,
                                            len(wt_heavy_seq),
                                            indices_hal,
                                            wt_seq=wt_seq,
                                            out_path=out_path)

    # sliced logos
    if len(indices_hal) > 0:
        seq_indices_ref = ''.join([wt_seq[t] for t in indices_hal])
        outfile_logo = os.path.join(out_path, 'logo.png')
        outfile_logo_ref = os.path.join(out_path, 'logo_ref.png')

        sequences_to_logo_without_weblogo(
            all_sequences,
            dict_residues,
            outfile_logo=outfile_logo,
            ref_seq=seq_indices_ref,
            outfile_logo_ref=outfile_logo_ref)
        
        if cdr_name != '' and not (pssm_target is None):
            outfile_logo = os.path.join(out_path, 'logo_with_ref.png')
            sequences_to_logo_with_ref(all_sequences,
                                    seq_indices_ref,
                                    np.array(pssm_target),
                                    dict_residues=dict_residues,
                                    outfile_logo=outfile_logo)

    #Trajectory visualization
    if make_trajectory_movie:
        out_traj_path = os.path.join(out_path, 'trajs')
        os.makedirs(out_traj_path, exist_ok=True)
        outfile_logo_movie = os.path.join(out_traj_path,'traj_logo_{}.gif')
        sequence_list_to_logo_movie(seq_slice_lists_for_visualization,
                                    dict_residues,
                                    outfile_logo=outfile_logo_movie)

    return None


def _get_args():
    """Gets command line arguments"""
    desc = ('''
        Plot sequence logos for designed residues from hallucinated trajectories.
        ''')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--target_pdb',
                        type=str,
                        default='',
                        help='path to target structure')
    parser.add_argument('--trajectory_path',
                        type=str,
                        default='',
                        help='path to sequences dir')
    parser.add_argument('--outdir',
                        type=str,
                        default='results/',
                        help='path to sequences dir')
    parser.add_argument('--cdr',
                        type=str,
                        default='',
                        help='comma separated list of cdrs')
    parser.add_argument('--framework',
                        action='store_true',
                        default=False,
                        help='design framework residues. Default: false')
    parser.add_argument('--indices',
                        type=str,
                        default='',
                        help='comma separated list: h:12,20,31A/l:56,57')
    parser.add_argument('--exclude',
                        type=str,
                        default='',
                        help='exclude indices: h:31A,52,53/l:97,99')
    parser.add_argument('--hl_interface',
                        action='store_true',
                        default=False,
                        help='hallucinate hl interface')
    parser.add_argument('--disable_postprocess_plots',
                        action='store_true',
                        default=False,
                        help='just get designed sequences, do not make logos')
    parser.add_argument(
        '--make_trajectory_movie',
        action='store_true',
        default=False,
        help='Make logo gif movie for a hallucination trajectory.')
    parser.add_argument('--cdr_cluster_database',
                        type=str,
                        default='',
                        help='Current database downloaded from PyIgClassify website September 2021 or pickled file.' )
    
    return parser.parse_args()


def _cli():
    args = _get_args()
    dict_indices = {}
    dict_exclude = {}
    if args.indices != '':
        indices_str = args.indices
        dict_indices = comma_separated_chain_indices_to_dict(indices_str)
    if args.exclude != '':
        indices_str = args.exclude
        dict_exclude = comma_separated_chain_indices_to_dict(indices_str)

    output = process_hallucination_output(args.target_pdb,
                                          args.trajectory_path,
                                          out_path=args.outdir,
                                          cdr_name=args.cdr,
                                          framework=args.framework,
                                          hl_interface=args.hl_interface,
                                          include_indices=dict_indices,
                                          exclude_indices=dict_exclude,
                                          make_trajectory_movie=args.make_trajectory_movie,
                                          db_path=args.cdr_cluster_database,
                                          make_plots=(not args.disable_postprocess_plots)
                                          )
    return output
    

if __name__ == '__main__':
    _cli()
