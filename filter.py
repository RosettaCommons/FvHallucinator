
from importlib_metadata import sys, warnings
import os, argparse, json, glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from dask.distributed import Client, get_client
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import pyrosetta.distributed.dask
from pyrosetta import *
from src.hallucination.utils.pyrosetta_utils \
    import fast_relax_pose_complex,\
    score_pdb, relax_pose, mutate_pose, align_to_complex,\
    get_sapscores
from src.deepab.models.AbResNet import load_model
from src.deepab.models.ModelEnsemble import ModelEnsemble
from src.deepab.build_fv.build_cen_fa \
  import build_initial_fv, get_cst_file, refine_fv
from src.deepab.build_fv.utils import migrate_seq_numbering, get_constraint_set_mover
from src.deepab.build_fv.score_functions import get_sf_fa
from src.util.pdb import get_pdb_numbering_from_residue_indices, renumber_pdb,\
    get_pdb_chain_seq


from src.hallucination.utils.util\
    import get_indices_from_different_methods,\
    comma_separated_chain_indices_to_dict
from src.hallucination.utils.interfacemetrics_plotting_utils \
    import iam_score_df_from_pdbs, scatter_hist,\
         select_best_designs_by_sum
from src.hallucination.utils.sequence_utils import sequences_to_logo_without_weblogo
from src.hallucination.utils.rmsd_plotting_utils import \
        threshold_by_rmsd_filters, write_fastas_for_alphafold2,\
             plt_ff_publication_for_run

init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"
pyrosetta.init(init_string)

def output_filtered_designs(csv_dg, csv_rmsd,
                            target_pdb,
                            indices_hal=[],
                            rmsd_filter='H3,1.8',
                            rmsd_filter_json='',
                            outdir='.', 
                            suffix='DeepAb'
                            ):
    os.makedirs(outdir, exist_ok=True)
    df_dg = pd.read_csv(csv_dg, delimiter=',')
    df_dg['design_id'] = \
        [int(os.path.basename(t).split('.pdb')[0].split('_')[-2]) 
            for t in list(df_dg['filename'])]
    df_ff = pd.read_csv(csv_rmsd)
    if rmsd_filter !='':
        x=rmsd_filter.split(',')[0]
        outfile_png = os.path.join(outdir, 'histrmsdff-{}.png'.format(suffix))                                          
        plt_ff_publication_for_run(csv_rmsd, x=x, outfile=outfile_png)
    outfile = os.path.join(outdir, 'df_ff-{}_thresholded_{{}}.csv'.format(suffix))
    df_ff_thr, rmsd_suffix = threshold_by_rmsd_filters(df_ff, rmsd_filter=rmsd_filter,
                                              rmsd_filter_json=rmsd_filter_json,
                                              outfile=outfile)
    df_dg_ff_thr = pd.merge(df_dg, df_ff_thr, on=['design_id'], suffixes=['', '_ff'])
    outfile = os.path.join(outdir, 'df_ff-{}_dg_thresholded_{}.csv'.format(suffix, rmsd_suffix))
    df_dg_ff_thr.to_csv(outfile)
    outfile_png = os.path.join(outdir, 'df_ff-{}_thresholded_{}.png'.format(suffix, rmsd_suffix))
    if rmsd_filter !='':
        x=rmsd_filter.split(',')[0]
        outfile_png = os.path.join(outdir, 'histrmsdff-{}_thresholded_{}.png'.format(suffix, rmsd_suffix))                                          
        plt_ff_publication_for_run(outfile.format(rmsd_suffix), x=x, outfile=outfile_png)
    
    sequences_thresholded = list(df_dg_ff_thr['seq'])
    print('{} sequences meet the thresholds.'.format(len(sequences_thresholded)))
    if len(sequences_thresholded) > 0:
        dict_residues = {'reslist': indices_hal}
        labellist = \
            get_pdb_numbering_from_residue_indices(target_pdb, indices_hal)
        dict_residues.update({'labellist': labellist})
        outfile_logo = \
            os.path.join(outdir, 'logo_ff-{}_dg_thresholded_rmsd{}.png'.format(suffix, rmsd_suffix))
        sequences_to_logo_without_weblogo(sequences_thresholded, dict_residues=dict_residues,
                                        outfile_logo=outfile_logo)
        # write inputs for running alphafold
        outdir_af2 = os.path.join(outdir, 'ff-{}_ddg_thresholded_rmsd{}'.format(suffix, rmsd_suffix))
        os.makedirs(outdir_af2, exist_ok=True)
        write_fastas_for_alphafold2(list(df_dg_ff_thr['filename']), outdir_af2)
        
        # interface metrics
        select_by = ['dG_separated']
        design_pdbs = list(set(list(df_dg_ff_thr['filename'])))
        df_iam_mutants = iam_score_df_from_pdbs(design_pdbs)
        print('iam: ',df_iam_mutants)
        
        df_iam_ref = iam_score_df_from_pdbs([target_pdb])
        n_all = min(50, len(design_pdbs))
        pdb_dir = os.path.join(outdir, 'interface_metrics_pdbs')
        os.makedirs(pdb_dir, exist_ok=True)
        best_decoys = select_best_designs_by_sum(df_iam_mutants, by=select_by,
                                                n=n_all, pdb_dir=pdb_dir,
                                                out_path=pdb_dir)
        
        selected_decoys_dir = os.path.join(outdir, 'selected_decoys_iam') 
        os.makedirs(selected_decoys_dir, exist_ok=True)
        outfile = os.path.join(selected_decoys_dir, "scatterplot_dgneg.png")
        df_iam_mutants_neg = df_iam_mutants[df_iam_mutants['dG_separated'] < 0.0]
        if 'dG_separated' in df_iam_ref.columns:
            scatter_hist(df_iam_mutants_neg, ref=df_iam_ref, out=outfile, highlight=best_decoys, by=select_by)
            out_csv_iam = os.path.join(outdir, 'df_ref_iam.csv'.format(suffix, rmsd_suffix))
            df_iam_ref.to_csv(out_csv_iam)
        else:
            scatter_hist(df_iam_mutants_neg, out=outfile, highlight=best_decoys, by=select_by)
        
        df_combined = pd.merge(df_dg_ff_thr, df_iam_mutants, on=['filename'])
        out_csv_iam = os.path.join(outdir, 'df_ff-{}_dg_iam_thresholded_rmsd{}.csv'.format(suffix, rmsd_suffix))
        df_combined.to_csv(out_csv_iam)

        df_best_indices = df_iam_mutants.loc[best_decoys]
        df_combined_best = pd.merge(df_dg_ff_thr, df_best_indices, on=['filename'])
        out_csv_iam = \
            os.path.join(selected_decoys_dir, 
            'df_ff-{}_dg_thresholded_rmsd{}_bestdecoys.csv'.format(suffix, rmsd_suffix))
        df_combined_best.to_csv(out_csv_iam)

        sequences_iam = list(df_combined_best['seq'])
        outfile_logo = os.path.join(outdir, 
        'logo_ff-{}_dg_thresholded_rmsd{}_iam-top{}.png'.format(suffix, rmsd_suffix, n_all))
        sequences_to_logo_without_weblogo(sequences_iam, dict_residues=dict_residues,
                                        outfile_logo=outfile_logo)


def get_args():
    desc = ('''
        Filter designs that meet RMSD threshold and have improved binding energies.
        Usage: python3 filter.py <options>
        ''')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('target_pdb',
                        type=str,
                        help='path to target structure chothia numbered pdb file.\
                            Provide pdb for the target structure of the Fv with the antigen.')
    parser.add_argument('--rmsd_filter',
                        default='H3,1.8',
                        help='specify metric and threshold separated by a comma.\
                             Metric list: OCD, H1, H2, H3, L1, L2, L3, HFr, LFr'
                        )
    parser.add_argument('--rmsd_filter_json',
                        default='',
                        help='specify multiple metrics and threshold as a json dictionary.\
                             Metric list: OCD, H1, H2, H3, L1, L2, L3, HFr, LFr'
                        )
    parser.add_argument('--csv_forward_folded',
                        default='',
                        help='csv file generated by --plot_consolidated_funnels'\
                        )
    parser.add_argument('--csv_complexes',
                        default='',
                        help='csv file generated by --plot_consolidated_dG'
                        )
    parser.add_argument('--model_suffix',
                        default='DeepAb',
                        help='give optional suffix with --output_filtered_designs\
                            if --csv_forward_folded file was generated with a different model'
                        )
    parser.add_argument('--outdir',
                        type=str,
                        default='./',
                        help='path to sequences dir')
    parser.add_argument('--cdr_list',
                        type=str,
                        default='',
                        help='comma separated list of cdrs: l1,h2')
    parser.add_argument('--framework',
                        action='store_true',
                        default=False,
                        help='design framework residues. Default: false')
    parser.add_argument('--indices',
                        type=str,
                        default='',
                        help='comma separated list of chothia numbered residues to design: h:12,20,31A/l:56,57')
    parser.add_argument('--exclude',
                        type=str,
                        default='',
                        help='comma separated list of chothia numbered residues to exclude from design: h:31A,52,53/l:97,99')
    parser.add_argument('--hl_interface',
                        action='store_true',
                        default=False,
                        help='hallucinate hl interface')

    return parser.parse_args()


def get_hal_indices(args):
    
    dict_indices = {}
    dict_exclude = {}
    if args.indices != '':
        indices_str = args.indices
        print(indices_str)
        dict_indices = comma_separated_chain_indices_to_dict(indices_str)
    if args.exclude != '':
        indices_str = args.exclude
        dict_exclude = comma_separated_chain_indices_to_dict(indices_str)

    indices_hal = get_indices_from_different_methods(
        args.target_pdb, \
        cdr_list=args.cdr_list, \
        framework=args.framework, \
        hl_interface=args.hl_interface, \
        include_indices=dict_indices, \
        exclude_indices=dict_exclude )
    print("Indices hallucinated: ", indices_hal)
    return indices_hal


if __name__ == '__main__':
    args = get_args()
    
    indices_hal = get_hal_indices(args)
    output_filtered_designs(args.csv_complexes,
                            args.csv_forward_folded,
                            args.target_pdb,
                            rmsd_filter=args.rmsd_filter,
                            rmsd_filter_json=args.rmsd_filter_json,
                            indices_hal=indices_hal,
                            outdir=args.outdir,
                            suffix=args.model_suffix
                            )

    