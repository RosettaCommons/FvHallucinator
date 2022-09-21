
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
    comma_separated_chain_indices_to_dict, get_model_file
from src.hallucination.utils.developability_plots import plot_developability_param
from src.hallucination.utils.interfacemetrics_plotting_utils \
    import iam_score_df_from_pdbs, plot_scores_and_select_designs, scatter_hist,\
         select_best_designs_by_sum
from src.hallucination.utils.sequence_utils import sequences_to_logo_without_weblogo
from src.hallucination.utils.rmsd_plotting_utils import plot_logos_for_design_ids,\
        get_thresholded_df, threshold_by_rmsd_filters, plot_cdr_metric_distributions,\
            mutate_and_get_per_residue_rmsd, mutate_and_get_metrics,\
                write_fastas_for_alphafold2, plt_ff_publication_for_run,\
                    frscores_vs_rosettascores
import torch

init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"
pyrosetta.init(init_string)
torch.no_grad()

def plot_thresholded_logos(df_filtered_lowest,
                           path,
                           indices_hal,
                           region_metrics, 
                           thresholds, 
                           outfile='logo_{}_{}.png'):
    for met, met_thr in zip(region_metrics, thresholds):
            rms_thresholded = \
                df_filtered_lowest[df_filtered_lowest[met] < met_thr]
            threshold_sequence_ids = list(
                rms_thresholded["design_id"])
            
            pdb_file_name = '{}/pdb_{{}}.deepAb.pdb'.format(path)
            outfile_logo=outfile.format(met.replace(' ', ''), met_thr)
            plot_logos_for_design_ids(threshold_sequence_ids,
                                      pdb_file_name,
                                      indices_hal,
                                      outfile_logo)
            
    rms_thresholded_all = get_thresholded_df(df_filtered_lowest, region_metrics, thresholds)
    threshold_sequence_ids = list(
                rms_thresholded_all["design_id"])
    
    if len(threshold_sequence_ids) > 0:
        pdb_file_name = '{}/pdb_{{}}.deepAb.pdb'.format(path)
        outfile_logo=outfile.format('all', '')
        plot_logos_for_design_ids(threshold_sequence_ids,
                                      pdb_file_name,
                                      indices_hal,
                                      outfile_logo)


def plot_thresholded_metrics(filtered_lowest,
                             region_metrics,
                             thresholds,
                             outfile='consolidated_metrics_cdr{}_thr{}.png'):
    for met, met_thr in zip(region_metrics, thresholds):
        rms_thresholded = \
            filtered_lowest[filtered_lowest[met] < met_thr]
        plot_cdr_metric_distributions(rms_thresholded, met, met_thr, outfile.format(met, met_thr))
    
    rms_thresholded_all = get_thresholded_df(filtered_lowest, region_metrics, thresholds)
    plot_cdr_metric_distributions(rms_thresholded_all, outfile=outfile.format('all', ''))
    return rms_thresholded_all

def plot_thresholded_perres_metrics(df_perres,
                                    filtered_lowest,
                                    region_metrics,
                                    thresholds,
                                    outfile):
    for met, met_thr in zip(region_metrics, thresholds):
        rms_thresholded = \
            filtered_lowest[filtered_lowest[met] < met_thr]
        threshold_sequence_ids = list(
            rms_thresholded["design_id"])
        
        # perres thresholded
        df_perres_rms_thr = df_perres[df_perres.design_id.isin(threshold_sequence_ids)]
        if len(df_perres_rms_thr) < 5:
            continue
        sns.boxplot(data=df_perres_rms_thr, x='Position', y='RMSD')
        plt.xticks(rotation=45)
        if met != 'OCD':
            plt.ylabel(r'RMSD ($\AA$)')
        else:
            plt.ylabel('OCD')
        plt.tight_layout()
        plt.savefig(outfile.format(met, met_thr), dpi=600, transparent=True)
        plt.close()

    rms_thresholded_all = get_thresholded_df(filtered_lowest, region_metrics, thresholds)
    threshold_sequence_ids = list(
            rms_thresholded_all["design_id"])
        
    # perres thresholded
    df_perres_rms_thr = df_perres[df_perres.design_id.isin(threshold_sequence_ids)]
    if len(df_perres_rms_thr) > 5:
        sns.boxplot(data=df_perres_rms_thr, x='Position', y='RMSD')
        plt.xticks(rotation=45)
        if met != 'OCD':
            plt.ylabel(r'RMSD ($\AA$)')
        else:
            plt.ylabel('OCD')
        plt.tight_layout()
        plt.savefig(outfile.format('all', ''), dpi=600, transparent=True)
        plt.close()


def seek_and_plot_frscores(rosetta_scorefile, wt_rosetta_scorefile):
    
    path_hal_results = os.path.join(os.path.dirname(os.path.dirname(wt_rosetta_scorefile)), 'results')
    print(path_hal_results)
    if os.path.exists(path_hal_results):
        design_fr_scores_file = glob.glob('{}/FRScore_per_design_IG*.csv'.format(path_hal_results))
        
        wt_fr_scorefile = glob.glob('{}/FRScore_wt_IG*.npy'.format(path_hal_results))
        print(design_fr_scores_file)
        print(wt_fr_scorefile)
        if (len(design_fr_scores_file) == 1) and len(wt_fr_scorefile)==1:
            print(design_fr_scores_file)
            frscores_vs_rosettascores(design_fr_scores_file[0],
                                      rosetta_scorefile,
                                      wt_fr_scorefile[0],
                                      wt_rosetta_scorefile)


def _read_stat_files(stat_file, design_pdb_file, indices_hal):
    id = int(stat_file.split('_')[-1].rstrip('.csv'))
    hal_struct_metrics = pd.read_csv(stat_file,
                                        sep=',',
                                        names=[
                                            "Decoy", "Score", "OCD", "H Fr",
                                            "H1", "H2", "H3", "L Fr", "L1",
                                            "L2", "L3"
                                        ])
    hal_struct_metrics['design_id'] = [
        id for _ in range(len(hal_struct_metrics))
    ]
    hal_struct_metrics['Lowest'] = [
        0 for _ in range(len(hal_struct_metrics))
    ]
    hal_struct_metrics.at[hal_struct_metrics.idxmin()["Score"],
                            'Lowest'] = 1
    sequence_full = get_pdb_chain_seq(design_pdb_file,'H') + \
                    get_pdb_chain_seq(design_pdb_file, 'L')
    sequence_indices = ''.join([sequence_full[i] for i in indices_hal])
    hal_struct_metrics['seq'] = [
        sequence_indices for _ in range(len(hal_struct_metrics))
    ]
    return hal_struct_metrics


def _read_agg_stat_files(stat_file):
    hal_struct_metrics = pd.read_csv(stat_file,
                                        sep=',',
                                        names=[
                                            "Target", "Score", "OCD", "H Fr",
                                            "H1", "H2", "H3", "L Fr", "L1",
                                            "L2", "L3"
                                        ])
    hal_struct_metrics['design_id'] = [
        int(name.split('_')[1]) for name in list(hal_struct_metrics['Target'])
    ]

    return hal_struct_metrics


def plot_folded_structure_metrics(path,
                                  prev,
                                  last,
                                  filename='intermediate/stats_pdb',
                                  indices_hal=[],
                                  target_pdb=None
                                  ):
    
    csv_pattern = '{}/{}'.format(path, filename) + '_{}.csv'
    stat_files = [csv_pattern.format('%03d' %i) for i in range(prev, last)
                    if os.path.exists(csv_pattern.format('%03d' %i))
                    ]
    pdb_pattern = '{}/pdb_{{}}.deepAb.pdb'.format(path)
    ff_pdb_files = [pdb_pattern.format('%03d' %i) for i in range(prev, last)
                    if os.path.exists(pdb_pattern.format('%03d' %i))]
    print('{} pdb files found.'.format(len(ff_pdb_files)))
    print('{} stat files found.'.format(len(stat_files)))
    if len(stat_files) < 1:
        raise FileNotFoundError('No files {} found in {}'.format(
            filename, path))
    
    path_results = '{}/results'.format(path)
    os.makedirs(path_results, exist_ok=True)

    all_metrics = []
    for sf in stat_files:
        i = int(os.path.basename(sf).split('_')[-1].replace('.csv', ''))
        ff_pdb_file = pdb_pattern.format('%03d' %i)
        hal_struct_metrics = _read_stat_files(sf, ff_pdb_file, indices_hal)
        all_metrics.append(hal_struct_metrics)

    all_metrics_df = pd.concat(all_metrics)
    all_metrics_df.to_csv(
        os.path.join(
            path_results,
            'consolidated_metrics_N{}.csv'.format('%03d' % len(stat_files))))

    filtered_lowest = all_metrics_df[all_metrics_df["Lowest"] == 1].reset_index()
    outfile_csv = os.path.join(
        path_results, 'consolidated_ff_lowest_N{}.csv'.format('%03d' % len(stat_files)))
    print(filtered_lowest)
    filtered_lowest.to_csv(outfile_csv, sep=',')
    outfile = os.path.join(
        path_results, 'consolidated_funnels_N{}.png'.format('%03d' % len(stat_files)))
    xlims = [[0, 5], [0, 1.0], [0., 1.0], [0, 2.0], [0, 2.0], [0, 3.5],
             [0, 2.0], [0, 2.0], [0, 2.0]]
    region_metrics = ["OCD", "H Fr", "L Fr", "H1", "H2", "H3", "L1", "L2", "L3"]
    plt.figure(dpi=300)
    for i, col in enumerate(region_metrics):
        plt.subplot(3, 3, i + 1)
        sns.scatterplot(data=filtered_lowest,
                        x=col,
                        y="Score",
                        color="darkblue",
                        s=10,
                        legend=False)
        if col != "H3":
            plt.xlim((0, max(1, 1.2 * filtered_lowest[col].max())))
        else:
            plt.xlim(xlims[i])

    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()

    # Get SAP scores
    rosetta_indices = [t+1 for t in indices_hal]
    outfile_sap_scores = os.path.join(
        path_results, 'consolidated_sapscores_N{}.csv'.format('%03d' % len(stat_files)))
    if not os.path.exists(outfile_sap_scores):
        sap_scores = ['{}\t{}\n'.format(pdbfile, get_sapscores(pdbfile, rosetta_indices))
                    for pdbfile in ff_pdb_files]
        open(outfile_sap_scores, 'w').write(''.join(sap_scores))
    else:
        sap_scores = open(outfile_sap_scores, 'r').readlines()
    df = pd.DataFrame()
    param = 'SAP score'
    df[param] = [float(t.split()[1]) for t in sap_scores]
    if target_pdb != '' and os.path.exists(target_pdb):
        sap_wt = get_sapscores(target_pdb, rosetta_indices)
        df_wt = pd.DataFrame()
        df_wt[param] = [sap_wt]
    outfile = os.path.join(
        path_results, 'consolidated_sapscores_N{}.png'.format('%03d' % len(stat_files)))
    plot_developability_param(df, param, df_wt, outfile)

    agg_metrics = []
    agg_stat_files = glob.glob(path + '/stats_*.csv')
    for agg_sf in agg_stat_files:
        hal_mets = _read_agg_stat_files(agg_sf)
        agg_metrics.append(hal_mets)
    
    agg_metrics = pd.concat(agg_metrics).drop_duplicates('design_id').reset_index(drop=True)
    outfile_agg_scores = os.path.join(
        path_results, 'consolidated_funnels_aggregate_N{}.csv'.format('%03d' % len(stat_files)))
    agg_metrics.to_csv(outfile_agg_scores)
    if not target_pdb is None:
        outfile_wt_score = os.path.join(path, 'relaxed_wt_score.txt')
        if not os.path.exists(outfile_wt_score):
            score_wt = score_pdb(target_pdb, relax=True)
            open(outfile_wt_score, 'w').write(str(score_wt)+'\n')
        else:
            score_wt = float(open(outfile_wt_score, 'r').read().rstrip())
    outfile = os.path.join(
        path_results, 'consolidated_funnels_aggregate_N{}.png'.format('%03d' % len(stat_files)))
    xlims = [[0, 5], [0, 1.0], [0., 1.0], [0, 2.0], [0, 2.0], [0, 3.5],
             [0, 2.0], [0, 2.0], [0, 2.0]]
    region_metrics = ["OCD", "H Fr", "L Fr", "H1", "H2", "H3", "L1", "L2", "L3"]
    plt.figure(dpi=300)
    for i, col in enumerate(region_metrics):
        plt.subplot(3, 3, i + 1)
        sns.scatterplot(data=agg_metrics,
                        x=col,
                        y="Score",
                        color="darkblue",
                        s=10,
                        legend=False)
        if col != "H3":
            plt.xlim((0, max(1, 1.2 * agg_metrics[col].max())))
        else:
            plt.xlim(xlims[i])
        if not target_pdb is None:
            plt.axhline(score_wt, ls='--', lw=2.0, c='black', zorder=1)

    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()

    outfile = os.path.join(
        path_results, 'dist_TotalScore_N{}.png'.format('%03d' % len(stat_files)))
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
    import matplotlib
    matplotlib.rcParams.update(theme)
    fig = plt.figure(figsize=(5,4))
    sns.histplot(data=agg_metrics, x='Score', stat="count", color='darkblue',binwidth=3.0)
    if target_pdb is not None:
        plt.axvline(score_wt, ls='--', lw=2.0, c='black', zorder=1)
    ax = plt.gca()
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
    plt.xlabel('Total Score (REU)')
    plt.ylabel('count(Total Score)')
    plt.tight_layout()
    plt.savefig(outfile, transparent=True, dpi=600)
    plt.close()

    seek_and_plot_frscores(outfile_agg_scores, outfile_wt_score)
    
    outfile = os.path.join(
        path_results, 'consolidated_dist_N{}.png'.format('%03d' % len(stat_files)))
    plot_cdr_metric_distributions(filtered_lowest, outfile=outfile)
    
    if indices_hal != []:
        pdb_file_name = '{}/pdb_*.deepAb.pdb'.format(path)
        pdb_files = glob.glob(pdb_file_name)
        dict_residues = {'reslist': indices_hal}
        labellist = indices_hal
        if len(pdb_files) != 0:
            labellist = \
            get_pdb_numbering_from_residue_indices(pdb_files[0], indices_hal)
        dict_residues.update({'labellist': labellist})

        # plot per-residues distributions
        csv_pattern = '{}/'.format(path) + 'perresiduermsd_*.csv'
        files = glob.glob(csv_pattern)
        if len(files) != 0:
            
            df_list = []
            for f in files:
                try:
                    df = pd.read_csv(f)
                    df.columns = ['pdb_name'] + labellist
                    df_list.append(df)
                except:
                    continue
            if df_list != []:
                df_concat = pd.concat(df_list, ignore_index=True)
                df_perres = pd.melt(df_concat, id_vars=['pdb_name'], 
                value_vars=labellist, var_name='Position', value_name='RMSD', ignore_index=False)
                df_perres['design_id'] = [int(name.split('_')[1]) for name in list(df_perres['pdb_name'])]
                ax = sns.boxplot(data=df_perres, x='Position', y='RMSD')
                plt.xticks(rotation=45)
                plt.ylabel(r'RMSD ($\AA$)')
                plt.tight_layout()
                outfile = os.path.join(
                path_results, 'consolidated_perres_boxplot_N{}.png'.format('%03d' % len(stat_files)))
                plt.savefig(outfile, dpi=600, transparent=True)
                plt.close()
        thresholds_low = [3.7, 0.43, 0.42, 0.72, 0.85, 2.00, 0.55, 0.45, 0.86]
        outfile = os.path.join(path_results,
                'logo_N{}_cdr{{}}_thr-low{{}}.png'.format('%03d' % len(stat_files)))
        plot_thresholded_logos(filtered_lowest, path, indices_hal, region_metrics, thresholds_low, outfile)
    

def refine_fv_(mds_pdb_file, decoy_pdb_file, cst_file):
    import pyrosetta
    pyrosetta.init(init_string)
    try:
        if os.path.exists(decoy_pdb_file):
            pose = pyrosetta.pose_from_pdb(decoy_pdb_file)
            csm = get_constraint_set_mover(cst_file)
            csm.apply(pose)
            sf_fa_cst = get_sf_fa()
            score = sf_fa_cst(pose)

            return score

        return refine_fv(mds_pdb_file, decoy_pdb_file, cst_file)
    
    except:
        return 100.0


def plot_dG(df_dg, outfile, min_base_dg=None):
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
    import matplotlib
    matplotlib.rcParams.update(theme)
    fig = plt.figure(figsize=(5,4))
    sns.histplot(data=df_dg, x='dG', stat="probability")
    if min_base_dg is not None:
        plt.axvline(min_base_dg, ls='--', lw=2.0, c='black', zorder=1)
    ax = plt.gca()
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
    plt.xlabel('dG (REU)')
    plt.ylabel('P(dG)')
    plt.tight_layout()
    plt.savefig(outfile, transparent=True, dpi=600)
    plt.close()

    fig = plt.figure(figsize=(5,4))
    df_dg_neg = df_dg[df_dg['dG'] < -10.0]
    sns.histplot(data=df_dg_neg, x='dG', stat="probability")
    if min_base_dg is not None:
        plt.axvline(min_base_dg, ls='--', lw=2.0, c='black', zorder=1)
    ax = plt.gca()
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
    plt.xlabel('dG (REU)')
    plt.ylabel('P(dG)')
    plt.tight_layout()
    plt.savefig(outfile.replace('.png', '') + '_neg.png',
                transparent=True,
                dpi=600)
    plt.close()

    fig = plt.figure(figsize=(5,4))
    df_dg_neg = df_dg[df_dg['dG'] < -10.0]
    sns.histplot(data=df_dg_neg, x='dG', stat="count")
    if min_base_dg is not None:
        plt.axvline(min_base_dg, ls='--', lw=2.0, c='black', zorder=1)
    ax = plt.gca()
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
    plt.xlabel('dG (REU)')
    plt.ylabel('Count(dG)')
    plt.tight_layout()
    plt.savefig(outfile.replace('.png', '') + '_neg_count.png',
                transparent=True,
                dpi=600)
    plt.close()


def compile_and_plot_results(basename_data,
                             prev,
                             last,
                             wt_dG='wt_min_dG.json',
                             indices_hal=[],
                             target_pdb=''):
    dict_pattern = '{}/min_dG_decoys_{{}}.json'.format(basename_data)
    seq_dict_files = [
        dict_pattern.format(i) for i in range(prev, last)
        if os.path.exists(dict_pattern.format(i))
    ]
    seq_dicts = [
        json.loads(open(seq_dict, 'r').read()) for seq_dict in seq_dict_files
    ]

    dfs = [pd.DataFrame.from_dict(seq_dict, orient='index') \
           for seq_dict in seq_dicts]

    df_dg = pd.concat(dfs)
    df_dg['itraj'] = df_dg.index

    basename_results = os.path.join(basename_data, 'results')
    os.makedirs(basename_results, exist_ok=True)
    outfile_df = os.path.join(basename_results,
                              'dg_min_{}-{}.csv'.format(prev, last))
    df_dg.to_csv(outfile_df, ',')
    

    dict_pattern = '{}/all_decoys_{{}}.json'.format(basename_data)
    seq_dict_files = [
        dict_pattern.format(i) for i in range(prev, last)
        if os.path.exists(dict_pattern.format(i))
    ]
    dfs = [pd.DataFrame.from_dict(json.loads(open(seq_dict, 'r').read()), orient='index') \
           for seq_dict in seq_dict_files]

    df_all = pd.concat(dfs)
    df_all['itraj'] = df_all.index
    outfile_df = os.path.join(basename_results,
                              'decoys_scores_{}-{}.csv'.format(prev, last))
    df_all.to_csv(outfile_df, ',')

    outfile_dg_plot = os.path.join(basename_results,
                                    'dg_hist_{}-{}.png'.format(prev, last))

    if os.path.exists(wt_dG):
        wt_dG_dict = json.loads(open(wt_dG, 'r').read())
        wt_dg_value = wt_dG_dict['-1']['dG']

        dg_improved = df_dg[df_dg['dG'] < (wt_dg_value + 5)]

        dg_worse = df_dg[df_dg['dG'] > (wt_dg_value + 5)]

        dg_improved_sorted = dg_improved.sort_values(by='dG',
                                                       ascending=True)
        outfile_improved = os.path.join(
            basename_results,
            'improved_dG_sequences_{}-{}.csv'.format(prev, last))
        with open(outfile_improved, 'w') as f:
            f.write('filename,dG,seq\n')
            for _, row in dg_improved_sorted.iterrows():
                f.write('{},{},{}\n'.format(row['pdb'], row['dG'],
                                              row['seq']))

        plot_dG(df_dg, outfile_dg_plot, wt_dg_value)
    else:
        plot_dG(df_dg, outfile_dg_plot)

    if not indices_hal == [] and os.path.exists(wt_dG):
        dict_residues = {"reslist": indices_hal}
        print('target_pdb ', target_pdb)
        if os.path.exists(target_pdb):
            dict_residues["labellist"] = \
                get_pdb_numbering_from_residue_indices(target_pdb, indices_hal)
            print(dict_residues)
        seq_slices = list(dg_improved_sorted["seq"])
        if len(seq_slices) > 0:
            assert len(seq_slices[0]) == len(indices_hal)

            outfile = os.path.join(
                basename_results,
                'logo_dG_improved_threshold{}_{}-{}.png'.format(wt_dg_value + 5, prev,
                                                    last))
            sequences_to_logo_without_weblogo(seq_slices,
                                          dict_residues,
                                          outfile_logo=outfile)
        seq_slices = list(dg_worse["seq"])
        if len(seq_slices) > 0:
            assert len(seq_slices[0]) == len(indices_hal)

            outfile = os.path.join(
                basename_results,
                'logo_dG_worse_threshold{}_{}-{}.png'.format(wt_dg_value + 5, prev,
                                                    last))
            sequences_to_logo_without_weblogo(seq_slices,
                                          dict_residues,
                                          outfile_logo=outfile)


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

def renumber_from_target(pdb_file, native_pdb_file, renumbered_file):
    native_pose = pyrosetta.pose_from_pdb(native_pdb_file)
    pose = pyrosetta.pose_from_pdb(pdb_file)
    migrate_seq_numbering(native_pose, pose)
    pose.dump_pdb(renumbered_file)


def build_structure(model,
                    fasta_file,
                    out_dir,
                    target_pdb,
                    num_decoys=20,
                    target="out",
                    constraint_dir=None,
                    use_cluster=False):

    if constraint_dir == None:
        constraint_dir = os.path.join(out_dir, "constraints_{}".format(target))
    os.makedirs(constraint_dir, exist_ok=True)

    cst_file = os.path.join(constraint_dir, "hb_csm", "constraints.cst")
    cst_file = get_cst_file(model, fasta_file, constraint_dir)
    
    out_dir_int = os.path.join(out_dir, 'intermediate')
    if not os.path.exists(out_dir_int):
        os.makedirs(out_dir_int, exist_ok=True)
    mds_pdb_file = os.path.join(out_dir_int, "{}.mds.pdb".format(target))

    build_initial_fv(fasta_file, mds_pdb_file, model)
    renum_mds_file = os.path.join(out_dir_int,
                                  "{}.mds_renum.pdb".format(target))
    if not target_pdb is None:
        renumber_from_target(mds_pdb_file, target_pdb, renum_mds_file)
    else:
        # when we dont have a target structure available
        renumber_pdb(mds_pdb_file, renum_mds_file)

    decoy_pdb_pattern = os.path.join(out_dir_int,
                                     "{}.deepAb.{{}}.pdb".format(target))
    decoy_scores = []
    if use_cluster:
        client = get_client()
        for i in range(num_decoys):
            decoy_pdb_file = decoy_pdb_pattern.format(i)
            decoy_score = client.submit(refine_fv_, renum_mds_file,
                                        decoy_pdb_file, cst_file)
            decoy_scores.append(decoy_score)

        decoy_scores = client.gather(decoy_scores)
    else:
        for i in range(num_decoys):
            decoy_pdb_file = decoy_pdb_pattern.format(i)
            decoy_score = refine_fv_(renum_mds_file, decoy_pdb_file, cst_file)
            decoy_scores.append(decoy_score)

    best_decoy_i = np.argmin(decoy_scores)
    best_decoy_pdb = decoy_pdb_pattern.format(best_decoy_i)
    out_pdb = os.path.join(out_dir, "{}.deepAb.pdb".format(target))
    os.system("cp {} {}".format(best_decoy_pdb, out_pdb))

    decoy_stats = [[i, score] for i, score in enumerate(decoy_scores)]
    decoy_stats_file = os.path.join(out_dir_int, "stats_{}.csv".format(target))
    np.savetxt(decoy_stats_file,
               np.asarray(decoy_stats),
               delimiter=",",
               fmt="%s")

    os.system("rm -rf {}".format(constraint_dir))

    return out_pdb


def generate_pdb_from_model(sequences_file,
                            target_pdb,
                            model_files,
                            indices_hal,
                            out_dir='.',
                            num_decoys=1,
                            use_cluster=False,
                            start_from=0,
                            last=10000000,
                            relax_design=False):

    model = ModelEnsemble(load_model=load_model,
                          model_files=model_files,
                          eval_mode=True)

    out_dir_ff = os.path.join(out_dir, 'forward_folding')
    if not os.path.exists(out_dir_ff):
        os.makedirs(out_dir_ff, exist_ok=True)
    out_dir_plts = os.path.join(out_dir_ff, 'funnels')
    if not os.path.exists(out_dir_plts):
        os.makedirs(out_dir_plts, exist_ok=True)
    sequences_fasta = open(sequences_file, 'r').readlines()
    sequences_fasta = [t for t in sequences_fasta if t != '\n']

    sequences_fasta_hl = [
        ''.join(sequences_fasta[i:i + 4])
        for i in range(0, len(sequences_fasta), 4)
    ]
    
    start_seq = min([start_from, len(sequences_fasta_hl)])
    end_seq = min([last, len(sequences_fasta_hl)])
    ids = [int(t.split('_')[1]) for t in sequences_fasta if (t.find('>') !=-1) and (t.find(':H') !=-1)]
    dsequences = {}
    assert len(ids) == len(sequences_fasta_hl)
    for id, seq in zip(ids, sequences_fasta_hl):
        dsequences[id] = seq
    traj_ids = [t for t in dsequences if (t >= start_from) and (t <= end_seq)]
    traj_ids.sort()

    all_metrics = []
    all_per_residue_rmsds = []
    for i in traj_ids:
        fasta_file = os.path.join(out_dir_ff,
                                  'fullsequence_{}.fasta'.format('%03d' % i))
        open(fasta_file, 'w').write(sequences_fasta_hl[i])
        target = 'pdb_{}'.format('%03d' % i)
        deepab_pdb_file = os.path.join(out_dir_ff,
                                       '{}.deepAb.pdb'.format(target))
        if not os.path.exists(deepab_pdb_file):
            #skip files already processed.
            build_structure(model,
                            fasta_file,
                            out_dir_ff,
                            target_pdb,
                            target=target,
                            num_decoys=num_decoys,
                            use_cluster=use_cluster)

        if target_pdb is None:
            continue

        rmsd_metrics_all, rmsd_metrics_perres = \
            get_rmsd_metrics_for_pdb(deepab_pdb_file, target_pdb, out_dir_ff,
                                     target, indices_hal, num_decoys=num_decoys,
                                     relax_design=relax_design)

        all_metrics.append(rmsd_metrics_all)
        all_per_residue_rmsds.append(rmsd_metrics_perres)
    
    # all metrics for best decoy
    if not target_pdb is None:
        stats_file = os.path.join(out_dir_ff,
                                "stats_{}-{}.csv".format(start_seq, end_seq))
        np.savetxt(stats_file, all_metrics, delimiter=',', fmt="%s")
        stats_res_file = os.path.join(out_dir_ff,
                                "perresiduermsd_{}-{}.csv".format(start_seq, end_seq))
        np.savetxt(stats_res_file, all_per_residue_rmsds, delimiter=',', fmt="%s")


def get_rmsd_metrics_for_pdb(pdb_file,
                             target_pdb,
                             out_dir_ff,
                             target,
                             indices_hal,
                             per_residue_metrics=True,
                             intermediate_metrics=True,
                             num_decoys=5,
                             ff_suffix='deepAb',
                             relax_design=False
                             ):
        
    pose = pyrosetta.pose_from_pdb(pdb_file)
    native_pose = pyrosetta.pose_from_pdb(target_pdb)

    # lowest scoring decoy score and metrics
    metrics = mutate_and_get_metrics(native_pose, pose, indices_hal)
    if per_residue_metrics:
        per_residue_rmsd = mutate_and_get_per_residue_rmsd(native_pose, pose, indices_hal)
    
    # This part of the loop can be skipped
    # get metrics for all decoys
    if intermediate_metrics:
        decoy_pdb_pattern = os.path.join(
            out_dir_ff, "intermediate/{}.{}.{{}}.pdb".format(target, ff_suffix))
        #read scores from relax stats file
        decoy_stats_file = os.path.join(
            out_dir_ff, "intermediate/stats_{}.csv".format(target))
        decoy_scores = np.genfromtxt(decoy_stats_file,
                                    delimiter=",",
                                    dtype=float)[:, 1]

        #get antibody related metrics; append scores
        decoy_metrics = []
        for i in range(num_decoys):
            decoy_score_rosetta = score_pdb(decoy_pdb_pattern.format(i))
            decoy_pose = pyrosetta.pose_from_pdb(decoy_pdb_pattern.format(i))
            m = mutate_and_get_metrics(native_pose, decoy_pose, indices_hal)
            decoy_metrics.append([i, decoy_score_rosetta] + m)

        # overwriting decoy stats file with updated metrics
        np.savetxt(decoy_stats_file,
                np.asarray(decoy_metrics),
                delimiter=",",
                fmt="%s")
    
    score = score_pdb(pdb_file, relax=relax_design)

    if per_residue_metrics:
        return [target, score] + metrics, [target] + [per_residue_rmsd[k] for k in per_residue_rmsd]
    else:
        return [target, score] + metrics


def mutated_complexes_from_sequences(pdb,
                                     sequences_file,
                                     res_positions,
                                     chains,
                                     basename='.',
                                     dump_mutate=True,
                                     pre_mutated=False,
                                     use_cluster=False,
                                     decoys=2,
                                     skip_relax=False,
                                     dry_run=False,
                                     prev=0,
                                     last=1000000,
                                     basename_ff='',
                                     docking_res=[],
                                     csv_rmsd='',
                                     rmsd_filter='',
                                     rmsd_filter_json=''):
    """Generates pdb with given mutations from base pdb"""

    filtered_design_ids = None
    if csv_rmsd != '':
        df_ff = pd.read_csv(csv_rmsd)
        outfile = os.path.join(basename, 'filtered_designs_for_dG_calculation.csv')
        df_ff_thr, _ = threshold_by_rmsd_filters(df_ff, rmsd_filter=rmsd_filter,
                                                rmsd_filter_json=rmsd_filter_json,
                                                outfile=outfile)
        filtered_design_ids = list(set(list(df_ff_thr['design_id'])))
        print('Number of designs that meet rmsd filter: ', len(filtered_design_ids))
        if len(filtered_design_ids) < 1:
            print('No design has rmsd below specified rmsd filter. Exiting.')
            sys.exit()
        if len(filtered_design_ids) < 10:
            warnings.warn('!!! Less than 10 designs have rmsd below specified rmsd filter. !!!')
    
    base_pose = pose_from_pdb(pdb)
    lines = open(sequences_file, 'r').readlines()
    sequences = [t.rstrip() for t in lines if t.find('>') == -1]
    try:
        ids = [int(t.split('_')[1]) for t in lines if (t.find('>') !=-1)]
        assert len(ids) == len(sequences)

        ids_sequences_tuples = [(id, seq) for id, seq in zip(ids, sequences)]
        if not filtered_design_ids is None:
            ids_sequences_tuples = [(id, seq) for id, seq in ids_sequences_tuples if id in filtered_design_ids]
        dsequences = {}
        for (id, seq) in ids_sequences_tuples:
            dsequences[id] = seq
    except:
        dsequences = {}

    print('Number of designs: ', len(sequences))
    
    #Important - 1 indexed so add one
    ros_positions = [t + 1 for t in res_positions]

    pdb_basename = pdb.split('/')[-1]
    if dump_mutate:
        if not pre_mutated:
            basename_mutate = os.path.join(basename, 'mutants')
        else:
            basename_mutate = os.path.join(basename, 'mutants_ff_aligned')
        if not os.path.exists(basename_mutate):
            os.makedirs(basename_mutate, exist_ok=True)
        outfile_mutate = os.path.join(
            basename_mutate,
            pdb_basename.rstrip('.pdb') + '_design_{}.pdb')

    if pre_mutated:
        if not os.path.exists(basename_ff):
            raise FileNotFoundError('For pre_mutated option, \
                provide valid forward folded pdbs {}'.format(basename_ff))
        if docking_res == []:
            # make continuous
            max_ros_pos = max(ros_positions)
            min_ros_pos = min(ros_positions)
            docking_res = [min_ros_pos, max_ros_pos]
        basename_packed = os.path.join(basename, 'relaxed_ff_bb_mutants')
        basename_wt_data = os.path.join(basename, 'relaxed_bb_wt_data')
        new_best_decoy = os.path.join(basename, pdb_basename.rstrip('.pdb') + '_{}.relaxed_bb.pdb')
        new_best_decoy_wt = os.path.join(basename, pdb_basename.rstrip('.pdb') + '.wt.relaxed_bb.pdb')
    else:
        basename_packed = os.path.join(basename, 'relaxed_mutants')
        basename_wt_data = os.path.join(basename, 'relaxed_wt_data')
        new_best_decoy = os.path.join(basename, pdb_basename.rstrip('.pdb') + '_{}.relaxed.pdb')
        new_best_decoy_wt = os.path.join(basename, pdb_basename.rstrip('.pdb') + '.wt.relaxed.pdb')
        docking_res=[]

    if not os.path.exists(basename_packed):
        os.makedirs(basename_packed, exist_ok=True)
    outfile_relax = os.path.join(basename_packed,
                                 pdb_basename.rstrip('.pdb') + '_relax_{}.pdb')

    dict_scores = {}
    min_dG = {}
    outfile_int_dg_wt = os.path.join(basename_wt_data, 'min_dG_decoys_wt.json')
    print('Starting from: ', prev)
    if (not skip_relax) and prev == 0:
        print('Relaxing wt ...')
        os.makedirs(basename_wt_data, exist_ok=True)
        
        input_packed_poses = []
        if not (use_cluster):
            for index_decoy in range(decoys):
                score_tuple = fast_relax_pose_complex(
                    pdb,
                    chains,
                    index_decoy,
                    outfile=outfile_relax.format('input_%03d' % (index_decoy)),
                    dry_run=dry_run,
                    dock=pre_mutated,
                    induced_docking_res=docking_res)
                input_packed_poses.append(score_tuple)
        else:
            for index_decoy in range(decoys):
                client = get_client()
                score_tuple = client.submit(fast_relax_pose_complex,
                                            pdb,
                                            chains,
                                            index_decoy,
                                            outfile=outfile_relax.format(
                                                'input_%03d' % (index_decoy)),
                                            dry_run=dry_run,
                                            dock=pre_mutated,
                                            induced_docking_res=docking_res)
                input_packed_poses.append(score_tuple)
            input_packed_poses = client.gather(input_packed_poses)
            #print(input_packed_poses)

        sorted_score_input_poses = sorted(input_packed_poses,
                                          key=lambda p: p[1])
        sorted_dg_input_poses = sorted(input_packed_poses, key=lambda p: p[2])
        min_input_score = sorted_score_input_poses[0][1]
        min_input_dg = sorted_dg_input_poses[0][2]
        print('Min Input Score: ', min_input_score)
        print('Min input dg: ', min_input_dg)
        dict_scores[-1] = {
            'decoyid': [t[0] for t in sorted_dg_input_poses],
            'total_score': [t[1] for t in sorted_dg_input_poses],
            'dG': [t[2] for t in sorted_dg_input_poses],
            'seq': ''
        }
        outfile_int_all = os.path.join(basename_wt_data, 'all_decoys_wt.json')
        open(outfile_int_all, 'w').write(json.dumps(dict_scores))
        outfile_best_decoy = outfile_relax.format('input_%03d' %
                                 (sorted_dg_input_poses[0][0]))
        os.system('cp {} {}'.format(outfile_best_decoy, new_best_decoy_wt))

        min_dG[-1] = {
            'dG':
            sorted_dg_input_poses[0][2],
            'decoyid':
            sorted_dg_input_poses[0][0],
            'pdb':
            outfile_relax.format('input_%03d' %
                                 (sorted_dg_input_poses[0][0])),
            'seq':
            ''
        }
        open(outfile_int_dg_wt, 'w').write(json.dumps(min_dG))

    max_seq = min([len(sequences), last])
    if dry_run:
        max_seq = 2

    if not pre_mutated:
        basename_data = os.path.join(basename, 'relaxed_mutants_data')
    else:
        basename_data = os.path.join(basename, 'relaxed_ff_bb_mutants_data')
    
    os.makedirs(basename_data, exist_ok=True)
    
    if dsequences != {}:
        traj_ids = [t for t in dsequences if (t < max_seq) and (t >= prev)]
        traj_ids.sort()
    else:
        traj_ids = [t for t in range(sequences) if (t < max_seq) and (t >= prev)]
        traj_ids.sort()
    for iseq in tqdm(traj_ids):
        seq = dsequences[iseq]
        print(iseq, seq)
        min_dG = {}
        dict_scores = {}
        if not pre_mutated:
            _ = mutate_pose(base_pose, ros_positions, seq,
                            outfile_mutate.format('%03d' % iseq))
        else:
            ff_pose = pose_from_pdb('{}/pdb_{}.deepAb.pdb'.format(
                basename_ff, '%03d' % iseq))
            align_to_complex(ff_pose, base_pose, chains,
                             outfile_mutate.format('%03d' % iseq))
        if skip_relax:
            continue

        outfile_int_dg = os.path.join(basename_data,
                                       'min_dG_decoys_{}.json'.format(iseq))
        if os.path.exists(outfile_int_dg):
           #skip if already processed
           continue
        relaxed_poses = relax_pose(outfile_mutate.format('%03d' % iseq),
                                   outfile_relax,
                                   iseq,
                                   chains,
                                   seq=seq,
                                   use_cluster=use_cluster,
                                   decoys=decoys,
                                   dry_run=dry_run,
                                   dock=pre_mutated,
                                   induced_docking_res=docking_res)

        packed_poses_sorted_dg = sorted(relaxed_poses, key=lambda tup: tup[2])
        print(iseq, ' Min decoy dg: ', packed_poses_sorted_dg[0][2])

        #Save data
        dict_scores[iseq] = {
            'decoyid': [t[0] for t in packed_poses_sorted_dg],
            'total_score': [t[1] for t in packed_poses_sorted_dg],
            'dG': [t[2] for t in packed_poses_sorted_dg],
            'seq': seq
        }

        outfile_int_all = os.path.join(basename_data,
                                       'all_decoys_{}.json'.format(iseq))
        open(outfile_int_all, 'w').write(json.dumps(dict_scores))

        min_dG[iseq] = {
            'dG':
            packed_poses_sorted_dg[0][2],
            'decoyid':
            packed_poses_sorted_dg[0][0],
            'pdb':
            outfile_relax.format('%03d_%03d' %
                                 (iseq, packed_poses_sorted_dg[0][0])),
            'seq':
            seq
        }
        outfile_best_decoy = outfile_relax.format('%03d_%03d' %
                                 (iseq, packed_poses_sorted_dg[0][0]))
        os.system('cp {} {}'.format(outfile_best_decoy, new_best_decoy.format(iseq)))

        open(outfile_int_dg, 'w').write(json.dumps(min_dG))

    compile_and_plot_results(basename_data, prev, max_seq, outfile_int_dg_wt)
    
    # Interface metrics
    if not pre_mutated:
        basename_interface_metrics = os.path.join(basename, 'interface_metrics')
    else:
        basename_interface_metrics = os.path.join(basename, 'interface_metrics_ff_bb')
    os.makedirs(basename_interface_metrics, exist_ok=True)
    design_pdbs = list(sorted(glob.glob(new_best_decoy.replace('{}','*'))))
    mutants_interface_metrics_file = os.path.join(basename_interface_metrics,
                                       'interface_metrics_all.csv')
    df_mutants = iam_score_df_from_pdbs(design_pdbs, mutants_interface_metrics_file)

    ref_interface_metrics_file = os.path.join(basename_interface_metrics,
                                       'interface_metrics_wt.csv')
    ref_pdbs = list(sorted(glob.glob(new_best_decoy_wt)))
    df_ref = iam_score_df_from_pdbs(ref_pdbs, ref_interface_metrics_file)

    by = ['dG_separated']
    plot_scores_and_select_designs(df_mutants, df_ref, out_path=basename_interface_metrics,
                                   pdb_dir=basename_interface_metrics,
                                   by=by, n=25)


def get_args():
    desc = ('''
        Distributed relax and deltaG (with Rosetta) calculation for designed sequences.
        Designed sequences -> relaxed antibody/complex (pdbs) -> total score/dg calculation.
        Example usage:
        python3 src/generate_pdbs_from_sequences.py <target pdb chothia numbered>
        <hallucination_results_dir>/sequences_indices.fasta
        --get_relaxed_complex # relax and get complex dg
        --decoys 2  # number of decoys for relax: 20 is a good number to start with>
        --dry_run # dont relax just run the full protocol once to check setup>
        --outdir # output directory
        --indices h:95,96,97,98,99,100,100A,100B,100C,101 
        --partner_chains HL_X #chain names of antibody and antigen
        # Recommended option
        # set --slurm_cluster_config option to run with dask on a slurm cluster
        ''')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('target_pdb',
                        type=str,
                        help='path to target structure chothia numbered pdb file.\
                            For complex structures, provide pdb for the antibody-antigen complex.\
                            For antibody only structures from DeepAb, provide target structure of\
                            the antibody.')
    parser.add_argument(
        'designed_seq_file',
        type=str,
        help=
        'Sequence file from process_designs.py (sequences_indices.fasta for complex generation;\
                        sequences.fasta for antibody only generation'                                                                     )
    parser.add_argument(
        '--path_forward_folded',
        type=str,
        default='',
        help='path to forward folded ab structures from forward folding run')
    parser.add_argument('--get_relaxed_complex',
                        action='store_true',
                        default=False,
                        help='Make mutations to target pdb from sequence file,\
                             relax interface, calc dG, get best dG designs'                                                                             )
    parser.add_argument('--pdbs_from_model',
                        action='store_true',
                        default=False,
                        help='Forward fold full Ab from full sequence designs\
                           file with DeepAb/H3 model.'                                                      )
    parser.add_argument('--plot_consolidated_funnels',
                        action='store_true',
                        default=False,
                        help='plot all forward folded structures in the\
                             same funnel plot from forward folding runs\
                             file with DeepAb/H3 model. Must provide path\
                             for forward folding dir'                                                     )
    parser.add_argument(
        '--plot_consolidated_dG',
        action='store_true',
        default=False,
        help='compile dG calculated for sequences into a plot.\
            Path for individual data files assumed to be same as \
            --outdir + /virtual_binding/relaxed_mutants_data'
    )
    parser.add_argument('--output_filtered_designs',
                        action='store_true',
                        default=False,
                        help='requires csv dataframes from --plot_consolidated_dG\
                            and --plot_consolidated_funnels.\
                            Outputs dataframes filtered by\
                            both RMSD and dG.\
                            Also, calculates interface metrics for\
                            for filtered designs. Specify rmsd_filter with\
                            either --rmsd_filter (single metric) or\
                            --rmsd_filter_json (multiple metrics)'
                        )
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
    parser.add_argument('--iter_every',
                        type=int,
                        default=None,
                        help='pick every nth designed sequence')
    parser.add_argument('--decoys',
                        type=int,
                        default=2,
                        help='number of decoys per design for relax')
    parser.add_argument('--prev_run',
                        type=int,
                        default=0,
                        help='continuation run - start from Nth design')
    parser.add_argument('--end',
                        type=int,
                        default=10000000,
                        help='end at Nth design')
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
                        help='Not implemented! hallucinate hl interface')
    parser.add_argument('--abag_interface',
                        action='store_true',
                        default=False,
                        help='Not implemented! hallucinate paratope')

    parser.add_argument(
        '--slurm_cluster_config',
        type=str,
        default='',
        help='Dictionary for setting up slurm cluster. Recommended.\
                See example config.json. Please modify for your slurm cluster.\
                If not using, consider using fewer decoys for DeepAb e.g. 2.')
    parser.add_argument('--partner_chains',
                        type=str,
                        default='',
                        help='Specify complex chains: Eg. HL_X; \
                              where HL chains form one interacting partner\
                              and X the other'                                              )
    parser.add_argument('--dry_run',
                        action='store_true',
                        default=False,
                        help='run everything except relax.apply().')
    parser.add_argument('--skip_relax',
                        action='store_true',
                        default=False,
                        help='run everything except relax.apply().')
    parser.add_argument('--slurm_scale',
                        type=int,
                        default=10,
                        help='number of clients (dask) on slurm')
    parser.add_argument('--scratch_space',
                        type=str,
                        default='./tmp_scratch',
                        help='scratch space for dask')
    parser.add_argument('--model',
                        type=str,
                        default='',
                        help='path to trained model')
    parser.add_argument('--scatterplot',
                        action='store_true',
                        default=False,
                        help="Produce a scatter plot and select best designs based on score.")
    parser.add_argument('--n_select',
                        type=int,
                        default=20,
                        help='Number of designs to select based on interfaceanalyzer data')
    parser.add_argument('--relax_designs',
                        action='store_true',
                        default=False,
                        help='Additional relax of DeepAb folded designs - slow - skip unless evaluating HL interface')
    
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
    import json
    use_cluster_decoy = False
    if args.slurm_cluster_config != '':
        scratch_dir = os.path.join(args.scratch_space)
        os.system("mkdir -p {}".format(scratch_dir))
        use_cluster_decoy = True
        config_dict = json.load(open(args.slurm_cluster_config,'r'))
        cluster = SLURMCluster(**config_dict,
                                local_directory=scratch_dir,
                                job_extra=[
                "-o {}".format(os.path.join(scratch_dir, "slurm-%j.out"))
            ],
            extra=pyrosetta.distributed.dask.worker_extra(init_flags=init_string)
            )
        print(cluster.job_script())
        cluster.adapt(minimum_jobs=min(args.decoys, 2),
                      maximum_jobs=min(args.decoys, args.slurm_scale))

        client = Client(cluster)

    if args.plot_consolidated_dG:
        indices_hal = get_hal_indices(args)
        wt_min_path = os.path.join(
            args.outdir,
            'virtual_binding/relaxed_wt_data/min_dG_decoys_wt.json')
        if not os.path.exists(wt_min_path):
            print("Did not find WT dg at {}".format(wt_min_path))
        basename_mutant_data = os.path.join(
            args.outdir, 'virtual_binding/relaxed_mutants_data')
        if os.path.exists(basename_mutant_data):
            compile_and_plot_results(basename_mutant_data,
                                 args.prev_run,
                                 args.end,
                                 wt_dG=wt_min_path,
                                 indices_hal=indices_hal,
                                 target_pdb=args.target_pdb)

    if args.plot_consolidated_funnels:
        indices_hal = get_hal_indices(args)
        if not os.path.exists(args.path_forward_folded):
            raise FileNotFoundError('For --plot_consolidated_funnels option , \
                provide valid path for forward folded pdbs with --path_forward_folded'
                                    )
        plot_folded_structure_metrics(args.path_forward_folded,
                                      args.prev_run,
                                      args.end,
                                      indices_hal=indices_hal,
                                      target_pdb=args.target_pdb)

    if args.output_filtered_designs:
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

    if args.get_relaxed_complex:
        pre_mutated = False
        if args.path_forward_folded != '':
            import glob
            pre_mutated = True
            assert os.path.exists(args.path_forward_folded)
            ff_pattern = '{}/*.deepAb.pdb'.format(args.path_forward_folded)
            assert len(glob.glob(ff_pattern)) > 0

        indices_hal = get_hal_indices(args)

        out_path_pdbs = os.path.join(args.outdir, 'virtual_binding')
        if not os.path.exists(out_path_pdbs):
            os.makedirs(out_path_pdbs, exist_ok=True)
        mutated_complexes_from_sequences(args.target_pdb,
                                         args.designed_seq_file,
                                         indices_hal,
                                         args.partner_chains,
                                         pre_mutated=pre_mutated,
                                         basename_ff=args.path_forward_folded,
                                         basename=out_path_pdbs,
                                         use_cluster=use_cluster_decoy,
                                         decoys=args.decoys,
                                         dry_run=args.dry_run,
                                         skip_relax=args.skip_relax,
                                         prev=args.prev_run,
                                         last=args.end,
                                         csv_rmsd=args.csv_forward_folded,
                                         rmsd_filter=args.rmsd_filter,
                                         rmsd_filter_json=args.rmsd_filter_json
                                         )

    if args.pdbs_from_model:
        model_files = get_model_file(args.model)
        if args.target_pdb=='None':
            target_pdb = None
            indices_hal = []
        else:
            target_pdb = args.target_pdb
            indices_hal = get_hal_indices(args)

        generate_pdb_from_model(args.designed_seq_file,
                                target_pdb,
                                model_files,
                                indices_hal,
                                out_dir=args.outdir,
                                num_decoys=args.decoys,
                                use_cluster=use_cluster_decoy,
                                start_from=args.prev_run,
                                last=args.end,
                                relax_design=args.relax_designs
                                )
