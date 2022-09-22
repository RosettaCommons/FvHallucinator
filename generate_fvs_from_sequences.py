
import os, argparse, json, glob

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dask.distributed import Client, get_client
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import pyrosetta.distributed.dask
from pyrosetta import *
from src.hallucination.utils.pyrosetta_utils \
    import score_pdb, get_sapscores
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
from src.hallucination.utils.rmsd_plotting_utils import plot_logos_for_design_ids,\
        get_thresholded_df, plot_cdr_metric_distributions,\
            mutate_and_get_per_residue_rmsd, mutate_and_get_metrics,\
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
    parser.add_argument('--start',
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

    if args.plot_consolidated_funnels:
        indices_hal = get_hal_indices(args)
        if not os.path.exists(args.path_forward_folded):
            raise FileNotFoundError('For --plot_consolidated_funnels option , \
                provide valid path for forward folded pdbs with --path_forward_folded'
                                    )
        plot_folded_structure_metrics(args.path_forward_folded,
                                      args.start,
                                      args.end,
                                      indices_hal=indices_hal,
                                      target_pdb=args.target_pdb)

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
                                start_from=args.start,
                                last=args.end,
                                relax_design=args.relax_designs
                                )
