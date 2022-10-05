from src.hallucination.utils.pyrosetta_utils \
    import mutate_pose, \
    heavy_bb_rmsd_from_atom_map
from src.hallucination.utils.sequence_utils import\
    sequences_to_logo_without_weblogo
from src.deepab.metrics.rosetta_ab import get_ab_metrics_safe

from src.util.pdb import get_pdb_numbering_from_residue_indices, get_pdb_chain_seq
import pyrosetta.distributed.dask
from pyrosetta import *
import os, glob
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def write_fastas_for_alphafold2(pdb_files, outpath, chunk_size=20):
    for chunk_id in range(0, len(pdb_files)-1, chunk_size):
        outpath_chunk = os.path.join(outpath, str(chunk_id))
        if not os.path.exists(outpath_chunk):
            os.makedirs(outpath_chunk, exist_ok=False)
        chunk_id_end = min(len(pdb_files), chunk_id+chunk_size)
        for pdbfile in pdb_files[chunk_id:chunk_id_end]:
            seq_h = get_pdb_chain_seq(pdbfile, 'H')
            seq_l = get_pdb_chain_seq(pdbfile, 'L')
            outfile = '{}/{}.fasta'.format(outpath_chunk, os.path.basename(pdbfile).split('.pdb')[0])
            with open(outfile, 'w') as outf:
                outf.write('>H\n{}\n>L\n{}\n'.format(seq_h, seq_l))

def mutate_and_get_metrics(native_pose, deeph3_pose, res_positions):
    sequence = deeph3_pose.sequence()
    ros_positions = [t + 1 for t in res_positions]
    seq = [sequence[pos] for pos in res_positions]
    mutated_pose = mutate_pose(native_pose, ros_positions, seq, dump=False)
    metrics = get_ab_metrics_safe(mutated_pose, deeph3_pose)
    return metrics

def mutate_and_get_per_residue_rmsd(native_pose, deeph3_pose, res_positions):
    pyrosetta.rosetta.core.scoring.calpha_superimpose_pose(deeph3_pose, native_pose)
    sequence = deeph3_pose.sequence()
    ros_positions = [t + 1 for t in res_positions]
    seq = [sequence[pos] for pos in res_positions]
    mutated_pose = mutate_pose(native_pose, ros_positions, seq, dump=False)
    metrics = heavy_bb_rmsd_from_atom_map(deeph3_pose, mutated_pose, res_positions)
    return metrics


def plot_cdr_metric_distributions(df, met='', met_thr=None, outfile='metric_dist.png'):
    plt.figure(dpi=300)
    xlims = [[0, 5], [0, 1.0], [0., 1.0], [0, 2.0], [0, 2.0], [0, 3.0],
             [0, 2.0], [0, 2.0], [0, 2.0]]
    for i, col in enumerate(
        ["OCD", "H Fr", "L Fr", "H1", "H2", "H3", "L1", "L2", "L3"]):
        ax = plt.subplot(3, 3, i + 1)
        #also plot distributions
        sns.histplot(data=df,
                    x=col,
                    color="darkblue",
                    legend=False,
                    ax=ax)
        #if col != "H3":
        plt.xlim((0, max(1, 1.2 * df[col].max())))
        #else:
        #plt.xlim(xlims[i])
        if col==met and (not met_thr is None):
            ax.axvline(met_thr, ls='--', lw=2.0, c='red', zorder=1)

    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()


def get_thresholded_df(df_filtered_lowest, region_metrics, thresholds):
    import numpy as np
    rms_bool = (df_filtered_lowest[region_metrics[0]] > -np.Infinity)
    for met, met_thr in zip(region_metrics, thresholds):
        rms_bool = rms_bool &  (df_filtered_lowest[met] < met_thr)
    return df_filtered_lowest[rms_bool]

def threshold_by_rmsd_filters(df, 
                              rmsd_filter='',
                              rmsd_filter_json='',
                              outfile=None):
    import json

    if rmsd_filter_json != '':
        print("Reading rmsd from json file {}".format(rmsd_filter_json))
        rmsd_filter_dict = json.loads(open(rmsd_filter_json, 'r').read())
        print('RMSD filterdictionary: ', rmsd_filter_dict)
        df_thr = get_thresholded_df(df, list(rmsd_filter_dict.keys()), list(rmsd_filter_dict.values()))
        rmsd_suffix = os.path.basename(rmsd_filter_json).replace('.json', '')
        
    else:
        threshold_key, threshold_value = rmsd_filter.split(',')
        threshold_key_column = [col for col in list(df.columns) if threshold_key==col.replace(' ','')]
        assert len(threshold_key_column) == 1
        df_thr = df[df[threshold_key_column[0]]<= float(threshold_value)]
        rmsd_suffix = '{}-{}'.format(threshold_key, threshold_value)
        rmsd_filter_dict = {threshold_key_column[0]: threshold_value}

    df_thr.attrs['rmsd_filter_dict']=rmsd_filter_dict
    
    if not outfile is None:
        df_thr.to_csv(outfile.format(rmsd_suffix))
    
    return df_thr, rmsd_suffix


def plot_logos_for_design_ids(design_ids, pdb_file_name, indices_hal, outfile='logo.png'):
    pdb_files = glob.glob(pdb_file_name)
    dict_residues = {'reslist': indices_hal}
    labellist = indices_hal
    if len(pdb_files) != 0:
        labellist = \
        get_pdb_numbering_from_residue_indices(pdb_files[0], indices_hal)
    dict_residues.update({'labellist': labellist})

    seq_slices = []
    for id in design_ids:
        target_pdb = pdb_file_name.format('%03d' % id)
        if not os.path.exists(target_pdb):
            print(target_pdb, "not found")
            continue
        heavy_seq, light_seq = get_pdb_chain_seq(target_pdb,
                                                'H'), get_pdb_chain_seq(
                                                    target_pdb, 'L')
        sequence = heavy_seq + light_seq
        seq_slices.append(''.join([sequence[i] for i in indices_hal]))
    if len(seq_slices) > 1:
            sequences_to_logo_without_weblogo(seq_slices,
                                    dict_residues,
                                    outfile_logo=outfile)

def plt_ff_publication_for_run(csvfile, x='H3', outfile='hist_H3rmsd.png'):
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
    df = pd.read_csv(csvfile)
    fig = plt.figure(figsize=(6,5))
    ax = plt.gca()
    ax = sns.histplot(data=df, x=x, ax=ax, stat='probability')
    for pos in ['top', 'bottom', 'right', 'left']:
        ax.spines[pos].set_edgecolor('k')
    plt.xticks(rotation=45)
    ax.set_xlabel(r'{} RMSD ($\AA$)'.format(x))
    ax.set_ylabel('P(RMSD)')
    plt.tight_layout()
    plt.savefig(outfile, dpi=600, transparent=True)
    plt.close()


def frscores_vs_rosettascores(frscore_csv, rosetta_scores_csv,
                              wt_fr_score_array,
                              wt_rosetta_score_file):
    import os
    theme = {'axes.grid': True,
            'grid.linestyle': '',
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            "font.weight": 'regular',
            'xtick.color': 'black',
            'ytick.color': 'black',
            "axes.titlesize": 18,
            "axes.labelsize": 18
        }
    import matplotlib
    matplotlib.rcParams.update(theme)

    
    df_frscores = pd.read_csv(frscore_csv)
    df_rosetta_scores = pd.read_csv(rosetta_scores_csv)
    df_merged = pd.merge(df_frscores, df_rosetta_scores, on=['design_id'])

    sns.scatterplot(data=df_merged,
                    x='FR Score',
                    y='Score',
                    color='royalblue'
                    )
    wt_fr_score = np.load(wt_fr_score_array)
    wt_ros_score = float(open(wt_rosetta_score_file).read().rstrip())
    plt.axvline(wt_fr_score, ls='--', lw=2.0, c='black', zorder=1)
    plt.axhline(wt_ros_score, ls='--', lw=2.0, c='black', zorder=1)
    plt.xlabel('FR Score')
    plt.ylabel('Rosetta Total Score (REU)')
    plt.tight_layout()
    dirname = os.path.dirname(frscore_csv)
    plt.savefig('{}/frscore_vs_score.png'.format(dirname), dpi=600, transparent=True)
    plt.close()
    
    theme = {'axes.grid': True,
            'grid.linestyle': '',
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            "font.weight": 'regular',
            'xtick.color': 'black',
            'ytick.color': 'black',
            "axes.titlesize": 14,
            "axes.labelsize": 16
        }
    matplotlib.rcParams.update(theme)

    jp = sns.jointplot(data=df_merged,
                    x='FR Score',
                    y='Score',
                    kind='scatter',
                    color='royalblue',
                    height=4, ratio=2,
                    marginal_ticks=True
                    )
    wt_fr_score = np.load(wt_fr_score_array)
    wt_ros_score = float(open(wt_rosetta_score_file).read().rstrip())
    jp.refline(x=wt_fr_score, y=wt_ros_score, ls='--', lw=2.0, c='black', zorder=1)
    jp.set_axis_labels(xlabel='FR Score', ylabel='Rosetta Total Score (REU)' )
    plt.tight_layout()
    dirname = os.path.dirname(frscore_csv)
    plt.savefig('{}/joinplot_scatter_frscore_vs_score.png'.format(dirname), dpi=600, transparent=True)
    plt.close()

