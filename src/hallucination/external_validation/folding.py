from generate_pdbs_from_sequences \
     import get_rmsd_metrics_for_pdb, get_hal_indices, \
         plot_cdr_metric_distributions, plot_logos_for_design_ids, renumber_from_target, \
             plot_thresholded_metrics
import glob, os
import numpy as np
import pandas as pd
import argparse
from tempfile import NamedTemporaryFile
import seaborn as sns
import matplotlib.pyplot as plt

def plot_cdr_metric_distributions_multiple_datasets(df, met='', met_thr=None,
                                                    outfile='metric_dist.png',
                                                    hue='Model'):
    plt.figure(dpi=300)
    for i, col in enumerate(
        ["OCD", "H Fr", "L Fr", "H1", "H2", "H3", "L1", "L2", "L3"]):
        ax = plt.subplot(3, 3, i + 1)
        #also plot distributions
        sns.histplot(data=df,
                    x=col,
                    ax=ax,
                    hue=hue,
                    legend=False)
        plt.xlim((0, max(1, 1.2 * df[col].max())))
        if col==met and (not met_thr is None):
            ax.axvline(met_thr, ls='--', lw=2.0, c='red', zorder=1)

    plt.tight_layout()
    plt.savefig(outfile, transparent=True)
    plt.close()


def get_metrics_for_pdbs(pdb_path,
                         native_pdb,
                         design_pos,
                         suffix,
                         outdir='./'):

    pdb_files = glob.glob(pdb_path + '/*.pdb')
    pdb_files = list(sorted(pdb_files))
    all_metrics = []
    for i, pdb_file in enumerate(pdb_files):
        print(i, pdb_file)
        if not os.path.exists(pdb_file):
            print('Not found ', pdb_file)
            continue
        target_name = os.path.basename(pdb_file).split('.pdb')[0]
        renum_pdb = NamedTemporaryFile(delete=False)
        assert os.path.exists(renum_pdb.name)
        renumber_from_target(pdb_file, native_pdb, renum_pdb.name)
        
        rmsds_and_score = get_rmsd_metrics_for_pdb(renum_pdb.name,
                                 native_pdb,
                                 outdir,
                                 target_name,
                                 design_pos,
                                 per_residue_metrics=False,
                                 intermediate_metrics=False,
                                 ff_suffix=suffix)
        #print(rmsds_and_score)
        #assert sum(rmsds_and_score[2:]) != 0
        all_metrics.append(rmsds_and_score)
        # remove temp file
        os.unlink(renum_pdb.name)
        assert not os.path.exists(renum_pdb.name)
    
    stats_file = os.path.join(outdir,
                                "stats.csv")
    np.savetxt(stats_file, all_metrics, delimiter=',', fmt="%s")
        
def get_metrics_for_pdbs_for_af2(pdb_path,
                                native_pdb,
                                design_pos,
                                suffix,
                                outdir='./'):
    print(pdb_path + '/*/ranked_0.pdb')
    pdb_files = glob.glob(pdb_path + '/*/ranked_0.pdb')
    pdb_files = list(sorted(pdb_files))
    print(len(pdb_files))
    all_metrics = []
    for i, pdb_file in enumerate(pdb_files):
        print(i, pdb_file)
        if not os.path.exists(pdb_file):
            print('Not found ', pdb_file)
            continue
        target_name = '_'.join([t.strip() for t in pdb_file.split('/')[-2:]])
        print(target_name)
        renum_pdb = NamedTemporaryFile(delete=False)
        assert os.path.exists(renum_pdb.name)
        renumber_from_target(pdb_file, native_pdb, renum_pdb.name)
        
        rmsds_and_score = get_rmsd_metrics_for_pdb(renum_pdb.name,
                                 native_pdb,
                                 outdir,
                                 target_name,
                                 design_pos,
                                 per_residue_metrics=False,
                                 intermediate_metrics=False,
                                 ff_suffix=suffix)
        print(rmsds_and_score)
        assert sum(rmsds_and_score[2:]) != 0
        all_metrics.append(rmsds_and_score)
        # remove temp file
        os.unlink(renum_pdb.name)
        assert not os.path.exists(renum_pdb.name)
    
    stats_file = os.path.join(outdir,
                                "stats.csv")
    np.savetxt(stats_file, all_metrics, delimiter=',', fmt="%s")


def plot_metrics(stats_file, ref_stat_file='', suffix='IgFold', outdir=None,    
                 suffix_ref='DeepAb', threshold_by='H3',
                 indices_hal=[]):

    region_metrics = ["OCD", "H Fr", "L Fr", "H1", "H2", "H3", "L1", "L2", "L3"]
    if outdir is None:
        outdir = os.path.join(os.path.join( os.path.dirname(stats_file), '..' ), 'results_folding_methods')
        os.makedirs(outdir, exist_ok=True)
    metric_cols = [
        "pdb_name", "Score", "OCD", "H Fr", "H1", "H2", "H3", "L Fr", "L1", "L2",
        "L3"]
    metrics_df = pd.read_csv(stats_file,delimiter=',',
                              names=metric_cols)
    if suffix.lower() == 'IgFold'.lower():
        metrics_df['design_id'] = [int(t.split('_')[0]) for t in list(metrics_df['pdb_name'])]
    if suffix == 'AF':
        metrics_df['design_id'] = [int(t.split('_')[1]) for t in list(metrics_df['pdb_name'])]
    
    metrics_df.to_csv(os.path.join(outdir, 'rmsd_metrics_{}.csv'.format(suffix)))
    outfile = os.path.join(outdir, 'rmsd_metrics_dist_{}.png'.format(suffix))
    plot_cdr_metric_distributions(metrics_df, outfile=outfile)

    region_metrics = ["OCD", "H Fr", "L Fr", "H1", "H2", "H3", "L1", "L2", "L3"]
    thresholds_mean = [3.7, 0.43, 0.42, 0.72, 0.85, 2.33, 0.55, 0.45, 0.86]
    outdir_mean = os.path.join(outdir, 'threshold_deepAb_mean')
    os.makedirs(outdir_mean, exist_ok=True)
    outfile = os.path.join(
        outdir_mean, 'rmsd_dist_{}_cdr{{}}_thr-mean{{}}.png'.format(suffix))
    plot_thresholded_metrics(metrics_df, region_metrics, thresholds_mean, outfile)
    
    thresholds_select = [6.0, 0.5, 0.5, 1.0, 1.0, 2.00, 1.0, 1.0, 1.0]
    outdir_select = os.path.join(outdir, 'threshold_select')
    os.makedirs(outdir_select, exist_ok=True)
    outfile = os.path.join(
        outdir_select, 'rmsd_dist_{}_cdr{{}}_thr-select{{}}.png'.format(suffix))
    df_select_rmsd = plot_thresholded_metrics(metrics_df, region_metrics, thresholds_select, outfile)
    outfile = os.path.join(
        outdir_select, 'rmsd_dist_{}_All_thr-select.csv'.format(suffix))
    df_select_rmsd.to_csv(outfile)
    
    import matplotlib
    theme = {'axes.grid': True,
            'grid.linestyle': '',
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            "font.weight": 'regular',
            'xtick.color': 'black',
            'ytick.color': 'black',
            "axes.titlesize": 20,
            "axes.labelsize": 14
        }
    matplotlib.rcParams.update(theme)
    print(metrics_df)

    if ref_stat_file != '':
        df_ref = pd.read_csv(ref_stat_file)
        print(df_ref)

        df_combined = pd.merge(metrics_df, df_ref, on=['design_id'],
                                suffixes=['_'+suffix, '_'+suffix_ref])
        print(df_combined.columns)
        axis_label_pattern = r'{} {} RMSD ($\AA$)'
        axis_label_pattern_ocd = '{} {}'
        axis_label_pattern_score = '{} {} (REU)'
        for met in region_metrics+['Score']:
            ax = sns.scatterplot(data=df_combined, x='{}_{}'.format(met, suffix),
                                 y='{}_{}'.format(met, suffix_ref), color='darkblue')
            if met=='OCD':
                label_pattern = axis_label_pattern_ocd
            if met=='Score':
                label_pattern = axis_label_pattern_score
            else:
                label_pattern = axis_label_pattern 
            ax.set_xlabel(label_pattern.format(suffix, met))
            ax.set_ylabel(label_pattern.format(suffix_ref, met))
            met_clean = met.rstrip()
            outfile = os.path.join(
            outdir, 'compare_{}-{}_scatter_rmsd_{}.png'.format(suffix, suffix_ref, met_clean))
            plt.tight_layout()
            plt.savefig(outfile, dpi=600, transparent=True)
            plt.close()
        metrics_df['Model'] = [suffix for _ in list(metrics_df['design_id'])]
        df_ref['Model'] = [suffix_ref for _ in list(df_ref['design_id'])]
        ref_cols = list(df_ref.columns)
        data_cols = list(metrics_df.columns)
        common_cols = [t for t in ref_cols if t in data_cols]
        df_concat = pd.concat([metrics_df, df_ref], ignore_index=True, names=common_cols)
        outfile = os.path.join(
            outdir, 'compare_{}-{}_dist_rmsd.png'.format(suffix, suffix_ref))    
        plot_cdr_metric_distributions_multiple_datasets(df_concat, outfile=outfile)

        outfile = os.path.join(
            outdir, 'compare_{}-{}_dist_rmsd-H3.png'.format(suffix, suffix_ref))
        ax = sns.histplot(data=df_concat,
                    x='H3',
                    hue='Model')
        ax.set_xlabel(r'{} RMSD ($\AA$)'.format(threshold_by))
        plt.tight_layout()
        plt.savefig(outfile, transparent=True)
        plt.close()

        met_threshold = thresholds_select[region_metrics.index(threshold_by)]
        threshold_sequence_ids_ref = list(df_ref[df_ref['H3'] <= met_threshold]['design_id'])
        df_selection_ref = df_ref[df_ref.design_id.isin(threshold_sequence_ids_ref)]
        outfile_csv = os.path.join(outdir, 'thresholded_designs_{}_H3{}.csv'.format(suffix_ref, met_threshold))
        df_selection_ref.to_csv(outfile_csv)

        threshold_sequence_ids_data = list(metrics_df[metrics_df['H3'] <= met_threshold]['design_id'])
        df_selection_data = metrics_df[metrics_df.design_id.isin(threshold_sequence_ids_data)]
        outfile_csv = os.path.join(outdir, 'thresholded_designs_{}_H3{}.csv'.format(suffix, met_threshold))
        df_selection_data.to_csv(outfile_csv)


        threshold_sequence_ids = [int(t) for t in threshold_sequence_ids_ref if t in threshold_sequence_ids_data]
        df_selection = df_concat[df_concat.design_id.isin(threshold_sequence_ids)]
        outfile_csv = os.path.join(outdir, 'thresholded_designs_{}-{}_H3{}.csv'.format(suffix, suffix_ref, met_threshold))
        df_selection.to_csv(outfile_csv)
        outfile = os.path.join(
            outdir, 'compare_{}-{}_dist_rmsd-H3{}.png'.format(suffix, suffix_ref, met_threshold))
        ax = sns.histplot(data=df_selection,
                    x=threshold_by,
                    hue='Model')
        ax.set_xlabel(r'{} RMSD ($\AA$)'.format(threshold_by))
        plt.tight_layout()
        plt.savefig(outfile, transparent=True)
        plt.close()
        
        if indices_hal != []:
            outfile = outfile = os.path.join(
                outdir, 'logo_{}-{}_dist_rmsd-H3{}.png'.format(suffix, suffix_ref, met_threshold))
            deepab_ff_path = os.path.join( os.path.dirname(ref_stat_file), '..' )
            pdb_file_name = '{}/pdb_{{}}.deepAb.pdb'.format(deepab_ff_path)
            print(threshold_sequence_ids, len(threshold_sequence_ids))
            plot_logos_for_design_ids(threshold_sequence_ids, pdb_file_name, indices_hal, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('target_pdb', type=str, help='path to native pdb file' )
    parser.add_argument('suffix', type=str, help='specify string with no spaces for folding method')
    parser.add_argument('--pdbs_path',
                        type=str,
                        default='',
                        help='path for pdbfiles')
    parser.add_argument('--outdir', type=str, default='./', help='Path to directory for output')
    parser.add_argument('--indices', type=str, default='', help='design positions as defined for hallucinate.py')
    parser.add_argument('--cdr_list', type=str, default='', help='design positions as defined for hallucinate.py')
    parser.add_argument('--framework',
                        action='store_true',
                        default=False,
                        help='design framework residues. Default: false')
    parser.add_argument('--exclude',
                        type=str,
                        default='',
                        help='comma separated list of chothia numbered residues to exclude from design: h:31A,52,53/l:97,99')
    parser.add_argument('--hl_interface',
                        action='store_true',
                        default=False,
                        help='Not implemented! hallucinate hl interface')
    parser.add_argument('--ref_metrics_file',
                        type=str,
                        default='',
                        help='csv dataframe of DeepAb RMSD metrics')
    parser.add_argument('--suffix_ref',
                        type=str,
                        default='DeepAb',
                        help='reference model')
    
    args = parser.parse_args()
    design_positions = get_hal_indices(args)
    if args.pdbs_path != '':
        if args.suffix == 'AF':
            get_metrics_for_pdbs_for_af2(args.pdbs_path,
                                         args.target_pdb,
                                         design_positions,
                                         args.suffix,
                                         args.outdir)
        else:
            get_metrics_for_pdbs(args.pdbs_path, args.target_pdb, design_positions, args.suffix, args.outdir)
    
    metrics_file = stats_file = os.path.join(args.outdir,
                                "stats.csv")
    if os.path.exists(metrics_file):
        plot_metrics(metrics_file, ref_stat_file=args.ref_metrics_file,
                     suffix=args.suffix, indices_hal=design_positions,
                     suffix_ref=args.suffix_ref)