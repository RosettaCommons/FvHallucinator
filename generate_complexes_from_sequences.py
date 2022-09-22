
from importlib_metadata import sys, warnings
import os, argparse, json, glob
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
    relax_pose, mutate_pose, align_to_complex
from src.util.pdb import get_pdb_numbering_from_residue_indices

from src.hallucination.utils.util\
    import get_indices_from_different_methods,\
    comma_separated_chain_indices_to_dict
from src.hallucination.utils.interfacemetrics_plotting_utils \
    import iam_score_df_from_pdbs, plot_scores_and_select_designs, scatter_hist,\
         select_best_designs_by_sum
from src.hallucination.utils.sequence_utils import sequences_to_logo_without_weblogo
from src.hallucination.utils.rmsd_plotting_utils import threshold_by_rmsd_filters,\
                write_fastas_for_alphafold2, plt_ff_publication_for_run

init_string = "-mute all -check_cdr_chainbreaks false -detect_disulf true"
pyrosetta.init(init_string)


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
        python3 src/generate_complexes_from_sequences.py <target complexpdb chothia numbered>
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
                            ')
    parser.add_argument(
        'designed_seq_file',
        type=str,
        help=
        'Sequence file from process_designs.py (sequences_indices.fasta for complex generation);\
                        ')
    parser.add_argument('--get_relaxed_complex',
                        action='store_true',
                        default=False,
                        help='Make mutations to target pdb from sequence file,\
                             relax interface, calc dG, get best dG designs')
    parser.add_argument(
        '--plot_consolidated_dG',
        action='store_true',
        default=False,
        help='compile dG calculated for sequences into a plot.\
            Path for individual data files assumed to be same as \
            --outdir + /virtual_binding/relaxed_mutants_data'
    )
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
                        help='hallucinate hl interface')
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
    parser.add_argument('--csv_forward_folded',
                        default='',
                        help='csv file generated by --plot_consolidated_funnels\
                            .Only use designs that were filtered to fold into target structure\
                            from forward folding runs.'\
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
    # Not recommended
    parser.add_argument(
        '--path_forward_folded',
        type=str,
        default='',
        help='path to forward folded ab structures from forward folding run.\
             If you want to use forward folded structures for virtual screening.\
            Not recommended.')
    
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