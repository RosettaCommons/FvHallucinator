
from src.hallucination.utils.loss_plotting_utils import plot_all_losses
from src.hallucination.utils.command_line_utils import _get_args
from src.hallucination.utils.util import get_model_file,\
    comma_separated_chain_indices_to_dict,\
    get_indices_from_different_methods,\
    convert_chain_aa_to_index_aa_map
from src.hallucination.loss.setup_losses import setup_loss_components,\
    setup_loss_weights,\
    get_reference_losses,\
    debug_wt_losses
from src.hallucination.SequenceHallucinator import SequenceHallucinator
from src.util.preprocess import bin_value_matrix
from src.util.pdb import get_pdb_chain_seq, \
    protein_pairwise_geometry_matrix
from src.util.masking import mask_from_indices_list
from src.util.get_bins import get_dist_bins, get_dihedral_bins, get_planar_bins
from src.deepab.models.ModelEnsemble import ModelEnsemble
from src.util.util import _aa_dict
import json
import warnings
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import matplotlib
matplotlib.use('Agg')


def load_model_from_dict(model_files,
                         device=None):

    from src.deepab.models.AbResNet.AbResNet \
        import load_model
    return ModelEnsemble(load_model,
                         model_files,
                         eval_mode=True,
                         device=device)


def get_target_geometries(target_pdb):
    bin_getter = [get_dist_bins, get_dihedral_bins, get_planar_bins]
    n_bin_types = [3, 2, 1]
    out_bins = [37, 36, 36]
    bins = [
        bg(ob) for bg, bt, ob in zip(bin_getter, n_bin_types, out_bins)
        for _ in range(bt)
    ]
    target_geometries = [
        g for g in protein_pairwise_geometry_matrix(pdb_file=target_pdb)]
    target_geometries = [
        bin_value_matrix(g, b) for g, b in zip(target_geometries, bins)
    ]
    return target_geometries


def run_hallucination(model_path,
                      loss_weights_for_run,
                      outdir="test",
                      target_pdb="data/antibody_dataset/pdbs_testrun/1a0q.pdb",
                      cdr_list='',
                      framework=False,
                      include_indices={},
                      exclude_indices={},
                      hl_interface=False,
                      max_iters=100,
                      suffix='',
                      seed=0,
                      n_every=100,
                      restricted_positions_aa_freq={},
                      restricted_dict_keep_aas={},
                      disallowed_aas='',
                      use_manual_seed=True,
                      autostop=True,
                      seed_with_WT=False,
                      apply_lr_scheduler=True,
                      lr_dict={'learning_rate': 0.05,
                               'patience': 20, 'cooldown': 10},
                      pssm=None,
                      local_loss_only=True
                      ):

    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_type)
    print('Using {} as device'.format(str(device).upper()))

    if use_manual_seed:
        torch.manual_seed(seed)
    else:
        torch.random.seed()

    model = load_model_from_dict(model_path,
                                 device=device)
    target_geometries = get_target_geometries(target_pdb)

    wt_heavy_seq, wt_light_seq = get_pdb_chain_seq(target_pdb,
                                                   'H'), get_pdb_chain_seq(
        target_pdb, 'L')
    # Collect indices of positions to hallucinate
    indices_hal = get_indices_from_different_methods(target_pdb, cdr_list=cdr_list,
                                                     framework=framework,
                                                     hl_interface=hl_interface,
                                                     include_indices=include_indices,
                                                     exclude_indices=exclude_indices)

    wt_seq = wt_heavy_seq + wt_light_seq

    out_dir_losses = os.path.join(outdir, 'losses')
    os.makedirs(out_dir_losses, exist_ok=True)
    outnpy = os.path.join(out_dir_losses, "lossgeomfull_wt.npy".format(suffix))
    list_wt_mask = debug_wt_losses(wt_seq, wt_heavy_seq, model, target_geometries,
                                   device, outnpy)
    seq_design_mask = mask_from_indices_list(indices_hal, len(wt_seq))
    os.makedirs(outdir, exist_ok=True)
    mask_2d = seq_design_mask.unsqueeze(1).expand(-1, 10)
    plt.imshow(mask_2d, aspect='equal')
    plt.colorbar()
    plt.savefig('{}/design_mask.png'.format(outdir))
    plt.close()
    non_design_mask = None
    seq_for_hal = ''.join(['*' if i in indices_hal
                           else t for i, t in enumerate(wt_seq)])
    # seeding with WT sequence
    if seed_with_WT:
        sequence_seed = wt_seq
    else:
        sequence_seed = None
    print("Sequence input for design: ", seq_for_hal)

    restricted_positions_aa_freq_indexed = {}
    if restricted_positions_aa_freq != {}:
        restricted_positions_aa_freq_indexed = \
            convert_chain_aa_to_index_aa_map(restricted_positions_aa_freq,
                                             target_pdb,
                                             len(wt_heavy_seq))
        print('Positions with restricted AA freqs at Indices: ',
              restricted_positions_aa_freq_indexed)

    restricted_dict_keep_aas_indexed = {}
    if restricted_dict_keep_aas != {}:
        restricted_dict_keep_aas_indexed = \
            convert_chain_aa_to_index_aa_map(restricted_dict_keep_aas,
                                             target_pdb,
                                             len(wt_heavy_seq))
        print('Positions with restricted AA at Indices: ',
              restricted_dict_keep_aas_indexed)

    out_dir_losses = os.path.join(outdir, 'losses')
    os.makedirs(out_dir_losses, exist_ok=True)
    loss_components, loss_components_dict = \
        setup_loss_components(wt_seq, model,
                              len(wt_heavy_seq),
                              target_geometries,
                              loss_weights_for_run,
                              seq_design_mask,
                              device=device,
                              restricted_dict_aa_freqs=restricted_positions_aa_freq_indexed,
                              restricted_dict_keep_aas=restricted_dict_keep_aas_indexed,
                              non_design_mask=non_design_mask,
                              pssm=pssm,
                              wt_losses_mask=list_wt_mask,
                              outdir=out_dir_losses,
                              local_loss_only=local_loss_only
                              )
    print('Components in loss ', loss_components_dict)
    wt_geom_loss = None
    if 'geom' in loss_components_dict:
        wt_geom_loss, _ = get_reference_losses(wt_seq, wt_heavy_seq, model,
                                               loss_components, device,
                                               loss_components_dict)

    seq_design_mask = mask_from_indices_list(indices_hal, len(seq_for_hal))
    sequence_hallucinator = SequenceHallucinator(
        wt_seq,
        len(wt_heavy_seq) - 1,
        model,
        loss_components,
        design_mask=seq_design_mask,
        device=device,
        sequence_seed=sequence_seed,
        apply_lr_scheduler=apply_lr_scheduler,
        lr_config=lr_dict,
        disallowed_aas_at_initialization=disallowed_aas).to(device)

    traj_loss_dict = {}
    for key in loss_components_dict:
        traj_loss_dict[key] = []

    out_dir_trajs = os.path.join(outdir, 'trajectories')
    out_dir_int = os.path.join(outdir, 'intermediate')
    os.makedirs(out_dir_int, exist_ok=True)
    os.makedirs(out_dir_trajs, exist_ok=True)

    for itr in tqdm(range(max_iters)):

        list_losses = sequence_hallucinator.update_sequence(
            disallow_letters=disallowed_aas)

        if itr == 0:
            sequence_hallucinator.write_sequence_history_file(
                os.path.join(out_dir_int, "sequences_{}_init.fasta".format(suffix)))

        for key in traj_loss_dict:
            if key != 'reg_seq':
                traj_loss_dict[key].append(
                    list_losses[loss_components_dict[key]].numpy())
            else:
                heavy_ll = list_losses[loss_components_dict[key]].numpy()
                light_ll = list_losses[loss_components_dict[key] + 1].numpy()
                traj_loss_dict[key].append((heavy_ll, light_ll))

        # learning rate based autostop criterion
        if autostop:
            print(sequence_hallucinator.lr, sequence_hallucinator.start_lr)
            if sequence_hallucinator.start_lr / float(
                sequence_hallucinator.lr
            ) >= 100.0:
                print(
                    "Stopping at {} because learning rate has reached {} at iter {}"
                    .format(itr, sequence_hallucinator.lr, itr))
                break

        if (itr + 1) % n_every == 0:
            sequence_hallucinator.write_sequence_history_file(
                os.path.join(out_dir_int,
                             "sequences_{}_{}.fasta".format(suffix, itr)))

            outfile = os.path.join(out_dir_losses,
                                   "loss_{{}}_{}_{}.png".format(suffix, itr))
            plot_all_losses(traj_loss_dict, outfile,
                            max_iters, wt_geom_loss)

    # Write trajectory sequences
    sequence_hallucinator.write_sequence_history_file(
        os.path.join(out_dir_trajs, "sequences_{}_final.fasta".format(suffix)))

    # Save losses
    outfile_loss_mat = os.path.join(out_dir_losses,
                                    "lossdict_{}_final.npy".format(suffix))
    np.save(outfile_loss_mat, traj_loss_dict)

    # Plot losses
    outfile = os.path.join(out_dir_losses,
                           "loss_{{}}_{}_final.png".format(suffix))
    plot_all_losses(traj_loss_dict, outfile,
                    max_iters, wt_geom_loss)


def _indstr_to_dictlist_freqs(indices_str):
    chains = indices_str.split('/')
    dict_positions = {}
    for chstr in chains:
        ch = chstr.split(':')[0].lower()
        positions_dict_str = chstr.split(':')[1].split(',')
        for pds in positions_dict_str:
            pos = pds.split('-')[0]
            aa_freq_strs = pds.split('-')[1:]
            aa_freq_map = {}
            for aa_freq_str in aa_freq_strs:
                aa = aa_freq_str.split('=')[0]
                aa_freq = aa_freq_str.split('=')[1]
                aa_freq_map[aa] = float(aa_freq)
            dict_positions[(ch, pos)] = aa_freq_map
    return dict_positions


def _indstr_to_dictlist_keep_aas(indices_str, except_aas=False):
    chains = indices_str.split('/')
    dict_positions = {}
    for chstr in chains:
        ch = chstr.split(':')[0].lower()
        positions_dict_str = chstr.split(':')[1].split(',')
        for pds in positions_dict_str:
            pos = pds.split('-')[0]
            aa_str = pds.split('-')[1]
            specified_aas = [t for t in aa_str]
            if not except_aas:
                dict_positions[(ch, pos)] = specified_aas
            else:
                non_zero_aas = [
                    t for t in list(_aa_dict.keys()) if t not in specified_aas
                ]
                dict_positions[(ch, pos)] = non_zero_aas
    return dict_positions


def _cli():
    args = _get_args()
    dict_indices = {}
    dict_exclude = {}
    restricted_dict = {}
    restricted_dict_keep_aas = {}

    indices_str = args.indices
    if indices_str != '':
        dict_indices = comma_separated_chain_indices_to_dict(indices_str)

    if args.exclude != '':
        dict_exclude = comma_separated_chain_indices_to_dict(args.exclude)

    if args.restrict_positions_to_freq != '' and args.restrict_positions_to_aas != '' and \
            args.restrict_positions_to_aas_except != '':
        raise argparse.ArgumentError.message('--restrict_positions_to_freq or \
            --restrict_positions_to_aas or \
            --restrict_positions_to_aas_except \
            more than one of these options has been specified. \
            Current implementation does not support more than one at the same time.'
                                             )

    if args.restrict_positions_to_freq != '':
        restricted_dict = _indstr_to_dictlist_freqs(
            args.restrict_positions_to_freq)

    pssm_mat = None
    if args.apply_distribution_from_pssm != '':
        if os.path.exists(args.apply_distribution_from_pssm):
            pssm_mat = np.load(args.apply_distribution_from_pssm)

    if args.restrict_positions_to_aas != '':
        restricted_dict_keep_aas = _indstr_to_dictlist_keep_aas(
            args.restrict_positions_to_aas)

    if args.restrict_positions_to_aas_except != '':
        restricted_dict_keep_aas = _indstr_to_dictlist_keep_aas(
            args.restrict_positions_to_aas_except, except_aas=True)

    disallowed_aas = ''
    if args.disallow_aas_at_all_positions != '':
        disallowed_aas = args.disallow_aas_at_all_positions
        for aa in disallowed_aas:
            assert aa in _aa_dict

    model_file = get_model_file()
    loss_weights_for_run = setup_loss_weights(args)

    out_dir = args.prefix
    # Only run if the output file does not exist yet:
    expected_final_outfile = os.path.join(
        out_dir, 'trajectories', "sequences_{}_final.fasta".format(args.suffix))
    if os.path.exists(expected_final_outfile) and args.overwrite == False:
        print('Final fasta already exists. Not overwriting ',
              expected_final_outfile)
        pass
    else:
        os.makedirs(out_dir, exist_ok=True)
        outfile_json_args = os.path.join(out_dir, 'log_run_args.json')
        if not os.path.exists(outfile_json_args):
            args_dict = vars(args)
            open(outfile_json_args, 'w').write(json.dumps(args_dict))

        lr_settings_list = [t for t in args.lr_settings.split(',')]
        lr_config = dict(learning_rate=float(lr_settings_list[0]),
                         patience=int(lr_settings_list[1]),
                         cooldown=int(lr_settings_list[2]))
        if args.use_global_loss:
            warnings.warn('--use_global_loss given.\
                            Results not guaranteed. See command line help')
        run_hallucination(model_file,
                          loss_weights_for_run,
                          outdir=args.prefix,
                          target_pdb=args.target_pdb,
                          cdr_list=args.cdr_list,
                          framework=args.framework,
                          include_indices=dict_indices,
                          exclude_indices=dict_exclude,
                          hl_interface=args.hl_interface,
                          max_iters=args.iterations,
                          suffix=args.suffix,
                          seed=args.seed,
                          n_every=args.n_every,
                          restricted_positions_aa_freq=restricted_dict,
                          restricted_dict_keep_aas=restricted_dict_keep_aas,
                          disallowed_aas=disallowed_aas,
                          use_manual_seed=(not args.random_seed),
                          autostop=(not args.disable_autostop),
                          seed_with_WT=args.seed_with_WT,
                          apply_lr_scheduler=(not args.disable_lr_scheduler),
                          lr_dict=lr_config,
                          pssm=pssm_mat,
                          local_loss_only=(not args.use_global_loss))


if __name__ == '__main__':
    _cli()
