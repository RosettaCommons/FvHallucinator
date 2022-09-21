
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np

from src.util.util import letter_to_num, _aa_dict
from src.util.get_bins import get_dist_bins, get_dihedral_bins, get_planar_bins
from src.util.pdb import protein_pairwise_geometry_matrix
from src.util.preprocess import bin_value_matrix
from src.hallucination.loss.LossComponent import *
from src.hallucination.utils.util import get_loss, get_background_distributions,\
                                      get_restricted_freq_distributions,\
                                      get_restricted_aa_distributions,\
                                      get_restricted_seq_mask,\
                                      get_restricted_seq_aa_mask,\
                                      get_cce_aa_distributions

import src.hallucination.params as params

def debug_wt_losses(seq, heavy_seq, model, target_geometries, device, outfile_loss_full):
    
    heavy_chain_delimiter = torch.zeros((1, len(seq))).to(device)
    heavy_chain_delimiter[0, (len(heavy_seq) - 1)] = 1
    seq_onehot = F.one_hot(torch.tensor(letter_to_num(
        seq, _aa_dict)).long()).to(device)
    model_input = torch.cat(
        [seq_onehot.transpose(0, 1), heavy_chain_delimiter]).unsqueeze(0)
        
    lc = GeometryLoss(target_geometries,
                    device=device,
                    reduction='none',
                    local=False)
    with torch.no_grad():
        candidate_geometries = model(model_input.to(device))
        candidate_geometries = [cg.squeeze(0) for cg in candidate_geometries]
        loss_full = lc.calc_loss(candidate_geometries)
        loss_full = torch.stack(loss_full)
        np.save(outfile_loss_full, loss_full.detach().cpu().numpy())
        loss_per_res = torch.sum(loss_full, dim=-1)
        loss_full_upper_quartile = torch.quantile(loss_per_res, 0.90, dim=-1)
        wt_uq_mask = torch.sum(loss_full, dim=-1) < \
            loss_full_upper_quartile.unsqueeze(2).expand(-1, -1, loss_full.shape[2])
    wt_uq_mask.squeeze_(1)
    list_wt_uq_mask = torch.split(wt_uq_mask.expand(-1, loss_full.shape[-1]), 1, dim=0)
    list_wt_uq_mask = [t for t in list_wt_uq_mask]
    return list_wt_uq_mask


def get_target_geometries(target_pdb):
    bin_getter = [get_dist_bins, get_dihedral_bins, get_planar_bins]
    # 3: distance, 2: dihedrals, 1: planar
    n_bin_types=[3, 2, 1]
    # 37 bins for distance, 36 for dihedral, 36 for planar
    out_bins=[37, 36, 36]
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


def get_reference_losses(seq, heavy_seq, model_eval, \
                         loss_components, device,
                         loss_component_dict):

    heavy_chain_delimiter = torch.zeros((1, len(seq))).to(device)
    heavy_chain_delimiter[0, (len(heavy_seq) - 1)] = 1
    seq_onehot = F.one_hot(torch.tensor(letter_to_num(
        seq, _aa_dict)).long()).to(device)

    _, loss_list = get_loss(model_eval, loss_components,heavy_chain_delimiter.float(),\
                        seq_onehot.float(),seq_onehot.float())
    if 'geom' in loss_component_dict:
        # 0: tuple position of loss in losses in loss list; 1 is w for geom loss
        wt_geom_loss = (torch.tensor(
            loss_list[loss_component_dict['geom']][0])).numpy()
        #print('Wildtype Geometric Loss: ', wt_geom_loss, len(wt_geom_loss))
        wt_geom_loss_total = np.sum(wt_geom_loss)
        return wt_geom_loss, wt_geom_loss_total
    else:
        if 'entropy' in loss_component_dict:
            return np.empty(), 0.0
        raise KeyError('geom loss not found in loss components')


def setup_loss_weights(args):

    if args.geometric_loss_list == '':
        print('Using geometric loss weights: ',
              params.geometric_loss_dict)
        geometric_loss_list = params.geometric_loss_dict
    else:
        geometric_loss_list = [
            float(t) for t in (args.geometric_loss_list).split(',')
        ]

    weight_netcharge = 1.0 if args.restrict_total_charge else 0.0
    weight_max_aa_freq = 1.0 if args.restrict_max_aa_freq else 0.0

    print('Setting loss weights as\n: \
           sequence (non-design): {}\n geometric: {}\n\
           seq restriction: {}\n\
           '.format(args.seq_loss_weight, args.geometric_loss_weight,
             args.restricted_positions_kl_loss_weight))
    loss_params = \
        params.HallucinationLossParams(weight_seq=args.seq_loss_weight,
                                       weight_geom=args.geometric_loss_weight,
                                       weight_kl_res=\
                                        args.restricted_positions_kl_loss_weight,
                                       weight_seq_reg=args.avg_seq_reg_loss_weight,
                                       weight_netcharge=weight_netcharge,
                                       weight_aa_freq=weight_max_aa_freq,
                                       geometric_loss_list=\
                                        geometric_loss_list)

    return loss_params


def setup_loss_components(seq_for_hal,
                          model,
                          len_heavy_seq,
                          target_geometries,
                          loss_weights,
                          design_mask,
                          restricted_dict_aa_freqs={},
                          restricted_dict_keep_aas={},
                          device=None,
                          non_design_mask=None,
                          pssm=None,
                          wt_losses_mask=[],
                          local_loss_only=True,
                          outdir='./',
                          disable_nondesign_mask=False
                          ):

    loss_components = []
    loss_components_dict = {}
    loss_index = -1

    if loss_weights.loss_weight_kl_bg > 0:
        loss_index += 1
        loss_components_dict['kl_bg'] = loss_index

        loss_components.append(
            (loss_weights.loss_weight_kl_bg,
             KLDivLoss(background_distributions=get_background_distributions(
                 model, len(seq_for_hal), len_heavy_seq - 1, device),
                       device=device, loss_candidate='geom')))


    if loss_weights.loss_weight_kl_res > 0 and (not pssm is None) and (
            (restricted_dict_aa_freqs == {}) and (restricted_dict_keep_aas == {})):
        loss_index += 1
        loss_components_dict['kl_res'] = loss_index

        cce_normed, cce_mask = get_cce_aa_distributions(pssm, design_mask, device)

        loss_components.append((loss_weights.loss_weight_kl_res,
                                KLDivLoss(background_distributions=cce_normed,
                                          device=device,
                                          masks=cce_mask,
                                          maximize=False,
                                          loss_candidate='seq')))


    if loss_weights.loss_weight_kl_res > 0 and (restricted_dict_aa_freqs != {}):
        loss_index += 1
        loss_components_dict['kl_res'] = loss_index

        bk_dist = []
        restricted_mask = []
        if restricted_dict_aa_freqs != {}:
            bk_dist = get_restricted_freq_distributions(
                     len(seq_for_hal), restricted_dict_aa_freqs)
            restricted_mask = get_restricted_seq_mask(len(seq_for_hal),
                                               restricted_dict_aa_freqs,
                                               device)
        if not pssm is None:
            bk_dist, restricted_mask = get_cce_aa_distributions(
                pssm, design_mask, device, bk_dist=bk_dist[0], restricted_mask=restricted_mask[0])

        loss_components.append(
            (loss_weights.loss_weight_kl_res,
             KLDivLoss(
                 background_distributions=bk_dist,
                 device=device,
                 masks=restricted_mask,
                 maximize=False,
                 loss_candidate='seq')))

    if loss_weights.loss_weight_kl_res > 0 and (restricted_dict_keep_aas != {}):
        loss_index += 1
        loss_components_dict['kl_res'] = loss_index
        bk_dist = []
        restricted_mask = []
        if restricted_dict_keep_aas != {}:
            bk_dist = get_restricted_aa_distributions(
                        len(seq_for_hal), restricted_dict_keep_aas)
            restricted_mask = get_restricted_seq_aa_mask(len(seq_for_hal),
                                                    restricted_dict_keep_aas,
                                                    device)

        if not pssm is None:
            bk_dist, restricted_mask = get_cce_aa_distributions(
                pssm,
                design_mask,
                device,
                bk_dist=bk_dist[0],
                restricted_mask=restricted_mask[0])

        loss_components.append(
            (loss_weights.loss_weight_kl_res,
             KLDivLoss(
                 background_distributions=bk_dist,
                 device=device,
                 masks=restricted_mask,
                 maximize=False,
                 loss_candidate='seq')))

    if loss_weights.loss_weight_seq > 0:
        loss_index += 1
        loss_components_dict['seq'] = loss_index

        if non_design_mask is None:
            non_design_mask = torch.ones(design_mask.shape)
            non_design_mask[design_mask==1] = 0

        loss_components.append((loss_weights.loss_weight_seq,
                                SequenceLoss(seq_for_hal, device=device, mask=non_design_mask)))

    if loss_weights.loss_weight_geom > 0:
        loss_index += 1
        loss_components_dict['geom'] = loss_index

        mask_2d = design_mask.expand((len(seq_for_hal), len(seq_for_hal)))
        mask_2d = mask_2d | mask_2d.transpose(0, 1)
        plt.imshow(mask_2d, aspect='equal')
        plt.colorbar()
        plt.savefig('{}/geom_design_mask.png'.format(outdir))
        plt.close()
        
        if disable_nondesign_mask:
            masks = None
        elif wt_losses_mask != []:
            masks = []
            i_mask=0 #just use ca
            mask_2d_i = torch.tensor(mask_2d)
            losses_mask = (design_mask | wt_losses_mask[i_mask].cpu().squeeze(0))
            losses_mask_2d = losses_mask.expand((len(seq_for_hal), len(seq_for_hal)))
            mask_2d_i[losses_mask_2d==0] = 0
            mask_2d_i[losses_mask_2d.transpose(1,0)==0] = 0
            masks = [mask_2d_i for _ in target_geometries]
        else:
            masks = [mask_2d for _ in target_geometries]

        loss_components.append(
        (loss_weights.loss_weight_geom,
         GeometryLoss(target_geometries,
                      geometry_weights=loss_weights.geometric_loss_split,
                      device=device,
                      masks=masks,
                      local=local_loss_only
                      )
                      )
        )

    if loss_weights.loss_weight_seq_reg > 0:
        loss_index += 1
        loss_components_dict['reg_seq'] = loss_index

        from params import oas_heavy_seq_dist, oas_light_seq_dist
        h_mask = torch.zeros(len(seq_for_hal))
        h_mask[:len_heavy_seq] = 1
        l_mask = torch.zeros(len(seq_for_hal))
        l_mask[len_heavy_seq:] = 1

        loss_components.append(
            (loss_weights.loss_weight_seq_reg,
             SequenceDistributionLoss(oas_heavy_seq_dist,
                                      add_noise=True,
                                      mask=h_mask,
                                      device=device
                                      )))
        loss_index += 1
        loss_components.append(
            (loss_weights.loss_weight_seq_reg,
             SequenceDistributionLoss(oas_light_seq_dist,
                                      add_noise=True,
                                      mask=l_mask,
                                      device=device)))

    if loss_weights.loss_weight_netcharge > 0:
        loss_index += 1
        loss_components_dict['netcharge'] = loss_index 

        loss_components.append((loss_weights.loss_weight_netcharge,
                                TotalChargeLoss(len(seq_for_hal),
                                                mask=design_mask,
                                                device=device)))        

    if loss_weights.loss_aa_freq > 0:
        loss_index += 1
        loss_components_dict['max_aa_freq'] = loss_index

        loss_components.append((loss_weights.loss_aa_freq,
                                TotalAAFrequencyLoss(
                                                mask=design_mask,
                                                device=device)))        


    return loss_components, loss_components_dict

