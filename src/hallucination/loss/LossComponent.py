import torch
#import numpy as np
from src.util.model_out import generate_probabilities
from src.util.util import _aa_dict, letter_to_num

from .LossType import LossType

class LossComponent():
    def __init__(self, loss_type):
        self.loss_type = loss_type
        super().__init__()

    def calc_loss(self, output):
        exit("Invalid: generic loss component type")


class KLDivLoss(LossComponent):
    def __init__(self,
                 masks=None,
                 background_distributions=None,
                 device=None,
                 maximize=True,
                 loss_candidate='geom',
                 pre_argmax=False):
        super().__init__(LossType.kl_div_loss)

        self.loss_function = torch.nn.KLDivLoss(reduction="mean")

        self.masks = masks
        self.device = device

        self.background_distributions = background_distributions

        if device != None and type(background_distributions) != type(None):
            self.background_distributions = [
                dist.to(device) for dist in self.background_distributions
            ]

        # Maximize or minimize KL divergence loss
        self.prefactor = -1.0 if maximize else 1.0

        self.loss_candidate = loss_candidate
        self.pre_argmax = pre_argmax

    def calc_loss(self, candidate_distributions):
        if type(self.background_distributions) == type(None):
            self.background_distributions = []
            for dist in candidate_distributions:
                self.background_distributions.append(
                    0.05 * torch.ones(dist.shape[::-1]))

            if self.device != None:
                self.background_distributions = [
                    dist.to(self.device)
                    for dist in self.background_distributions
                ]

        if type(self.masks) == type(None):
            self.masks = [
                torch.ones(list(dist.shape)[1:])
                for dist in candidate_distributions
            ]

            if self.device != None:
                self.masks = [mask.to(self.device) for mask in self.masks]

        assert len(candidate_distributions) == len(self.masks)

        log_probs = [
            torch.log(generate_probabilities(dist) + 1e-8)
            for dist in candidate_distributions
        ]
        
        losses = [
            self.prefactor * self.loss_function(
                log_prob[mask == 1], background_distribution[mask == 1])
            for log_prob, background_distribution, mask in zip(
                log_probs, self.background_distributions, self.masks)
        ]

        return losses


class SequenceLoss(LossComponent):
    def __init__(self,
                 target_sequence,
                 mask=None,
                 device=None,
                 ignore_index=-999):
        super().__init__(LossType.candidate_sequence_loss)

        self.loss_function = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index)

        self.target_sequence = target_sequence
        self.target_sequence_label = torch.Tensor([
            int(_aa_dict[aa] if aa in _aa_dict else ignore_index)
            for aa in target_sequence
        ]).long()

        if type(mask) != type(None):
            self.target_sequence_label[mask == 0] = ignore_index

        if device != None:
            self.target_sequence_label = self.target_sequence_label.to(device)

    def calc_loss(self, sequence_distribution):
        loss = self.loss_function(sequence_distribution,
                                  self.target_sequence_label)

        return [loss]


class SequenceDistributionLoss(LossComponent):
    def __init__(self,
                 label_distribution,
                 add_noise=True,
                 pre_argmax=False,
                 mask=None,
                 device=None,
                 mean_dist=True):
        super().__init__(LossType.candidate_sequence_loss)

        self.loss_function = torch.nn.KLDivLoss(reduction="mean")

        self.label_distribution = label_distribution
        self.add_noise = add_noise
        self.pre_argmax = pre_argmax
        self.mask = mask
        self.mean_dist = mean_dist

        if device != None:
            self.label_distribution = self.label_distribution.to(device)

    def calc_loss(self, sequence_distribution):

        if self.mask is not None:
            sequence_distribution = sequence_distribution[self.mask == 1]
            if not self.mean_dist:
                self.label_distribution = self.label_distribution[self.mask ==
                                                                  1]
        
        if self.pre_argmax:
            if self.mean_dist:
                sequence_distribution = sequence_distribution.mean(0)
                sequence_distribution[sequence_distribution==0.0] = 0.025
            else:
                sequence_distribution = sequence_distribution
        else:
            oh_sequence = torch.nn.functional.one_hot(
                sequence_distribution.argmax(dim=-1), num_classes=20)
            oh_sequence = oh_sequence - sequence_distribution.detach(
            ) + sequence_distribution
            if self.mean_dist:
                sequence_distribution = oh_sequence.mean(0)
                #print('post argmax: ', sequence_distribution)
                # set zeros to a small value
                sequence_distribution[sequence_distribution==0.0] = 0.025
            else:
                sequence_distribution = oh_sequence
        
        if self.add_noise:
            sequence_distribution += torch.abs(
                torch.randn_like(sequence_distribution) * 1e-3).float()
            sequence_distribution /= sequence_distribution.sum()

        loss = self.loss_function(torch.log(sequence_distribution),
                                  self.label_distribution)
        
        return [loss]

class GeometryLoss(LossComponent):
    def __init__(self,
                 target_geometries,
                 geometry_weights=None,
                 masks=None,
                 device=None,
                 ignore_index=-999,
                 reduction='mean',
                 local=True,
                 distance_bin_cutoff=20):
        super().__init__(LossType.candidate_geometry_loss)

        self.loss_function = torch.nn.CrossEntropyLoss(
                ignore_index=ignore_index, reduction=reduction)
        self.target_geometries = target_geometries

        if masks != None:
            if len(target_geometries) == len(masks):
                for i, mask in enumerate(masks):
                    self.target_geometries[i][mask == 0] = ignore_index
            else:
                for i in range(len(target_geometries)):
                    self.target_geometries[i][masks == 0] = ignore_index

        if local:
            local_mask = target_geometries[0] >= distance_bin_cutoff
            for i in range(len(target_geometries)):
                self.target_geometries[i][local_mask] = ignore_index

        if device != None:
            for i in range(len(self.target_geometries)):
                self.target_geometries[i] = self.target_geometries[i].to(
                    device)
        
        if geometry_weights == None:
            geometry_weights = [1] * len(target_geometries)
        self.geometry_weights = geometry_weights

    def calc_loss(self, candidate_geometry_distributions):

        losses = [
            weight *
            self.loss_function(output.unsqueeze(0), label.unsqueeze(0))
            for output, label, weight in zip(candidate_geometry_distributions,
                                             self.target_geometries,
                                             self.geometry_weights)
        ]

        return losses

class TotalChargeLoss(LossComponent):
    def __init__(self,
                 seq_len,
                 mask=None,
                 device=None,
                 max_sequence_charge=2.001,
                 debug=False):
        super().__init__(LossType.candidate_sequence_loss)

        charge_dict = {'K': 1, 'R': 1, 'D': -1, 'E': -1, 'H': 0.1}
        self.loss_function = torch.nn.L1Loss(reduction="mean")
        self.mask = mask
        self.charge_tensor = torch.zeros((1, 20))
        for key in charge_dict:
            self.charge_tensor[:,letter_to_num(key, _aa_dict)] = charge_dict[key]
        self.charge_tensor = self.charge_tensor.expand(seq_len, -1)

        self.max_sequence_charge = torch.tensor(max_sequence_charge)

        if not device is None:
            self.charge_tensor = self.charge_tensor.to(device)
            self.max_sequence_charge = self.max_sequence_charge.to(device)
            if not self.mask is None:
                self.mask = self.mask.to(device)
                self.mask = self.mask.unsqueeze(-1).expand(-1, self.charge_tensor.shape[1])
                self.charge_tensor = self.charge_tensor * self.mask

        if not self.mask is None:
            if debug:
                import matplotlib.pyplot as plt
                plt.imshow(self.charge_tensor.detach().cpu().numpy(),
                           vmin=-1, vmax=1, cmap='vlag_r', aspect='auto')
                plt.xticks([t for t in range(20)], [key for key in _aa_dict])
                plt.colorbar()
                plt.savefig('masked_charge.png', dpi=600)
                plt.close()
                
        
    def calc_loss(self, sequence_distribution):

        oh_sequence = torch.nn.functional.one_hot(
            sequence_distribution.argmax(dim=-1), num_classes=20)
        oh_sequence = oh_sequence - sequence_distribution.detach(
        ) + sequence_distribution

        total_sequence_charge = oh_sequence * self.charge_tensor
        total_sequence_charge = torch.sum(total_sequence_charge.flatten(), 0)
        
        loss = self.loss_function(total_sequence_charge,
                                  self.max_sequence_charge)

        if total_sequence_charge.detach().item() <= self.max_sequence_charge.item():
            loss = loss * 0
        
        return [loss]

class TotalAAFrequencyLoss(LossComponent):
    def __init__(self,
                 mask=None,
                 device=None,
                 max_aa_freq=4):
        super().__init__(LossType.candidate_sequence_loss)

        self.loss_function = torch.nn.L1Loss(reduction="none")
        self.mask = mask
        
        self.max_aa_freq_tensor = torch.full((1, 20), max_aa_freq)
        self.max_aa_freq_value = max_aa_freq
        
        if not device is None:
            self.max_aa_freq_tensor = self.max_aa_freq_tensor.to(device)
        
    def calc_loss(self, sequence_distribution):

        if self.mask is not None:
            sequence_distribution = sequence_distribution[self.mask == 1]
            
        oh_sequence = torch.nn.functional.one_hot(
            sequence_distribution.argmax(dim=-1), num_classes=20)
        oh_sequence = oh_sequence - sequence_distribution.detach(
        ) + sequence_distribution

        candidate_aa_freq = oh_sequence.sum(0)
        
        loss = self.loss_function(candidate_aa_freq,
                                  self.max_aa_freq_tensor).squeeze(0)
        loss[candidate_aa_freq <= self.max_aa_freq_value] = 0.0
        loss = loss.sum(0)
        
        return [loss]
