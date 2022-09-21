from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F

from src.util.util import _aa_dict, letter_to_num, one_hot_seq
from src.hallucination.loss.LossComponent import LossComponent
from src.hallucination.utils.util import get_loss


class SequenceHallucinator(torch.nn.Module):
    def __init__(self,
                 init_sequence: str,
                 delimiter_position: int,
                 forward_model: torch.nn.Module,
                 loss_components: List[Tuple[int, LossComponent]],
                 design_mask: Optional[torch.LongTensor] = None,
                 device: Optional[torch.device] = None,
                 sequence_seed: Optional[str] = None,
                 apply_lr_scheduler: Optional[bool] = False,
                 lr_config: Optional[Dict] = {'learning_rate': 0.05, 'patience': 20, 'cooldown': 10},
                 initialize_fullseq_to_random: Optional[bool] = False,
                 disallowed_aas_at_initialization='CP'):
        super().__init__()
        
        sequence_len = len(init_sequence)
        self.delimiter_position = delimiter_position

        self.heavy_chain_delimiter = torch.zeros((1, sequence_len))
        self.heavy_chain_delimiter[0, self.delimiter_position] = 1

        if design_mask is None:
            self.restrict_mask = torch.zeros(sequence_len).bool()
        else:
            self.restrict_mask = (~(design_mask == 1)).bool()

        sequence = torch.rand((len(init_sequence), 20))
        sequence[:, letter_to_num(disallowed_aas_at_initialization, _aa_dict)] = 0.0
        # initialize non-design residues from init sequence
        sequence[self.restrict_mask] = one_hot_seq(
                init_sequence).float()[self.restrict_mask]
        
        if sequence_seed is not None:
            # seed sequence has slightly higher prob than uniform at design positions
            sequence[design_mask == 1] = one_hot_seq(sequence_seed).float()[
                design_mask == 1] * 0.5 + sequence[design_mask == 1]

        if device != None:
            self.heavy_chain_delimiter = self.heavy_chain_delimiter.to(device)
            self.restrict_mask = self.restrict_mask.to(device)
            sequence = sequence.to(device)

        self.sequence = torch.nn.Parameter(sequence)
        self.sequence_history = [self.get_sequence()]

        self.forward_model = forward_model

        self.optimizer = torch.optim.SGD([self.sequence],
                                         lr=lr_config['learning_rate'],
                                         weight_decay=0.01)

        self.pssm = None
        self.sequence_grad = None
        self.sequence_grad_prenorm = None

        self.apply_lr_scheduler = apply_lr_scheduler
        self.start_lr = lr_config['learning_rate']
        self.lr = lr_config['learning_rate']

        if self.apply_lr_scheduler:
            
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                verbose=True,
                threshold=0.1,
                patience=lr_config['patience'],
                cooldown=lr_config['cooldown'])

        self.loss_components = loss_components

    def get_sequence(self):
        aa_nums = self.sequence.argmax(dim=-1)
        aa_dict = {int(num): aa for aa, num in _aa_dict.items()}
        aa_sequence = [aa_dict[num.item()] for num in aa_nums]
        aa_sequence = "".join(aa_sequence)

        heavy, light = aa_sequence[:self.delimiter_position +
                                   1], aa_sequence[self.delimiter_position +
                                                   1:]

        return heavy, light

    def update_sequence(self, disallow_letters='C'):
        self.optimizer.zero_grad()

        sm_sequence = F.softmax(self.sequence, dim=-1)
        oh_sequence = F.one_hot(sm_sequence.argmax(dim=-1),
                                num_classes=20).float()
        oh_sequence = oh_sequence - sm_sequence.detach() + sm_sequence
        
        total_loss, loss_weight_list = get_loss(self.forward_model,self.loss_components,\
                                        self.heavy_chain_delimiter,oh_sequence,sm_sequence)

        total_loss.backward()
        self.sequence_grad_prenorm = self.sequence.grad.detach()
        self.sequence.grad = F.normalize(self.sequence.grad, dim=-1)
        #zero out gradient for non-design residues
        self.sequence.grad[self.restrict_mask] = 0.
        
        self.sequence.grad[:, letter_to_num(disallow_letters, _aa_dict)] = 0.
        self.optimizer.step()
        self.sequence_grad = self.sequence.grad.detach()
        self.pssm = sm_sequence.detach()

        # Pass the loss to scheduler - after optimizer step
        if self.apply_lr_scheduler:
            self.scheduler.step(total_loss)
            for param_group in self.optimizer.param_groups:
                self.lr = param_group['lr']

        self.sequence_history.append(self.get_sequence())
        
        return [
            weight * torch.tensor(losses)
            for losses, weight in loss_weight_list
        ]

    def write_sequence_history_file(self, out_fasta, outfile_pssm=None, outfile_grad=None):
        with open(out_fasta, "w") as f:
            for i, (heavy, light) in enumerate(self.sequence_history):
                f.write(">HS{0}:H\n{1}\n>LS{0}:L\n{2}\n\n".format(
                    i, heavy, light))
        if not outfile_pssm is None:
            import numpy as np
            last_mats = self.pssm.cpu().numpy()
            np.save(outfile_pssm, last_mats)
        if not outfile_grad is None:
            import numpy as np
            last_mats = self.sequence_grad_prenorm.cpu().numpy()
            np.save(outfile_grad, last_mats)

    def update_loss_components(self, loss_index, loss_component):
        self.loss_components[loss_index] = loss_component
