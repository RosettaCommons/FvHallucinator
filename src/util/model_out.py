import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio import SeqIO
import numpy as np

from src.util.get_bins import get_bin_values
from src.util.util import load_full_seq, one_hot_seq


def bin_matrix(in_tensor, are_logits=True, method='max'):
    """
    Bins a 3d tensor of shape (logits, N, N). This assumes that the channels
    are logits to generate probabilities from.

    :param in_tensor: The tensor to bin.
    :type in_tensor: torch.Tensor
    :param are_logits:
        Whether or not the tensor consists of logits to turn into
        probabilities. If not, they are assumed to be probabilities.
    :param method:
        The binning method. Can either be 'max' or 'average'. 'max' will
        assign an element to the bin with the highest probability and 'average'
        will assign an element to the weighted average of the bins
    :return:
    """
    if are_logits:
        probs = generate_probabilities(in_tensor)
    else:
        probs = in_tensor
    if method == 'max':
        # Predict the bins with the highest probability
        return probs.max(len(probs.shape) - 1)[1]
    elif method == 'avg':
        # Predict the bin that is closest to the average of the probability dist
        # predicted_bins[i][j] = round(sum(bin_index * P(bin_index at i,j)))
        bin_indices = torch.arange(probs.shape[-1]).float()
        predicted_bins = torch.round(
            torch.sum(probs.mul(bin_indices), dim=len(probs.shape) - 1))
        return predicted_bins
    else:
        raise ValueError('method must be in {\'avg\',\'max\'}')


def get_inputs_from_fasta(fasta_file):
    seq = one_hot_seq(load_full_seq(fasta_file)).float()

    # Add chain delimiter
    seq = nn.functional.pad(seq, (0, 1, 0, 0))
    h_len = 0
    for chain in SeqIO.parse(fasta_file, 'fasta'):
        if ':H' in chain.id:
            h_len = len(chain.seq)
    if h_len == 0:
        raise ValueError('No heavy chain detected. Cannot add chain delimiter')
    seq[h_len - 1, seq.shape[1] - 1] = 1

    seq = seq.unsqueeze(0).transpose(1, 2)

    return seq


def get_logits_from_model(model, fasta_file, device=None):
    """Gets the probability distribution output of a H3ResNet model"""
    seq = get_inputs_from_fasta(fasta_file)
    if not device == None:
        seq = seq.to(device)

    with torch.no_grad():
        out = model(seq)
        # Remove batch dim from output tensors
        out = [torch.squeeze(o, dim=0) for o in out]
        return out


def generate_probabilities(logits):
    """Transforms a tensor of logits of shape (logits, L, [L...]) to probabilities"""
    if len(logits.shape) < 2:
        raise ValueError(
            'Expected a shape with at least dimensions (channels, L, [L...]), got {}'
            .format(logits.shape))

    # For 2D output: Transform from [channels, L_i, L_j] to [L_i, L_j, channels]
    # For 1D output: Transform from [channels, L_i] to [L_i, channels]
    for i in range(0, len(logits.size()) - 1):
        logits = logits.transpose(i, i + 1)

    # Get the probabilities of each bin at each position and predict the bins
    return nn.Softmax(dim=-1)(logits)


def get_probs_from_model(model, fasta_file, **kwargs):
    """Gets the probability distribution output of a H3ResNet model"""
    logits_list = get_logits_from_model(model, fasta_file, **kwargs)
    probs = []
    for logits in logits_list:
        probs.append(generate_probabilities(logits))

    return probs


def binned_mat_to_values(binned_mat, bins):
    bin_values = get_bin_values(bins)
    value_mat = torch.zeros(binned_mat.shape)
    for i in range(value_mat.shape[0]):
        if len(binned_mat.shape) == 1:
            if binned_mat[i] < len(bin_values):
                value = bin_values[binned_mat[i]]
            else:
                value = torch.tensor(np.nan)
        else:
            value = binned_mat_to_values(binned_mat[i], bins)

        value_mat[i] = value

    value_mat[binned_mat == -999] = -999

    return value_mat
