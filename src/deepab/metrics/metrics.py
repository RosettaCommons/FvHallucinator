import torch
import numpy as np
import math
from astropy.stats import circcorrcoef
import matplotlib.pyplot as plt

def calc_binned_mat_likelihood(probs, binned_mat, mask=None, fig_name=None):
    # assert len(probs.shape) == 3
    # assert len(binned_mat.shape) == 2

    binned_mat = binned_mat.unsqueeze(-1)
    # Must remove -999 values, mask should account for them
    binned_mat[binned_mat < 0] = 0
    likelihood_mat = probs.gather(-1, binned_mat)

    if type(mask) == type(None):
        mask = torch.ones(likelihood_mat.shape)

    if not fig_name == None:
        nan_likelihood_mat = likelihood_mat.squeeze(2).float().numpy()
        nan_likelihood_mat[mask.numpy() == 0] = np.nan

        plt.imshow(nan_likelihood_mat)
        plt.colorbar()
        plt.savefig(fig_name)
        plt.close()

    mean_likelihood = likelihood_mat[mask == 1].mean()

    return mean_likelihood


def calc_matrix_element_correlation(mat_1,
                                    mat_2,
                                    mask=None,
                                    circular=False,
                                    fig_name=None,
                                    fig_bins=20,
                                    fig_bounds=None):
    assert mat_1.shape == mat_2.shape

    if type(mask) == type(None):
        mask = torch.ones(mat_1.shape)

    mat_1 = mat_1[mask == 1].numpy()
    mat_2 = mat_2[mask == 1].numpy()

    if circular:
        corr = circcorrcoef(mat_1 * np.pi / 180, mat_2 * np.pi / 180).item()
    else:
        corr = np.corrcoef(mat_1, mat_2)[0, 1]

    if not fig_name == None:
        if fig_bounds == None:
            fig_bounds = [
                min(mat_1.min(), mat_2.min()),
                max(mat_1.max(), mat_2.max())
            ]

        plt.hist2d(mat_1, mat_2, bins=fig_bins, range=(fig_bounds, fig_bounds))
        plt.colorbar()
        plt.annotate("n = {}\nR = {}".format(len(mat_1), round(corr, 3)),
                     xy=(0, 0),
                     xycoords='axes fraction',
                     color='w')
        plt.savefig(fig_name)
        plt.close()

    return corr


def calc_matrix_element_error(pred_mat,
                              native_mat,
                              mask=None,
                              angle=False,
                              fig_name=None,
                              probs=None):
    assert pred_mat.shape == native_mat.shape
    if not probs is None:
        assert len(probs.shape) == len(pred_mat.shape) + 1

    if type(mask) == type(None):
        mask = torch.ones(pred_mat.shape)

    pred_mat = pred_mat[mask == 1].numpy()
    native_mat = native_mat[mask == 1].numpy()

    error = pred_mat - native_mat

    if angle:
        error = (error + 180) % 360 - 180
        error *= np.pi / 180

    error = np.abs(error)

    if not (fig_name is None or probs is None):
        max_probs = probs.max(dim=-1)[0]
        max_probs = max_probs[mask == 1].numpy()

        plt.scatter(max_probs, error)
        plt.xlim([0, 1])
        plt.xlabel("Modal Probability")
        plt.ylabel("Error")
        plt.savefig(fig_name)
        plt.close()

    mean_error = error.mean().item()

    return mean_error


def calc_element_error_by_dist(pred_mat,
                               native_mat,
                               dist_mat,
                               mask=None,
                               angle=False,
                               fig_name=None,
                               fig_bins=20,
                               dist_bounds=None):
    assert pred_mat.shape == native_mat.shape
    assert pred_mat.shape == dist_mat.shape

    if type(mask) == type(None):
        mask = torch.ones(pred_mat.shape)

    pred_mat = pred_mat[mask == 1].numpy()
    native_mat = native_mat[mask == 1].numpy()
    dist_mat = dist_mat[mask == 1].numpy()

    error = pred_mat - native_mat

    if angle:
        error = (error + 180) % 360 - 180
        error *= np.pi / 180

    if not fig_name is None:
        if dist_bounds == None:
            dist_bounds = [error.min(), error.max()]

        plt.hist2d(dist_mat,
                   error,
                   bins=fig_bins,
                   range=(dist_bounds, [error.min(), error.max()]))
        plt.colorbar()
        plt.savefig(fig_name)
        plt.close()

    dist_error_mat = np.concatenate(
        (dist_mat.reshape(len(dist_mat), 1), error.reshape(len(error), 1)),
        axis=1)

    return dist_error_mat
