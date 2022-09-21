import torch


def bin_value_matrix(value_mat, bins):

    # Note: Checking lower_bin <= x <= upper_bin to handle case where
    # x == upper_bin.  Values where x == upper_bin(i) == lower_bin(i+1)
    # be placed in the (i+1) bin
    binned_matrix = value_mat.clone().detach().long()
    for i, (lower_bound, upper_bound) in enumerate(bins):
        bin_mask = (value_mat >= lower_bound).__and__(value_mat <= upper_bound)
        binned_matrix[bin_mask] = i

    return binned_matrix


def bin_dist_angle_matrix(dist_angle_mat, dist_bins, omega_bins, theta_bins,
                          phi_bins):

    # Note: Checking lower_bin <= x <= upper_bin to handle case where
    # x == upper_bin.  Values where x == upper_bin(i) == lower_bin(i+1)
    # be placed in the (i+1) bin
    binned_matrix = dist_angle_mat.clone().detach().long()
    for i, (lower_bound, upper_bound) in enumerate(dist_bins):
        bin_mask = (dist_angle_mat[0] >= lower_bound).__and__(
            dist_angle_mat[0] <= upper_bound)
        binned_matrix[0][bin_mask] = i
    for i, (lower_bound, upper_bound) in enumerate(omega_bins):
        bin_mask = (dist_angle_mat[1] >= lower_bound).__and__(
            dist_angle_mat[1] <= upper_bound)
        binned_matrix[1][bin_mask] = i
    for i, (lower_bound, upper_bound) in enumerate(theta_bins):
        bin_mask = (dist_angle_mat[2] >= lower_bound).__and__(
            dist_angle_mat[2] <= upper_bound)
        binned_matrix[2][bin_mask] = i
    for i, (lower_bound, upper_bound) in enumerate(phi_bins):
        bin_mask = (dist_angle_mat[3] >= lower_bound).__and__(
            dist_angle_mat[3] <= upper_bound)
        binned_matrix[3][bin_mask] = i

    return binned_matrix


def bin_phi_psi_matrix(phi_psi_mat, phi_psi_bins):
    # Note: Checking lower_bin <= x <= upper_bin to handle case where
    # x == upper_bin.  Values where x == upper_bin(i) == lower_bin(i+1)
    # be placed in the (i+1) bin
    binned_matrix = phi_psi_mat.clone().detach().long()
    for i, (lower_bound, upper_bound) in enumerate(phi_psi_bins):
        bin_mask = (phi_psi_mat >= lower_bound).__and__(
            phi_psi_mat <= upper_bound)
        binned_matrix[bin_mask] = i

    return binned_matrix