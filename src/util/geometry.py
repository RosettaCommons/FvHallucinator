import math
import random
import torch


def calc_dist_mat(a_coords, b_coords):
    assert a_coords.shape == b_coords.shape
    mat_shape = (len(a_coords), len(a_coords), 3)

    a_coords = a_coords.unsqueeze(0).expand(mat_shape)
    b_coords = b_coords.unsqueeze(1).expand(mat_shape)

    dist_mat = (a_coords - b_coords).norm(dim=-1)

    return dist_mat


def calc_dihedral(a_coords,
                  b_coords,
                  c_coords,
                  d_coords,
                  convert_to_degree=False):
    b1 = a_coords - b_coords
    b2 = b_coords - c_coords
    b3 = c_coords - d_coords

    n1 = torch.cross(b1, b2)
    n1 = torch.div(n1, n1.norm(dim=-1, keepdim=True))
    n2 = torch.cross(b2, b3)
    n2 = torch.div(n2, n2.norm(dim=-1, keepdim=True))
    m1 = torch.cross(n1, torch.div(b2, b2.norm(dim=-1, keepdim=True)))

    dihedral = torch.atan2((m1 * n2).sum(-1), (n1 * n2).sum(-1))

    if convert_to_degree:
        dihedral = dihedral * 180 / math.pi

    return dihedral


def calc_planar(a_coords, b_coords, c_coords, convert_to_degree=False):
    v1 = a_coords - b_coords
    v2 = c_coords - b_coords

    a = (v1 * v2).sum(-1)
    b = v1.norm(dim=-1) * v2.norm(dim=-1)

    planar = torch.acos(a / b)

    if convert_to_degree:
        planar = planar * 180 / math.pi

    return planar


def get_masked_mat(input_mat, mask, mask_fill_value=-999, device=None):
    out_mat = torch.ones(input_mat.shape)
    if device is not None:
        mask = mask.to(device)
        out_mat = out_mat.to(device)

    out_mat[mask == 0] = mask_fill_value
    out_mat[mask == 1] = input_mat[mask == 1]

    return out_mat
