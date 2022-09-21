import enum


class ConstraintType(enum.Enum):
    cbca_distance = 1
    ca_distance = 2
    cb_distance = 3
    no_distance = 4
    omega_dihedral = 5
    theta_dihedral = 6
    phi_planar = 7
    bb_phi_dihedral = 8
    bb_psi_dihedral = 9
