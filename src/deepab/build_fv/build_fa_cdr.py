import os
import torch
import pyrosetta

from .score_functions import get_sf_fa
from .utils import get_constraint_set_mover
from src.deepab.constraints import ConstraintType, get_constraint_residue_pairs, get_filtered_constraint_file
from src.deepab.constraints.custom_filters import hb_dist_filter
from src.util.util import get_heavy_seq_len

pyrosetta.init('--mute all')

def get_cst_file(model: torch.nn.Module, fasta_file: str,
                 constraint_dir: str) -> str:
    """
    Generate constraint files for Fv builder
    """

    heavy_seq_len = get_heavy_seq_len(fasta_file)
    residue_pairs = get_constraint_residue_pairs(model, fasta_file,
                                                 heavy_seq_len)

    all_cst_file = get_filtered_constraint_file(
        residue_pairs=residue_pairs,
        constraint_dir=os.path.join(constraint_dir, "all_csm"),
        local=True,
        threshold=0.1,
        constraint_types=[
            ConstraintType.cb_distance, ConstraintType.ca_distance,
            ConstraintType.omega_dihedral, ConstraintType.theta_dihedral,
            ConstraintType.phi_planar
        ])
    hb_cst_file = get_filtered_constraint_file(
        residue_pairs=residue_pairs,
        constraint_dir=os.path.join(constraint_dir, "hb_csm"),
        threshold=0.1,
        constraint_types=[ConstraintType.no_distance],
        constraint_filters=[hb_dist_filter])

    os.system("cat {} >> {}".format(all_cst_file, hb_cst_file))

    return hb_cst_file


def get_fastrelax_mover(
        relax_max_iter: int = 400,
        min_max_iter: int = 1000) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create full-atom relax mover
    """

    sf_fa = get_sf_fa()
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 1)
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0)

    mmap = pyrosetta.rosetta.core.kinematics.MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf_fa)
    relax.max_iter(relax_max_iter)
    relax.cartesian(True)
    relax.set_movemap(mmap)
    relax.ramp_down_constraints(True)

    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf_fa, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover.max_iter(min_max_iter)
    min_mover.cartesian(True)

    seq_mover = pyrosetta.rosetta.protocols.moves.SequenceMover()
    seq_mover.add_mover(relax)
    seq_mover.add_mover(min_mover)

    rep_mover = pyrosetta.rosetta.protocols.moves.RepeatMover(seq_mover, 3)

    return rep_mover


def get_cdr_refine_mover(
        pose: pyrosetta.Pose,
        constraint_scale: float = 0.05,
        relax_max_iter: int = 400,
        min_max_iter: int = 1000) -> pyrosetta.rosetta.protocols.moves.Mover:
    """
    Create CDR + H/L docking relax, minimization mover
    """

    sf_fa = get_sf_fa(constraint_scale=constraint_scale)
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.cart_bonded, 1)
    sf_fa.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.pro_close, 0)

    ab_info = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose)
    cdr_loops = ab_info.get_CDR_loops(pose, overhang=1)
    mmap = ab_info.get_MoveMap_for_LoopsandDock(pose, cdr_loops)

    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf_fa)
    relax.max_iter(relax_max_iter)
    relax.cartesian(True)
    relax.set_movemap(mmap)
    relax.ramp_down_constraints(True)

    min_mover = pyrosetta.rosetta.protocols.minimization_packing.MinMover(
        mmap, sf_fa, 'lbfgs_armijo_nonmonotone', 0.0001, True)
    min_mover.max_iter(min_max_iter)
    min_mover.cartesian(True)

    seq_mover = pyrosetta.rosetta.protocols.moves.SequenceMover()
    seq_mover.add_mover(relax)
    seq_mover.add_mover(min_mover)
    rep_mover = pyrosetta.rosetta.protocols.moves.RepeatMover(seq_mover, 3)

    return rep_mover


def refine_fv(in_pdb_file: str,
              out_pdb_file: str,
              cst_file: str,
              refine_constraint_scale: float = 0.05) -> float:
    """
    Run constrained relax protocol on initial pdb file and return final score
    """

    ############################################################################
    # Load initial pose
    ############################################################################
    pose = pyrosetta.pose_from_pdb(in_pdb_file)

    csm = get_constraint_set_mover(cst_file)
    csm.apply(pose)

    ############################################################################
    # Apply full atom relax/refine loop
    ############################################################################
    fastrelax_mover = get_fastrelax_mover()
    fastrelax_mover.apply(pose)

    ############################################################################
    # Apply full atom CDR/VH-VL refinement
    ############################################################################
    refine_mover = get_cdr_refine_mover(
        pose, constraint_scale=refine_constraint_scale)
    refine_mover.apply(pose)

    pose.dump_pdb(out_pdb_file)

    sf_fa_cst = get_sf_fa(constraint_scale=refine_constraint_scale)
    score = sf_fa_cst(pose)

    return score
