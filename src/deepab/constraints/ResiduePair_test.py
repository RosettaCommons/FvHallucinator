from .ConstraintType import ConstraintType
from .Constraint import Constraint
from .ResiduePair import ResiduePair
from .Residue import Residue


def get_constraint_residue_pair():
    residue_1 = Residue(index=100, identity="G")
    residue_2 = Residue(index=101, identity="A")

    x_vals = [0, 0.5, 1, 1.5, 2]
    y_vals = [0.4, 0.1, 0.0, 0.2, 0.3]
    constraint1 = Constraint(constraint_type=ConstraintType.omega_dihedral,
                             residue_1=residue_1,
                             residue_2=residue_2,
                             x_vals=x_vals,
                             y_vals=y_vals)
    constraint2 = Constraint(constraint_type=ConstraintType.cb_distance,
                             residue_1=residue_1,
                             residue_2=residue_2,
                             x_vals=x_vals,
                             y_vals=y_vals[::-1])

    constraint_residue_pair = ResiduePair(
        residue_1=residue_1,
        residue_2=residue_2,
        constraints=[constraint1, constraint2])

    return constraint_residue_pair


def test_get_constraints():
    constraint_residue_pair = get_constraint_residue_pair()

    assert len(constraint_residue_pair.get_constraints(modal_x_min=1)) == 1
    assert len(constraint_residue_pair.get_constraints(modal_x_max=1)) == 1

    assert len(constraint_residue_pair.get_constraints(average_x_min=0.5)) == 2
    assert len(constraint_residue_pair.get_constraints(average_x_max=0.5)) == 0

    assert len(
        constraint_residue_pair.get_constraints(custom_filters=[
            lambda residue_pair, constraint: constraint.constraint_type ==
            ConstraintType.cb_distance
        ])) == 1
