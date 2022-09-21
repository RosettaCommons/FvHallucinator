from pytest import approx

from .ConstraintType import ConstraintType
from .Constraint import Constraint
from .Residue import Residue


def test_constraint_init():
    x_vals = [0, 0.5, 1, 1.5, 2]
    y_vals = [0.4, 0.1, 0.0, 0.2, 0.3]

    residue_1 = Residue(index=100, identity="G")
    residue_2 = Residue(index=101, identity="A")

    constraint = Constraint(constraint_type=ConstraintType.cb_distance,
                            residue_1=residue_1,
                            residue_2=residue_2,
                            x_vals=x_vals,
                            y_vals=y_vals)

    assert constraint.modal_x == approx(0)
    assert constraint.modal_y == approx(0.4)

    assert constraint.average_x == approx(1)
    assert constraint.average_y == approx(0)