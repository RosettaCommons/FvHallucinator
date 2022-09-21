import enum

class LossType(enum.Enum):
    kl_div_loss = 1
    candidate_sequence_loss = 2
    candidate_geometry_loss = 3