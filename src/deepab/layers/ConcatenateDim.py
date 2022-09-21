import torch
import torch.nn as nn


class ConcatenateDim(nn.Module):
    """Combines set of output tensors of matching shape along channel dim"""
    def __init__(self):
        super(ConcatenateDim, self).__init__()

    def forward(self, x):
        """
        :param x: A FloatTensor to propagate forward
        :type x: torch.Tensor
        """
        x = torch.stack(x).transpose(0, 1)
        if len(x.shape) != 5:
            raise ValueError(
                'Expected five dimensional shape, got shape {}'.format(
                    x.shape))

        out = torch.transpose(x, 0, 1)
        out = torch.cat([out[0], out[1], out[2], out[3]], dim=1)

        return out
