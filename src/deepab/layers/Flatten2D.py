import torch
import torch.nn as nn


class Flatten2D(nn.Module):
    """Transforms pairwise data to sequential data by summing row_i and column_i for all in in L."""
    def __init__(self, sum_row_col=True):
        super(Flatten2D, self).__init__()

        self.sum_row_col = sum_row_col

    def forward(self, x):
        """
        :param x: A FloatTensor to propagate forward
        :type x: torch.Tensor
        """
        if len(x.shape) != 4:
            raise ValueError(
                'Expected three dimensional shape, got shape {}'.format(
                    x.shape))

        # Switch shape from [batch, filter/channel, timestep/length_i, timestep/length_j]
        #                to [batch, filter/channel, timestep/length_i]
        if self.sum_row_col:
            row_flatten = torch.sum(x, dim=3)
            col_flatten = torch.sum(torch.transpose(x, 2, 3), dim=3)
            out_tensor = row_flatten + col_flatten
        # Switch shape from [batch, filter/channel, timestep/length_i, timestep/length_j]
        #                to [batch, 2 * filter/channel, timestep/length_i]
        else:
            row_flatten = torch.sum(x, dim=3)
            col_flatten = torch.sum(torch.transpose(x, 2, 3), dim=3)
            out_tensor = torch.cat([row_flatten, col_flatten], dim=1)

        return out_tensor
