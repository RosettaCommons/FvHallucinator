import torch
import torch.nn.functional as F


def max_shape(data):
    """Gets the maximum length along all dimensions in a list of Tensors"""
    shapes = torch.Tensor([_.shape for _ in data])
    return torch.max(shapes.transpose(0, 1), dim=1)[0].int()


def pad_data_to_same_shape(tensor_list, pad_value=0):
    target_shape = max_shape(tensor_list)

    padded_dataset_shape = [len(tensor_list)] + list(target_shape)
    padded_dataset = torch.Tensor(*padded_dataset_shape).type_as(
        tensor_list[0])

    for i, data in enumerate(tensor_list):
        # Get how much padding is needed per dimension
        padding = reversed(target_shape - torch.Tensor(list(data.shape)).int())

        # Add 0 every other index to indicate only right padding
        padding = F.pad(padding.unsqueeze(0).t(), (1, 0, 0, 0)).view(-1, 1)
        padding = padding.view(1, -1)[0].tolist()

        padded_data = F.pad(data, padding, value=pad_value)
        padded_dataset[i] = padded_data

    return padded_dataset


def fill_diagonally_(matrix,
                     diagonal_index,
                     fill_value=0,
                     fill_method='below'):
    """Destructively fills an nxm tensor somehow with respect to a diagonal.
    :param matrix:
    :type matrix: torch.Tensor
    :param diagonal_index:
    :param fill_value:
    :type fill_value: numeric
    :param fill_method:
    :type fill_method: str
    :return:
    """
    num_rows = matrix.shape[0]
    if fill_method == 'symmetric':
        mask = torch.ones(matrix.shape)
        fill_diagonally_(mask,
                         diagonal_index - 1,
                         fill_method='between',
                         fill_value=0)
        matrix[mask.byte()] = fill_value
        return

    for i in range(num_rows):
        if fill_method == 'below':
            left_bound = 0
            right_bound = min(num_rows, max(i - diagonal_index + 1, 0))
        elif fill_method == 'above':
            left_bound = min(num_rows, max(i - diagonal_index, 0))
            right_bound = num_rows
        elif fill_method == 'between':
            left_bound = min(num_rows, max(i - diagonal_index, 0))
            right_bound = min(num_rows, min(i + diagonal_index + 1, num_rows))
        else:
            msg = ('{} is an invalid fill_method. The fill_method must be in '
                   '\'below\', \'above\', \'symmetric\', \'between\'')
            raise ValueError(msg.format(fill_method))

        matrix[i, left_bound:right_bound] = fill_value
