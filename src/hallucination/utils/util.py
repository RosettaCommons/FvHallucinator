from src.util.util import letter_to_num, _aa_dict
import torch
from src.hallucination.loss.LossType import LossType
from src.util.pdb import get_pdb_chain_seq,\
                            get_residue_indices_from_pdb_numbering,\
                            get_indices_for_framework,\
                            get_indices_for_cdrs
import glob

def comma_separated_chain_indices_to_dict(indices_str):
    chains = indices_str.split('/')
    dict_indices = {}
    for t in chains:
        ch = t.split(':')[0].lower()
        positions = t.split(':')[1]
        dict_indices[ch] = positions.split(',')
    return dict_indices


def remove_indices_from_list(target_pdb, len_heavy_seq, indices_list,
                             exclude_indices):

    indices_hal_ex = []
    for key in exclude_indices:
        indices_hal_ex += \
        get_residue_indices_from_pdb_numbering(target_pdb,
                                            exclude_indices[key],
                                            key,\
                                            len_heavy_seq)
    indices_list = [t for t in indices_list if not t in indices_hal_ex]
    return indices_list


def get_indices_from_different_methods(target_pdb, cdr_list='', framework=False, hl_interface=False, \
    include_indices={}, exclude_indices ={}, paratope=False, complex_pdb=None):
    # Design selected CDR loops
    heavy_seq, light_seq = get_pdb_chain_seq(target_pdb,
                                             "H"), get_pdb_chain_seq(
                                                 target_pdb, "L")
    seq = heavy_seq + light_seq

    indices_hal = []

    if cdr_list != '':
        cdr_indices_list = get_indices_for_cdrs(
            target_pdb,
            cdr_list,
        )
        indices_hal += cdr_indices_list
        #print("Following CDR loops are set to design: ", cdr_list)
        #print("Indices for CDR residues set to design: ", cdr_indices_list)

    # Design framework region
    if framework:
        framework_indices_list = get_indices_for_framework(target_pdb)
        indices_hal += framework_indices_list
        #print("Indices for framework residues set to design: ",
        #      framework_indices_list)

    # Design VH/VL interface. Interface is determined by Rosetta's InterGroupInterfaceByVectorSelector
    if hl_interface:
        from src.util.pdb import get_indices_for_interface_residues
        hl_indices_list = get_indices_for_interface_residues(
            target_pdb, seq, "H", "L")
        indices_hal += hl_indices_list
        #print("Indices for H/L interface set to design: ", hl_indices_list)

    # Design entire paratope (within x Angstrom of antigen)
    # get_indices_for_interface_residues(target_pdb, "HL")

    # Design select indices of the sequence
    if include_indices != {}:
        include_indices_list = []
        for key in include_indices:
            include_indices_list += get_residue_indices_from_pdb_numbering(target_pdb,
                                                   include_indices[key],
                                                   key,\
                                                   len(heavy_seq))
        indices_hal += include_indices_list
        #print("Indices set to design: ", include_indices_list)

    # If no residues are explicitly selected, select all
    if indices_hal == []:
        #print("Setting all residues for design!")
        indices_hal = range(len(seq))

    # Exclude indices from design (overrides any residues that have been set to design before)
    if exclude_indices != {}:
        exclude_indices_list = []
        for key in exclude_indices:
            exclude_indices_list += get_residue_indices_from_pdb_numbering(target_pdb,
                                               include_indices[key],
                                               key,\
                                               len(heavy_seq))
        #print("Indices for excluded from design: ", exclude_indices_list)

        indices_hal = remove_indices_from_list(target_pdb, len(heavy_seq),
                                               indices_hal, exclude_indices)

    #print("Final selection of residue indices set to design: ", indices_hal)
    return (indices_hal)


def convert_chain_aa_to_index_aa_map(restricted_dict, target_pdb,
                                     len_heavy_seq):
    restricted_positions_indexed = {}
    for pdb_ind in restricted_dict:
        pos_ind = \
        get_residue_indices_from_pdb_numbering(target_pdb,
                                                [pdb_ind[1]],
                                                pdb_ind[0],
                                                len_heavy_seq)

        for pos_ind_i in pos_ind:
            restricted_positions_indexed[pos_ind_i] = \
                                        restricted_dict[pdb_ind]
    return restricted_positions_indexed


def convert_aalist_to_index_aa_map(values, positions, existing_dict={}):

    restricted_positions_indexed = {}
    for ind in positions:
        restricted_positions_indexed[ind] = values

    for key in restricted_positions_indexed:
        if key in existing_dict:
            restricted_positions_indexed[key] = \
                list(set(existing_dict[key] + restricted_positions_indexed[key]))

    return restricted_positions_indexed


def get_loss(model, loss_components, heavy_chain_delimiter, input_sequence,
             sm_sequence, **kargs):
    model_input = torch.cat(
        [input_sequence.transpose(0, 1), heavy_chain_delimiter]).unsqueeze(0)

    candidate_geometries = model(model_input)

    candidate_geometries = [cg.squeeze(0) for cg in candidate_geometries]

    loss_weight_list = []
    for w, lc in loss_components:
        losses = [0]
        if lc.loss_type == LossType.candidate_sequence_loss:
            losses = lc.calc_loss(sm_sequence)
        elif lc.loss_type == LossType.candidate_geometry_loss:
            losses = lc.calc_loss(candidate_geometries)
        elif lc.loss_type == LossType.kl_div_loss:
            if lc.loss_candidate == 'geom':
                losses = lc.calc_loss(candidate_geometries)
            elif lc.loss_candidate == 'seq':
                losses = lc.calc_loss([sm_sequence])
            else:
                ValueError('Unimplemented kl_div loss \
                    candidate type {}'                                                                                                                                                        .format(lc.loss_candidate))
        elif lc.loss_type == LossType.entropy_loss:
            losses = lc.calc_loss(candidate_geometries)
        else:
            raise KeyError.error('Unimplemented loss type {}'.format(
                lc.loss_type))

        loss_weight_list.append((losses, w))

    total_loss = sum(
        [weight * sum(losses) for losses, weight in loss_weight_list])
    return total_loss, loss_weight_list


def get_background_distributions(m, sequence_len, delimiter_position,\
                                device):
    from src.util.model_out import generate_probabilities

    heavy_chain_delimiter = torch.zeros((1, sequence_len))
    heavy_chain_delimiter[0, delimiter_position] = 1
    model_input = torch.cat([
        torch.abs(torch.randn((20, sequence_len)) * 0.05),
        heavy_chain_delimiter
    ]).unsqueeze(0)

    bg_logits = m(model_input.to(device))
    bg_probs = [generate_probabilities(l.squeeze(0)) for l in bg_logits]
    bg_dists = [cg.detach() for cg in bg_probs]

    return bg_dists


def get_restricted_freq_distributions(sequence_len, restricted_dict):
    '''
    restricted_dict: {pos: {a:0.33,c:0,f:0.33, w:0.33}}
    '''

    restricted_seq_pssm = torch.zeros((20, sequence_len))
    for pos in restricted_dict:
        for aa in restricted_dict[pos]:
            aa_one_hot_index = letter_to_num(aa, _aa_dict)
            aa_freq = restricted_dict[pos][aa]
            restricted_seq_pssm[aa_one_hot_index[0], pos] = aa_freq
            #print(aa_one_hot_index[0], pos, restricted_seq_pssm[aa_one_hot_index[0], pos])
        restricted_seq_pssm[:, pos] = restricted_seq_pssm[:, pos] \
                                / torch.sum(restricted_seq_pssm[:,pos], dim=0)

    #debug
    #import matplotlib.pyplot as plt
    #plt.imshow(restricted_seq_pssm)
    #plt.colorbar()
    #plt.show()
    #plt.close()

    return [restricted_seq_pssm]


def get_restricted_aa_distributions(sequence_len, restricted_dict):
    '''
    restricted_dict: {pos: [a,c,w]}
    '''

    restricted_seq_pssm = torch.zeros((20, sequence_len))
    src = torch.ones((20, sequence_len))
    index_tensor = torch.tensor(list(restricted_dict.keys()))
    index_tensor = index_tensor.unsqueeze(0).expand(20, -1)

    restricted_seq_pssm.scatter_(1, index_tensor, src)

    for pos in restricted_dict:
        zero_aas = [aa for aa in _aa_dict if aa not in restricted_dict[pos]]
        for aa in zero_aas:
            aa_one_hot_index = letter_to_num(aa, _aa_dict)
            restricted_seq_pssm[aa_one_hot_index[0], pos] = 0.0

        restricted_seq_pssm[:, pos] = restricted_seq_pssm[:, pos] \
                                / torch.sum(restricted_seq_pssm[:, pos], dim=0)

    #debug
    import matplotlib.pyplot as plt
    plt.imshow(restricted_seq_pssm)
    plt.gca().set_xticks([t for t in range(0, sequence_len, 20)])
    plt.colorbar()
    plt.savefig('restricted_seq_pssm.png')
    plt.close()

    return [restricted_seq_pssm]


def get_cce_aa_distributions(cce_dist, design_mask, device, bk_dist=None, restricted_mask=None):
    cce_tensor = torch.tensor(cce_dist)
    cce_normed = torch.nn.functional.softmax(cce_tensor, dim=0).float()
    mask_seq_aa = design_mask.unsqueeze(0).expand(20, -1)
    cce_normed[mask_seq_aa == 0] = 0

    import matplotlib.pyplot as plt
    plt.imshow(cce_normed, aspect='equal')
    plt.colorbar()
    plt.savefig('./cce_dist.png')
    plt.close()
    
    cce_normed = cce_normed.to(device)
    cce_mask = mask_seq_aa.to(device)
    
    if bk_dist != None:

        #do not apply cce distribution on restricted positions
        cce_normed[restricted_mask == 1] = 0.0
        plt.imshow(cce_normed, aspect='equal')
        plt.colorbar()
        plt.savefig('./cce_dist_sans_restricted.png')
        plt.close()

        cce_normed = bk_dist + cce_normed
        cce_mask = restricted_mask.long() | cce_mask.long()
        plt.imshow(cce_normed, aspect='equal')
        plt.colorbar()
        plt.savefig('./bk_dist_comb.png')
        plt.close()
        plt.imshow(cce_mask, aspect='equal')
        plt.colorbar()
        plt.savefig('./restricted_mask_comb.png')
        plt.close()

    return [cce_normed.to(device)], [cce_mask.to(device)]



def get_restricted_seq_mask(sequence_len, restricted_dict, device):
    mask = torch.zeros(20, sequence_len)
    unmask_indices_tensor = torch.tensor(list(
        restricted_dict.keys())).unsqueeze(0)

    #unmask seq dims for res positions
    unmask_indices_tensor = unmask_indices_tensor.expand(20, -1)
    mask.scatter_(1, unmask_indices_tensor, torch.ones(20, sequence_len))

    #debug
    #import matplotlib.pyplot as plt
    #plt.imshow(mask)
    #plt.colorbar()
    #plt.show()
    #plt.close()

    return [mask.to(device)]


def get_restricted_seq_aa_mask(sequence_len, restricted_pos_to_aas_dict,
                               device):
    mask = torch.zeros(20, sequence_len)
    unmask_indices_tensor = torch.tensor(
        list(restricted_pos_to_aas_dict.keys())).unsqueeze(0)

    #unmask seq dims for res positions
    unmask_indices_tensor = unmask_indices_tensor.expand(20, -1)
    mask.scatter_(1, unmask_indices_tensor, torch.ones(20, sequence_len))

    #debug
    import matplotlib.pyplot as plt
    plt.imshow(mask)
    plt.colorbar()
    plt.savefig('mask_aa.png')
    plt.close()

    return [mask.to(device)]


def get_model_file(model_path):

    model_file = glob.glob(model_path + '/*/*.p*')
    print(model_file)
    if len(model_file) == 0:
        raise FileNotFoundError(
            'No model files found in {}'.format(model_path + '/*/*.p*'))
    return model_file
