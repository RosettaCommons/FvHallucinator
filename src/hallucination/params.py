import torch
import numpy as np


class HallucinationLossParams():
    def __init__(self,
                 weight_kl_bg=0,
                 weight_seq=0,
                 weight_geom=1,
                 weight_kl_res=50,
                 weight_seq_reg=0,
                 weight_entropy=0,
                 weight_netcharge=0,
                 weight_aa_freq=0.0,
                 geometric_loss_list=[]):

        super().__init__()

        self.loss_weight_kl_bg = weight_kl_bg
        self.loss_weight_geom = weight_geom
        self.loss_weight_seq = weight_seq
        self.loss_weight_kl_res = weight_kl_res
        self.geometric_loss_split = geometric_loss_list
        self.loss_weight_seq_reg = weight_seq_reg
        self.loss_weight_entropy = weight_entropy
        self.loss_weight_netcharge = weight_netcharge
        self.loss_aa_freq = weight_aa_freq


latest_models = ['latest']
model_dict = {'latest': 'AbResNet'}
geometric_loss_dict = {
    'latest': [1, 1, 1, 1, 1, 1]
}
oas_heavy_seq_dist = torch.tensor(
    np.loadtxt("data/ab_seq_distributions/all_heavy_dist.csv")).float()

oas_light_seq_dist = torch.tensor(
    np.loadtxt("data/ab_seq_distributions/all_light_dist.csv")).float()

germline_data_path = "data/h_germline_enrichment"
