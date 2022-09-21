import math
from os.path import isfile
import torch.nn as nn
import torch
from torch.utils.checkpoint import checkpoint
from src.deepab.models.PairedSeqLSTM import PairedSeqLSTM
from src.deepab.resnets import ResNet1D, ResBlock1D, ResNet2D, ResBlock2D, RCCAModule
from src.deepab.layers import OuterConcatenation2D
from src.util.tensor import pad_data_to_same_shape


class AbResNet(nn.Module):
    """
    Predicts binned output distributions for CA-distance, CB-distance, NO-distance, 
    omega and theta dihedrals, and phi planar angle from a one-hot encoded sequence 
    of heavy and light chain resides.
    """
    def __init__(self,
                 in_planes,
                 lstm_model,
                 rnn_planes=128,
                 num_out_bins=37,
                 num_blocks1D=3,
                 num_blocks2D=21,
                 dilation_cycle=5,
                 dropout_proportion=0.2,
                 lstm_mean=None,
                 lstm_scale=None):
        super(AbResNet, self).__init__()
        if isinstance(num_blocks1D, list):
            if len(num_blocks1D) > 1:
                raise NotImplementedError('Multi-layer resnets not supported')
            num_blocks1D = num_blocks1D[0]
        if isinstance(num_blocks2D, int):
            num_blocks2D = [num_blocks2D]

        norm1D = nn.BatchNorm1d
        norm2D = nn.BatchNorm2d

        self.lstm_model = lstm_model
        self.lstm_mean = torch.zeros(1, ) if lstm_mean is None else lstm_mean
        self.lstm_scale = torch.ones(1, ) if lstm_scale is None else lstm_scale
        
        self._num_out_bins = num_out_bins
        self.resnet1D = ResNet1D(in_planes,
                                 ResBlock1D, [num_blocks1D],
                                 init_planes=32,
                                 kernel_size=17,
                                 norm=norm1D)
        self.seq2pairwise = OuterConcatenation2D()

        # Calculate the number of planes output from the seq2pairwise layer
        expansion1D = int(math.pow(2, self.resnet1D.num_layers - 1))
        out_planes1D = self.resnet1D.init_planes * expansion1D
        in_planes2D = 2 * (out_planes1D + rnn_planes)

        self.resnet2D = ResNet2D(in_planes2D,
                                 ResBlock2D,
                                 num_blocks2D,
                                 init_planes=64,
                                 kernel_size=5,
                                 dilation_cycle=dilation_cycle,
                                 norm=norm2D)

        # Calculate the number of planes output from the ResNet2D layer
        expansion2D = int(math.pow(2, self.resnet2D.num_layers - 1))
        out_planes2D = self.resnet2D.init_planes * expansion2D

        self.out_dropout = nn.Dropout2d(p=dropout_proportion)

        # Output convolution to reduce/expand to the number of bins
        self.out_ca_dist = nn.Sequential(
            nn.Conv2d(out_planes2D,
                      num_out_bins,
                      kernel_size=self.resnet2D.kernel_size,
                      padding=self.resnet2D.kernel_size // 2),
            RCCAModule(in_channels=num_out_bins,
                       kernel_size=self.resnet2D.kernel_size,
                       return_attn=True,
                       norm=norm2D))
        self.out_cb_dist = nn.Sequential(
            nn.Conv2d(out_planes2D,
                      num_out_bins,
                      kernel_size=self.resnet2D.kernel_size,
                      padding=self.resnet2D.kernel_size // 2),
            RCCAModule(in_channels=num_out_bins,
                       kernel_size=self.resnet2D.kernel_size,
                       return_attn=True,
                       norm=norm2D))
        self.out_no_dist = nn.Sequential(
            nn.Conv2d(out_planes2D,
                      num_out_bins,
                      kernel_size=self.resnet2D.kernel_size,
                      padding=self.resnet2D.kernel_size // 2),
            RCCAModule(in_channels=num_out_bins,
                       kernel_size=self.resnet2D.kernel_size,
                       return_attn=True,
                       norm=norm2D))
        self.out_omega = nn.Sequential(
            nn.Conv2d(out_planes2D,
                      num_out_bins,
                      kernel_size=self.resnet2D.kernel_size,
                      padding=self.resnet2D.kernel_size // 2),
            RCCAModule(in_channels=num_out_bins,
                       kernel_size=self.resnet2D.kernel_size,
                       return_attn=True,
                       norm=norm2D))
        self.out_theta = nn.Sequential(
            nn.Conv2d(out_planes2D,
                      num_out_bins,
                      kernel_size=self.resnet2D.kernel_size,
                      padding=self.resnet2D.kernel_size // 2),
            RCCAModule(in_channels=num_out_bins,
                       kernel_size=self.resnet2D.kernel_size,
                       return_attn=True,
                       norm=norm2D))
        self.out_phi = nn.Sequential(
            nn.Conv2d(out_planes2D,
                      num_out_bins,
                      kernel_size=self.resnet2D.kernel_size,
                      padding=self.resnet2D.kernel_size // 2),
            RCCAModule(in_channels=num_out_bins,
                       kernel_size=self.resnet2D.kernel_size,
                       return_attn=True,
                       norm=norm2D))

    def get_lstm_input(self, x):
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_type)

        seq_start, seq_end, seq_delim = torch.tensor(
            [20]).byte().to(device), torch.tensor(
                [21]).byte().to(device), torch.tensor([22]).byte().to(device)

        input_seqs = x.transpose(1, 2)[:, :, :-1].argmax(-1).to(device)
        input_delims = x.transpose(1, 2)[:, :, -1].argmax(-1).to(device)

        lstm_input = [
            torch.cat([
                seq_start, seq[:d + 1].byte(), seq_delim, seq[d + 1:].byte(),
                seq_end
            ]) for seq, d in zip(input_seqs, input_delims)
        ]
        lstm_input = pad_data_to_same_shape(lstm_input, pad_value=22)
        lstm_input = torch.stack(
            [nn.functional.one_hot(seq.long()) for seq in lstm_input])
        lstm_input = lstm_input.transpose(0, 1)

        return lstm_input, input_delims

    def get_lstm_encoding(self, inputs, detach_lstm=True):
        with torch.no_grad():
            lstm_input, input_delims = self.get_lstm_input(inputs)

            if detach_lstm:
                enc = self.lstm_model.encoder(src=lstm_input)[0].detach()
            else:
                enc = self.lstm_model.encoder(src=lstm_input)[0]
            enc = enc.permute(1, 0, 2)

            no_delim_enc = []
            for i in range(len(enc)):
                no_delim_enc.append(
                    torch.cat([
                        enc[i][1:input_delims[i]],
                        enc[i][input_delims[i] + 1:-1]
                    ]))
            enc = torch.stack(no_delim_enc).permute(0, 2, 1)

            enc = (enc - self.lstm_mean.view(1, -1, 1)) / self.lstm_scale.view(
                1, -1, 1)

            return enc

    def get_lstm_pssm(self, inputs, teacher_forcing_ratio=1):
        with torch.no_grad():
            lstm_input, input_delims = self.get_lstm_input(inputs)

            pssm = self.lstm_model(
                src=lstm_input,
                trg=lstm_input,
                teacher_forcing_ratio=teacher_forcing_ratio).softmax(2)
            pssm = pssm.transpose(0, 1)

            no_delim_pssm = []
            for i in range(len(pssm)):
                no_delim_pssm.append(
                    torch.cat([
                        pssm[i][1:input_delims[i]],
                        pssm[i][input_delims[i] + 1:-1]
                    ]))
            pssm = torch.stack(no_delim_pssm).permute(0, 2, 1)

            # Remove delimiter probabilities
            pssm = pssm[:, :20]
            # Renormalize over AAs
            pssm = (pssm.transpose(1, 2) /
                    pssm.transpose(1, 2).sum(2, keepdims=True)).transpose(
                        1, 2)

            return pssm

    def forward(self, x, detach_lstm=True):
        out = self.resnet1D(x)
        lstm_enc = self.get_lstm_encoding(x, detach_lstm=detach_lstm)
        out = torch.cat([out, lstm_enc], dim=1)

        out = self.seq2pairwise(out)

        # out = self.resnet2D(out)
        out = checkpoint(self.resnet2D, out)
        out = self.out_dropout(out)

        out_ca_dist = self.out_ca_dist(out)[0]
        out_cb_dist = self.out_cb_dist(out)[0]
        out_no_dist = self.out_no_dist(out)[0]
        out_omega = self.out_omega(out)[0]
        out_theta = self.out_theta(out)[0]
        out_phi = self.out_phi(out)[0]

        out_ca_dist = out_ca_dist + out_ca_dist.transpose(2, 3)
        out_cb_dist = out_cb_dist + out_cb_dist.transpose(2, 3)
        out_omega = out_omega + out_omega.transpose(2, 3)
        l = [
            out_ca_dist, out_cb_dist, out_no_dist, out_omega, out_theta,
            out_phi
        ]
        return [
            out_ca_dist, out_cb_dist, out_no_dist, out_omega, out_theta,
            out_phi
        ]

    def forward_attn(self, x, detach_lstm=True):
        out = self.resnet1D(x)

        lstm_enc = self.get_lstm_encoding(x, detach_lstm=detach_lstm)
        out = torch.cat([out, lstm_enc], dim=1)

        out = self.seq2pairwise(out)

        out = checkpoint(self.resnet2D, out)
        out = self.out_dropout(out)

        out_ca_dist = self.out_ca_dist(out)[1]
        out_cb_dist = self.out_cb_dist(out)[1]
        out_no_dist = self.out_no_dist(out)[1]
        out_omega = self.out_omega(out)[1]
        out_theta = self.out_theta(out)[1]
        out_phi = self.out_phi(out)[1]
        
        return [
            out_ca_dist, out_cb_dist, out_no_dist, out_omega, out_theta,
            out_phi
        ]


def load_model(model_file,
               eval_mode=True,
               device=None,
               scaled=True,
               strict=True):
    if not isfile(model_file):
        raise FileNotFoundError("No file at {}".format(model_file))
    checkpoint_dict = torch.load(model_file, map_location='cpu')
    model_state = checkpoint_dict['model_state_dict']

    dilation_cycle = 0 if not 'dilation_cycle' in checkpoint_dict else checkpoint_dict[
        'dilation_cycle']

    num_out_bins = checkpoint_dict['num_out_bins']
    in_planes = 21
    num_blocks1D = checkpoint_dict['num_blocks1D']
    num_blocks2D = checkpoint_dict['num_blocks2D']

    if scaled:
        lstm_checkpoint_dict = checkpoint_dict['lstm_checkpoint_dict']
        lstm_model = PairedSeqLSTM(
            seq_dim=lstm_checkpoint_dict['seq_dim'],
            enc_hid_dim=lstm_checkpoint_dict['enc_hid_dim'],
            dec_hid_dim=lstm_checkpoint_dict['dec_hid_dim'])
        lstm_mean = torch.tensor(lstm_checkpoint_dict['scaler_mean']).float()
        lstm_scale = torch.tensor(lstm_checkpoint_dict['scaler_scale']).float()

        if type(device) is not type(None):
            lstm_model = lstm_model.to(device)
            lstm_mean = lstm_mean.to(device)
            lstm_scale = lstm_scale.to(device)

        model = AbResNet(in_planes=in_planes,
                         lstm_model=lstm_model,
                         num_out_bins=num_out_bins,
                         num_blocks1D=num_blocks1D,
                         num_blocks2D=num_blocks2D,
                         dilation_cycle=dilation_cycle,
                         lstm_mean=lstm_mean,
                         lstm_scale=lstm_scale)
    else:
        if 'lstm_checkpoint_dict' in checkpoint_dict:
            lstm_checkpoint_dict = checkpoint_dict['lstm_checkpoint_dict']
            lstm_model = PairedSeqLSTM(
                seq_dim=lstm_checkpoint_dict['seq_dim'],
                enc_hid_dim=lstm_checkpoint_dict['enc_hid_dim'],
                dec_hid_dim=lstm_checkpoint_dict['dec_hid_dim'])
        else:
            # try loading with default dimensions if lstm_checkpoint_dict not found
            lstm_model = PairedSeqLSTM()

        model = AbResNet(in_planes=in_planes,
                         lstm_model=lstm_model,
                         num_out_bins=num_out_bins,
                         num_blocks1D=num_blocks1D,
                         num_blocks2D=num_blocks2D,
                         dilation_cycle=dilation_cycle)

    model.load_state_dict(model_state, strict=strict)

    if eval_mode:
        model.eval()

    return model
