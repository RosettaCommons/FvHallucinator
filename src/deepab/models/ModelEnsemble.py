import src
import torch
from torch import nn


class ModelEnsemble(nn.Module):
    def __init__(self, load_model, model_files, eval_mode=False, device=None,strict=True):
        super(ModelEnsemble, self).__init__()
        self.load_model = load_model
        self.model_files = model_files

        if type(device) == type(None):
            self.models = [
                load_model(mf, eval_mode=eval_mode, strict=strict) for mf in model_files
            ]
        else:
            self.models = [
                load_model(mf, eval_mode=eval_mode, device=device, strict=strict).to(device)
                for mf in model_files
            ]
        self._num_out_bins = self.models[0]._num_out_bins

        for model in self.models:
            assert type(self.models[0]) == type(model)

    def model_type(self):
        return type(self.models[0])

    def forward(self, x, **kwargs):
        out = [model(x, **kwargs) for model in self.models]
        out = [torch.mean(torch.stack(list(o)), dim=0) for o in zip(*out)]

        return out