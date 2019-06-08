import torch.nn as nn
from .global_discriminator import GlobalDiscriminator
from .local_discriminator import LocalDiscriminator
from .layers import Concatenate


class ContextDiscriminator(nn.Module):
    def __init__(self, local_input_shape, global_input_shape, dataset='celeba'):
        super(ContextDiscriminator, self).__init__()
        self.dataset = dataset
        self.input_shape = [local_input_shape, global_input_shape]
        self.output_shape = (1,)
        self.model_ld = LocalDiscriminator(local_input_shape)
        self.model_gd = GlobalDiscriminator(global_input_shape, dataset=dataset)
        in_features = self.model_ld.output_shape[-1] + self.model_gd.output_shape[-1]
        self.concat1 = Concatenate(dim=-1)
        self.linear1 = nn.Linear(in_features, 1)
        self.act1 = nn.Sigmoid()

    def forward(self, x):
        x_ld, x_gd = x
        x_ld = self.model_ld(x_ld)
        x_gd = self.model_gd(x_gd)
        out = self.act1(self.linear1(self.concat1([x_ld, x_gd])))
        return out
