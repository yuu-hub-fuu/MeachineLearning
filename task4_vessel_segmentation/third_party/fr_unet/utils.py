"""Original He initialization; unused timm dependency replaced by torch builtin.

Upstream: lseventeen/FR-UNet (MIT), utils.py blob
ab1d8a6f80531df32cd8e4e5d1c055e2212c58a2.
"""
from torch import nn


class InitWeights_He:
    def __init__(self, neg_slope=1e-2):
        self.neg_slope=neg_slope

    def __call__(self,module):
        if isinstance(module,(nn.Conv2d,nn.Conv3d,nn.ConvTranspose2d,nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(module.weight,a=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias,0)
        elif isinstance(module,nn.Linear):
            nn.init.trunc_normal_(module.weight,std=self.neg_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias,0)
        elif isinstance(module,nn.LayerNorm):
            nn.init.constant_(module.bias,0)
            nn.init.constant_(module.weight,1.)
