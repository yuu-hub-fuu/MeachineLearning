"""LP-FT adaptation of DINOv2-L, updating its final four blocks and head.

Reuses the fixed-epoch 5-fold runner in plant_finetune.py. The pretrained prefix
stays frozen to fit an 8GB GPU. Both global and green-weighted patch descriptors
are used. This is a partial-finetuning adaptation of the LP-FT idea.
"""
import sys
import torch
from torch import nn
import torch.nn.functional as F
import plant_finetune as runner


class DinoFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=torch.hub.load('facebookresearch/dinov2','dinov2_vitl14',trust_repo=True)

    def forward(self,x):
        layers=self.encoder.get_intermediate_layers(x,n=4,return_class_token=True)
        components=[F.normalize(c.float(),dim=1) for _,c in layers]
        patch=layers[-1][0].float(); side=int(patch.shape[1]**.5)
        # Soft positive excess-green weighting; detached from representation gradients.
        rgb=x.float()*x.new_tensor(runner.STD).view(1,3,1,1)+x.new_tensor(runner.MEAN).view(1,3,1,1)
        r,g,b=rgb.unbind(1)
        green=F.relu(2*g-r-b-.035)*((g>r)&(g>b))
        weights=F.adaptive_avg_pool2d(green[:,None],(side,side)).flatten(1)
        weights=torch.where(weights.sum(1,keepdim=True)>.05,weights,torch.ones_like(weights))
        fg=(patch*weights[...,None]).sum(1)/weights.sum(1,keepdim=True)
        components.extend([F.normalize(patch.mean(1),dim=1),F.normalize(fg,dim=1)])
        return F.normalize(torch.cat(components,dim=1),dim=1)


class DinoNetwork(nn.Module):
    def __init__(self):
        super().__init__();self.features=DinoFeatures();self.classifier=nn.Sequential(nn.Identity())

    def forward(self,x): return self.classifier(self.features(x))

    def requires_grad_(self,requires_grad=True):
        super().requires_grad_(requires_grad)
        if requires_grad:
            self.features.encoder.requires_grad_(False)
            for block in self.features.encoder.blocks[-4:]:block.requires_grad_(True)
            self.features.encoder.norm.requires_grad_(True)
        return self


def make_model(device): return DinoNetwork().to(device).eval()


if __name__=='__main__':
    # Preserve an unambiguous backbone identity and wrapper source hash in config.
    runner.features_model=make_model
    runner.__file__=__file__
    if '--run-dir' not in sys.argv: sys.argv.extend(['--run-dir','outputs/stronger_20260929/dino_lpft'])
    if '--size' not in sys.argv: sys.argv.extend(['--size','336'])
    if '--epochs' not in sys.argv: sys.argv.extend(['--epochs','15'])
    if '--batch-size' not in sys.argv: sys.argv.extend(['--batch-size','4'])
    if '--backbone-lr' not in sys.argv: sys.argv.extend(['--backbone-lr','0.00001'])
    if '--head-lr' not in sys.argv: sys.argv.extend(['--head-lr','0.0001'])
    runner.main()
