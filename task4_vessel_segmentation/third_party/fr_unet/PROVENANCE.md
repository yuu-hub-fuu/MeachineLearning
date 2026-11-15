# Source and adaptation

Official implementation of Liu et al., Full-Resolution Network and Dual-Threshold Iteration for Retinal Vessel and Coronary Angiograph Segmentation, IEEE JBHI 26(9):4623-4634, 2022.

Repository: https://github.com/lseventeen/FR-UNet (master, fetched 2026-11-03).

- `fr_unet.py`: original code, Git blob bc464f5fe3e125f43c5327cd0ec51e5ecce6143d.
- `utils_original.py.txt`: original initialization utility, blob ab1d8a6f80531df32cd8e4e5d1c055e2212c58a2.
- `utils.py`: same initialization, replacing the unused timm trunc_normal_ import with torch.nn.init.trunc_normal_.
- MIT license retained in LICENSE. Copyright (c) 2022 liuwentao.

Task-specific training uses four image channels and two output heads: submitted-mask segmentation and optional native-core supervision. The default upstream model uses one input/output channel. Only the source architecture is reused. NO upstream pretrained weights, external retinal images, or external retinal annotations are loaded.
