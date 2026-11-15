# Local compatibility changes

Original URLs and downloaded source SHA256 values are in `sources.json`.
The MIT licenses from QCS and RMN remain in their respective directories.

- RMN `resnet.py` and `resmasking.py`: replace the empty author utility shim's
  `load_state_dict_from_url` import with `torch.hub.load_state_dict_from_url`.
  The instantiated ResMasking constructor does not call this downloader.
- QCS vendor files are unchanged. `upgrade_models.py` reconstructs only the
  author's single-image inference branch; training-only cross branches are
  explicitly excluded when loading, while all inference tensors must match.
- NLA is reconstructed using torchvision's matching ResNet18 structure and
  the author's `features`, `features2`, `fc`, `embed` key layout. Strict loading
  confirms the actual checkpoint keys. Its RAF-DB normalization is preserved.
- RMN uses OpenCV INTER_LINEAR as in the author inference code; QCS uses
  resize 232 then center crop 224, as in the active author test transform.

No model weights are distributed in this source package. They remain on D:.
