"""Fixed inference kernels; preserves the exact source used for training.

Training uses cuDNN benchmarking for speed. This entry point disables algorithm
autotuning during inference so threshold-boundary pixels reproduce exactly on
the tested machine. The original training source and its hash stay unchanged.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
import pipeline


def main():
    torch.set_num_threads(4)
    pipeline.seed_all(20261103)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    pipeline.infer(None)


if __name__ == '__main__':
    main()
