import argparse
import os
from pathlib import Path

from fovea.common import load_manifest, split_rows, write_submission
from fovea.data import prepare


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('work'))
    p.add_argument('--dark-dir', type=Path, default=Path('third_party/DarkPose'))
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--archive', type=Path, required=True)
    t = sub.add_parser('train')
    t.add_argument('--name', default='hrnet_s384')
    t.add_argument('--size', type=int, default=384)
    t.add_argument('--epochs', type=int, default=40)
    t.add_argument('--batch', type=int, default=12)
    t.add_argument('--lr', type=float, default=2e-4)
    t.add_argument('--seed', type=int, default=20261027)
    t.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    s = sub.add_parser('submit-file')
    s.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    s.add_argument('--size', type=int, default=384)
    s.add_argument('--output', type=Path, default=Path('submission.csv'))
    return p


def main():
    args = build_parser().parse_args()
    if args.command == 'prepare':
        print(prepare(args.archive, args.root))
        return
    os.environ.setdefault('DARKPOSE_DIR', str(args.dark_dir.resolve()))
    from fovea.hrnet_dark import HRNetDark
    import cv2
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(0)
    torch.backends.cudnn.benchmark = True
    rows = load_manifest(args.root)
    model = HRNetDark(args.root, args.size)
    if args.command == 'train':
        for fold in args.folds:
            print(model.train_fold(rows, fold, args.name, args.epochs, args.batch, args.lr, args.seed))
    else:
        _, test_rows = split_rows(rows)
        preds = [model.predict(model.load(c), test_rows) for c in args.checkpoints]
        pred = sum(preds) / len(preds)
        print(write_submission(args.output, args.root, test_rows, pred))


if __name__ == '__main__':
    main()
