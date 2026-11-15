"""DINOv2 frozen-feature classifier for the five-species plant dataset.

Training data layout:
    train/<species name>/<image file>

Submission manifest columns: ID,Category. The Category values in the provided
sample submission are placeholders; prediction replaces them in row order.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def discover_training_images(train_dir: str):
    root = Path(train_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Training directory does not exist: {root}")
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    rows = []
    counts = {}
    for class_dir in class_dirs:
        paths = sorted(p for p in class_dir.rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        counts[class_dir.name] = len(paths)
        rows.extend((p.relative_to(root.parent).as_posix(), class_dir.name)
                    for p in paths)
    if len(class_dirs) < 2 or any(count == 0 for count in counts.values()):
        raise ValueError(f"Expected non-empty species subfolders under {root}; got {counts}")
    if not rows:
        raise ValueError(f"No images found under {root}")
    return rows, counts, root.parent


class ImageRows(Dataset):
    def __init__(self, rows, image_root: Path, image_size: int):
        self.rows = rows
        self.image_root = image_root
        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        relative_path, label = self.rows[index]
        image_path = self.image_root / relative_path
        with Image.open(image_path) as im:
            image = self.transform(im.convert("RGB"))
        return image, label if label is not None else ""


def read_submission(path: str):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"ID", "Category"}.issubset(reader.fieldnames or []):
            raise ValueError("Submission CSV must have ID and Category columns")
        rows = list(reader)
    if not rows:
        raise ValueError("Submission CSV is empty")
    ids = [r["ID"] for r in rows]
    if any(not item for item in ids):
        raise ValueError("Submission CSV contains an empty ID")
    if len(set(ids)) != len(ids):
        raise ValueError("Submission CSV contains duplicate IDs")
    return rows


def audit(args):
    train_rows, counts, _ = discover_training_images(args.train_dir)
    classes = sorted(counts)
    if len(classes) != args.expected_classes:
        raise ValueError(f"Expected {args.expected_classes} species, found {len(classes)}: {classes}")
    if args.expected_per_class:
        wrong = {name: n for name, n in counts.items() if n != args.expected_per_class}
        if wrong:
            raise ValueError(f"Unexpected training image counts (expected {args.expected_per_class} each): {wrong}")

    rows = read_submission(args.input_csv)
    if args.expected_test_count and len(rows) != args.expected_test_count:
        raise ValueError(f"Expected {args.expected_test_count} submission rows, found {len(rows)}")
    test_root = Path(args.test_dir).expanduser().resolve()
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_root}")
    available = {p.relative_to(test_root).as_posix() for p in test_root.rglob("*")
                 if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
    requested = {row["ID"] for row in rows}
    missing = sorted(requested - available)
    extra = sorted(available - requested)
    if missing or extra:
        raise ValueError(f"Test/submission IDs differ: missing_images={len(missing)}, "
                         f"images_without_submission_rows={len(extra)}; "
                         f"first_missing={missing[:3]}, first_extra={extra[:3]}")

    print(f"Training classes ({len(classes)}): {', '.join(classes)}")
    print(f"Training images: {len(train_rows)}; per-class counts: {counts}")
    print(f"Test images and submission IDs match exactly: {len(rows)}")
    print("Submission columns: ID,Category; Category values are ignored during prediction.")


def load_backbone(model_name: str, device: torch.device):
    # Official DINOv2 PyTorch Hub entry point; weights are cached by torch.hub.
    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.inference_mode()
def extract_features(model, loader, device: torch.device, use_amp: bool,
                     last_blocks: int = 4):
    all_features, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        # Horizontal-flip test-time averaging reduces sensitivity to leaf direction.
        two_views = torch.cat((images, images.flip(dims=[3])), dim=0)
        amp_context = (torch.autocast(device_type="cuda", dtype=torch.float16)
                       if use_amp and device.type == "cuda" else nullcontext())
        with amp_context:
            # Official DINOv2 linear-evaluation recipe: concatenate CLS tokens
            # from the final blocks and append the last block's mean patch token.
            layers = model.get_intermediate_layers(
                two_views, n=last_blocks, return_class_token=True
            )
            cls = torch.cat([F.normalize(class_token.float(), dim=1)
                             for _, class_token in layers], dim=1)
            patches = F.normalize(layers[-1][0].float().mean(dim=1), dim=1)
            descriptors = F.normalize(torch.cat((cls, patches), dim=1), dim=1)
            n = images.shape[0]
            descriptors = F.normalize((descriptors[:n] + descriptors[n:]) * 0.5,
                                      dim=1)
        all_features.append(descriptors.cpu().numpy())
        all_labels.extend(labels)
    return np.concatenate(all_features).astype(np.float32), all_labels


def train(args):
    rows, counts, image_root = discover_training_images(args.train_dir)
    classes = sorted(counts)
    if args.expected_classes and len(classes) != args.expected_classes:
        raise ValueError(f"Expected {args.expected_classes} classes, found {len(classes)}: {classes}")
    if args.expected_per_class:
        wrong = {name: n for name, n in counts.items() if n != args.expected_per_class}
        if wrong:
            raise ValueError(f"Expected {args.expected_per_class} training images per class; got {wrong}")
    print("Training images per species:")
    for name in classes:
        print(f"  {name}: {counts[name]}")

    dataset = ImageRows(rows, image_root, args.image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    backbone = load_backbone(args.model, device)
    features, labels = extract_features(backbone, loader, device, args.amp, args.last_blocks)
    del backbone
    label_array = np.asarray(labels)

    min_class_count = min(counts.values())
    n_splits = min(args.folds, min_class_count)
    if n_splits < 2:
        raise ValueError("Each species needs at least two labeled images for validation")
    probe = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=4000, solver="lbfgs", random_state=args.seed
        )),
    ])
    # Compare linear probes and a small RBF-SVM grid. Scaling stays inside each
    # CV fit, and the nested outer fold remains untouched by model selection.
    param_grid = [
        {"classifier": [LogisticRegression(
            class_weight="balanced", max_iter=4000, solver="lbfgs", random_state=args.seed
        )], "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        {"classifier": [LinearSVC(
            class_weight="balanced", max_iter=20000, random_state=args.seed
        )], "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        {"classifier": [SVC(class_weight="balanced", kernel="rbf")],
         "classifier__C": [0.1, 1.0, 10.0, 100.0],
         "classifier__gamma": ["scale", 0.001, 0.01]},
    ]

    # Nested CV keeps each outer fold out of the regularization-selection loop.
    outer_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    outer_scores = []
    for fold, (fit_idx, eval_idx) in enumerate(outer_cv.split(features, label_array), start=1):
        y_fit = label_array[fit_idx]
        class_counts = np.unique(y_fit, return_counts=True)[1]
        inner_splits = min(args.inner_folds, int(class_counts.min()))
        if inner_splits < 2:
            raise ValueError("Not enough examples per class for nested cross-validation")
        inner_cv = StratifiedKFold(
            n_splits=inner_splits, shuffle=True, random_state=args.seed + fold
        )
        fold_search = GridSearchCV(
            clone(probe), param_grid, scoring="balanced_accuracy", cv=inner_cv,
            refit=True, n_jobs=args.cv_jobs, return_train_score=False,
        )
        fold_search.fit(features[fit_idx], y_fit)
        fold_pred = fold_search.predict(features[eval_idx])
        fold_score = balanced_accuracy_score(label_array[eval_idx], fold_pred)
        outer_scores.append(fold_score)
        selected_params = {k.split("__", 1)[-1]: v for k, v in fold_search.best_params_.items()
                           if k != "classifier"}
        print(f"outer_fold={fold} balanced_accuracy={fold_score:.4f} "
              f"inner_model={fold_search.best_params_['classifier'].__class__.__name__} "
              f"inner_params={selected_params}")
    print(f"Nested {n_splits}-fold balanced accuracy: "
          f"{np.mean(outer_scores):.4f} +/- {np.std(outer_scores, ddof=1):.4f}")

    # Select the final C on all labeled examples, then refit the probe on all features.
    final_cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=args.seed + n_splits)
    search = GridSearchCV(
        probe, param_grid, scoring="balanced_accuracy", cv=final_cv,
        refit=True, n_jobs=args.cv_jobs, return_train_score=False,
    )
    search.fit(features, label_array)
    print(f"Selected classifier: {search.best_params_['classifier'].__class__.__name__}; "
          f"parameters: { {k: v for k, v in search.best_params_.items() if k != 'classifier'} }")

    checkpoint = {
        "estimator": search.best_estimator_,
        "classes": classes,
        "model_name": args.model,
        "image_size": args.image_size,
        "last_blocks": args.last_blocks,
        "feature_recipe": "DINOv2 last-block CLS tokens + final mean patch tokens; horizontal-flip average",
    }
    output = Path(args.checkpoint).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(checkpoint, output)
    print(f"Saved fitted classifier: {output.resolve()}")


def predict(args):
    rows = read_submission(args.input_csv)
    image_root = Path(args.image_dir).expanduser().resolve()
    missing = [r["ID"] for r in rows if not (image_root / r["ID"]).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} submission images are missing; first: {missing[0]}")

    checkpoint = joblib.load(args.checkpoint)
    dataset_rows = [(r["ID"], None) for r in rows]
    dataset = ImageRows(dataset_rows, image_root, checkpoint["image_size"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    backbone = load_backbone(checkpoint["model_name"], device)
    features, _ = extract_features(backbone, loader, device, args.amp,
                                   checkpoint.get("last_blocks", 1))
    predictions = checkpoint["estimator"].predict(features)
    allowed = set(checkpoint["classes"])
    if len(predictions) != len(rows) or not set(predictions).issubset(allowed):
        raise RuntimeError("Prediction count or class labels failed validation")

    output = Path(args.output_csv).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "Category"])
        writer.writeheader()
        writer.writerows({"ID": row["ID"], "Category": str(pred)}
                         for row, pred in zip(rows, predictions))
    print(f"Wrote {len(predictions)} predictions to {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_audit = subparsers.add_parser("audit", help="Validate train classes, image counts, and submission IDs")
    p_audit.add_argument("--train-dir", required=True)
    p_audit.add_argument("--test-dir", required=True)
    p_audit.add_argument("--input-csv", required=True)
    p_audit.add_argument("--expected-classes", type=int, default=5)
    p_audit.add_argument("--expected-per-class", type=int, default=100)
    p_audit.add_argument("--expected-test-count", type=int, default=378)
    p_audit.set_defaults(func=audit)

    p_train = subparsers.add_parser("train", help="Extract frozen DINOv2 features and fit a validated linear probe")
    p_train.add_argument("--train-dir", required=True)
    p_train.add_argument("--checkpoint", default="plant_dinov2.joblib")
    p_train.add_argument("--model", default="dinov2_vitb14",
                         choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14"])
    p_train.add_argument("--image-size", type=int, default=518)
    p_train.add_argument("--last-blocks", type=int, default=4,
                         help="Number of final DINOv2 blocks whose CLS tokens are concatenated")
    p_train.add_argument("--batch-size", type=int, default=2)
    p_train.add_argument("--workers", type=int, default=4)
    p_train.add_argument("--folds", type=int, default=5)
    p_train.add_argument("--inner-folds", type=int, default=4)
    p_train.add_argument("--cv-jobs", type=int, default=1)
    p_train.add_argument("--expected-classes", type=int, default=5)
    p_train.add_argument("--expected-per-class", type=int, default=100,
                         help="Expected images per species; set to 0 to disable this check")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--amp", action="store_true", help="Use CUDA mixed precision for frozen feature extraction")
    p_train.set_defaults(func=train)

    p_predict = subparsers.add_parser("predict", help="Write predictions in submission CSV order")
    p_predict.add_argument("--input-csv", required=True)
    p_predict.add_argument("--image-dir", required=True)
    p_predict.add_argument("--checkpoint", required=True)
    p_predict.add_argument("--output-csv", default="submission.csv")
    p_predict.add_argument("--batch-size", type=int, default=2)
    p_predict.add_argument("--workers", type=int, default=4)
    p_predict.add_argument("--amp", action="store_true")
    p_predict.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
