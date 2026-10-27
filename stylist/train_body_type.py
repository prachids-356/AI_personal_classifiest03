"""
Train the body-shape classifier on your Roboflow dataset.

    # download + train (get your API key from Roboflow > Settings > API Keys)
    python train_body_type.py --roboflow-key YOUR_KEY --version 1

    # or train on an already-downloaded folder
    python train_body_type.py --data path/to/dataset

Works with Roboflow "Folder structure" (classification) exports, and with YOLO-format
(object-detection) exports, where the class of the largest box in each image is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from body_type import BodyTypeClassifier
from clip_encoder import ClipEncoder

IMG = {".jpg", ".jpeg", ".png", ".webp"}
WORKSPACE, PROJECT = "prachi-singh-lkcsm", "woman-body-shape-nwnj8-0adzt"


def download(key, version, dest):
    from roboflow import Roboflow
    ver = Roboflow(api_key=key).workspace(WORKSPACE).project(PROJECT).version(version)
    for fmt in ("folder", "yolov8"):
        try:
            return ver.download(fmt, location=str(dest)).location
        except Exception as e:
            print(f"'{fmt}' export failed ({e}); trying next format")
    raise SystemExit("Could not download. Check API key / version number.")


def yolo_label(label_file: Path, names):
    best, area = None, -1
    for line in label_file.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        a = float(p[3]) * float(p[4]) if len(p) == 5 else 1.0
        if a > area:
            best, area = names[int(p[0])], a
    return best


def collect(root: Path):
    samples = []
    names = None
    yaml_file = root / "data.yaml"
    if yaml_file.exists():
        n = yaml.safe_load(yaml_file.read_text()).get("names")
        names = list(n.values()) if isinstance(n, dict) else n
    for split in ("train", "valid", "val", "test"):
        d = root / split
        if not d.is_dir():
            continue
        if (d / "images").is_dir() and (d / "labels").is_dir():       # YOLO layout
            for img in (d / "images").iterdir():
                lf = d / "labels" / (img.stem + ".txt")
                if img.suffix.lower() in IMG and lf.exists() and names:
                    lab = yolo_label(lf, names)
                    if lab:
                        samples.append((img, lab))
        else:                                                         # folder-per-class layout
            for cdir in d.iterdir():
                if cdir.is_dir():
                    samples += [(p, cdir.name) for p in cdir.iterdir() if p.suffix.lower() in IMG]
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--roboflow-key")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--out", default="models/body_type.joblib")
    a = ap.parse_args()

    root = Path(a.data) if a.data else Path(download(a.roboflow_key, a.version, "body_shape_dataset"))
    samples = collect(root)
    if not samples:
        raise SystemExit(f"No labelled images found under {root}")
    counts = Counter(l for _, l in samples)
    print(f"{len(samples)} images, classes: {dict(counts)}")

    enc = ClipEncoder()
    X = enc.encode_paths([p for p, _ in samples], progress=lambda i, n: print(f"\rembedding {i}/{n}", end=""))
    print()
    y = [l for _, l in samples]

    strat = y if min(counts.values()) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=strat)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(Xtr, ytr)
    print(classification_report(yte, clf.predict(Xte), zero_division=0))

    clf.fit(X, y)  # final model uses all data
    BodyTypeClassifier(clf, clf.classes_).save(a.out)
    print("Saved", a.out)


if __name__ == "__main__":
    main()
