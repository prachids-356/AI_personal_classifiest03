"""
Build catalog/catalog.csv + catalog/images/ from real fashion product images.

Option A - Kaggle "Fashion Product Images" (supports styles.csv metadata or images.csv):
    python build_catalog.py --kaggle-dir path/to/fashion-dataset --gender Women --per-category 150

Option B - any folder of clothing images (tags colour/category/occasion with CLIP zero-shot):
    python build_catalog.py --images-dir path/to/my_images --gender Women
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from catalog import CATEGORIES, COLORS, OCCASIONS

DEFAULT_KAGGLE_TYPES = ["Dresses", "Tops", "Tshirts", "Kurtas", "Jeans", "Trousers", "Skirts",
                        "Jumpsuit", "Jackets", "Blazers", "Sweaters", "Palazzos", "Shirts",
                        "Tunics", "Leggings", "Shorts", "Sarees"]


def save_resized(src: Path, dst: Path, max_side=512):
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=90)


def from_kaggle(kdir: Path, out: Path, gender, per_cat, types):
    metadata = kdir / "styles.csv"
    if not metadata.exists():
        metadata = kdir / "images.csv"
        if not metadata.exists():
            raise FileNotFoundError(f"Expected styles.csv or images.csv under {kdir}")
        images = pd.read_csv(metadata).dropna(subset=["filename"])
        images = images[images["filename"].astype(str).str.lower().str.endswith(".jpg")]
        images = images.sample(min(len(images), per_cat * max(len(types), 1)), random_state=0)
        rows = []
        for r in images.itertuples():
            filename = Path(str(r.filename)).name
            src = kdir / "images" / filename
            if not src.exists():
                continue
            try:
                item_id = int(Path(filename).stem)
                save_resized(src, out / "images" / filename)
            except (ValueError, OSError) as e:
                print("skip", src, e)
                continue
            rows.append(dict(item_id=item_id, image=f"images/{filename}",
                             name=f"Fashion item {item_id}", category="Unknown",
                             gender=gender or "Unisex", color="", occasion="", season=""))
        return pd.DataFrame(rows)

    styles = pd.read_csv(metadata, on_bad_lines="skip")
    styles = styles.dropna(subset=["id", "articleType"])
    if gender:
        styles = styles[styles["gender"].isin([gender, "Unisex"])]
    styles = styles[styles["articleType"].isin(types)]
    styles = (styles.groupby("articleType", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), per_cat), random_state=0)))
    rows = []
    for r in styles.itertuples():
        src = kdir / "images" / f"{int(r.id)}.jpg"
        if not src.exists():
            continue
        try:
            save_resized(src, out / "images" / f"{int(r.id)}.jpg")
        except Exception as e:
            print("skip", src, e)
            continue
        rows.append(dict(item_id=int(r.id), image=f"images/{int(r.id)}.jpg",
                         name=getattr(r, "productDisplayName", "") or r.articleType,
                         category=r.articleType, gender=r.gender,
                         color=getattr(r, "baseColour", ""), occasion=getattr(r, "usage", ""),
                         season=getattr(r, "season", "")))
    return pd.DataFrame(rows)


def zero_shot(encoder, img_emb, labels, template):
    t = encoder.encode_texts([template.format(l) for l in labels])
    return [labels[i] for i in (img_emb @ t.T).argmax(axis=1)]


def from_folder(idir: Path, out: Path, gender, encoder):
    files = sorted(p for p in idir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    rows, kept = [], []
    for i, p in enumerate(files, 1):
        dst = out / "images" / f"{i}.jpg"
        try:
            save_resized(p, dst)
        except Exception as e:
            print("skip", p, e)
            continue
        kept.append(dst)
        rows.append(dict(item_id=i, image=f"images/{i}.jpg", gender=gender or "Unisex", season=""))
    emb = encoder.encode_paths(kept)
    cats = zero_shot(encoder, emb, CATEGORIES, "a photo of a {} clothing item")
    cols = zero_shot(encoder, emb, COLORS, "a {} coloured garment")
    occ = zero_shot(encoder, emb, OCCASIONS, "a {} outfit")
    for r, c, co, o in zip(rows, cats, cols, occ):
        r.update(category=c, color=co.title(), occasion=o.title(), name=f"{co.title()} {c}".title())
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle-dir")
    ap.add_argument("--images-dir")
    ap.add_argument("--out", default="catalog")
    ap.add_argument("--gender", default="Women")
    ap.add_argument("--per-category", type=int, default=150)
    ap.add_argument("--types", nargs="*", default=DEFAULT_KAGGLE_TYPES)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.kaggle_dir:
        df = from_kaggle(Path(a.kaggle_dir), out, a.gender, a.per_category, a.types)
    elif a.images_dir:
        from clip_encoder import ClipEncoder
        df = from_folder(Path(a.images_dir), out, a.gender, ClipEncoder())
    else:
        ap.error("give --kaggle-dir or --images-dir")
    if df.empty:
        raise SystemExit("No items found - check the paths.")
    df.to_csv(out / "catalog.csv", index=False)
    (out / "embeddings.npz").unlink(missing_ok=True)
    print(f"Wrote {len(df)} items to {out/'catalog.csv'}")


if __name__ == "__main__":
    main()
