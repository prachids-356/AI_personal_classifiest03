"""
Build catalog/catalog.csv + catalog/images/ from real fashion product images.

Option A - Kaggle "Fashion Product Images" (has styles.csv with colour/category/gender/usage):
    python build_catalog.py --kaggle-dir path/to/fashion-dataset --gender Women --per-category 150

Option B - any folder of clothing images (tags colour/category/occasion with CLIP zero-shot):
    python build_catalog.py --images-dir path/to/my_images --gender Women
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from catalog import CATEGORIES, CATEGORY_PROMPTS, COLORS, OCCASIONS

DEFAULT_KAGGLE_TYPES = ["Dresses", "Tops", "Tshirts", "Kurtas", "Jeans", "Trousers", "Skirts",
                        "Jumpsuit", "Jackets", "Blazers", "Sweaters", "Palazzos", "Shirts",
                        "Tunics", "Leggings", "Shorts", "Sarees"]

COLUMN_ALIASES = {
    "articleType": ["articleType", "category", "type", "subCategory"],
    "productDisplayName": ["productDisplayName", "productName", "name"],
    "baseColour": ["baseColour", "baseColor", "primaryColour", "colour", "color"],
    "usage": ["usage", "usageType", "occasion", "event"],
    "gender": ["gender", "genderName", "targetGender"],
    "season": ["season", "seasonName"],
}


def _clean(value):
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    if value.lower() in {"nan", "none", "unknown", "n/a"}:
        return ""
    return value


def _pick(row, *names):
    for name in names:
        if hasattr(row, name):
            val = _clean(getattr(row, name))
            if val:
                return val
    return ""


def save_resized(src: Path, dst: Path, max_side=512):
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, "JPEG", quality=90)


def from_kaggle(kdir: Path, out: Path, gender, per_cat, types):
    styles = pd.read_csv(kdir / "styles.csv", on_bad_lines="skip")
    if styles.empty:
        return pd.DataFrame()

    id_col = next((c for c in ["id", "item_id"] if c in styles.columns), None)
    article_col = next((c for c in ["articleType", "category", "type", "subCategory"] if c in styles.columns), None)
    display_col = next((c for c in ["productDisplayName", "productName", "name"] if c in styles.columns), None)
    base_col = next((c for c in ["baseColour", "baseColor", "primaryColour", "colour", "color"] if c in styles.columns), None)
    usage_col = next((c for c in ["usage", "usageType", "occasion", "event"] if c in styles.columns), None)
    gender_col = next((c for c in ["gender", "genderName", "targetGender"] if c in styles.columns), None)
    season_col = next((c for c in ["season", "seasonName"] if c in styles.columns), None)

    if id_col is None or article_col is None:
        return pd.DataFrame()

    styles = styles[[id_col, article_col, *( [display_col] if display_col else [] ), *( [base_col] if base_col else [] ), *( [usage_col] if usage_col else [] ), *( [gender_col] if gender_col else [] ), *( [season_col] if season_col else [] )]].copy()
    styles = styles.rename(columns={
        id_col: "id",
        article_col: "articleType",
        **({display_col: "productDisplayName"} if display_col else {}),
        **({base_col: "baseColour"} if base_col else {}),
        **({usage_col: "usage"} if usage_col else {}),
        **({gender_col: "gender"} if gender_col else {}),
        **({season_col: "season"} if season_col else {}),
    })

    styles["id"] = pd.to_numeric(styles["id"], errors="coerce")
    styles = styles.dropna(subset=["id"]).copy()
    styles["articleType"] = styles["articleType"].map(_clean)
    styles["productDisplayName"] = styles.get("productDisplayName", pd.Series("", index=styles.index)).map(_clean)
    styles["baseColour"] = styles.get("baseColour", pd.Series("", index=styles.index)).map(_clean)
    styles["usage"] = styles.get("usage", pd.Series("", index=styles.index)).map(_clean)
    styles["gender"] = styles.get("gender", pd.Series("", index=styles.index)).map(_clean)
    styles["season"] = styles.get("season", pd.Series("", index=styles.index)).map(_clean)

    styles = styles[styles["articleType"].ne("")]
    if gender:
        allowed_gender = {str(gender).lower(), "unisex"}
        styles = styles[styles["gender"].fillna("").astype(str).str.lower().isin(allowed_gender)]

    allowed_types = {str(t).strip().lower() for t in (types or []) if str(t).strip()}
    if allowed_types:
        type_matches = set()
        for t in allowed_types:
            type_matches.add(t)
            if t.endswith("s"):
                type_matches.add(t[:-1])
            else:
                type_matches.add(t + "s")
        styles = styles[styles["articleType"].fillna("").astype(str).str.strip().str.lower().isin(type_matches)]

    if styles.empty:
        return styles

    sampled = []
    for _, g in styles.groupby("articleType", sort=False):
        sampled.append(g.sample(n=min(len(g), per_cat), random_state=0))
    styles = pd.concat(sampled, ignore_index=True) if sampled else styles.iloc[0:0]

    rows = []
    for _, r in styles.iterrows():
        article_type = str(r.get("articleType", "") or "").strip()
        if not article_type:
            continue
        item_id = int(r["id"])
        src = kdir / "images" / f"{item_id}.jpg"
        if not src.exists():
            continue
        try:
            save_resized(src, out / "images" / f"{item_id}.jpg")
        except Exception as e:
            print("skip", src, e)
            continue
        rows.append(dict(
            item_id=item_id,
            image=f"images/{item_id}.jpg",
            name=(str(r.get("productDisplayName", "") or article_type)).strip() or article_type,
            category=article_type,
            gender=(str(r.get("gender", "") or str(gender) or "Unisex")).strip() or (gender or "Unisex"),
            color=str(r.get("baseColour", "") or ""),
            occasion=str(r.get("usage", "") or ""),
            season=str(r.get("season", "") or ""),
        ))
    return pd.DataFrame(rows)


def zero_shot(encoder, img_emb, labels, template):
    t = encoder.encode_texts([template.format(l) for l in labels])
    return [labels[i] for i in (img_emb @ t.T).argmax(axis=1)]


def classify_categories(encoder, img_emb):
    prompts = [CATEGORY_PROMPTS.get(label, f"a product photo of {label} clothing") for label in CATEGORIES]
    text_emb = encoder.encode_texts(prompts)
    return [CATEGORIES[i] for i in (img_emb @ text_emb.T).argmax(axis=1)]


def from_folder(idir: Path, out: Path, gender, encoder):
    files = sorted(p for p in idir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    source_dir = idir.resolve()
    output_dir = out.resolve()
    overlaps = source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents
    if overlaps:
        with tempfile.TemporaryDirectory() as temp_dir:
            staged_files = []
            for index, src in enumerate(files):
                staged = Path(temp_dir) / f"{index}{src.suffix.lower()}"
                shutil.copy2(src, staged)
                staged_files.append(staged)
            return _build_folder_catalog(staged_files, out, gender, encoder)
    return _build_folder_catalog(files, out, gender, encoder)


def _build_folder_catalog(files, out: Path, gender, encoder):
    rows, kept = [], []
    for i, p in enumerate(files, 1):
        dst = out / "images" / f"{i}.jpg"
        try:
            save_resized(p, dst)
        except Exception as e:
            print("skip", p, e)
            continue
        kept.append(dst)
        rows.append(dict(item_id=i, image=f"images/{i}.jpg", gender=gender or "Unisex", season="All season"))
    emb = encoder.encode_paths(kept)
    cats = classify_categories(encoder, emb)
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
