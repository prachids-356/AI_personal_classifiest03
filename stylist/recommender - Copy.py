"""Visual style matching + filters + body-shape fit."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from body_type import body_fit

OPTIONAL_TEXT_COLS = ["name", "category", "gender", "color", "occasion", "season"]


def crop_clothing(img: Image.Image) -> Image.Image:
    """Drop head/feet/background margins so CLIP focuses on the outfit."""
    w, h = img.size
    return img.crop((int(w * 0.10), int(h * 0.15), int(w * 0.90), int(h * 0.90)))


class StyleRecommender:
    def __init__(self, catalog_dir, encoder):
        self.dir = Path(catalog_dir)
        csv = self.dir / "catalog.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv} not found. Run build_catalog.py first (see README).")
        df = pd.read_csv(csv)
        if "image" not in df.columns:
            raise ValueError("catalog.csv needs an 'image' column (path relative to the catalog folder)")
        for c in OPTIONAL_TEXT_COLS:
            if c not in df.columns:
                df[c] = ""
            df[c] = df[c].fillna("").astype(str)
        if "item_id" not in df.columns:
            df["item_id"] = np.arange(1, len(df) + 1)
        df["image_path"] = df["image"].apply(lambda p: str(self.dir / p))
        ok = df["image_path"].apply(os.path.exists)
        self.skipped = int((~ok).sum())
        df = df[ok].reset_index(drop=True)
        if df.empty:
            raise RuntimeError("No catalog images found on disk.")
        self.df, self.encoder = df, encoder
        self.emb = self._load_embeddings()

    def _load_embeddings(self):
        key = hashlib.sha1(
            "|".join(f"{p}:{os.path.getsize(p)}" for p in self.df["image_path"]).encode()
        ).hexdigest()
        cache = self.dir / "embeddings.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            if str(z["key"]) == key:
                return z["emb"]
        emb = self.encoder.encode_paths(list(self.df["image_path"]))
        np.savez(cache, emb=emb, key=np.array(key))
        return emb

    def embed_query(self, image, text=None, text_weight=0.5, focus=True):
        q = self.encoder.encode_images([crop_clothing(image) if focus else image])[0]
        if text and text.strip():
            q = q + text_weight * self.encoder.encode_texts([text.strip()])[0]
            q = q / np.linalg.norm(q)
        return q

    def recommend(self, image, *, body_type=None, filters=None, max_price=None, text=None,
                  body_weight=0.25, focus=True, feedback=None, top_n=12):
        """
        filters: {"color": [...], "category": [...], "gender": [...], "occasion": [...], "season": [...]}
        feedback: {item_id: +1 / -1} from earlier likes/dislikes
        """
        df = self.df
        mask = np.ones(len(df), bool)
        for col, allowed in (filters or {}).items():
            if allowed and col in df.columns:
                low = {str(a).lower() for a in allowed}
                col_vals = df[col].str.lower()
                m = col_vals.isin(low)
                if col == "gender":              # Unisex fits everyone
                    m |= col_vals.eq("unisex")
                mask &= m.to_numpy()
        if max_price is not None and "price" in df.columns:
            mask &= (pd.to_numeric(df["price"], errors="coerce").fillna(0) <= max_price).to_numpy()
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return df.iloc[0:0].assign(visual_score=[], body_score=[], final_score=[], reasons=[])

        q = self.embed_query(image, text, focus=focus)
        sims = self.emb[idx] @ q
        visual = np.clip((sims - 0.5) / 0.45, 0, 1)   # display scale: cosine 0.5 -> 0, 0.95 -> 1

        sub = df.iloc[idx].copy()
        fits, why = zip(*[body_fit(c, n, body_type) for c, n in zip(sub["category"], sub["name"])])
        fits = np.array(fits)
        wb = body_weight if body_type else 0.0
        fb = np.array([np.sign((feedback or {}).get(int(i), 0)) for i in sub["item_id"]]) * 0.08

        sub["visual_score"] = visual
        sub["body_score"] = fits
        sub["final_score"] = np.clip((1 - wb) * visual + wb * (fits + 1) / 2 + fb, 0, 1)
        sub["reasons"] = [
            " · ".join(filter(None, [f"Visual match {v:.0%}", w, "You liked this" if f > 0 else ("You disliked this" if f < 0 else None)]))
            for v, w, f in zip(visual, why, fb)
        ]
        return sub.sort_values("final_score", ascending=False).head(top_n).reset_index(drop=True)
