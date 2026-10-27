"""Body-shape classifier (CLIP embedding + logistic regression) and styling rules."""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np

BODY_TYPES = ["hourglass", "pear", "apple", "rectangle", "inverted triangle"]

# General styling heuristics (same idea as the body-type fashion papers).
# Edit freely: "prefer"/"avoid" are matched as substrings against "<category> <product name>".
BODY_RULES = {
    "hourglass": {
        "prefer": ["wrap", "fit and flare", "bodycon", "sheath", "belt", "high waist",
                   "pencil", "jumpsuit", "dress"],
        "avoid": ["oversized", "boxy", "shapeless"],
        "tip": "Highlight the waist: wrap, belted and fitted silhouettes.",
    },
    "pear": {
        "prefer": ["a-line", "flared", "fit and flare", "off shoulder", "boat neck",
                   "blazer", "jacket", "structured", "bootcut", "wide leg", "palazzo", "peplum"],
        "avoid": ["skinny", "pencil", "cargo", "jegging"],
        "tip": "Draw attention upward; A-line and flared bottoms balance the hips.",
    },
    "apple": {
        "prefer": ["empire", "v-neck", "tunic", "kurta", "a-line", "flowy", "wrap", "straight"],
        "avoid": ["bodycon", "crop", "skinny", "tight"],
        "tip": "Empire waists, V-necks and flowy fabrics skim the midsection.",
    },
    "rectangle": {
        "prefer": ["peplum", "ruffle", "belt", "wrap", "layer", "pleated", "flared",
                   "fit and flare", "skater", "high waist"],
        "avoid": ["shift", "boxy"],
        "tip": "Create curves with belts, ruffles, peplum and layering.",
    },
    "inverted triangle": {
        "prefer": ["a-line", "flared", "wide leg", "palazzo", "skirt", "full", "bootcut", "scoop", "v-neck"],
        "avoid": ["shoulder pad", "puff", "halter", "off shoulder", "padded", "boat neck"],
        "tip": "Add volume below the waist; keep the shoulder line simple.",
    },
}


def canonical_body_type(label) -> str | None:
    """Map whatever class names your Roboflow dataset uses onto the 5 standard shapes."""
    s = re.sub(r"[^a-z]", "", str(label).lower())
    if "invert" in s:
        return "inverted triangle"
    if "hourglass" in s:
        return "hourglass"
    if "pear" in s or s in ("triangle", "spoon"):
        return "pear"
    if any(k in s for k in ("apple", "round", "oval", "diamond")):
        return "apple"
    if any(k in s for k in ("rect", "straight", "banana", "column")):
        return "rectangle"
    return None


def body_fit(category: str, name: str, body_type: str | None):
    """Return (score in {-1,0,1}, reason or None)."""
    bt = canonical_body_type(body_type) if body_type else None
    if bt is None:
        return 0.0, None
    rule = BODY_RULES[bt]
    text = f"{category} {name}".lower()
    for w in rule["avoid"]:
        if w in text:
            return -1.0, f"Less ideal for {bt} ({w})"
    for w in rule["prefer"]:
        if w in text:
            return 1.0, f"Flatters {bt} shape ({w})"
    return 0.0, None


class BodyTypeClassifier:
    def __init__(self, clf, classes):
        self.clf, self.classes = clf, list(classes)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        return cls(d["clf"], d["classes"])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "classes": self.classes}, path)

    def predict(self, encoder, image):
        """Return [(label, probability)] sorted high to low. Use a full-body photo."""
        emb = encoder.encode_images([image])
        p = self.clf.predict_proba(emb)[0]
        order = np.argsort(-p)
        return [(str(self.clf.classes_[i]), float(p[i])) for i in order]
