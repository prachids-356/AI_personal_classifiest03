"""Runs without CLIP: uses a fake encoder to verify catalog, filters, scoring, DB and training."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

from clip_encoder import EncoderBase, _l2
from recommender import StyleRecommender
from body_type import BodyTypeClassifier, canonical_body_type, body_fit


class FakeEncoder(EncoderBase):
    def encode_images(self, images):
        return _l2(np.stack([np.asarray(im.resize((8, 8))).astype(np.float32).ravel() for im in images]))
    def encode_texts(self, texts):
        return _l2(np.stack([np.random.RandomState(abs(hash(t)) % 2**31).rand(192) for t in texts]).astype(np.float32))


def make_catalog(d):
    (d / "images").mkdir(parents=True)
    rows = []
    specs = [("Black", (10, 10, 10), "Dresses", "A-Line Dress"), ("Blue", (20, 40, 200), "Jeans", "Skinny Jeans"),
             ("Red", (200, 20, 20), "Skirts", "Flared Skirt"), ("Black", (30, 30, 30), "Tops", "Crop Top"),
             ("White", (240, 240, 240), "Kurtas", "Straight Kurta")]
    for i, (col, rgb, cat, name) in enumerate(specs, 1):
        Image.new("RGB", (64, 96), rgb).save(d / "images" / f"{i}.jpg")
        rows.append(dict(item_id=i, image=f"images/{i}.jpg", name=name, category=cat, gender="Women",
                         color=col, occasion="Casual", season="Summer"))
    pd.DataFrame(rows).to_csv(d / "catalog.csv", index=False)


def test_all():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        make_catalog(t / "cat")
        rec = StyleRecommender(t / "cat", FakeEncoder())
        q = Image.new("RGB", (200, 300), (15, 15, 15))              # dark query
        out = rec.recommend(q, top_n=5)
        assert out.iloc[0]["color"] == "Black" and len(out) == 5, out
        out = rec.recommend(q, filters={"color": ["blue"]})
        assert list(out["name"]) == ["Skinny Jeans"]
        assert rec.recommend(q, filters={"color": ["Pink"]}).empty
        out = rec.recommend(q, body_type="pear", body_weight=0.5, filters={"color": ["black", "red"]})
        assert out.iloc[0]["name"] in ("A-Line Dress", "Flared Skirt")
        out = rec.recommend(q, feedback={4: -1, 1: 1}, filters={"color": ["black"]})
        assert out.iloc[0]["item_id"] == 1
        rec2 = StyleRecommender(t / "cat", FakeEncoder())             # loads cache
        assert np.allclose(rec.emb, rec2.emb)
        assert canonical_body_type("InvertedTriangle") == "inverted triangle"
        assert canonical_body_type("Pear") == "pear" and canonical_body_type("Hourglass") == "hourglass"
        assert body_fit("Skirts", "Pencil Skirt", "pear")[0] == -1
        X = FakeEncoder().encode_images([Image.new("RGB", (8, 8), c) for c in [(255, 0, 0)] * 3 + [(0, 0, 255)] * 3])
        clf = LogisticRegression().fit(X, ["pear"] * 3 + ["apple"] * 3)
        bc = BodyTypeClassifier(clf, clf.classes_); bc.save(t / "m.joblib")
        lab, p = BodyTypeClassifier.load(t / "m.joblib").predict(FakeEncoder(), Image.new("RGB", (8, 8), (255, 0, 0)))[0]
        assert lab == "pear"
        import database as db
        db.DB_PATH = t / "x.db"; db.create_tables()
        db.save_feedback("u", 1, "Like"); db.save_feedback("u", 1, "Dislike"); db.save_item("u", 2, "n", "p")
        assert db.get_feedback_scores("u") == {1: -1} and len(db.get_saved("u")) == 1
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test_all()
