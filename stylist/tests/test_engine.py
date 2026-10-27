"""Runs without CLIP: uses a fake encoder to verify catalog, filters, scoring, DB and training."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression

from clip_encoder import EncoderBase, _l2
from recommender import StyleRecommender
from build_catalog import from_folder, from_kaggle
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


def test_kaggle_alias_fields():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "images").mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 96), (255, 0, 0)).save(t / "images" / "1.jpg")
        pd.DataFrame([
            {"id": 1, "category": "Dress", "colour": "Red", "gender": "Women",
             "occasion": "Party", "season": "Summer", "productName": "Red Party Dress"}
        ]).to_csv(t / "styles.csv", index=False)
        df = from_kaggle(t, t, "Women", 10, ["Dress"])
        assert not df.empty
        assert df.iloc[0]["category"] == "Dress"
        assert df.iloc[0]["color"] == "Red"
        assert df.iloc[0]["occasion"] == "Party"
        assert df.iloc[0]["season"] == "Summer"
        assert df.iloc[0]["name"] == "Red Party Dress"


def test_folder_catalog_rebuild_sets_metadata():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "images").mkdir()
        Image.new("RGB", (64, 96), (255, 0, 0)).save(t / "images" / "1.jpg")
        Image.new("RGB", (64, 96), (0, 0, 255)).save(t / "images" / "2.jpg")
        df = from_folder(t, t, "Women", FakeEncoder())
        assert len(df) == 2
        assert set(df["category"]) - {""}  # non-empty tags
        assert set(df["color"]) - {""}
        assert set(df["occasion"]) - {""}


def test_folder_rebuild_stages_overlapping_source_before_renaming():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        image_dir = t / "images"
        image_dir.mkdir()
        Image.new("RGB", (64, 96), (255, 0, 0)).save(image_dir / "1.jpg")
        Image.new("RGB", (64, 96), (0, 255, 0)).save(image_dir / "10.jpg")
        Image.new("RGB", (64, 96), (0, 0, 255)).save(image_dir / "2.jpg")

        df = from_folder(t, t, "Women", FakeEncoder())

        assert len(df) == 3
        assert Image.open(image_dir / "2.jpg").getpixel((20, 20))[1] > 200
        assert Image.open(image_dir / "3.jpg").getpixel((20, 20))[2] > 200


def test_all():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        make_catalog(t / "cat")
        pd.DataFrame([
            dict(item_id=6, image="images/6.jpg", name="Sneakers", category="sneakers",
                 gender="Women", color="White", occasion="Casual", season="Summer"),
            dict(item_id=7, image="images/7.jpg", name="Boxer Briefs", category="boxer briefs",
                 gender="Men", color="Black", occasion="Casual", season="Summer"),
        ]).to_csv(
            t / "cat" / "catalog.csv", mode="a", header=False, index=False
        )
        Image.new("RGB", (64, 96), (240, 240, 240)).save(t / "cat" / "images" / "6.jpg")
        Image.new("RGB", (64, 96), (20, 20, 20)).save(t / "cat" / "images" / "7.jpg")
        rec = StyleRecommender(t / "cat", FakeEncoder())
        q = Image.new("RGB", (200, 300), (15, 15, 15))              # dark query
        out = rec.recommend(q, top_n=5)
        assert out.iloc[0]["color"] == "Black" and len(out) == 5, out
        assert "sneakers" not in set(out["category"].str.lower())
        assert "boxer briefs" not in set(out["category"].str.lower())
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
