# AI Personal Fashion Stylist (CLIP + body shape)

Upload a photo -> CLIP finds visually similar outfits in a real catalog -> results are
filtered (colour, category, occasion, season, gender, price) and re-ranked by body-shape fit.

## Setup
```bash
pip install -r requirements.txt
```
(Delete your old `visual_recommender.py`, `recommendation.py`, `ml_recommender.py`, `image_analysis.py`,
`myntra_ui.py`, `outfits.csv` and `fashion.db`; they are replaced by the files here.)

## 1. Build the catalog (needs real product images)
Kaggle "Fashion Product Images" (has colour/category/gender/occasion in styles.csv):
```bash
python build_catalog.py --kaggle-dir path/to/fashion-dataset --gender Women --per-category 150
```
Or any folder of clothing images (CLIP auto-tags category/colour/occasion):
```bash
python build_catalog.py --images-dir path/to/images --gender Women
```

## 2. Train the body-shape classifier on your Roboflow dataset
```bash
python train_body_type.py --roboflow-key YOUR_KEY --version 1
```
Prints per-class accuracy, saves `models/body_type.joblib`. Class names are mapped to
hourglass / pear / apple / rectangle / inverted triangle in `body_type.py`.

## 3. Run
```bash
streamlit run app.py
```

## Check the logic without downloading CLIP
```bash
python tests/test_engine.py
```

## Files
- `clip_encoder.py` CLIP via Hugging Face transformers
- `recommender.py` similarity search, filters, body-fit and feedback re-ranking
- `body_type.py` classifier + styling rules (edit `BODY_RULES` to taste)
- `build_catalog.py`, `train_body_type.py`, `database.py`, `app.py`
