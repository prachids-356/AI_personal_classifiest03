# Style Muse

## Run the web app

Build the React frontend once, then start the API and frontend from the `stylist` directory:

```powershell
cd stylist/frontend
npm install
npm run build
cd ..
..\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Open http://localhost:8000. On first startup, the API loads the CLIP model and indexes the local catalog; later starts use the saved image embeddings.

For frontend development, run the API as above and use another terminal:

```powershell
cd stylist/frontend
npm run dev
```

The Vite server proxies `/api` and `/catalog` requests to the Python API.

## Legacy interface

The Streamlit app remains available with `python -m streamlit run app.py`, but the React client is the primary interface.
# AI Personal Fashion Stylist (CLIP + body shape)

Upload a photo -> CLIP finds visually similar outfits in a real catalog -> results are
filtered (colour, category, occasion, season, gender, price) and re-ranked by body-shape fit.

## Setup
```bash
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
