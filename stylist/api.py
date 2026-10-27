"""HTTP API for the fashion stylist web client."""
from __future__ import annotations

import io
import json
import math
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field

from body_type import BODY_RULES, BODY_TYPES, BodyTypeClassifier, canonical_body_type
from catalog import CATEGORIES, CATEGORY_PROMPTS
from clip_encoder import ClipEncoder, open_rgb
from database import create_tables, get_feedback_scores, get_saved, save_feedback, save_item
from recommender import StyleRecommender

BASE_DIR = Path(__file__).resolve().parent
CATALOG_DIR = BASE_DIR / "catalog"
BODY_MODEL = BASE_DIR / "models" / "body_type.joblib"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    encoder = ClipEncoder()
    app.state.encoder = encoder
    app.state.recommender = StyleRecommender(CATALOG_DIR, encoder)
    app.state.body_classifier = BodyTypeClassifier.load(BODY_MODEL) if BODY_MODEL.exists() else None
    app.state.garment_labels = CATEGORIES
    app.state.garment_text_embeddings = encoder.encode_texts([
        CATEGORY_PROMPTS.get(category, f"a catalog product photo of {category}")
        for category in CATEGORIES
    ])
    yield


app = FastAPI(title="Style Muse API", lifespan=lifespan)


@app.middleware("http")
async def browser_session(request: Request, call_next):
    response = await call_next(request)
    if not request.cookies.get("style_session"):
        response.set_cookie(
            "style_session", secrets.token_urlsafe(24), httponly=True,
            samesite="lax", secure=False, max_age=60 * 60 * 24 * 365,
        )
    return response


def _user(request: Request) -> str:
    return request.cookies.get("style_session", "guest")


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _image_url(image: str) -> str:
    return "/catalog/" + quote(str(image).replace("\\", "/"), safe="/")


def _item_payload(row) -> dict:
    item_value = row.get if isinstance(row, dict) else lambda key, default=None: row[key] if key in row.index else default
    item = {
        "item_id": int(item_value("item_id")),
        "name": str(item_value("name", "Fashion look")),
        "category": str(item_value("category", "")),
        "color": str(item_value("color", "")),
        "occasion": str(item_value("occasion", "")),
        "season": str(item_value("season", "")),
        "image_url": _image_url(row["image"]),
    }
    for key in ("final_score", "visual_score", "reasons"):
        value = item_value(key)
        if value is not None:
            item[key] = _clean_value(value)
    return item


def _catalog_item(app: FastAPI, item_id: int):
    matches = app.state.recommender.df[app.state.recommender.df["item_id"].astype(int).eq(item_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Look not found")
    return matches.iloc[0]


async def _uploaded_image(image: UploadFile):
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WEBP image")
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 12 MB or smaller")
    try:
        return open_rgb(io.BytesIO(content))
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="That image could not be opened") from exc


class FeedbackInput(BaseModel):
    vote: str = Field(pattern="^(Like|Dislike)$")


class SavedInput(BaseModel):
    item_id: int


@app.get("/api/session")
def session():
    return {"ready": True}


@app.get("/api/options")
def options(request: Request):
    df = request.app.state.recommender.df
    samples = []
    for category in ("dress", "kurta", "t-shirt", "top"):
        candidates = df[df["category"].astype(str).str.lower().eq(category)]
        if not candidates.empty:
            samples.append(_item_payload(candidates.iloc[0]))
    return {
        "gender": sorted(x for x in df["gender"].dropna().astype(str).unique() if x),
        "category": sorted(x for x in df["category"].dropna().astype(str).unique() if x),
        "color": sorted(x for x in df["color"].dropna().astype(str).unique() if x),
        "occasion": sorted(x for x in df["occasion"].dropna().astype(str).unique() if x),
        "season": sorted(x for x in df["season"].dropna().astype(str).unique() if x),
        "body_types": BODY_TYPES,
        "body_tips": {key: value["tip"] for key, value in BODY_RULES.items()},
        "catalog_count": int(len(df)),
        "body_model_available": request.app.state.body_classifier is not None,
        "sample_images": samples,
    }


@app.post("/api/classify")
async def classify_image(request: Request, image: UploadFile = File(...)):
    query_image = await _uploaded_image(image)
    image_embedding = request.app.state.encoder.encode_images([query_image])[0]
    similarities = request.app.state.garment_text_embeddings @ image_embedding
    top_indices = np.argsort(similarities)[::-1][:5]
    predictions = [
        {"category": request.app.state.garment_labels[index], "similarity": round(float(similarities[index]), 4)}
        for index in top_indices
    ]
    return {
        "predictions": predictions,
        "score_label": "CLIP image-text similarity",
        "note": "Similarity ranks candidate labels; it is not a calibrated probability.",
    }


@app.post("/api/recommendations")
async def recommendations(
    request: Request,
    image: UploadFile = File(...),
    options: str = "{}",
):
    try:
        query_image = await _uploaded_image(image)
        settings = json.loads(options)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="That image could not be opened") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Recommendation settings are invalid") from exc

    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="Recommendation settings are invalid")

    body_type = settings.get("body_type") or None
    body_probs = []
    classifier = request.app.state.body_classifier
    if settings.get("auto_body_type") and classifier:
        body_probs = classifier.predict(request.app.state.encoder, query_image)
        body_type = canonical_body_type(body_probs[0][0]) if body_probs else None
    elif body_type:
        body_type = canonical_body_type(body_type)

    allowed_filters = ("gender", "category", "color", "occasion", "season")
    filters = {key: settings.get("filters", {}).get(key, []) for key in allowed_filters}
    try:
        results = request.app.state.recommender.recommend(
            query_image,
            body_type=body_type,
            filters=filters,
            max_price=settings.get("max_price"),
            text=str(settings.get("vibe", "")),
            body_weight=float(settings.get("body_weight", 0.25)),
            focus=bool(settings.get("focus", True)),
            feedback=get_feedback_scores(_user(request)),
            top_n=max(1, min(int(settings.get("top_n", 12)), 24)),
            variety=max(0.0, min(float(settings.get("variety", 0.45)), 1.0)),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Recommendation settings are invalid") from exc

    return {
        "items": [_item_payload(row) for _, row in results.iterrows()],
        "body_type": body_type,
        "body_probabilities": [{"label": label, "probability": float(prob)} for label, prob in body_probs[:3]],
        "body_tip": BODY_RULES[body_type]["tip"] if body_type in BODY_RULES else None,
    }


@app.post("/api/items/{item_id}/feedback")
def feedback(item_id: int, data: FeedbackInput, request: Request):
    _catalog_item(request.app, item_id)
    save_feedback(_user(request), item_id, data.vote)
    return {"saved": True, "vote": data.vote}


@app.get("/api/saved-items")
def saved_items(request: Request):
    user = _user(request)
    saved = get_saved(user)
    items = []
    for saved_item in saved:
        row = _catalog_item(request.app, int(saved_item["item_id"]))
        items.append(_item_payload(row))
    return {"items": items}


@app.post("/api/saved-items")
def add_saved_item(data: SavedInput, request: Request):
    row = _catalog_item(request.app, data.item_id)
    save_item(_user(request), data.item_id, row["name"], row["image_path"])
    return {"saved": True, "item": _item_payload(row)}


app.mount("/catalog", StaticFiles(directory=CATALOG_DIR), name="catalog")

FRONTEND_DIST = BASE_DIR / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")

    @app.get("/{path:path}")
    def frontend(path: str):
        candidate = FRONTEND_DIST / path
        if path and candidate.is_file() and FRONTEND_DIST in candidate.resolve().parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")