from pathlib import Path
from urllib.parse import quote

import streamlit as st
from PIL import Image

from body_type import BODY_RULES, BODY_TYPES, BodyTypeClassifier, canonical_body_type
from clip_encoder import ClipEncoder, open_rgb
from database import create_tables, get_feedback_scores, get_saved, save_feedback, save_item
from recommender import StyleRecommender

BASE_DIR = Path(__file__).resolve().parent
CATALOG_DIR = BASE_DIR / "catalog"
BODY_MODEL = BASE_DIR / "models" / "body_type.joblib"
USER = "guest"

st.set_page_config(page_title="AI Personal Fashion Stylist", page_icon="👗", layout="wide")
st.markdown(
    """<style>
    .block-container{max-width:1250px;padding-top:2rem}
    .chip{background:#ffe3ea;color:#ff3f6c;border-radius:99px;padding:.15rem .6rem;font-size:.8rem;font-weight:700}
    </style>""",
    unsafe_allow_html=True,
)
create_tables()


@st.cache_resource(show_spinner="Loading CLIP model (first run downloads ~600 MB)...")
def get_encoder():
    return ClipEncoder()


@st.cache_resource(show_spinner="Indexing catalog (only slow the first time)...")
def get_recommender():
    return StyleRecommender(CATALOG_DIR, get_encoder())


@st.cache_resource
def get_body_clf():
    return BodyTypeClassifier.load(BODY_MODEL) if BODY_MODEL.exists() else None


def on_feedback(item_id, vote):
    save_feedback(USER, item_id, vote)
    st.toast("Thanks - I'll use that next time" if vote == "Like" else "Noted, fewer like this")


def on_save(row):
    save_item(USER, row["item_id"], row["name"], row["image_path"])
    st.toast("Saved ❤️")


# ---------------- setup checks ----------------
if not (CATALOG_DIR / "catalog.csv").exists():
    st.error("No catalog yet. Build one first (see README):")
    st.code("python build_catalog.py --kaggle-dir path/to/fashion-dataset --gender Women", language="bash")
    st.stop()

rec = get_recommender()
body_clf = get_body_clf()
df = rec.df

# ---------------- header ----------------
st.title("👗 AI Personal Fashion Stylist")
st.caption("Upload a look you like. I find similar styles in the catalog, tuned to your body shape and filters.")

# ---------------- sidebar filters ----------------
with st.sidebar:
    st.header("Filters")
    genders = sorted(g for g in df["gender"].unique() if g)
    gender = st.multiselect("Gender", genders, default=[g for g in genders if g == "Women"])
    category = st.multiselect("Category", sorted(c for c in df["category"].unique() if c))
    color = st.multiselect("Colour", sorted(c for c in df["color"].unique() if c))
    occasion = st.multiselect("Occasion", sorted(c for c in df["occasion"].unique() if c))
    season = st.multiselect("Season", sorted(c for c in df["season"].unique() if c))
    max_price = None
    if "price" in df.columns:
        top = int(df["price"].max())
        max_price = st.slider("Max price", 0, top, top)

    st.header("Body shape")
    options = ["Auto-detect from photo" if body_clf else "(classifier not trained)", "None"] + [b.title() for b in BODY_TYPES]
    body_choice = st.selectbox("Body shape", options)
    body_weight = st.slider("Body-fit influence", 0.0, 0.6, 0.25, 0.05)

    st.header("Style")
    vibe = st.text_input("Extra vibe (optional)", placeholder="e.g. minimal, boho, office-ready")
    focus = st.checkbox("Focus on the outfit area of the photo", value=True)
    top_n = st.slider("Results", 4, 24, 12, 4)

# ---------------- upload ----------------
uploaded = st.file_uploader("Upload a photo (full-body works best)", type=["jpg", "jpeg", "png", "webp"])

if st.button("✨ Find my looks", type="primary"):
    if uploaded is None:
        st.warning("Please upload a photo first.")
    else:
        try:
            image = open_rgb(uploaded)
            body_probs, body_type = [], None
            if body_choice.startswith("Auto") and body_clf:
                body_probs = body_clf.predict(get_encoder(), image)
                body_type = canonical_body_type(body_probs[0][0])
            elif body_choice not in ("None",) and not body_choice.startswith("("):
                body_type = body_choice.lower()

            results = rec.recommend(
                image,
                body_type=body_type,
                filters={"gender": gender, "category": category, "color": color,
                         "occasion": occasion, "season": season},
                max_price=max_price,
                text=vibe,
                body_weight=body_weight,
                focus=focus,
                feedback=get_feedback_scores(USER),
                top_n=top_n,
            )
            st.session_state.update(results=results, image=image, body_type=body_type, body_probs=body_probs)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ---------------- results ----------------
if "results" in st.session_state:
    results = st.session_state["results"]
    left, right = st.columns([1, 2])
    with left:
        st.image(st.session_state["image"], caption="Your photo")
    with right:
        bt = st.session_state["body_type"]
        if bt:
            st.subheader(f"Body shape: {bt.title()}")
            for label, p in st.session_state["body_probs"][:3]:
                st.progress(min(max(p, 0.0), 1.0), text=f"{label} {p:.0%}")
            st.info(BODY_RULES[bt]["tip"])
        else:
            st.subheader("No body-shape filter applied")

    st.header("Shop your look")
    if results.empty:
        st.warning("No items match these filters. Try removing a few.")
    cols = st.columns(4)
    for i, row in results.iterrows():
        with cols[i % 4]:
            st.image(row["image_path"])
            st.markdown(f"**{row['name'][:60]}**")
            st.markdown(f'<span class="chip">{row["final_score"]:.0%} match</span>', unsafe_allow_html=True)
            st.caption(" · ".join(x for x in (row["category"], row["color"], row["occasion"]) if x))
            st.caption(row["reasons"])
            b1, b2, b3 = st.columns(3)
            iid = int(row["item_id"])
            b1.button("👍", key=f"up_{iid}", on_click=on_feedback, args=(iid, "Like"))
            b2.button("👎", key=f"dn_{iid}", on_click=on_feedback, args=(iid, "Dislike"))
            b3.button("❤️", key=f"sv_{iid}", on_click=on_save, args=(row.to_dict(),))
            st.markdown(f"[Find on Myntra](https://www.myntra.com/{quote(row['name'].replace(' ', '-'))})")

# ---------------- saved ----------------
st.markdown("---")
st.header("❤️ My saved outfits")
if st.button("Show saved outfits"):
    saved = get_saved(USER)
    if not saved:
        st.info("Nothing saved yet.")
    cols = st.columns(4)
    for i, s in enumerate(saved):
        with cols[i % 4]:
            if Path(s["image_path"]).exists():
                st.image(s["image_path"])
            st.caption(s["name"])
