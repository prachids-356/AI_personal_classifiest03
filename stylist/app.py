from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
from PIL import Image

from body_type import BODY_RULES, BODY_TYPES, BodyTypeClassifier, canonical_body_type
from build_catalog import from_folder
from clip_encoder import ClipEncoder, open_rgb
from database import create_tables, get_feedback_scores, get_saved, save_feedback, save_item
from recommender import StyleRecommender

BASE_DIR = Path(__file__).resolve().parent
CATALOG_DIR = BASE_DIR / "catalog"
BODY_MODEL = BASE_DIR / "models" / "body_type.joblib"
USER = "guest"

st.set_page_config(
    page_title="AI Personal Fashion Stylist",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """<style>
    :root{--ink:#273238;--muted:#748087;--line:#ece8e9;--rose:#e44f7b;--rose-soft:#fff1f5}
    .block-container{max-width:1440px;padding:1rem 2.1rem 3rem}
    [data-testid="stAppViewContainer"]{background:#fff}
    [data-testid="stSidebar"]{background:#fff8fa;border-right:1px solid #f1e4e8}
    [data-testid="stSidebar"] h2{font-family:Georgia,serif;color:var(--ink);font-size:1.25rem}
    h1,h2,h3{color:var(--ink);letter-spacing:0}
    h1,h2{font-family:Georgia,"Times New Roman",serif}
    .topbar{height:54px;border-bottom:1px solid #f1edef;display:flex;align-items:center;justify-content:space-between;margin-bottom:3rem;color:#45535a}
    .wordmark{font:italic 600 1.7rem Georgia,"Times New Roman",serif;color:var(--rose)}
    .topline-meta{font-size:.66rem;font-weight:700;letter-spacing:.16em;color:#43535b}
    .fashion-header{padding:1rem 0 .9rem;margin:0}
    .fashion-kicker,.mini-label{letter-spacing:.15em;text-transform:uppercase;font-size:.68rem;font-weight:800;color:var(--rose);margin-bottom:.7rem}
    .fashion-title{font:300 2.8rem/1.2 "Segoe UI Light","Aptos Light",sans-serif!important;text-transform:uppercase;margin:0;color:var(--rose)!important;max-width:640px}
    .fashion-subtitle{max-width:620px;font-size:1rem;color:#7a858b!important;margin-top:1rem;line-height:1.8}
    .stat-pill{background:transparent;border:0;border-left:2px solid #f1a9bd;border-radius:0;padding:.1rem .65rem;font-size:.76rem;color:#657178;display:inline-flex;align-items:center;gap:.4rem;margin:.75rem .45rem 0 0}
    .stat-pill strong{color:#39464c;text-transform:uppercase;font-size:.68rem;letter-spacing:.06em}
    [data-testid="stFileUploader"]{background:#fff;border:1px dashed #ed9ab2;border-radius:4px;padding:.35rem}
    [data-testid="stFileUploader"] section{background:transparent}
    [data-testid="stButton"] button{border-radius:4px;transition:transform .15s ease,box-shadow .15s ease}
    [data-testid="stButton"] button:hover{transform:translateY(-1px);box-shadow:0 5px 14px rgba(32,40,44,.09)}
    [data-testid="stBaseButton-primary"]{background:var(--rose);border-color:var(--rose);min-height:48px;padding:0 1.5rem;font-size:.82rem;font-weight:800;letter-spacing:.04em}
    [data-testid="stImage"] img{border-radius:4px}
    [data-testid="stVerticalBlockBorderWrapper"]{border-color:var(--line)!important;border-radius:6px!important;background:#fff}
    div[class*="st-key-result-card-"]{background:#fff;border:1px solid var(--line);border-radius:6px;padding:.7rem;height:100%;box-shadow:0 7px 20px rgba(32,40,44,.025);transition:transform .18s ease,box-shadow .18s ease}
    div[class*="st-key-result-card-"]:hover{transform:translateY(-3px);box-shadow:0 14px 28px rgba(32,40,44,.08)}
    div[class*="st-key-result-card-"] [data-testid="stImage"] img{width:100%;height:280px;object-fit:contain;background:#faf9f9}
    div[class*="st-key-hero-photo"] [data-testid="stImage"] img{height:420px;object-fit:cover;object-position:center 22%;background:#fff;border-radius:2px}
    .result-title{font:600 1.08rem/1.25 Georgia,"Times New Roman",serif;margin:.65rem 0 .4rem;color:var(--ink)}
    .result-score{background:#fff0f4;color:#ae385c;border:1px solid #f7d5df;border-radius:3px;padding:.2rem .5rem;font-size:.7rem;font-weight:800;display:inline-block}
    .catalog-badge{background:#fff6f8;color:#8f4c60;border:1px solid #f3e1e6;border-radius:3px;padding:.22rem .48rem;font-size:.68rem;font-weight:700;display:inline-block;margin:.15rem .2rem .05rem 0}
    .chip-soft{background:#f8f7f7;color:#69747a;border:1px solid #eeebec;border-radius:3px;padding:.2rem .45rem;font-size:.68rem;font-weight:700;display:inline-block;margin:.16rem .2rem 0 0}
    .result-reasons{color:var(--muted);font-size:.76rem;line-height:1.45;margin-top:.5rem;min-height:38px}
    .filter-strip{background:#fff5f7;border-left:3px solid #e887a0;border-radius:3px;padding:.7rem .85rem;margin:.8rem 0 1rem;color:#55464b;font-size:.82rem;line-height:1.6}
    .analysis-panel{background:#fff;border:1px solid var(--line);border-radius:6px;padding:.8rem}
    @media(max-width:700px){.block-container{padding:.6rem .8rem 2rem}.topbar{margin-bottom:1.2rem}.topline-meta{font-size:.55rem;letter-spacing:.08em}.fashion-title{font-size:2rem!important}.fashion-subtitle{line-height:1.6}div[class*="st-key-hero-photo"] [data-testid="stImage"] img{height:280px}div[class*="st-key-result-card-"] [data-testid="stImage"] img{height:220px}}
    </style>""",
    unsafe_allow_html=True,
)
create_tables()


def ensure_catalog_ready():
    csv_path = CATALOG_DIR / "catalog.csv"
    if csv_path.exists():
        try:
            df = __import__("pandas").read_csv(csv_path)
            nonempty = {
                key: bool(df.get(key, pd.Series([""] * len(df))).astype("string").fillna("").str.strip().ne("").any())
                for key in ["category", "color", "occasion", "season"]
            }
            if nonempty.get("category") and nonempty.get("color") and nonempty.get("occasion"):
                return
        except Exception:
            pass

    images_dir = CATALOG_DIR / "images"
    if images_dir.exists() and any(images_dir.rglob("*")):
        st.warning("The catalog looks incomplete, so I’m rebuilding it from the local image folder to restore filters and style matching.")
        df = from_folder(CATALOG_DIR, CATALOG_DIR, "Women", get_encoder())
        df.to_csv(csv_path, index=False)
        return

    st.error("No usable catalog found. Rebuild it first: python build_catalog.py --images-dir path/to/images --gender Women")
    st.stop()


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

ensure_catalog_ready()
rec = get_recommender()
body_clf = get_body_clf()
df = rec.df
catalog_meta_missing = {
    key: int(df[key].fillna("").astype(str).str.strip().ne("").sum())
    for key in ["category", "color", "occasion", "season"]
}
if any(v == 0 for v in catalog_meta_missing.values()):
    st.warning(
        "This catalog is missing metadata for some filters. Rebuild it from a real dataset or a photo folder: "
        "python build_catalog.py --images-dir path/to/images --gender Women"
    )

# ---------------- header ----------------
st.markdown(
    "<div class='topbar'><span class='wordmark'>Style Muse</span><span class='topline-meta'>PERSONAL STYLE • FIND YOUR LOOK</span></div>",
    unsafe_allow_html=True,
)
hero_items = df[df["category"].astype(str).str.lower().eq("dress")]
hero_image = hero_items.iloc[0]["image_path"] if not hero_items.empty else df.iloc[0]["image_path"]
hero_copy, hero_visual = st.columns([1.05, 0.95], gap="large", vertical_alignment="center")
with hero_copy:
    st.markdown(
        """
        <div class="fashion-header">
          <div class="fashion-kicker">Your personal fashion edit</div>
          <h1 class="fashion-title">Find your next favorite look</h1>
          <div class="fashion-subtitle">Upload a look that inspires you. Discover pieces with the same feeling, matched to your color, occasion, and personal style.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("Choose an inspiration photo", type=["jpg", "jpeg", "png", "webp"])
    find_looks = st.button("FIND MY LOOKS", type="primary", use_container_width=True)
    st.caption(f"Explore {len(df):,} catalog pieces. Style and color filters are available in the filter panel.")
with hero_visual, st.container(key="hero-photo"):
    st.image(hero_image, use_container_width=True)

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
    style_mode = st.selectbox("Recommendation mode", ["Closest match", "Similar vibe / varied looks"], index=1)
    variety = st.slider("Style variety", 0.0, 1.0, 0.45 if style_mode == "Similar vibe / varied looks" else 0.0, 0.05)
    focus = st.checkbox("Focus on the outfit area of the photo", value=True)
    top_n = st.slider("Results", 4, 24, 12, 4)

if find_looks:
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
                variety=variety,
            )
            st.session_state.update(results=results, image=image, body_type=body_type, body_probs=body_probs)
        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ---------------- results ----------------
if "results" in st.session_state:
    results = st.session_state["results"]
    left, right = st.columns([1, 2])
    with left, st.container(border=True, key="inspiration-panel"):
        st.image(st.session_state["image"], caption="Your inspiration look", use_container_width=True)
    with right:
        bt = st.session_state["body_type"]
        if bt:
            st.markdown(f'<div class="mini-label">Body profile</div>', unsafe_allow_html=True)
            st.subheader(f"{bt.title()} silhouette")
            for label, p in st.session_state["body_probs"][:3]:
                st.progress(min(max(p, 0.0), 1.0), text=f"{label} {p:.0%}")
            st.markdown(f'<div class="filter-strip">{BODY_RULES[bt]["tip"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="mini-label">Body profile</div>', unsafe_allow_html=True)
            st.subheader("No body-shape filter applied")

    st.markdown('<div class="mini-label">Curated matches</div>', unsafe_allow_html=True)
    st.header("Shop your vibe")
    active_filters = [
        f"Gender: {', '.join(gender) or 'Any'}",
        f"Category: {', '.join(category) or 'Any'}",
        f"Colour: {', '.join(color) or 'Any'}",
        f"Occasion: {', '.join(occasion) or 'Any'}",
        f"Season: {', '.join(season) or 'Any'}",
    ]
    st.markdown(
        f"<div class='filter-strip'><strong>Active look filters</strong><br>"
        + " · ".join(active_filters)
        + "</div>",
        unsafe_allow_html=True,
    )

    if results.empty:
        st.warning("No items match these filters. Try removing a few.")
    cols = st.columns(4)
    for i, row in results.iterrows():
        iid = int(row["item_id"])
        with cols[i % 4], st.container(border=True, key=f"result-card-{iid}"):
            st.image(row["image_path"], use_container_width=True)
            st.markdown(f'<div class="result-score">{row["final_score"]:.0%} match</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-title">{row["name"][:60]}</div>', unsafe_allow_html=True)
            badges = "".join(f'<span class="catalog-badge">{x}</span>' for x in (row.get("category", ""), row.get("color", ""), row.get("occasion", "")) if str(x).strip())
            if badges:
                st.markdown(f"<div>{badges}</div>", unsafe_allow_html=True)
            chips = "".join(f'<span class="chip-soft">{x}</span>' for x in (row.get("season", ""),) if str(x).strip())
            if chips:
                st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
            st.markdown(f'<div class="result-reasons">{row["reasons"]}</div>', unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3, gap="small")
            b1.button("👍", key=f"up_{iid}", on_click=on_feedback, args=(iid, "Like"), help="More like this")
            b2.button("👎", key=f"dn_{iid}", on_click=on_feedback, args=(iid, "Dislike"), help="Fewer like this")
            b3.button("♡", key=f"sv_{iid}", on_click=on_save, args=(row.to_dict(),), help="Save this look")
            st.link_button("Shop similar", f"https://www.myntra.com/{quote(row['name'].replace(' ', '-'))}", use_container_width=True)

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
