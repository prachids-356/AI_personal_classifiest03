import { useEffect, useRef, useState } from "react";
import {
  ArrowDown, ArrowRight, Check, ChevronDown, Heart, ImagePlus, LoaderCircle,
  SlidersHorizontal, Sparkles, ThumbsDown, ThumbsUp, X,
} from "lucide-react";

const EMPTY_FILTERS = { gender: [], category: [], color: [], occasion: [], season: [] };

async function request(path, options = {}) {
  const response = await fetch(path, { credentials: "same-origin", ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || "Something went wrong. Please try again.");
  return data;
}

function displayLabel(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function App() {
  const [options, setOptions] = useState(null);
  const [upload, setUpload] = useState(null);
  const [preview, setPreview] = useState("");
  const [filters, setFilters] = useState(EMPTY_FILTERS);
  const [bodyType, setBodyType] = useState("");
  const [vibe, setVibe] = useState("");
  const [mode, setMode] = useState("varied");
  const [results, setResults] = useState([]);
  const [classification, setClassification] = useState(null);
  const [bodyInfo, setBodyInfo] = useState(null);
  const [saved, setSaved] = useState([]);
  const [busy, setBusy] = useState(false);
  const [busyMode, setBusyMode] = useState("classify");
  const [samplesOpen, setSamplesOpen] = useState(false);
  const [wardrobeCategory, setWardrobeCategory] = useState("All looks");
  const [filterOpen, setFilterOpen] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const fileInput = useRef(null);

  useEffect(() => {
    request("/api/session")
      .then(() => Promise.all([request("/api/options"), request("/api/saved-items")]))
      .then(([catalogOptions, savedItems]) => {
        setOptions(catalogOptions);
        setSaved(savedItems.items);
      })
      .catch((cause) => setError(cause.message));
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.14 });
    document.querySelectorAll(".reveal-on-scroll").forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, [results.length, Boolean(classification), saved.length]);

  useEffect(() => {
    if (!upload) return undefined;
    const url = URL.createObjectURL(upload);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [upload]);

  function chooseFile(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("Choose a JPG, PNG, or WEBP image.");
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      setError("Image must be 12 MB or smaller.");
      return;
    }
    setError("");
    setUpload(file);
    setClassification(null);
    setResults([]);
  }

  async function classifyLook(event) {
    event.preventDefault();
    if (!upload) {
      setError("Upload a look to get started.");
      return;
    }
    setBusy(true);
    setBusyMode("classify");
    setError("");
    const form = new FormData();
    form.append("image", upload);
    try {
      const data = await request("/api/classify", { method: "POST", body: form });
      setClassification(data);
      setNotice("Analysis complete");
      document.getElementById("analysis")?.scrollIntoView({ behavior: "smooth", block: "center" });
    } catch (cause) {
      setError(cause.message);
    } finally {
      setBusy(false);
    }
  }

  async function findLooks() {
    if (!upload) return;
    setBusy(true);
    setBusyMode("recommend");
    setError("");
    const form = new FormData();
    form.append("image", upload);
    form.append("options", JSON.stringify({
      filters,
      body_type: bodyType || null,
      vibe,
      top_n: 12,
      body_weight: bodyType ? 0.25 : 0,
      focus: true,
      variety: mode === "varied" ? 0.55 : 0,
    }));
    try {
      const data = await request("/api/recommendations", { method: "POST", body: form });
      setResults(data.items);
      setBodyInfo(data.body_type ? { type: data.body_type, tip: data.body_tip } : null);
      setNotice(`${data.items.length} looks found`);
      document.getElementById("matches")?.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (cause) {
      setError(cause.message);
    } finally {
      setBusy(false);
    }
  }

  async function trySample(sample) {
    try {
      const response = await fetch(sample.image_url);
      if (!response.ok) throw new Error("Could not load sample image");
      const blob = await response.blob();
      chooseFile(new File([blob], `${sample.name.toLowerCase().replaceAll(" ", "-")}.jpg`, { type: blob.type || "image/jpeg" }));
      setSamplesOpen(false);
    } catch (cause) {
      setError(cause.message);
    }
  }

  async function vote(itemId, voteValue) {
    try {
      await request(`/api/items/${itemId}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vote: voteValue }),
      });
      setNotice(voteValue === "Like" ? "We’ll find more like this" : "Got it. We’ll show fewer like this");
    } catch (cause) {
      setError(cause.message);
    }
  }

  async function save(item) {
    try {
      await request("/api/saved-items", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ item_id: item.item_id }),
      });
      setSaved((current) => current.some((savedItem) => savedItem.item_id === item.item_id)
        ? current
        : [item, ...current]);
      setNotice("Saved to your wardrobe");
    } catch (cause) {
      setError(cause.message);
    }
  }

  function toggleFilter(group, value) {
    setFilters((current) => ({
      ...current,
      [group]: current[group].includes(value)
        ? current[group].filter((entry) => entry !== value)
        : [...current[group], value],
    }));
  }

  const activeCount = Object.values(filters).reduce((total, values) => total + values.length, 0);

  return (
    <main>
      <header className="topbar">
        <a className="wordmark" href="#top" aria-label="Style Muse home">style<span>muse</span><i>✳</i></a>
        <nav className="topnav" aria-label="Main navigation">
          <a className="nav-active" href="#top">Discover</a>
          <a href="#matches">Your matches</a>
          <a href="#saved">Saved looks {saved.length > 0 && <small>{saved.length}</small>}</a>
        </nav>
        <a className="nav-cta" href="#inspiration">Find your look <ArrowRight size={15} /></a>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <div className="eyebrow hero-reveal"><span /> PERSONAL STYLE, REIMAGINED</div>
          <h1><span className="headline-line"><span>Find a look.</span></span><span className="headline-line"><em>Make it yours.</em></span></h1>
          <p className="hero-description">Bring us a look you love. We’ll find the pieces with the same energy, made to fit your style and your shape.</p>

          <form id="inspiration" className="upload-form" onSubmit={classifyLook}>
            <div
              className={`upload-zone ${preview ? "has-image" : ""}`}
              onClick={() => fileInput.current?.click()}
              onDragOver={(event) => event.preventDefault()}
              onDrop={(event) => { event.preventDefault(); chooseFile(event.dataTransfer.files[0]); }}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => { if (event.key === "Enter") fileInput.current?.click(); }}
            >
              {preview ? <img src={preview} alt="Your uploaded inspiration" /> : <span className="upload-icon"><ImagePlus size={20} /></span>}
              <span className="upload-copy">
                <strong>{upload ? upload.name : "Drop your inspiration here"}</strong>
                <small>{upload ? "Click to choose a different image" : "or browse from your device · JPG, PNG, WEBP"}</small>
              </span>
              {upload && <button className="remove-upload" type="button" aria-label="Remove image" onClick={(event) => { event.stopPropagation(); setUpload(null); }}><X size={15} /></button>}
              <input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" hidden onChange={(event) => chooseFile(event.target.files?.[0])} />
            </div>
            <div className="hero-actions">
              <button className="primary-button" type="submit" disabled={busy || !options}>
                {busy && busyMode === "classify" ? <><LoaderCircle className="spin" size={17} /> Reading your look</> : <><Sparkles size={16} /> Analyze my look <ArrowRight size={16} /></>}
              </button>
              <span className="privacy-note">Your photo stays private</span>
            </div>
            <button type="button" className="sample-toggle" onClick={() => setSamplesOpen((open) => !open)}><ImagePlus size={14} /> Try a sample look</button>
            {samplesOpen && options?.sample_images?.length > 0 && <div className="sample-picker">{options.sample_images.map((sample) => <button type="button" key={sample.item_id} onClick={() => trySample(sample)}><img src={sample.image_url} alt="" /><span>{sample.name}</span></button>)}</div>}
          </form>

          <button className={`filter-toggle ${filterOpen ? "open" : ""}`} type="button" onClick={() => setFilterOpen((value) => !value)}>
            <SlidersHorizontal size={16} /> Shape your search {activeCount > 0 && <span className="count-badge">{activeCount}</span>} <ChevronDown size={15} />
          </button>
          {filterOpen && options && (
            <div className="filter-panel">
              {[["gender", "For"], ["category", "Pieces"], ["color", "Color"], ["occasion", "Occasion"], ["season", "Season"]].map(([group, label]) => (
                <fieldset key={group}>
                  <legend>{label}</legend>
                  <div className="filter-options">
                    {options[group].map((value) => (
                      <button key={value} type="button" className={filters[group].includes(value) ? "filter-chip selected" : "filter-chip"} onClick={() => toggleFilter(group, value)}>
                        {filters[group].includes(value) && <Check size={12} />}{displayLabel(value)}
                      </button>
                    ))}
                  </div>
                </fieldset>
              ))}
              <label className="select-label">Body shape
                <select value={bodyType} onChange={(event) => setBodyType(event.target.value)}>
                  <option value="">No fit preference</option>
                  {(options.body_types || []).map((value) => <option key={value} value={value}>{displayLabel(value)}</option>)}
                </select>
              </label>
              <label className="select-label">Style direction
                <select value={mode} onChange={(event) => setMode(event.target.value)}>
                  <option value="varied">Same energy, more variety</option>
                  <option value="closest">Closest visual matches</option>
                </select>
              </label>
              <label className="select-label vibe-label">Add a vibe
                <input value={vibe} onChange={(event) => setVibe(event.target.value)} placeholder="minimal, romantic, streetwear…" />
              </label>
              <button className="clear-filters" type="button" onClick={() => setFilters(EMPTY_FILTERS)}>Clear all filters</button>
            </div>
          )}
        </div>

        <div className="hero-art" aria-label="Featured fashion look">
          <img src="/catalog/images/19.jpg" alt="Model wearing a black dress" />
          <div className="image-caption"><span>THE STYLE EDIT</span><span>01 / 04</span></div>
          <div className="hero-stamp"><Sparkles size={15} /><span>LOOKS THAT<br />FEEL LIKE YOU</span></div>
          <div className="art-index">01</div>
        </div>
        <div className="hero-bottomline"><span>DISCOVER YOUR NEXT SIGNATURE LOOK</span><ArrowDown size={15} /></div>
      </section>

      {error && <div className="toast toast-error" role="alert">{error}<button onClick={() => setError("")} aria-label="Dismiss"><X size={15} /></button></div>}
      {notice && <div className="toast toast-notice" role="status">{notice}<button onClick={() => setNotice("")} aria-label="Dismiss"><X size={15} /></button></div>}

      {upload && <section className="analysis-section reveal-on-scroll" id="analysis">
        <div className="analysis-heading"><span className="eyebrow">THE AI FASHION LAB</span><h2>{classification ? "Here’s what we see." : busy && busyMode === "classify" ? "Reading the details." : "Ready when you are."}</h2><p>Garment labels are matched directly against the uploaded image using CLIP.</p></div>
        <div className="analysis-layout">
          <div className={`analysis-photo ${busy && busyMode === "classify" ? "scanning" : ""}`}><img src={preview} alt="Uploaded look under analysis" />{busy && busyMode === "classify" && <><div className="scan-line" /><div className="scan-status"><LoaderCircle className="spin" size={15} /> Comparing garment details</div></>}</div>
          <div className="analysis-results">
            {classification ? <>
              <div className="prediction-kicker">TOP MATCH</div>
              <div className="prediction-main">{displayLabel(classification.predictions[0]?.category || "Look")}</div>
              <p className="score-disclaimer">{classification.score_label}. These are similarity scores, not probabilities.</p>
              <div className="prediction-list">{classification.predictions.map((prediction, index) => {
                const maxScore = classification.predictions[0]?.similarity || 1;
                return <div className="prediction-row" key={prediction.category}><span className="prediction-rank">0{index + 1}</span><span className="prediction-name">{displayLabel(prediction.category)}</span><div className="prediction-track"><span style={{ width: `${Math.max(4, prediction.similarity / maxScore * 100)}%` }} /></div><span className="prediction-score">{prediction.similarity.toFixed(3)}</span></div>;
              })}</div>
              <button className="primary-button recommendation-button" type="button" onClick={findLooks} disabled={busy}>{busy && busyMode === "recommend" ? <><LoaderCircle className="spin" size={16} /> Finding related styles</> : <>Discover similar styles <ArrowRight size={16} /></>}</button>
            </> : <div className="analysis-empty"><Sparkles size={22} /><h3>Your garment readout</h3><p>Run the analysis to reveal the top matching garment categories. No scores are shown until the model returns.</p></div>}
          </div>
        </div>
      </section>}

      <section className="matches-section reveal-on-scroll" id="matches">
        <div className="section-heading">
          <div><span className="eyebrow">CURATED FOR YOUR POINT OF VIEW</span><h2>{results.length ? "Your style, found." : "A little inspiration goes a long way."}</h2></div>
          <p>{results.length ? `${results.length} pieces with a similar point of view` : "Upload a look above to see your personal edit."}</p>
        </div>
        {bodyInfo && <div className="body-note"><span>{displayLabel(bodyInfo.type)} fit note</span>{bodyInfo.tip}</div>}
        {results.length > 0 && (
          <div className="results-grid">
            {results.map((item, index) => (
              <article className="product-tile" key={item.item_id}>
                <div className="product-photo">
                  <img src={item.image_url} alt={item.name} loading="lazy" />
                  <span className="match-score">{Math.round(item.final_score * 100)}% match</span>
                  <button className="save-button" aria-label="Save look" onClick={() => save(item)}><Heart size={17} /></button>
                </div>
                <div className="product-info">
                  <span className="product-index">LOOK {String(index + 1).padStart(2, "0")}</span>
                  <h3>{item.name}</h3>
                  <div className="product-tags">{[item.category, item.color, item.occasion].filter(Boolean).map((tag) => <span key={tag}>{displayLabel(tag)}</span>)}</div>
                  <p>{item.reasons}</p>
                  <div className="product-actions">
                    <button aria-label="More like this" title="More like this" onClick={() => vote(item.item_id, "Like")}><ThumbsUp size={15} /></button>
                    <button aria-label="Fewer like this" title="Fewer like this" onClick={() => vote(item.item_id, "Dislike")}><ThumbsDown size={15} /></button>
                    <a href={`https://www.myntra.com/${encodeURIComponent(item.name.replaceAll(" ", "-"))}`} target="_blank" rel="noreferrer">Explore similar <ArrowRight size={14} /></a>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <section className="saved-section reveal-on-scroll" id="saved">
        <div className="section-heading saved-heading"><div><span className="eyebrow">YOUR PERSONAL COLLECTION</span><h2>Your digital wardrobe.</h2></div><Heart size={21} /></div>
        {saved.length ? <>
          <div className="wardrobe-toolbar"><span>{saved.length} saved {saved.length === 1 ? "look" : "looks"}</span><label>Filter by category<select value={wardrobeCategory} onChange={(event) => setWardrobeCategory(event.target.value)}><option>All looks</option>{[...new Set(saved.map((item) => displayLabel(item.category)))].map((category) => <option key={category}>{category}</option>)}</select></label></div>
          <div className="saved-row">{saved.filter((item) => wardrobeCategory === "All looks" || displayLabel(item.category) === wardrobeCategory).map((item) => <a className="saved-item" href="#matches" key={item.item_id}><img src={item.image_url} alt={item.name} /><span>{item.name}</span></a>)}</div>
        </> : <p className="saved-empty">Your favorite finds will live here.</p>}
      </section>

      <footer className="footer"><a className="wordmark" href="#top">style<span>muse</span><i>✳</i></a><span>PERSONAL STYLE, REIMAGINED</span><a href="#top">Back to top ↑</a></footer>
    </main>
  );
}

export default App;