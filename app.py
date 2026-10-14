"""
app.py — Streamlit demo for the Customer Review Sentiment Analysis project.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (main file: app.py)

The app loads the saved pipeline (models/best_model_sentiment.pkl), which
accepts RAW review text, so no preprocessing is needed here.
"""

import io
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import streamlit as st

from text_utils import clean_text, standardize_channel  # also required to unpickle the model

# ---------------------------------------------------------------------------
# Page setup & constants
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="💬", layout="wide")

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
PALETTE = {"negative": "#e34948", "neutral": "#9a9890", "positive": "#2a78d6"}
EMOJI = {"negative": "😠", "neutral": "😐", "positive": "😊"}
CATEGORIES = ["Audio", "Wearables", "Smart Home", "Accessories"]

ASPECTS = {
    "Connectivity": r"disconnect|connection|pairing|bluetooth|cuts out|cutting out|wifi",
    "App / Software": r"\bapp\b|crash|sync|logs me out",
    "Battery": r"battery|charge\b|drains",
    "Shipping / Delivery": r"shipping|delivery|arrived|package|ship",
    "Customer Support": r"support|customer service|refund",
    "Build / Durability": r"build|cheap|flimsy|broke|cracked|frayed|\bport\b|durable",
    "Price / Value": r"overpriced|not worth|waste of money|price",
    "Sound / ANC": r"sound|audio|bass|noise|anc|hiss",
    "Charging": r"charging|charges|hot",
    "Comfort / Fit": r"comfort|hurts|falling out|rash|fit",
}

EXAMPLES = {
    "— choose an example —": ("", "Audio", ""),
    "Happy customer": ("Love it", "Audio",
                       "Sound quality blew me away!! Battery lasts for days and pairing was instant 😍"),
    "Angry customer": ("Disappointed", "Smart Home",
                       "The app keeps crashing and it wont connect to my wifi.<br/>Support never replied. "
                       "Requesting a refund 😡"),
    "Lukewarm customer": ("", "Accessories", "its ok. charges fine, a bit warm. price is reasonable i guess"),
    "Support ticket (no title)": ("", "Smart Home",
                                  "hi, camera video freezes all the time at night and setup failed 3 times. pls help"),
}

plt.rcParams.update({"axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e6e5e0",
                     "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("models/best_model_sentiment.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_reviews():
    """Load and clean the historical dataset (same steps as the notebook)."""
    df = pd.read_csv("data/customer_reviews.csv").drop_duplicates()
    df = df[~df["review_text"].fillna("").str.strip().eq("")].copy()
    df["review_title"] = df["review_title"].fillna("")
    df["sales_channel"] = df["sales_channel"].apply(standardize_channel)
    df["review_date"] = pd.to_datetime(df["review_date"])
    df["year_month"] = df["review_date"].dt.to_period("M").astype(str)
    return df.reset_index(drop=True)


model = load_model()
reviews = load_reviews()


def predict(df):
    """Return df with predicted sentiment, class probabilities and confidence."""
    data = df.copy()
    if "review_title" not in data:
        data["review_title"] = ""
    if "product_category" not in data:
        data["product_category"] = "Unknown"
    data["review_text"] = data["review_text"].fillna("").astype(str)
    proba = model.predict_proba(data)
    out = df.copy()
    out["predicted_sentiment"] = model.classes_[proba.argmax(axis=1)]
    for i, c in enumerate(model.classes_):
        out[f"prob_{c}"] = proba[:, i].round(3)
    out["confidence"] = proba.max(axis=1).round(3)
    return out


# ---------------------------------------------------------------------------
# Chart helpers (matplotlib)
# ---------------------------------------------------------------------------
def show(fig):
    """Render a matplotlib figure in Streamlit and free its memory."""
    st.pyplot(fig)
    plt.close(fig)


def chart_probabilities(probs):
    fig, ax = plt.subplots(figsize=(6, 2.2))
    vals = [probs[s] for s in SENTIMENT_ORDER]
    ax.barh(SENTIMENT_ORDER, vals, color=[PALETTE[s] for s in SENTIMENT_ORDER], height=0.6)
    for i, v in enumerate(vals):
        if v > 0.85:
            ax.text(v - 0.02, i, f"{v:.0%}", va="center", ha="right", color="white", fontsize=10)
        else:
            ax.text(v + 0.02, i, f"{v:.0%}", va="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title("Class probabilities", loc="left", fontsize=11)
    return fig


def chart_distribution(series):
    counts = series.value_counts().reindex(SENTIMENT_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[s] for s in counts.index], width=0.55)
    total = counts.sum() or 1
    for b, n in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(n):,} ({n / total:.0%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, max(counts.max(), 1) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_title("Predicted sentiment", loc="left", fontsize=11)
    return fig


def chart_trend(df, highlight_product=None):
    monthly = df.groupby("year_month")["sentiment"].apply(lambda s: (s == "negative").mean() * 100)
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(monthly.index, monthly.values, color="#9a9890", lw=2, marker="o", ms=3, label="Selected products")
    if highlight_product:
        sub = df[df["product_name"] == highlight_product]
        m2 = sub.groupby("year_month")["sentiment"].apply(lambda s: (s == "negative").mean() * 100)
        ax.plot(m2.index, m2.values, color="#e34948", lw=2, marker="o", ms=3, label=highlight_product)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Monthly share of negative reviews", loc="left", fontsize=11)
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False)
    return fig


def chart_products(df):
    prod = pd.crosstab(df["product_name"], df["sentiment"], normalize="index")
    prod = prod.reindex(columns=SENTIMENT_ORDER, fill_value=0) * 100
    prod = prod.sort_values("negative")
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(prod) + 1.2))
    left = np.zeros(len(prod))
    for s in SENTIMENT_ORDER:
        ax.barh(prod.index, prod[s], left=left, color=PALETTE[s], label=s.title(),
                edgecolor="white", linewidth=2, height=0.7)
        if s == "negative":
            for y, v in enumerate(prod[s]):
                if v > 6:
                    ax.text(v / 2, y, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9)
        left += prod[s].values
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", visible=False)
    ax.set_title("Sentiment mix by product", loc="left", fontsize=11)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    return fig


def chart_aspects(df):
    neg = df[df["sentiment"] == "negative"]
    share = pd.Series({a: neg["review_text"].str.contains(p, case=False, regex=True).mean() * 100
                       for a, p in ASPECTS.items()}).fillna(0).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(share.index, share.values, color="#e34948", height=0.65)
    for i, v in enumerate(share.values):
        ax.text(v + 0.4, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.grid(axis="y", visible=False)
    ax.set_title("% of negative reviews mentioning each aspect", loc="left", fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 💬 Review Sentiment Analyzer")
    st.caption("Data science case study · NLP")
    st.markdown(
        "**Business context:** U.S.-based direct-to-consumer consumer electronics brand "
        "(audio, wearables, smart home, charging accessories)."
    )
    st.markdown("**Model performance (test set)**")
    c1, c2 = st.columns(2)
    c1.metric("Macro F1", "0.859")
    c2.metric("Neg. recall", "0.885")
    st.markdown("**Model:** tuned Logistic Regression + TF-IDF (1–2 grams)")
    st.caption("Company and product names are fictional. Dataset built to mirror a real review export.")
    st.markdown("Built by **Yunido Baheramsyah**  \n[GitHub repository](https://github.com/)")

st.title("Customer Review Sentiment Analyzer")
st.write("Classify customer reviews, support tickets or social comments as **positive**, "
         "**neutral** or **negative**, and explore what drives customer dissatisfaction.")

tab_single, tab_batch, tab_dash, tab_about = st.tabs(
    ["🔍 Single review", "📂 Batch prediction", "📊 Insights dashboard", "ℹ️ About the model"])

# ---------------------------------------------------------------------------
# Tab 1 — Single review
# ---------------------------------------------------------------------------
with tab_single:
    example = st.selectbox("Try an example or write your own review", list(EXAMPLES))
    ex_title, ex_cat, ex_text = EXAMPLES[example]

    col_a, col_b = st.columns([3, 1])
    with col_a:
        title = st.text_input("Review title (optional)", value=ex_title)
    with col_b:
        category = st.selectbox("Product category", CATEGORIES, index=CATEGORIES.index(ex_cat))
    text = st.text_area("Review text", value=ex_text, height=130,
                        placeholder="e.g. Battery drains way too fast and the app keeps crashing...")

    if st.button("Analyze sentiment", type="primary"):
        if not text.strip():
            st.warning("Please enter a review text first.")
        else:
            res = predict(pd.DataFrame([{"review_title": title, "review_text": text,
                                         "product_category": category}])).iloc[0]
            label = res["predicted_sentiment"]
            probs = {s: res[f"prob_{s}"] for s in SENTIMENT_ORDER}

            r1, r2 = st.columns([1, 2])
            with r1:
                st.markdown(
                    f"<div style='padding:18px;border-radius:10px;background:{PALETTE[label]};color:white;'>"
                    f"<div style='font-size:14px;opacity:.9'>Predicted sentiment</div>"
                    f"<div style='font-size:30px;font-weight:700'>{EMOJI[label]} {label.title()}</div>"
                    f"<div style='font-size:14px'>Confidence: {res['confidence']:.0%}</div></div>",
                    unsafe_allow_html=True)
                if res["confidence"] < 0.5:
                    st.caption("⚠️ Low confidence — the text is mixed or ambiguous. Worth a quick human check.")
                elif label == "negative" and probs["negative"] >= 0.6:
                    st.caption("🚨 Suggested action: respond within 24 hours.")
            with r2:
                show(chart_probabilities(probs))

            with st.expander("See how the model cleaned the text"):
                st.code(clean_text(f"{title}. {text}") or "(empty)", language=None)

# ---------------------------------------------------------------------------
# Tab 2 — Batch prediction
# ---------------------------------------------------------------------------
with tab_batch:
    st.markdown("Upload a CSV with at least a **`review_text`** column. "
                "Optional columns: `review_title`, `product_category` (Audio, Wearables, Smart Home, Accessories). "
                "Any other columns (IDs, dates, channels) are kept in the output.")

    template = pd.DataFrame({
        "review_id": ["R-1", "R-2", "R-3"],
        "review_title": ["Great product", "", "Returned"],
        "product_category": ["Audio", "Accessories", "Smart Home"],
        "review_text": ["Love these earbuds, sound is crisp!", "charges fine, a bit warm",
                        "Setup failed three times and support never replied."],
    })
    st.download_button("⬇️ Download CSV template", template.to_csv(index=False).encode("utf-8"),
                       "review_template.csv", "text/csv")

    uploaded = st.file_uploader("Upload reviews (CSV)", type=["csv"])
    use_sample = st.checkbox("…or use 200 sample reviews from the dataset", value=uploaded is None)

    batch = None
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the file: {e}")
    elif use_sample:
        batch = reviews.sample(200, random_state=1)[["review_id", "product_category", "review_title",
                                                      "review_text"]].reset_index(drop=True)

    if batch is not None:
        if "review_text" not in batch.columns:
            st.error("The file must contain a `review_text` column.")
        else:
            scored = predict(batch)
            scored = scored.sort_values("prob_negative", ascending=False).reset_index(drop=True)
            scored["suggested_action"] = np.select(
                [scored["prob_negative"] >= 0.6, scored["prob_negative"] >= 0.3],
                ["Respond within 24h", "Review this week"], default="No action needed")

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Reviews scored", f"{len(scored):,}")
            for k, s in zip([k2, k3, k4], SENTIMENT_ORDER):
                k.metric(f"{EMOJI[s]} {s.title()}", f"{(scored['predicted_sentiment'] == s).mean():.0%}")

            c1, c2 = st.columns([1, 2])
            with c1:
                show(chart_distribution(scored["predicted_sentiment"]))
            with c2:
                st.markdown("**Priority queue** (sorted by probability of being negative)")
                show_cols = [c for c in ["review_id", "product_category", "review_text", "predicted_sentiment",
                                         "prob_negative", "confidence", "suggested_action"] if c in scored]
                st.dataframe(scored[show_cols], height=320)

            buf = io.StringIO()
            scored.to_csv(buf, index=False)
            st.download_button("⬇️ Download scored reviews (CSV)", buf.getvalue().encode("utf-8"),
                               "scored_reviews.csv", "text/csv", type="primary")

# ---------------------------------------------------------------------------
# Tab 3 — Insights dashboard (historical labelled data)
# ---------------------------------------------------------------------------
with tab_dash:
    st.markdown("Historical reviews (Jan 2024 – Dec 2025) labelled by the client's CX team.")
    f1, f2 = st.columns([2, 1])
    with f1:
        cats = st.multiselect("Product category", CATEGORIES, default=CATEGORIES)
    with f2:
        channels = sorted(reviews["sales_channel"].unique())
        chans = st.multiselect("Sales channel", channels, default=channels)

    view = reviews[reviews["product_category"].isin(cats) & reviews["sales_channel"].isin(chans)]
    if view.empty:
        st.warning("No reviews match the selected filters.")
    else:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Reviews", f"{len(view):,}")
        for k, s in zip([k2, k3, k4], SENTIMENT_ORDER):
            k.metric(f"{EMOJI[s]} {s.title()}", f"{(view['sentiment'] == s).mean():.1%}")

        products = ["(none)"] + sorted(view["product_name"].unique())
        default_idx = products.index("AirPulse Pro Earbuds") if "AirPulse Pro Earbuds" in products else 0
        hl = st.selectbox("Highlight a product in the trend", products, index=default_idx)
        show(chart_trend(view, None if hl == "(none)" else hl))

        d1, d2 = st.columns(2)
        with d1:
            show(chart_products(view))
        with d2:
            show(chart_aspects(view))

        st.markdown("""
**Key findings**
- **AirPulse Pro Earbuds:** negative share jumped to **51–55% in Aug–Oct 2025**; most complaints mention connectivity.
- **Smart Home** products have the highest negative share (~33%); most of their complaints mention the **app**.
- **Customer support** and **shipping** are the most frequent aspects in negative reviews — fixable without hardware changes.
""")

# ---------------------------------------------------------------------------
# Tab 4 — About
# ---------------------------------------------------------------------------
with tab_about:
    st.markdown("""
### Business problem
The client's CX team tagged reviews manually and could only read a weekly sample, so product issues were
detected late. The goal was a model that classifies sentiment **from text alone** (no star rating), so it can
also be used on support tickets, chats and social posts.

### Approach
1. **Data cleaning** — duplicates, blank reviews, inconsistent channel / verified labels.
2. **Text preprocessing** — HTML & URL removal, emojis → sentiment tokens, slang normalisation,
   stop-word removal **keeping negations** (*not*, *never*, *but*).
3. **Features** — TF-IDF (unigrams + bigrams), word count, exclamation count, upper-case ratio, product category.
4. **Leakage control** — `rating` and `helpful_votes` excluded.
5. **Models compared (5-fold CV, macro F1)** — Logistic Regression, Linear SVM, Complement NB,
   Random Forest, KNN.
6. **Tuning** — GridSearchCV on Logistic Regression (C, n-grams, min_df).
""")
    st.table(pd.DataFrame({
        "Metric (test set, 1,177 reviews)": ["Macro F1", "Recall — negative", "Accuracy"],
        "Target": ["≥ 0.80", "≥ 0.85", "—"],
        "Result": ["0.859", "0.885", "0.885"],
    }))
    st.caption("Full analysis: notebook.ipynb · Inference example: model_inference.ipynb")
