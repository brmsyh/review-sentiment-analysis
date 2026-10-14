"""
generate_dummy_data.py
----------------------
Creates the customer-review dataset used in this case study.

The script builds a realistic dataset with the same structure and the same
kind of "messiness" found in real e-commerce review exports:
  - duplicates, missing values, inconsistent category labels
  - HTML tags, URLs, emojis, slang, typos, ALL-CAPS rants
  - mixed-sentiment reviews and a small share of label noise
  - hidden business patterns (a firmware-related complaint spike,
    holiday shipping delays, app issues on smart-home devices)

Run:  python generate_dummy_data.py
Output: data/customer_reviews.csv
"""

import random
from datetime import date, timedelta

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)
random.seed(SEED)

N_REVIEWS = 6000

# ---------------------------------------------------------------------------
# Product catalogue (fictional product names)
# ---------------------------------------------------------------------------
PRODUCTS = {
    "AirPulse Pro Earbuds":        ("Audio",       "these earbuds",   59.99),
    "BassCore Over-Ear Headphones": ("Audio",       "these headphones", 129.99),
    "WaveBox Mini Speaker":        ("Audio",       "this speaker",     39.99),
    "FitTrack S2 Smartwatch":      ("Wearables",   "this watch",       149.99),
    "PulseBand Lite":              ("Wearables",   "this band",        49.99),
    "HomeLink Smart Plug (2-pack)": ("Smart Home",  "these plugs",      24.99),
    "EyeGuard 2K Security Camera": ("Smart Home",  "this camera",      79.99),
    "VoltMax 20K Power Bank":      ("Accessories", "this power bank",  45.99),
    "QuickCharge 65W Charger":     ("Accessories", "this charger",     34.99),
}
PRODUCT_WEIGHTS = [0.20, 0.10, 0.10, 0.12, 0.08, 0.12, 0.10, 0.10, 0.08]

CATEGORY_ASPECTS = {
    "Audio":       ["sound", "anc", "battery", "connectivity", "comfort"],
    "Wearables":   ["tracking", "battery", "comfort", "app", "connectivity"],
    "Smart Home":  ["app", "setup", "connectivity", "image"],
    "Accessories": ["charging", "battery", "durability"],
}
COMMON_ASPECTS = ["build", "shipping", "packaging", "support", "price"]

# ---------------------------------------------------------------------------
# Phrase banks: aspect -> sentiment -> list of phrases ({p} = product noun)
# ---------------------------------------------------------------------------
PHRASES = {
    "sound": {
        "pos": ["the sound is crisp and the bass is punchy", "audio quality is amazing for the price",
                "clear highs and deep bass, really impressed", "music sounds rich and detailed",
                "sound quality blew me away"],
        "neg": ["the sound is muddy and distorted at high volume", "audio quality is terrible, everything sounds tinny",
                "there is a constant hissing noise", "bass is weak and the mids are muffled",
                "sound cuts out randomly"],
        "neu": ["sound is decent, nothing special", "audio is fine for podcasts but average for music",
                "sound quality is ok for the price", "the sound is about what you'd expect"],
    },
    "anc": {
        "pos": ["noise cancelling works great on the subway", "the ANC blocks out almost everything",
                "noise cancellation is excellent on flights"],
        "neg": ["noise cancelling barely does anything", "the ANC creates a weird pressure and a hiss",
                "noise cancellation is useless in the office"],
        "neu": ["noise cancelling is so-so", "ANC is fine but not as strong as advertised"],
    },
    "battery": {
        "pos": ["battery lasts for days", "battery life is excellent", "easily gets through a full week on one charge",
                "the battery lasts way longer than my old one"],
        "neg": ["battery drains way too fast", "the battery died after two hours", "battery life is a joke",
                "it won't hold a charge anymore"],
        "neu": ["battery life is average", "battery gets me through the day, barely",
                "battery is ok, not as long as advertised"],
    },
    "connectivity": {
        "pos": ["pairing was instant and the connection is rock solid", "bluetooth connection never drops",
                "connects to my phone in seconds"],
        "neg": ["keeps disconnecting from my phone", "the bluetooth connection drops every few minutes",
                "pairing fails constantly", "connection is unstable and keeps cutting out"],
        "neu": ["connection is mostly stable with an occasional drop", "pairing took a couple of tries but works now"],
    },
    "comfort": {
        "pos": ["super comfortable even after hours", "fits perfectly and stays in place", "so light I forget I'm wearing it"],
        "neg": ["hurts my ears after 30 minutes", "uncomfortable and keeps falling out", "the strap gives me a rash"],
        "neu": ["comfort is ok for short sessions", "fit is fine, a bit tight at first"],
    },
    "tracking": {
        "pos": ["step and heart rate tracking are spot on", "sleep tracking is surprisingly accurate",
                "fitness tracking is very accurate"],
        "neg": ["heart rate readings are way off", "step count is wildly inaccurate", "GPS tracking is useless"],
        "neu": ["tracking is roughly accurate", "sleep data is hit or miss"],
    },
    "app": {
        "pos": ["the app is clean and easy to use", "companion app works flawlessly", "app is intuitive and fast"],
        "neg": ["the app keeps crashing", "app is buggy and logs me out constantly", "the app is a nightmare to use",
                "app won't sync no matter what I try"],
        "neu": ["the app is ok but could be more intuitive", "app does the basics, nothing more"],
    },
    "setup": {
        "pos": ["setup took less than five minutes", "installation was a breeze", "super easy to set up"],
        "neg": ["setup was a total headache", "could not get it to connect to my wifi", "setup failed three times"],
        "neu": ["setup was a bit confusing but I figured it out", "installation took longer than expected"],
    },
    "image": {
        "pos": ["video is sharp even at night", "night vision is impressively clear", "picture quality is excellent"],
        "neg": ["video is blurry and laggy", "night vision is grainy and useless", "the feed freezes all the time"],
        "neu": ["video quality is fine during the day", "image is acceptable, not really 2K"],
    },
    "charging": {
        "pos": ["charges my laptop and phone super fast", "fast charging works as advertised", "charging speed is fantastic"],
        "neg": ["charges painfully slow", "gets extremely hot while charging", "stopped charging after a week"],
        "neu": ["charging speed is ok", "charges fine, a bit warm"],
    },
    "durability": {
        "pos": ["feels very durable", "survived several drops without a scratch", "solid and built to last"],
        "neg": ["the cable frayed within a month", "broke after a few weeks", "the port came loose quickly"],
        "neu": ["seems durable enough so far", "time will tell how long it lasts"],
    },
    "build": {
        "pos": ["build quality feels premium", "looks and feels high quality", "very well made"],
        "neg": ["feels cheap and plasticky", "build quality is flimsy", "arrived with a cracked case"],
        "neu": ["build quality is fine", "the design is plain but functional"],
    },
    "shipping": {
        "pos": ["arrived two days early", "shipping was fast", "delivery was quick and smooth"],
        "neg": ["shipping took almost three weeks", "the package arrived late and damaged", "delivery was a mess",
                "still waiting for the replacement to ship"],
        "neu": ["shipping was on time", "delivery took about a week"],
    },
    "packaging": {
        "pos": ["packaging was neat and eco-friendly", "nicely packaged, feels like a gift"],
        "neg": ["the box was crushed", "packaging was damaged and parts were missing"],
        "neu": ["packaging was basic", "simple packaging, nothing fancy"],
    },
    "support": {
        "pos": ["customer support was super helpful", "support replaced it with no questions asked",
                "the support team responded within an hour"],
        "neg": ["customer service never replied", "support was rude and unhelpful", "refund request was ignored for weeks"],
        "neu": ["support answered after a few days", "had to contact support once, they eventually helped"],
    },
    "price": {
        "pos": ["great value for money", "worth every penny", "unbeatable at this price"],
        "neg": ["way overpriced for what you get", "not worth the money", "total waste of money"],
        "neu": ["fair price", "price is reasonable, could be cheaper", "decent value I guess"],
    },
}

OPENERS = {
    "pos": ["Love {p}!", "Really happy with {p}.", "Absolutely love it.", "Best purchase this year.",
            "Highly recommend {p}.", "Exceeded my expectations.", "", "", "", "", ""],
    "neg": ["Very disappointed.", "Do not buy {p}.", "Regret buying {p}.", "Terrible experience.",
            "Returned it.", "Worst purchase ever.", "", "", "", "", ""],
    "neu": ["It's ok.", "Mixed feelings about {p}.", "Does the job.", "Not bad, not great.",
            "Average product.", "It works, I guess.", "", ""],
}
CLOSERS = {
    "pos": ["Would buy again.", "Recommend to everyone.", "Gonna buy another one as a gift.", "", "", ""],
    "neg": ["Avoid.", "Requesting a refund.", "Never again.", "Save your money.", "", "", ""],
    "neu": ["Might keep it.", "Can't decide if I'd buy again.", "Fine for now.", "", "", ""],
}
TITLES = {
    "pos": ["Great product", "Love it", "Worth it", "Excellent", "Amazing quality", "Very satisfied", "Solid buy"],
    "neg": ["Disappointed", "Waste of money", "Broke quickly", "Not as advertised", "Terrible", "Returned", "Avoid"],
    "neu": ["It's okay", "Average", "Decent", "Mixed feelings", "Meh", "Not bad", "As expected"],
}

SLANG = {"great": "gr8", "you": "u", "to be honest": "tbh", "really": "rly", "with": "w/",
         "because": "bc", "though": "tho", "please": "pls", "probably": "prob"}
EMOJI = {"pos": ["😍", "👍", "🔥", "❤️", "😊"], "neg": ["😡", "👎", "🤬", "😤", "💩"], "neu": ["😐", "🤷", "🙂"]}

CHANNELS = ["Official Webstore", "Amazon", "Walmart Marketplace", "Best Buy Marketplace"]
CHANNEL_W = [0.30, 0.40, 0.18, 0.12]
CHANNEL_VARIANTS = {
    "Official Webstore": ["official webstore", "Official Webstore ", "WEBSTORE", "Webstore"],
    "Amazon": ["amazon", "AMAZON", " Amazon", "Amazon.com"],
    "Walmart Marketplace": ["walmart", "Walmart", "WALMART MARKETPLACE"],
    "Best Buy Marketplace": ["bestbuy", "Best Buy", "best buy marketplace"],
}
REGIONS = ["US - Northeast", "US - Midwest", "US - South", "US - West", "Canada", "UK"]
REGION_W = [0.20, 0.17, 0.25, 0.20, 0.10, 0.08]

START = date(2024, 1, 1)
END = date(2025, 12, 31)


def random_date():
    return START + timedelta(days=int(rng.integers(0, (END - START).days + 1)))


def base_sentiment_probs(product, d, channel):
    """Base probabilities [pos, neu, neg] with hidden business patterns."""
    p = np.array([0.62, 0.15, 0.23])
    # Pattern 1: firmware v2.3 issue on earbuds (Aug - Oct 2025)
    if product == "AirPulse Pro Earbuds" and date(2025, 8, 1) <= d <= date(2025, 10, 31):
        p = np.array([0.30, 0.15, 0.55])
    # Pattern 2: smart-home app issues all year round
    if PRODUCTS[product][0] == "Smart Home":
        p = p + np.array([-0.08, 0.0, 0.08])
    # Pattern 3: holiday shipping delays on marketplaces (Dec)
    if d.month == 12 and channel != "Official Webstore":
        p = p + np.array([-0.07, 0.0, 0.07])
    # Pattern 4: power bank seen as overpriced -> more neutral
    if product == "VoltMax 20K Power Bank":
        p = p + np.array([-0.08, 0.08, 0.0])
    p = np.clip(p, 0.02, None)
    return p / p.sum()


def pick_aspects(product, d, channel, sentiment):
    cat = PRODUCTS[product][0]
    pool = CATEGORY_ASPECTS[cat] + COMMON_ASPECTS
    k = int(rng.choice([1, 2, 3], p=[0.35, 0.45, 0.20]))
    aspects = list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    # Force the hidden patterns into the text
    if product == "AirPulse Pro Earbuds" and date(2025, 8, 1) <= d <= date(2025, 10, 31) and sentiment == "neg":
        if rng.random() < 0.7:
            aspects[0] = "connectivity"
    if cat == "Smart Home" and sentiment == "neg" and rng.random() < 0.5:
        aspects[0] = "app"
    if d.month == 12 and channel != "Official Webstore" and sentiment == "neg" and rng.random() < 0.5:
        aspects[-1] = "shipping"
    if product == "VoltMax 20K Power Bank" and rng.random() < 0.4:
        aspects[-1] = "price"
    return list(dict.fromkeys(aspects))


def compose(product, sentiment, aspects):
    noun = PRODUCTS[product][1]
    parts = []
    opener = random.choice(OPENERS[sentiment]).format(p=noun)
    if opener:
        parts.append(opener)
    for i, a in enumerate(aspects):
        s = sentiment
        # mixed reviews: positive/negative reviews often contain one contrasting clause
        if i > 0 and sentiment in ("pos", "neg") and rng.random() < 0.35:
            s = "neg" if sentiment == "pos" else "pos"
            phrase = random.choice(PHRASES[a][s])
            parts.append(random.choice(["But", "However,", "Only downside:", "On the other hand,"]) + " " + phrase + ".")
            continue
        if sentiment == "neu" and rng.random() < 0.55:
            s = random.choice(["pos", "neg"])
        phrase = random.choice(PHRASES[a][s])
        parts.append(phrase[0].upper() + phrase[1:] + ".")
    closer = random.choice(CLOSERS[sentiment])
    if closer:
        parts.append(closer)
    return " ".join(parts)


def add_noise(text, sentiment):
    r = rng.random
    if r() < 0.25:
        text = text.lower()
    for k, v in SLANG.items():
        if k in text and r() < 0.3:
            text = text.replace(k, v)
    if sentiment == "neg" and r() < 0.10:
        text = text.upper()
    if r() < 0.20:
        text += " " + random.choice(EMOJI[sentiment]) * int(rng.integers(1, 4))
    if r() < 0.15:
        text = text.replace(". ", random.choice([".<br/>", ".<br><br>", ". <p>"]), 1)
    if r() < 0.04:
        text += " more info: https://www." + random.choice(["youtube.com/watch?v=abc123", "myblog.net/review", "reddit.com/r/gadgets"])
    if r() < 0.12:
        text = text.replace("so ", "sooo ", 1).replace("very", "veryyy", 1)
    if r() < 0.10:
        text = text.replace("!", "!!!")
    if r() < 0.08:
        text = "  " + text.replace(" ", "  ", 2) + "   "
    if r() < 0.06:  # simple typos
        words = text.split(" ")
        j = int(rng.integers(0, len(words)))
        w = words[j]
        if len(w) > 4:
            i = int(rng.integers(1, len(w) - 1))
            words[j] = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        text = " ".join(words)
    return text


def rating_from_sentiment(s):
    if s == "pos":
        return int(rng.choice([4, 5], p=[0.35, 0.65]))
    if s == "neg":
        return int(rng.choice([1, 2], p=[0.65, 0.35]))
    return int(rng.choice([2, 3, 4], p=[0.15, 0.70, 0.15]))


def helpful_votes(s):
    lam = {"pos": 1.2, "neu": 0.8, "neg": 2.6}[s]
    v = int(rng.poisson(lam))
    if rng.random() < 0.015:  # viral reviews -> outliers
        v += int(rng.integers(40, 400))
    return v


LABEL = {"pos": "positive", "neu": "neutral", "neg": "negative"}

rows = []
for i in range(N_REVIEWS):
    product = str(rng.choice(list(PRODUCTS), p=PRODUCT_WEIGHTS))
    d = random_date()
    channel = str(rng.choice(CHANNELS, p=CHANNEL_W))
    sentiment = str(rng.choice(["pos", "neu", "neg"], p=base_sentiment_probs(product, d, channel)))

    aspects = pick_aspects(product, d, channel, sentiment)
    text_sent = sentiment
    if rng.random() < 0.09:  # label noise: annotators disagree / sarcasm
        text_sent = str(rng.choice([s for s in ["pos", "neu", "neg"] if s != sentiment]))
    text = add_noise(compose(product, text_sent, aspects), text_sent)

    channel_raw = channel if rng.random() > 0.15 else str(rng.choice(CHANNEL_VARIANTS[channel]))
    verified = rng.random() < 0.82
    verified_raw = ("Yes" if verified else "No") if rng.random() > 0.1 else str(rng.choice(["Y", "N", "TRUE", "FALSE", "yes", "no"]))

    rows.append({
        "review_id": f"RV-{i + 1:06d}",
        "customer_id": f"C{int(rng.integers(10000, 99999))}",
        "review_date": d.isoformat(),
        "product_name": product,
        "product_category": PRODUCTS[product][0],
        "sales_channel": channel_raw,
        "customer_region": str(rng.choice(REGIONS, p=REGION_W)),
        "verified_purchase": verified_raw,
        "rating": rating_from_sentiment(sentiment),
        "helpful_votes": helpful_votes(sentiment),
        "review_title": random.choice(TITLES[text_sent if rng.random() > 0.15 else str(rng.choice(["pos", "neu", "neg"]))]),
        "review_text": text,
        "sentiment": LABEL[sentiment],
    })

df = pd.DataFrame(rows)

# Messiness: missing values
for col, frac in [("review_text", 0.015), ("customer_region", 0.03), ("review_title", 0.05),
                  ("verified_purchase", 0.01), ("helpful_votes", 0.01)]:
    idx = df.sample(frac=frac, random_state=int(rng.integers(0, 1e6))).index
    df.loc[idx, col] = np.nan
# Messiness: blank-only text
idx = df.sample(n=25, random_state=7).index
df.loc[idx, "review_text"] = "   "
# Messiness: exact duplicates (double-submitted reviews)
dups = df.sample(n=130, random_state=11)
df = pd.concat([df, dups]).sample(frac=1, random_state=SEED).reset_index(drop=True)

df.to_csv("data/customer_reviews.csv", index=False)
print(df.shape)
print(df["sentiment"].value_counts(normalize=True).round(3))
