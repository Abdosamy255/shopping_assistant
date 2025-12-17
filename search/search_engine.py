# search/search_engine.py
import pandas as pd
import re

def relevance_score(row, attrs):
    """
    Calculate relevance score for ranking
    """
    score = 0

    title = str(row.get("product_name", "")).lower()
    price = row.get("price", 0)
    rating = row.get("rating", 0)

    intents = attrs.get("intents", [])
    features = attrs.get("features", {})
    price_range = attrs.get("price_range", {})

    # 1️⃣ Brand match
    if attrs.get("brand") and attrs["brand"] in title:
        score += 30

    # 2️⃣ Price range handling
    if price_range.get("max") and price <= price_range["max"]:
        score += 20

    if price_range.get("min") and price >= price_range["min"]:
        score += 10

    # around price → مرونة
    if "around_price" in intents and price_range.get("max"):
        diff = abs(price - price_range["max"])
        score += max(0, 10 - diff / 500)

    # 3️⃣ Cheap intent
    if "cheap" in intents and price:
        score += max(0, 20 - price / 1000)

    # 4️⃣ Quality intent (rating)
    if rating and "quality" in intents:
        score += (rating / 5) * 25
    elif rating:
        score += (rating / 5) * 15

    # 5️⃣ Feature match
    for feat in features.values():
        if str(feat).lower() in title:
            score += 5

    return score


# ---- Keyword maps ----
PHONE_KEYWORDS = [
    "galaxy", "iphone", "redmi", "note", "infinix",
    "tecno", "vivo", "realme", "poco", "pixel", "nokia"
]

EXCLUDE_KEYWORDS = ["buds", "earbuds", "airpods", "headphone", "earphone", "earbud"]

def load_products(path="data/products.csv"):
    """
    Load cleaned dataset
    """
    return pd.read_csv(path)

def search_products(df, attrs, top_n=5):
    """
    Product filtering and ranking
    """
    result = df.copy()

    # Convert to lowercase strings
    for col in ["product_name", "category", "brand"]:
        if col in result.columns:
            result[col] = result[col].astype(str).str.lower()

    # ---- Filter by product type ----
    product = attrs.get("product")

    if product == "phone":
        phone_pattern = "|".join(re.escape(k) for k in PHONE_KEYWORDS)
        result = result[result["product_name"].str.contains(phone_pattern, na=False)]

        exclude_pattern = "|".join(EXCLUDE_KEYWORDS)
        result = result[~result["product_name"].str.contains(exclude_pattern, na=False)]

    # ---- Budget ----
    if attrs.get("budget") is not None and "price" in result.columns:
        result = result[result["price"] <= attrs["budget"]]

    # ---- Price range ----
    price_range = attrs.get("price_range", {})
    if "price" in result.columns:
        if price_range.get("max") is not None:
            result = result[result["price"] <= price_range["max"]]
        if price_range.get("min") is not None:
            result = result[result["price"] >= price_range["min"]]



    # ---- Brand ----
    brand = attrs.get("brand")
    if brand:
        b = str(brand).lower()
        result = result[result["product_name"].str.contains(b, na=False)]

    # ---- Relevance scoring ----
    result["score"] = result.apply(
    lambda row: relevance_score(row, attrs),
    axis=1
    )

    result = result.sort_values(by="score", ascending=False)

