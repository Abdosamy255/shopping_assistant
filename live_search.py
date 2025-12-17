# live_search.py

import pandas as pd
import subprocess
import shlex
import re

from search.search_engine import search_products

# ---------------------------
# 1) Clean Price from Amazon
# ---------------------------
def clean_price_amazon(raw):
    """
    Amazon Egypt sometimes returns:
        "29,900 EGP"
        "29900"
        "29900.00"
        "29 900"
        etc...

    Amazon prices are in “pounds & cents” format
    so 29900 => 299.00 EGP
    """
    if not isinstance(raw, str):
        return pd.to_numeric(raw, errors="coerce")

    # remove text
    x = raw.replace("EGP", "").replace("ج.م", "")
    x = x.replace("جنيه", "").strip()
    x = x.replace(" ", "").replace(",", "")

    # keep digits only
    x = re.sub(r"[^\d]", "", x)

    if not x.isdigit():
        return pd.NA

    # convert
    val = int(x)
    result = val / 100
    return round(result, 2)


# ---------------------------
# 2) Live Search
# ---------------------------
def live_search(attrs: dict, max_products: int = 30):
    """
    - Build query from attributes
    - Run crawler to generate CSV
    - Load CSV
    - Clean data
    - Rank using search_engine
    """
    # 1) Build query text from attributes
    parts = []

    if attrs.get("brand"):
        parts.append(attrs["brand"])

    if attrs.get("product"):
        parts.append(attrs["product"])

    if attrs.get("color"):
        parts.append(attrs["color"])

    if attrs.get("size"):
        parts.append(str(attrs["size"]))

    query = " ".join([p for p in parts if p]).strip()
    if not query:
        query = "best deals"

    print("🔍 Live query:", query)

    # 2) Output CSV path
    output_file = "data/live_amazon.csv"

    # 3) Run crawler
    cmd = f'python crawlir.py -q "{query}" --pages 1 --max-products {max_products} --output {output_file}'
    subprocess.run(shlex.split(cmd), check=False)

    # 4) Load CSV
    try:
        df = pd.read_csv(output_file)
    except FileNotFoundError:
        return pd.DataFrame()

    if df.empty:
        return df

    # 5) Clean price
    df["price"] = df["price"].apply(clean_price_amazon)
    df = df.dropna(subset=["price"])

    # 6) Rank using search engine
    ranked = search_products(df, attrs, top_n=5)

    return ranked
