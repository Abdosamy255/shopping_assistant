import pandas as pd
import numpy as np
import re

RAW_PATH = r"P:\a_universty\Third_year\NLP\a_project\data\raw_products.csv"
OUT_PATH =r"P:\a_universty\Third_year\NLP\a_project\data\products.csv"

def clean_price(x):
    # تحويل "Rs. 8,299" إلى رقم 8299
    if isinstance(x, str):
        x = re.sub(r"[^\d]", "", x)
    return pd.to_numeric(x, errors="coerce")

def main():
    df = pd.read_csv(RAW_PATH)

    out = pd.DataFrame()
    out["product_name"] = df["product_name"].astype(str)

    # category
    if "product_category" in df.columns:
        out["category"] = df["product_category"].astype(str)
    else:
        out["category"] = ""

    # price
    out["price"] = df["product_price"].apply(clean_price)

    # rating
    if "product_ratings" in df.columns:
        out["rating"] = pd.to_numeric(df["product_ratings"], errors="coerce")
    else:
        out["rating"] = np.nan

    # link + image
    out["link"] = df["product_link"].astype(str)
    out["image_url"] = df["product_image"].astype(str)

    # brand — نستخرجه من product_name بحيلة بسيطة
    def guess_brand(name):
        parts = str(name).split()
        return parts[0] if parts else ""
    out["brand"] = df["product_name"].apply(guess_brand)

    # empty fields now
    out["color"] = ""
    out["size"] = ""

    # drop invalid prices
    out = out.dropna(subset=["price"])

    # sample if dataset large
    if len(out) > 3000:
        out = out.sample(n=3000, random_state=42)

    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("Exported:", OUT_PATH, "with", len(out), "rows")

if __name__ == "__main__":
    main()
