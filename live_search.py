import pandas as pd
import subprocess
import shlex
import re
import pandas as pd
import re
import pandas as pd

def clean_price_amazon(raw):
    if not isinstance(raw, str):
        return pd.to_numeric(raw, errors="coerce")

    # remove text
    x = raw.replace("EGP", "").replace("ج.م", "").strip()
    x = x.replace(" ", "")

    # remove symbols
    x = re.sub(r"[^\d]", "", x)

    if not x.isdigit():
        return pd.NA

    val = int(x)

    # convert (pounds & cents)
    result = val / 100

    return round(result, 2)


def live_search(attrs):
    # 1) بنبني الـ query
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

    # 2) path CSV
    output_file = "data/live_amazon.csv"

    # 3) بنادي الكراولر من streamlit
    cmd = f'python crawlir.py -q "{query}" --pages 1 --max-products 30 --output {output_file}'
    subprocess.run(shlex.split(cmd), check=False)

    # 4) نقرأ الـ CSV
    try:
        df = pd.read_csv(output_file)
        df["price"] = df["price"].apply(clean_price)
        df = df.dropna(subset=["price"])

    except:
        return pd.DataFrame()

    return df
