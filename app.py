import os
import sys
import time
from datetime import datetime

import streamlit as st
import pandas as pd

from crawlir import crawl_amazon_to_csv  # 👈 ضيف دي
from nlp.preprocessing import preprocess_text
from nlp.attribute_extraction import extract_attributes
from live_search import live_search,clean_price_amazon
from search.search_engine import search_products

#عشان تظبط شكل السعر
def clean_price(x):
    if not isinstance(x, str):
        return pd.to_numeric(x, errors="coerce")

    x = x.replace("EGP", "").replace("ج.م", "").replace("جنيه", "")
    x = x.replace(" جنيه", "").replace("ريال", "").strip()
    x = x.replace(" ", "")
    x = x.replace(",", "")
    x = re.sub(r"[^\d.]", "", x)

    val = pd.to_numeric(x, errors="coerce")
    try:
        if val.is_integer():
            return int(val)
        return val
    except:
        return val

# لو في مشكلة imports نضمن إن الجذر في الـ path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from nlp.preprocessing import preprocess_text
from nlp.attribute_extraction import extract_attributes

# =========================
# إعداد الصفحة
# =========================
st.set_page_config(
    page_title="Smart Shopping Assistant",
    page_icon="🛒",
    layout="wide"
)

# تهيئة الـ session_state للتاريخ
if "history" not in st.session_state:
    st.session_state.history = []  # كل عنصر: {"time", "query", "attrs", "count"}

# =========================
# دوال مساعدة
# =========================

def run_search(user_input: str):
    """NLP فقط: يرجّع tokens + attrs ونسيب حتة البحث للـ live search."""
    tokens, lang, intents = preprocess_text(user_input)
    attrs = extract_attributes(tokens, lang)
    attrs["intents"] = intents
    return tokens, attrs


def apply_ui_filters(results: pd.DataFrame,
                     sort_by: str,
                     sort_dir: str,
                     max_price: float | None,
                     brand_filter: str | None):
    """تطبيق الفلاتر والـ sorting على DataFrame النتائج."""
    df = results.copy()

    # فلتر السعر
    if max_price is not None and "price" in df.columns:
        df = df[df["price"] <= max_price]

    # فلتر البراند (لو فيه عمود brand)
    if brand_filter and brand_filter.strip() and "brand" in df.columns:
        bf = brand_filter.strip().lower()
        df = df[df["brand"].fillna("").str.lower().str.contains(bf, na=False)]

    # ترتيب
    if sort_by and sort_by in df.columns:
        ascending = (sort_dir == "Ascending")
        df = df.sort_values(by=sort_by, ascending=ascending)

    return df
import re

def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    if "price" not in df.columns:
        return df
    df = df.copy()
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .replace("", None)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    return df.dropna(subset=["price"])


def render_product_card(row: pd.Series):
    """عرض كارت منتج واحد بـ HTML بسيط."""
    # في الـ Live search الأعمدة غالبًا: title, price, rating, image_url, product_link
    name = row.get("title", "Unknown product")
    price = row.get("price", "-")
    rating = row.get("rating", "-")
    link = row.get("product_link", "#")
    img = row.get("image_url", None)
    brand = row.get("brand", "-")

    left, right = st.columns([1, 3])

    with left:
        if isinstance(img, str) and img.strip():
            st.image(img, use_column_width=True)
        else:
            st.markdown(
                "<div style='width:100%;height:100px;border-radius:12px;"
                "background:linear-gradient(135deg,#00D09C33,#ffffff11);"
                "display:flex;align-items:center;justify-content:center;font-size:32px;'>🛍</div>",
                unsafe_allow_html=True
            )

    with right:
        st.markdown(
            f"""
<div style="background:#1A1D23;padding:14px;border-radius:12px;margin-bottom:6px;">
  <div style="font-size:20px;font-weight:700;margin-bottom:4px;">{name}</div>
  <div>🔖 <b>Brand:</b> {brand}</div>
  <div>💲 <b>Price:</b> {price} جنيه</div>
  <div>⭐ <b>Rating:</b> {rating}</div>
  <div style="margin-top:8px;">
    <a href="{link}" target="_blank"
       style="background:#00D09C;color:black;padding:6px 10px;border-radius:6px;
              text-decoration:none;font-weight:600;">
      رابط الشراء
    </a>
  </div>
</div>
""",
            unsafe_allow_html=True
        )


# =========================
# الهيدر العام
# =========================

st.markdown(
    """
<style>
html, body, [class*="css"]  {
    font-family: "Segoe UI", "Cairo", sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align:center; font-size:40px; font-weight:700; margin-bottom:4px;">
🛒 Smart Shopping Assistant
</div>
<p style="text-align:center; color:#aaaaaa; margin-top:0;">
Arabic NLP → Attribute Extraction → Live Product Ranking from Amazon
</p>
<hr>
""",
    unsafe_allow_html=True,
)

# =========================
# Sidebar (فلترة + Sorting)
# =========================

st.sidebar.header("⚙️ Controls")

# أقصى سعر
max_price_val = st.sidebar.number_input(
    "أقصى سعر (اختياري)",
    min_value=0,
    value=0,
    step=100
)
if max_price_val == 0:
    max_price_val = None
else:
    max_price_val = float(max_price_val)

# فلتر البراند (تكست حر بدل ما نعتمد على df_products)
brand_filter = st.sidebar.text_input(
    "فلترة حسب البراند (مثال: samsung, xiaomi)",
    value=""
).strip() or None

# Sorting options بناءً على أعمدة الـ live_search
sort_by = st.sidebar.selectbox(
    "ترتيب حسب",
    options=["price", "rating", "title"],
    index=0
)
sort_dir = st.sidebar.radio("اتجاه الترتيب", ["Ascending", "Descending"], index=0)

st.sidebar.markdown("---")
st.sidebar.write("✳️ كل بحث جديد بيتسجل في صفحة **History**.")

# =========================
# Tabs
# =========================

tab_search, tab_history, tab_about = st.tabs(["🔍 Search", "🕒 History", "ℹ️ About"])

# ---------- TAB 1: Search ----------
with tab_search:
    st.markdown("### 🔍 اكتب وصف المنتج")

    default_text = "عايز كوتش اسود مقاس 46 تحت 1500"
    user_input = st.text_area(
        "وصف المنتج",
        placeholder=default_text,
        height=80
    )

    search_clicked = st.button("🚀 ابحث", use_container_width=True)

if search_clicked:
    if not user_input.strip():
        st.warning("اكتب وصف المنتج الأول.")
    else:
        with st.spinner("جاري تحليل النص والبحث عن أفضل المنتجات..."):

            # 1) NLP: Preprocessing + Attributes (عشان نعرضهم للدكتور)
            tokens, lang, intents = preprocess_text(user_input)
            attrs = extract_attributes(tokens, lang)
            attrs["intents"] = intents

            # 2) نبني الـ Query من الـ tokens بعد الـ preprocessing
            query = " ".join(tokens).strip()
            if not query:
                st.error("بعد الـ preprocessing مابقاش فيه كلمات مفيدة في الـ Query.")
                st.stop()

            # 3) ننده الكراولر عشان يكتب في CSV
            csv_path = os.path.join("data", "live_amazon.csv")

            crawl_amazon_to_csv(
                query=query,
                output_path=csv_path,
                language="en",       # لو حابب تخليها "ar" أو تـ switch حسب lang
                pages=1,
                detailed=False,
                max_products=30,
                append=False         # False = كل بحث يكتب ملف جديد
            )

            # 4) نقرأ من CSV
            try:
                raw_results = pd.read_csv(csv_path)
                raw_results['price'] = raw_results['price'].apply(clean_price_amazon)
                raw_results = raw_results.dropna(subset=['price'])
                

            except FileNotFoundError:
                st.error("ملف الـ CSV مش موجود، حصلت مشكلة في الكراولر.")
                st.stop()

            # 5) لو عاملين search_query في الكراولر نفلتر بيه (لو ضفته)
            if "search_query" in raw_results.columns:
                raw_results = raw_results[raw_results["search_query"] == query]

            # 6) تنظيف السعر
            raw_results = clean_price_column(raw_results)

            # 7) إعادة تسمية الأعمدة عشان تمشي مع الكروت القديمة
            # الكراولر بيطلع: title, price, rating, image, product_link
            # الواجهة القديمة كانت متعودة على: product_name, image_url, link
            results = raw_results.rename(
                columns={
                    "title": "product_name",
                    "image": "image_url",
                    "product_link": "link",
                }
            )
            # 8️) Search + Ranking using NLP attributes
            final_results = search_products(
                results,
                attrs,
                top_n=50
            )

            time.sleep(0.3)


        # 9) حفظ في الـ history
        st.session_state.history.insert(
            0,
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": user_input,
                "attrs": attrs,
                "count": int(len(final_results))
            }
        )

        # 10) عرض الـ Tokens و الـ Attributes
        col_tokens, col_attrs = st.columns(2)
        with col_tokens:
            st.markdown("### 🔤 Tokens بعد الـ Preprocessing")
            st.code(tokens)

        with col_attrs:
            st.markdown("### 🧠 السمات المستخرجة (Attributes)")
            st.json(attrs)

        st.markdown("---")
        st.markdown("### 🛍 النتائج")

        # 11) عرض النتائج
        if final_results.empty:
            st.info("❌ لا توجد نتائج مطابقة بناءً على الوصف والفلاتر الحالية.")
        else:
            st.success(f"👍 تم إيجاد {len(final_results)} نتيجة بعد الفلاتر")

            # Cards
            top_cards = final_results.head(20)
            cols = st.columns(3)
            for i, (_, row) in enumerate(top_cards.iterrows()):
                with cols[i % 3]:
                    st.image(row.get("image_url", ""), width=120)
                    st.markdown(f"**{row.get('product_name','منتج بدون اسم')}**")
                    price = str(row['price'])

# لو مفيش نقطة ونهاية السعر رقمين
                    if "." not in price and len(price) ==7:
                     price = price[:5] + "." + price[5:]

                     st.markdown(f"💸 {price} EGP")
                    elif "." not in price and len(price) ==6:
                     price = price[:4] + "." + price[4:]
                     st.markdown(f"💸 {price} EGP")
                    elif "." not in price and len(price) ==5:
                     price = price[:3] + "." + price[3:]
                     st.markdown(f"💸 {price} EGP")

                    

                    st.markdown(f"⭐ {row.get('rating','-')}")
                    st.link_button("عرض على Amazon", row.get("link", "#"))
                    st.markdown("---")

            with st.expander("عرض كل النتائج في جدول"):
                st.dataframe(final_results.reset_index(drop=True))


# ---------- TAB 2: History ----------
with tab_history:
    st.markdown("### 🕒 Search History")

    if not st.session_state.history:
        st.info("لسه ماعملتش أي بحث.")
    else:
        for item in st.session_state.history:
            st.markdown(
                f"""
- **{item['time']}**  
  - Query: `{item['query']}`  
  - Results: **{item['count']}**  
  - Attributes: `{item['attrs']}`
"""
            )

        st.markdown("---")
        st.markdown("#### Summary Table")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df)

        if st.button("🧹 مسح التاريخ بالكامل"):
            st.session_state.history = []
            st.experimental_rerun()

# ---------- TAB 3: About ----------
with tab_about:
    st.markdown("### ℹ️ About Project")
    st.write(
        """
هذا المشروع هو **Smart Shopping Assistant** لمادة **NLP**:

- يفهم وصف المنتج بالعربي (Natural Language).
- يطبق خطوات Text Preprocessing:
  - Normalization
  - Tokenization
  - Stopwords Removal
- يستخرج السمات (Attributes) مثل:
  - نوع المنتج (Product)
  - اللون (Color)
  - المقاس (Size)
  - الميزانية (Budget)
  - البراند (Brand)
- يستخدم Live Search من Amazon عبر Web Scraping (crawlir.py)
  بدل الاعتماد فقط على ملف CSV ثابت.

يمكن تطويره لاحقًا لاستخدام APIs رسمية أو Models أقوى (BERT, LLMs) أو دعم مواقع متعددة (Jumia / Noon / Amazon).
"""
    )
    st.markdown("---")
    st.markdown("👨‍💻 *Built by: Your Team (Third Year AI / NLP)*")
