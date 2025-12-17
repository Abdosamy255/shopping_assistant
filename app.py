import os
import sys
import time
import re
from datetime import datetime
from typing import Dict, Optional

import streamlit as st
import pandas as pd

from crawlir import crawl_amazon_to_csv
from nlp.preprocessing import preprocess_text
from nlp.attribute_extraction_enhanced import extract_enhanced_attributes
from nlp.utils import clean_price_egp
from search.search_engine_enhanced import search_products_enhanced, calculate_relevance_score

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Smart Shopping Assistant",
    page_icon="🛒",
    layout="wide"
)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# Helper Functions
# =========================

def apply_ui_filters(
    results: pd.DataFrame,
    sort_by: str,
    sort_dir: str,
    max_price: Optional[float],
    min_rating: Optional[float],
    brand_filter: Optional[str]
) -> pd.DataFrame:
    """Apply UI filters and sorting to results DataFrame."""
    df = results.copy()

    # Price filter
    if max_price is not None and max_price > 0 and "price" in df.columns:
        df = df[df["price"] <= max_price]

    # Rating filter
    if min_rating is not None and min_rating > 0 and "rating_numeric" in df.columns:
        df = df[df["rating_numeric"] >= min_rating]

    # Brand filter
    if brand_filter and brand_filter.strip() and "product_name" in df.columns:
        bf = brand_filter.strip().lower()
        df = df[df["product_name"].fillna("").str.lower().str.contains(bf, na=False)]

    # Sorting
    if sort_by and sort_by in df.columns:
        ascending = (sort_dir == "Ascending")
        df = df.sort_values(by=sort_by, ascending=ascending)

    return df


def clean_price_column(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert price column using unified utility."""
    if "price" not in df.columns:
        return df
    
    df = df.copy()
    df["price"] = df["price"].apply(clean_price_egp)
    df = df.dropna(subset=["price"])
    
    return df


def extract_rating_numeric(rating_str: str) -> float:
    """Extract numeric rating from string like '4.5 out of 5 stars'."""
    if pd.isna(rating_str) or not isinstance(rating_str, str):
        return 0.0
    
    match = re.search(r'(\d+\.?\d*)', rating_str)
    if match:
        try:
            return float(match.group(1))
        except:
            return 0.0
    return 0.0


def format_price_display(price: float) -> str:
    """Format price for display with proper separators."""
    if pd.isna(price):
        return "N/A"
    
    price_str = f"{price:.2f}"
    parts = price_str.split(".")
    
    # Add thousands separator
    integer_part = parts[0]
    if len(integer_part) > 3:
        integer_part = f"{integer_part[:-3]},{integer_part[-3:]}"
    
    return f"{integer_part}.{parts[1]} EGP"


def render_product_card_enhanced(row: pd.Series, show_relevance: bool = True):
    """Enhanced product card with better styling and relevance score."""
    name = row.get("product_name", "Unknown product")
    price = row.get("price", 0)
    rating = row.get("rating", "-")
    rating_numeric = row.get("rating_numeric", 0)
    link = row.get("link", "#")
    img = row.get("image_url", None)
    relevance = row.get("relevance_score", 0)

    col1, col2 = st.columns([1, 3])

    with col1:
        if isinstance(img, str) and img.strip():
            st.image(img, width=300)

        else:
            st.markdown(
                "<div style='width:100%;height:120px;border-radius:12px;"
                "background:linear-gradient(135deg,#00D09C33,#ffffff11);"
                "display:flex;align-items:center;justify-content:center;font-size:36px;'>🛍</div>",
                unsafe_allow_html=True
            )

    with col2:
        # Relevance badge
        if show_relevance and relevance > 0:
            relevance_color = "#00D09C" if relevance >= 50 else "#FFA500" if relevance >= 30 else "#FF6B6B"
            st.markdown(
                f'<span style="background:{relevance_color};color:white;'
                f'padding:4px 10px;border-radius:6px;font-size:12px;font-weight:600;">'
                f'Match: {relevance:.0f}%</span>',
                unsafe_allow_html=True
            )

        st.markdown(f"### {name[:80]}{'...' if len(name) > 80 else ''}")
        
        # Price with formatting
        price_display = format_price_display(price)
        st.markdown(f"💰 **{price_display}**")
        
        # Rating with stars
        if rating_numeric > 0:
            stars = "⭐" * int(rating_numeric)
            st.markdown(f"{stars} ({rating_numeric:.1f}/5.0)")
        else:
            st.markdown(f"⭐ {rating}")
        
        # Link button
        if link and link != "#":
            st.link_button("🛒 View on Amazon", link, use_container_width=True)


# =========================
# Main UI
# =========================

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Segoe UI", "Cairo", sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align:center; font-size:42px; font-weight:700; margin-bottom:6px;">
🛒 Smart Shopping Assistant
</div>
<p style="text-align:center; color:#aaaaaa; margin-top:0; font-size:16px;">
Enhanced NLP Pipeline → Intelligent Ranking → Live Amazon Results
</p>
<hr>
""",
    unsafe_allow_html=True,
)

# =========================
# Sidebar Controls
# =========================

st.sidebar.header("⚙️ Search Filters")

# Price filter
max_price_val = st.sidebar.number_input(
    "🏷️ Maximum Price (EGP)",
    min_value=0,
    value=0,
    step=500,
    help="Leave at 0 for no limit"
)
max_price_val = None if max_price_val == 0 else float(max_price_val)

# Rating filter
min_rating_val = st.sidebar.slider(
    "⭐ Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5,
    help="Filter products by minimum rating"
)
min_rating_val = None if min_rating_val == 0 else min_rating_val

# Brand filter
brand_filter = st.sidebar.text_input(
    "🔍 Filter by Brand",
    value="",
    placeholder="e.g., samsung, xiaomi, apple"
).strip() or None

# Sorting options
st.sidebar.markdown("### 📊 Sorting")
sort_by = st.sidebar.selectbox(
    "Sort by",
    options=["relevance_score", "price", "rating_numeric"],
    index=0,
    format_func=lambda x: {
        "relevance_score": "Relevance",
        "price": "Price",
        "rating_numeric": "Rating"
    }.get(x, x)
)
sort_dir = st.sidebar.radio(
    "Direction",
    ["Descending", "Ascending"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip:** Try queries like:\n- 'Samsung phone under 10000'\n- 'عايز كوتش اسود مقاس 42'\n- 'laptop with 16GB RAM'")

# =========================
# Main Tabs
# =========================

tab_search, tab_history, tab_about = st.tabs(["🔍 Search", "🕒 History", "ℹ️ About"])

# ---------- TAB 1: Search ----------
with tab_search:
    st.markdown("### 🔍 Enter Product Description")

    user_input = st.text_area(
        "Describe what you're looking for",
        placeholder="search for products ",
        height=100,
        help="Describe the product in Arabic or English"
    )

    col1, col2,col3 = st.columns([2, 1, 1])
    with col1:
        search_clicked = st.button("🚀 Search Products", use_container_width=True, type="primary")
    with col2:
        max_results = st.number_input("Results", min_value=10, max_value=50, value=30, step=5)
    with col3:
        no_pages = st.number_input("Pages", min_value=1, max_value=5, value=1, step=1)

    if search_clicked:
        if not user_input.strip():
            st.warning("⚠️ Please enter a product description")
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Step 1: Preprocessing
                status_text.text("🔍 Analyzing your query...")
                progress_bar.progress(20)
                tokens, lang = preprocess_text(user_input)
                time.sleep(0.3)

                # Step 2: Enhanced attribute extraction
                status_text.text("🧠 Extracting attributes...")
                progress_bar.progress(40)
                attrs = extract_enhanced_attributes(tokens, user_input, lang)
                time.sleep(0.3)

                # Step 3: Build search query
                query = " ".join(tokens).strip()
                if not query:
                    st.error("❌ Could not extract meaningful search terms from your query.")
                    progress_bar.empty()
                    status_text.empty()
                    st.stop()

                # Step 4: Crawl Amazon
                status_text.text("🛒 Searching Amazon Egypt...")
                progress_bar.progress(60)
                
                csv_path = os.path.join("data", "live_amazon.csv")
                os.makedirs("data", exist_ok=True)

                crawl_amazon_to_csv(
                    query=query,
                    output_path=csv_path,
                    language="en",
                    pages=no_pages,
                    detailed=True,
                    max_products=int(max_results),
                     
                    append=False
                )
                time.sleep(0.3)

                # Step 5: Load and process results
                status_text.text("📊 Processing results...")
                progress_bar.progress(80)

                try:
                    raw_results = pd.read_csv(csv_path)
                except FileNotFoundError:
                    st.error("❌ Failed to fetch results from Amazon. Please try again.")
                    progress_bar.empty()
                    status_text.empty()
                    st.stop()

                if raw_results.empty:
                    st.warning("😕 No products found. Try different keywords.")
                    progress_bar.empty()
                    status_text.empty()
                    st.stop()

                # Clean and prepare data
                raw_results = clean_price_column(raw_results)
                
                # Rename columns for compatibility
                results = raw_results.rename(columns={
                    "title": "product_name",
                    "image": "image_url",
                    "product_link": "link"
                })

                # Extract numeric rating
                if "rating" in results.columns:
                    results["rating_numeric"] = results["rating"].apply(extract_rating_numeric)
                else:
                    results["rating_numeric"] = 0.0

                # Step 6: Calculate relevance and rank
                status_text.text("🎯 Ranking by relevance...")
                progress_bar.progress(90)
                
                ranked_results = search_products_enhanced(results, attrs, top_n=len(results))
                
                # Apply UI filters
                final_results = apply_ui_filters(
                    ranked_results,
                    sort_by=sort_by,
                    sort_dir=sort_dir,
                    max_price=max_price_val,
                    min_rating=min_rating_val,
                    brand_filter=brand_filter
                )

                progress_bar.progress(100)
                status_text.text("✅ Search complete!")
                st.success("Products loaded successfully 🎉")

                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                # Save to history
                st.session_state.history.insert(0, {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "query": user_input,
                    "attrs": attrs,
                    "count": len(final_results)
                })

                # Display extracted information
                st.markdown("---")
                col_tokens, col_attrs = st.columns(2)
                
                with col_tokens:
                    st.markdown("### 🔤 Processed Tokens")
                    st.code(", ".join(tokens), language="text")

                with col_attrs:
                    st.markdown("### 🧠 Extracted Attributes")
                    st.json(attrs)

                # Display results
                st.markdown("---")
                st.markdown("### 🛍 Search Results")

                if final_results.empty:
                    st.info("❌ No products match your filters. Try adjusting the filters in the sidebar.")
                else:
                    st.success(f"✅ Found {len(final_results)} products matching your criteria")

                    # Top results in cards
                    top_results = final_results.head(20)
                    
                    for idx, (_, row) in enumerate(top_results.iterrows()):
                        with st.container():
                            render_product_card_enhanced(row, show_relevance=True)
                            if idx < len(top_results) - 1:
                                st.markdown("<hr style='margin: 10px 0;'>", unsafe_allow_html=True)

                    # Expandable table view
                    with st.expander("📋 View All Results as Table"):
                        display_cols = []
                        for col in ["product_name", "price", "rating_numeric", "relevance_score", "link"]:
                            if col in final_results.columns:
                                display_cols.append(col)
                        
                        display_df = final_results[display_cols].copy()
                        display_df["price"] = display_df["price"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                        
                        st.dataframe(
                            display_df.reset_index(drop=True),
                            use_container_width=True,
                            hide_index=True
                        )

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

# ---------- TAB 2: History ----------
with tab_history:
    st.markdown("### 🕒 Search History")

    if not st.session_state.history:
        st.info("📭 No search history yet. Start by searching for products!")
    else:
        for idx, item in enumerate(st.session_state.history):
            with st.expander(f"🔍 Search #{idx + 1} - {item['time']}"):
                st.markdown(f"**Query:** `{item['query']}`")
                st.markdown(f"**Results Found:** {item['count']}")
                st.markdown("**Extracted Attributes:**")
                st.json(item['attrs'])

        st.markdown("---")
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

# ---------- TAB 3: About ----------
with tab_about:
    st.markdown("### ℹ️ About This Project")
    
    st.markdown("""
    ## 🎓 Smart Shopping Assistant
    
    This is an **NLP-powered shopping assistant** built for university coursework, demonstrating:
    
    ### 🔧 Core NLP Pipeline
    - **Text Preprocessing**: Normalization, tokenization, stopword removal
    - **Bilingual Support**: Handles both Arabic and English queries
    - **Attribute Extraction**: Extracts product type, brand, color, size, price range
    - **Feature Detection**: Identifies technical specs (5G, RAM, storage, etc.)
    - **Intent Understanding**: Differentiates between "cheap" vs "premium" queries
    
    ### 🚀 Technical Features
    - **Live Web Scraping**: Real-time product data from Amazon Egypt
    - **Intelligent Ranking**: Multi-factor relevance scoring algorithm
    - **Price Range Handling**: Understands "under 5000", "between 2000 and 3000"
    - **Concurrent Fetching**: Fast parallel data collection
    - **Error Handling**: Robust retry logic and timeout management
    
    ### 📊 Ranking Algorithm
    Products are scored based on:
    - Brand match (30 points)
    - Price fit to budget (25 points)
    - Customer rating (15 points)
    - Feature matching (20 points)
    - Color/specification match (10 points)
    
    ### 🔮 Future Enhancements
    - Multi-platform support (Jumia, Noon, B.Tech)
    - Transformer models (BERT/ArabicBERT)
    - Sentiment analysis on reviews
    - Price tracking and alerts
    - Conversational chatbot interface
    
    ---
    
    ### 👨‍💻 Development Team
    **Built by:** Third Year AI/CS Students  
    **Course:** Natural Language Processing  
    **Technology Stack:** Python, Streamlit, BeautifulSoup, Pandas
    
    ### 📚 Key Improvements Applied
    ✅ Fixed missing imports (`re`, `os`)  
    ✅ Unified price cleaning logic  
    ✅ Enhanced attribute extraction  
    ✅ Implemented relevance scoring  
    ✅ Added proper error handling  
    ✅ Improved UI/UX with progress tracking  
    ✅ Added comprehensive filtering options
    """)
    
    st.markdown("---")
    st.info("💡 **Tips for Best Results:**\n- Be specific with brands and features\n- Include price ranges for better filtering\n- Try both Arabic and English\n- Use technical terms (5G, 128GB, etc.)")