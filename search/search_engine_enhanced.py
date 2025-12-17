"""
search/search_engine_enhanced.py

Enhanced search engine with intelligent relevance scoring.
FIXED: Better product filtering to exclude accessories
"""

import pandas as pd
import re
from typing import Dict, Any, Optional


def calculate_relevance_score(product_row: pd.Series, attrs: Dict[str, Any]) -> float:
    """
    Calculate comprehensive relevance score for a product.
    
    Scoring breakdown (100 points total):
    - Brand match: 30 points
    - Price fit: 25 points
    - Feature match: 20 points
    - Rating quality: 15 points
    - Color match: 10 points
    
    Args:
        product_row: Product data from DataFrame
        attrs: Extracted attributes from query
        
    Returns:
        float: Relevance score (0-100)
    """
    score = 0.0
    
    product_name = str(product_row.get("product_name", "")).lower()
    product_price = product_row.get("price", 0)
    
    # 1. Brand match (30 points)
    brand = attrs.get("brand")
    if brand:
        if brand.lower() in product_name:
            score += 30
        else:
            # Partial match - check for similar brands
            brand_variations = {
                "samsung": ["galaxy", "samsung"],
                "apple": ["iphone", "apple"],
                "xiaomi": ["redmi", "poco", "xiaomi"],
            }
            variations = brand_variations.get(brand, [brand])
            for var in variations:
                if var.lower() in product_name:
                    score += 20  # Partial brand match
                    break
    
    # 2. Price fit (25 points)
    price_range = attrs.get("price_range", {})
    
    if product_price and product_price > 0:
        max_price = price_range.get("max")
        min_price = price_range.get("min")
        target_price = price_range.get("target")
        
        if max_price:
            if product_price <= max_price:
                # Closer to max = higher score
                # Products well below max get slightly lower score
                ratio = product_price / max_price
                if ratio > 0.7:  # Within 70-100% of max budget
                    score += 25
                elif ratio > 0.5:  # Within 50-70% of max budget
                    score += 20
                else:  # Below 50% of budget
                    score += 15
        
        if min_price and product_price >= min_price:
            score += 5  # Bonus for meeting minimum
        
        if target_price:
            # Closeness to target price
            diff_ratio = abs(product_price - target_price) / target_price
            if diff_ratio <= 0.1:  # Within 10%
                score += 10
            elif diff_ratio <= 0.2:  # Within 20%
                score += 5
    
    # 3. Feature matching (20 points max)
    features = attrs.get("features", {})
    feature_score = 0
    
    for feature_key, feature_value in features.items():
        feature_str = str(feature_value).lower()
        if feature_str in product_name:
            feature_score += 5  # 5 points per feature match
    
    score += min(feature_score, 20)  # Cap at 20 points
    
    # 4. Rating quality (15 points)
    rating = product_row.get("rating_numeric", 0)
    if rating and rating > 0:
        # Scale rating to 15 points (5 stars = 15 points)
        score += (rating / 5.0) * 15
    
    # 5. Color match (10 points)
    color = attrs.get("color")
    if color:
        if color.lower() in product_name:
            score += 10
    
    # 6. Product type match (bonus 5 points)
    product_type = attrs.get("product")
    if product_type:
        type_keywords = {
            "phone": ["phone", "smartphone", "mobile"],
            "laptop": ["laptop", "notebook"],
            "shoes": ["shoe", "sneaker", "boot"],
            "watch": ["watch"],
        }
        keywords = type_keywords.get(product_type, [product_type])
        for keyword in keywords:
            if keyword in product_name:
                score += 5
                break
    
    # 7. Quality level match (bonus)
    quality_level = attrs.get("quality_level")
    if quality_level:
        if quality_level == "cheap" and product_price:
            # Reward lower prices for cheap queries
            if product_price < 5000:
                score += 5
        elif quality_level == "premium" and product_price:
            # Reward higher prices for premium queries
            if product_price > 15000:
                score += 5
    
    return round(score, 2)


def is_accessory(product_name: str) -> bool:
    """
    Check if product is an accessory (case, cover, charger, etc.)
    Returns True if it's an accessory, False if it's the actual product.
    """
    product_lower = product_name.lower()
    
    # Accessory keywords
    accessory_keywords = [
        "case", "cover", "كفر", "جراب",
        "screen protector", "واقي شاشة", "حماية",
        "charger", "شاحن", "cable", "كابل",
        "adapter", "محول", "earphone", "سماعة",
        "holder", "حامل", "stand", "ستاند",
        "skin", "sticker", "ملصق",
        "tempered glass", "زجاج",
        "tpu", "silicone", "سيليكون",
        "pouch", "جيب", "bag", "حقيبة",
        "strap", "حزام", "band"
    ]
    
    # Check if any accessory keyword exists
    for keyword in accessory_keywords:
        if keyword in product_lower:
            return True
    
    return False


def search_products_enhanced(
    df: pd.DataFrame, 
    attrs: Dict[str, Any], 
    top_n: int = 20
) -> pd.DataFrame:
    """
    Enhanced product search with intelligent ranking.
    FIXED: Better filtering to exclude accessories
    
    Args:
        df: DataFrame containing product catalog
        attrs: Extracted attributes from user query
        top_n: Number of top results to return
        
    Returns:
        DataFrame with ranked products and relevance scores
    """
    if df.empty:
        return df
    
    result = df.copy()
    
    # Ensure required columns exist
    if "product_name" not in result.columns:
        if "title" in result.columns:
            result = result.rename(columns={"title": "product_name"})
    
    # CRITICAL FIX: Remove accessories first!
    product_type = attrs.get("product")
    if product_type in ["phone", "laptop", "tablet", "watch"]:
        print(f"🔍 Filtering accessories... (before: {len(result)} items)")
        result = result[~result["product_name"].apply(is_accessory)]
        print(f"✅ After filtering: {len(result)} items")
    
    # Basic filtering for product type
    if product_type:
        # Product type keywords for filtering
        type_patterns = {
            "phone": r"(?i)(smartphone|mobile|galaxy|iphone|redmi|note|pro|plus|max|ultra)",
            "laptop": r"(?i)(laptop|notebook|macbook|thinkpad|ideapad|vivobook)",
            "shoes": r"(?i)(shoe|sneaker|boot|nike|adidas|puma|running)",
            "watch": r"(?i)(watch|smartwatch|fitbit|garmin)",
        }
        
        if product_type in type_patterns:
            pattern = type_patterns[product_type]
            filtered = result[
                result["product_name"].str.contains(pattern, case=False, na=False, regex=True)
            ]
            # Only apply filter if we have results
            if not filtered.empty:
                result = filtered
    
    # Brand filtering (not too strict)
    brand = attrs.get("brand")
    if brand:
        # Try strict brand filter first
        strict_filter = result[
            result["product_name"].str.contains(brand, case=False, na=False)
        ]
        
        # If we have results, use them; otherwise keep all
        if not strict_filter.empty:
            result = strict_filter
    
    # Price range filtering (hard limits)
    price_range = attrs.get("price_range", {})
    if price_range.get("max") and "price" in result.columns:
        result = result[result["price"] <= price_range["max"]]
    
    if price_range.get("min") and "price" in result.columns:
        result = result[result["price"] >= price_range["min"]]
    
    # If no results after all filtering, try relaxing constraints
    if result.empty:
        print("⚠️ No results after filtering. Relaxing constraints...")
        result = df.copy()
        
        # Remove accessories
        if product_type in ["phone", "laptop", "tablet", "watch"]:
            result = result[~result["product_name"].apply(is_accessory)]
        
        # Apply only price filter
        if price_range.get("max") and "price" in result.columns:
            result = result[result["price"] <= price_range["max"]]
    
    # Calculate relevance scores for all remaining products
    if not result.empty:
        result["relevance_score"] = result.apply(
            lambda row: calculate_relevance_score(row, attrs), 
            axis=1
        )
        
        # Sort by relevance first, then by price (ascending)
        result = result.sort_values(
            by=["relevance_score", "price"], 
            ascending=[False, True]
        )
    
    return result.head(top_n)


def search_products(
    df: pd.DataFrame, 
    attrs: Dict[str, Any], 
    top_n: int = 5
) -> pd.DataFrame:
    """
    Backward compatible wrapper for legacy code.
    """
    return search_products_enhanced(df, attrs, top_n)


def filter_by_attributes(
    df: pd.DataFrame,
    brand: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[float] = None,
    color: Optional[str] = None,
) -> pd.DataFrame:
    """
    Utility function for explicit attribute filtering.
    
    Args:
        df: Product DataFrame
        brand: Brand name to filter
        min_price: Minimum price
        max_price: Maximum price
        min_rating: Minimum rating
        color: Color to filter
        
    Returns:
        Filtered DataFrame
    """
    result = df.copy()
    
    if brand and "product_name" in result.columns:
        result = result[
            result["product_name"].str.contains(brand, case=False, na=False)
        ]
    
    if min_price is not None and "price" in result.columns:
        result = result[result["price"] >= min_price]
    
    if max_price is not None and "price" in result.columns:
        result = result[result["price"] <= max_price]
    
    if min_rating is not None and "rating_numeric" in result.columns:
        result = result[result["rating_numeric"] >= min_rating]
    
    if color and "product_name" in result.columns:
        result = result[
            result["product_name"].str.contains(color, case=False, na=False)
        ]
    
    return result


def get_top_products_by_category(
    df: pd.DataFrame,
    category: str,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Get top products in a specific category.
    
    Args:
        df: Product DataFrame
        category: Product category
        top_n: Number of results
        
    Returns:
        Top products in category
    """
    if df.empty or "product_name" not in df.columns:
        return pd.DataFrame()
    
    category_patterns = {
        "phones": r"(?i)(phone|smartphone|mobile)",
        "laptops": r"(?i)(laptop|notebook)",
        "shoes": r"(?i)(shoe|sneaker)",
        "watches": r"(?i)(watch)",
    }
    
    pattern = category_patterns.get(category.lower(), category)
    
    filtered = df[
        df["product_name"].str.contains(pattern, case=False, na=False, regex=True)
    ]
    
    # Remove accessories
    filtered = filtered[~filtered["product_name"].apply(is_accessory)]
    
    # Sort by rating and return top N
    if "rating_numeric" in filtered.columns:
        filtered = filtered.sort_values("rating_numeric", ascending=False)
    
    return filtered.head(top_n)