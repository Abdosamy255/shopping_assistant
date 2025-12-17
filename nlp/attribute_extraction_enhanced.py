"""
nlp/attribute_extraction_enhanced.py

Enhanced attribute extraction with:
- Multi-word brand detection
- Feature extraction (5G, RAM, storage, etc.)
- Better price range handling
- Quality indicators (cheap, premium)
- Size detection for various product types
"""

import re
from typing import Dict, List, Optional, Any


# Expanded brand mappings with Arabic variants
BRAND_KEYWORDS = {
    "samsung": ["سامسونج", "samsung", "galaxy", "سامسونغ"],
    "apple": ["ابل", "apple", "iphone", "ايفون", "آيفون"],
    "xiaomi": ["شاومي", "xiaomi", "redmi", "poco", "شياومي"],
    "huawei": ["هواوي", "huawei", "هواواي"],
    "oppo": ["اوبو", "oppo", "أوبو"],
    "vivo": ["فيفو", "vivo", "ڤيفو"],
    "realme": ["ريلمي", "realme", "ريل مي"],
    "infinix": ["انفينكس", "infinix", "انفنكس"],
    "tecno": ["تكنو", "tecno", "تيكنو"],
    "nokia": ["نوكيا", "nokia", "نوكيه"],
    "oneplus": ["ون بلس", "oneplus", "وان بلس"],
    "sony": ["سوني", "sony", "سونى"],
    "lg": ["ال جي", "lg", "إل جي"],
    "motorola": ["موتورولا", "motorola", "موتو"],
    "lenovo": ["لينوفو", "lenovo", "لنوفو"],
    "asus": ["اسوس", "asus", "أسوس"],
    "hp": ["اتش بي", "hp", "إتش بي"],
    "dell": ["ديل", "dell"],
    "acer": ["ايسر", "acer", "أيسر"],
    "nike": ["نايك", "nike", "نايكي"],
    "adidas": ["اديداس", "adidas", "أديداس"],
    "puma": ["بوما", "puma"],
}

# Product type keywords
PRODUCT_KEYWORDS = {
    "phone": ["موبايل", "جوال", "هاتف", "تليفون", "phone", "smartphone", "mobile"],
    "laptop": ["لابتوب", "حاسوب محمول", "كمبيوتر محمول", "laptop", "notebook"],
    "tablet": ["تابلت", "لوح", "tablet", "pad"],
    "shoes": ["كوتش", "حذاء", "جزمة", "shoes", "sneakers", "boots"],
    "watch": ["ساعة", "watch", "smartwatch"],
    "headphones": ["سماعة", "سماعات", "headphones", "earphones", "earbuds"],
    "camera": ["كاميرا", "camera"],
    "tv": ["تلفزيون", "تليفزيون", "tv", "television"],
}

# Color keywords
COLOR_KEYWORDS = {
    "black": ["اسود", "أسود", "black"],
    "white": ["ابيض", "أبيض", "white"],
    "blue": ["ازرق", "أزرق", "blue", "navy"],
    "red": ["احمر", "أحمر", "red"],
    "green": ["اخضر", "أخضر", "green"],
    "yellow": ["اصفر", "أصفر", "yellow"],
    "gray": ["رمادي", "grey", "gray", "رصاصي"],
    "silver": ["فضي", "silver", "سيلفر"],
    "gold": ["ذهبي", "gold", "جولد"],
    "pink": ["وردي", "بينك", "pink", "rose"],
    "purple": ["بنفسجي", "purple", "موف"],
}

# Feature extraction patterns
FEATURE_PATTERNS = {
    "storage": [
        (r'(\d+)\s*(?:gb|جيجا|جيجابايت|giga)', "storage_gb"),
        (r'(\d+)\s*(?:tb|تيرا|تيرابايت|tera)', "storage_tb"),
    ],
    "ram": [
        (r'(\d+)\s*(?:gb|جيجا)\s*(?:ram|رام|ذاكرة)', "ram_gb"),
    ],
    "camera": [
        (r'(\d+)\s*(?:mp|ميجا|ميجابكسل|mega)', "camera_mp"),
    ],
    "display": [
        (r'(\d+\.?\d*)\s*(?:inch|انش|بوصة|")', "screen_inch"),
        (r'(amoled|oled|lcd|ips|retina)', "display_type"),
    ],
    "network": [
        (r'(5g|4g|lte)', "network_type"),
    ],
    "processor": [
        (r'(snapdragon|mediatek|helio|exynos|a\d+\s*bionic|intel|amd|ryzen|core\s*i\d)', "processor"),
    ],
}

# Price range patterns
PRICE_PATTERNS = [
    (r'(?:تحت|اقل من|under|below|less than)\s*(\d+)', "max"),
    (r'(?:فوق|اكثر من|above|over|more than)\s*(\d+)', "min"),
    (r'(?:من|between|from)\s*(\d+)\s*(?:الى|لـ|to|and|-)\s*(\d+)', "range"),
    (r'(?:حوالي|تقريبا|around|approximately|about)\s*(\d+)', "target"),
]

# Quality indicators
QUALITY_INDICATORS = {
    "cheap": ["رخيص", "ارخص", "cheap", "budget", "affordable", "اقتصادي"],
    "premium": ["فخم", "غالي", "premium", "flagship", "high-end", "راقي", "ممتاز"],
    "good": ["جيد", "كويس", "good", "quality", "جودة"],
}


def extract_brand(text: str, tokens: List[str]) -> Optional[str]:
    """Extract brand name from text."""
    text_lower = text.lower()
    
    # Check each brand and its keywords
    for brand, keywords in BRAND_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return brand
    
    # Check tokens directly
    for token in tokens:
        token_lower = token.lower()
        for brand, keywords in BRAND_KEYWORDS.items():
            if token_lower in [k.lower() for k in keywords]:
                return brand
    
    return None


def extract_product_type(text: str, tokens: List[str]) -> Optional[str]:
    """Extract product type from text."""
    text_lower = text.lower()
    
    for product, keywords in PRODUCT_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return product
    
    # Check tokens
    for token in tokens:
        token_lower = token.lower()
        for product, keywords in PRODUCT_KEYWORDS.items():
            if token_lower in [k.lower() for k in keywords]:
                return product
    
    return None


def extract_color(text: str, tokens: List[str]) -> Optional[str]:
    """Extract color from text."""
    text_lower = text.lower()
    
    for color, keywords in COLOR_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return color
    
    return None


def extract_size(text: str, tokens: List[str], product_type: Optional[str]) -> Optional[int]:
    """Extract size based on product type."""
    numbers = [int(t) for t in tokens if t.isdigit()]
    
    if not numbers:
        return None
    
    # Shoe size (typically 35-50)
    if product_type == "shoes":
        for num in numbers:
            if 35 <= num <= 50:
                return num
    
    # Screen size for electronics (typically 5-100 inches)
    if product_type in ["phone", "laptop", "tablet", "tv"]:
        for num in numbers:
            if 5 <= num <= 100:
                return num
    
    return None


def extract_price_range(text: str) -> Dict[str, Optional[int]]:
    """Extract min/max price from text."""
    result = {"min": None, "max": None, "target": None}
    
    text_lower = text.lower()
    
    for pattern, ptype in PRICE_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            if ptype == "max":
                result["max"] = int(match.group(1))
            elif ptype == "min":
                result["min"] = int(match.group(1))
            elif ptype == "range":
                result["min"] = int(match.group(1))
                result["max"] = int(match.group(2))
            elif ptype == "target":
                target = int(match.group(1))
                result["target"] = target
                # Set range as ±20% of target
                result["min"] = int(target * 0.8)
                result["max"] = int(target * 1.2)
    
    return result


def extract_features(text: str) -> Dict[str, Any]:
    """Extract technical features from text."""
    features = {}
    text_lower = text.lower()
    
    for category, patterns in FEATURE_PATTERNS.items():
        for pattern, key in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                value = match.group(1) if match.lastindex >= 1 else match.group(0)
                features[key] = value
    
    return features


def extract_quality_level(text: str) -> Optional[str]:
    """Extract quality/price level indicators."""
    text_lower = text.lower()
    
    for level, keywords in QUALITY_INDICATORS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return level
    
    return None


def extract_enhanced_attributes(
    tokens: List[str], 
    original_text: str, 
    lang: str
) -> Dict[str, Any]:
    """
    Enhanced attribute extraction with comprehensive feature detection.
    
    Args:
        tokens: Preprocessed tokens from the query
        original_text: Original unprocessed query text
        lang: Detected language ("ar" or "en")
        
    Returns:
        dict: Comprehensive attribute dictionary
    """
    attrs = {
        "lang": lang,
        "product": None,
        "brand": None,
        "color": None,
        "size": None,
        "price_range": {"min": None, "max": None, "target": None},
        "features": {},
        "quality_level": None,
        "tokens": tokens,
    }
    
    # Extract brand
    attrs["brand"] = extract_brand(original_text, tokens)
    
    # Extract product type
    attrs["product"] = extract_product_type(original_text, tokens)
    
    # Extract color
    attrs["color"] = extract_color(original_text, tokens)
    
    # Extract size (depends on product type)
    attrs["size"] = extract_size(original_text, tokens, attrs["product"])
    
    # Extract price range
    attrs["price_range"] = extract_price_range(original_text)
    
    # Extract features
    attrs["features"] = extract_features(original_text)
    
    # Extract quality level
    attrs["quality_level"] = extract_quality_level(original_text)
    
    # Legacy budget field (for backward compatibility)
    if attrs["price_range"]["max"]:
        attrs["budget"] = attrs["price_range"]["max"]
    else:
        attrs["budget"] = None
    
    return attrs


# Backward compatibility function
def extract_attributes(tokens: List[str], lang: str, original_text: str = "") -> Dict[str, Any]:
    """
    Backward compatible wrapper for enhanced extraction.
    
    Args:
        tokens: Preprocessed tokens
        lang: Language code
        original_text: Original query (optional, will reconstruct if not provided)
        
    Returns:
        dict: Extracted attributes
    """
    if not original_text:
        original_text = " ".join(tokens)
    
    return extract_enhanced_attributes(tokens, original_text, lang)