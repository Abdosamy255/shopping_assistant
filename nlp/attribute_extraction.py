# nlp/attribute_extraction.py


import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from preprocessing import preprocess_text


COLORS = [
    
    "اسود", "ابيض", "احمر", "ازرق", "اخضر",
    "رمادي", "بني", "اصفر", "فضي", "ذهبي", "وردي",
    # English
    "black", "white", "red", "blue", "green",
    "gray", "grey", "brown", "yellow", "silver", "gold", "pink"
]


PRODUCTS_MAP = {
    "shoes": ["كوتش", "حذاء", "جزمة", "جزمه", "shoes", "shoe", "sneaker", "sneakers"],
    "laptop": ["لاب", "لابتوب", "حاسوب", "كمبيوتر", "notebook", "laptop", "pc"],
    "phone": ["موبايل", "جوال", "هاتف", "iphone", "ايفون", "samsung", "سامسونج", "phone"],
    "watch": ["ساعة", "watch", "smartwatch"],
    "tshirt": ["تيشيرت", "قميص", "tshirt", "t-shirt", "shirt"],
    # زوّد اللي تحتاجه
}
# map من اسم البراند بالعربي → الإنجليزي اللي موجود في الداتا
AR_BRAND_MAP = {
    "سامسونج": "samsung",
    "شاومي": "xiaomi",
    "ايفون": "iphone",
    "ابل": "apple",
    "هواوي": "huawei",
    "انفينكس": "infinix",
    "ريلمي": "realme",
    "نوكيا": "nokia",
    "تكنو": "tecno",
    "اوبو": "oppo",
}

def extract_attributes(tokens, lang: str = "ar"):
    product = None
    color = None
    size = None
    budget = None
    brand = None
    features = []

    for t in tokens:
        # اللون
        if t in COLORS and color is None:
            color = t

        # أرقام
        if t.isdigit():
            num = int(t)
            if num <= 70 and size is None:
                size = num
            elif num > 70 and budget is None:
                budget = num

        # براند من العربي → الإنجليزي
        if t in AR_BRAND_MAP and brand is None:
            brand = AR_BRAND_MAP[t]

        # براند إنجليزي (لو المستخدم كتب samsung مثلاً)
        if any("a" <= ch.lower() <= "z" for ch in t) and brand is None:
            brand = t

        # نوع المنتج
        if product is None:
            for cat, words in PRODUCTS_MAP.items():
                if t in words:
                    product = cat
                    break

    return {
        "lang": lang,
        "product": product,
        "color": color,
        "size": size,
        "budget": budget,
        "brand": brand,
        "features": features
    }
import re

BRANDS = {
    "samsung": ["samsung", "سامسونج", "galaxy"],
    "apple": ["apple", "iphone", "ايفون"],
    "xiaomi": ["xiaomi", "redmi", "poco", "شاومي"],
}

def extract_brand(text):
    text = text.lower()
    for brand, keywords in BRANDS.items():
        for kw in keywords:
            if kw in text:
                return brand
    return None


FEATURE_PATTERNS = {
    "ram": r"(\d+)\s*(gb)\s*(ram|رام)",
    "storage": r"(\d+)\s*(gb|جيجا)",
    "network": r"(5g|4g|lte)",
    "display": r"(amoled|oled|ips|lcd)",
}

def extract_features(text):
    features = {}
    text = text.lower()
    for key, pattern in FEATURE_PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            features[key] = match.group(0)
    return features


def extract_price_range(text):
    text = text.lower()
    result = {"min": None, "max": None}

    if m := re.search(r"(تحت|under)\s*(\d+)", text):
        result["max"] = int(m.group(2))

    if m := re.search(r"(فوق|above)\s*(\d+)", text):
        result["min"] = int(m.group(2))

    if m := re.search(r"(من|between)\s*(\d+).*(لـ|to|and)\s*(\d+)", text):
        result["min"] = int(m.group(2))
        result["max"] = int(m.group(4))

    return result
