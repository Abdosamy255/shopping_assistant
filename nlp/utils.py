"""
nlp/utils.py

Unified utility functions for the NLP pipeline.
Consolidates price cleaning and other common operations.
"""

import re
import pandas as pd
from typing import Union, Optional


def clean_price_egp(raw_price: Union[str, int, float]) -> Union[float, pd.NA]:
    """
    Unified price cleaning for Egyptian Pounds.
    
    Handles various formats:
    - "65,500.00 EGP"
    - "65500"
    - "65.500 EGP" (European format)
    - "29,900 EGP"
    - "299.00"
    
    Args:
        raw_price: Raw price string or number
        
    Returns:
        float: Cleaned price in EGP, or pd.NA if invalid
        
    Examples:
        >>> clean_price_egp("29,900 EGP")
        29900.0
        >>> clean_price_egp("1,234.56")
        1234.56
        >>> clean_price_egp("invalid")
        <NA>
    """
    # Handle empty/null values
    if pd.isna(raw_price) or raw_price == "" or raw_price is None:
        return pd.NA
    
    # If already numeric, validate and return
    if not isinstance(raw_price, str):
        try:
            value = float(raw_price)
            return value if value > 0 else pd.NA
        except (ValueError, TypeError):
            return pd.NA
    
    # String processing
    text = str(raw_price).upper().strip()
    
    # Remove currency indicators
    text = re.sub(r'(EGP|ج\.م|جنيه|POUND|LE|£)', '', text, flags=re.IGNORECASE)
    
    # Remove whitespace
    text = text.strip()
    
    # Handle empty after cleaning
    if not text:
        return pd.NA
    
    # Remove all non-digit characters except decimal point
    # Keep first decimal point, remove others
    parts = text.split('.')
    if len(parts) > 2:
        # Multiple decimal points - keep first
        text = parts[0] + '.' + ''.join(parts[1:])
    
    # Remove commas and other separators
    text = re.sub(r'[,\s]', '', text)
    
    # Final validation
    if not text or text == '.':
        return pd.NA
    
    # Convert to float
    try:
        value = float(text)
        # Validate reasonable price range (0.01 to 1,000,000 EGP)
        if 0.01 <= value <= 1_000_000:
            return round(value, 2)
        else:
            return pd.NA
    except (ValueError, TypeError):
        return pd.NA


def normalize_text(text: str, language: str = "auto") -> str:
    """
    Normalize text for better processing.
    
    Args:
        text: Input text
        language: Language hint ("ar", "en", or "auto")
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Arabic-specific normalization
    if language == "ar" or language == "auto":
        # Normalize Arabic letters
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'[ًٌٍَُِّْ]', '', text)  # Remove diacritics
    
    return text.strip()


def extract_numbers(text: str) -> list:
    """
    Extract all numbers from text.
    
    Args:
        text: Input text
        
    Returns:
        list: List of numbers found
        
    Examples:
        >>> extract_numbers("phone with 128GB and 8GB RAM under 5000")
        [128, 8, 5000]
    """
    numbers = re.findall(r'\d+(?:\.\d+)?', text)
    return [float(n) if '.' in n else int(n) for n in numbers]


def detect_language(text: str) -> str:
    """
    Simple language detection based on character set.
    
    Args:
        text: Input text
        
    Returns:
        str: "ar" for Arabic, "en" for English, "mixed" for both
    """
    if not text:
        return "en"
    
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total_chars = arabic_chars + english_chars
    
    if total_chars == 0:
        return "en"
    
    arabic_ratio = arabic_chars / total_chars
    
    if arabic_ratio > 0.7:
        return "ar"
    elif arabic_ratio < 0.3:
        return "en"
    else:
        return "mixed"


def clean_text(text: Optional[str]) -> str:
    """
    General text cleaning utility.
    
    Args:
        text: Input text
        
    Returns:
        str: Cleaned text
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s\u0600-\u06FF.,!?()-]', '', text)
    
    return text.strip()


def format_currency(amount: float, currency: str = "EGP") -> str:
    """
    Format currency amount for display.
    
    Args:
        amount: Numeric amount
        currency: Currency code
        
    Returns:
        str: Formatted currency string
        
    Examples:
        >>> format_currency(12345.67)
        '12,345.67 EGP'
    """
    if pd.isna(amount):
        return "N/A"
    
    try:
        formatted = f"{amount:,.2f}"
        return f"{formatted} {currency}"
    except:
        return "N/A"


# Price range helper
def parse_price_indicators(text: str) -> dict:
    """
    Parse price-related keywords from text.
    
    Args:
        text: Input text
        
    Returns:
        dict: Contains 'min', 'max', 'target' price hints
        
    Examples:
        >>> parse_price_indicators("phone under 5000")
        {'max': 5000, 'min': None, 'target': None}
    """
    result = {
        "min": None,
        "max": None,
        "target": None
    }
    
    text_lower = text.lower()
    
    # Extract "under X" / "below X" / "less than X"
    under_patterns = [
        r'(?:under|below|less than|تحت|اقل من)\s+(\d+)',
        r'(\d+)\s+(?:او اقل|or less)'
    ]
    
    for pattern in under_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["max"] = int(match.group(1))
            break
    
    # Extract "above X" / "over X" / "more than X"
    over_patterns = [
        r'(?:above|over|more than|فوق|اكثر من)\s+(\d+)',
        r'(\d+)\s+(?:او اكثر|or more)'
    ]
    
    for pattern in over_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["min"] = int(match.group(1))
            break
    
    # Extract "between X and Y"
    range_patterns = [
        r'(?:between|من)\s+(\d+)\s+(?:and|to|الى|لـ)\s+(\d+)',
        r'(\d+)\s+(?:to|الى|لـ)\s+(\d+)'
    ]
    
    for pattern in range_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["min"] = int(match.group(1))
            result["max"] = int(match.group(2))
            break
    
    # Extract "around X" / "approximately X"
    around_patterns = [
        r'(?:around|approximately|about|حوالي|تقريبا)\s+(\d+)'
    ]
    
    for pattern in around_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result["target"] = int(match.group(1))
            # Set range as ±20% of target
            target = result["target"]
            result["min"] = int(target * 0.8)
            result["max"] = int(target * 1.2)
            break
    
    return result