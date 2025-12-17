# nlp/preprocessing.py
# -----------------------------------
# Text Preprocessing module (Arabic + English)
# Steps:
#   1) Language detection
#   2) Normalization (Arabic / English)
#   3) Cleaning (URLs, emails, emojis, extra spaces)
#   4) Tokenization
#   5) Stopwords removal
#   6) Phrase handling (e.g. "اقل من" → "اقل_من")
# -----------------------------------

import re
import string

# -------------------------
# 1) Stopwords (Arabic + English)
# -------------------------

ARABIC_STOPWORDS = set("""
انا نحن انت انتي انتم انهما انهم انتن
هو هي هم هن
هذا هذه هؤلاء هذي ذلك تلك هناك هنا
في على عن من الى إلي حتي حتى او ام أم
ثم بل لكن لان لأن لو لم لما لن لا
كل بعض اي أي اى ايه أيه
لقد قد كان ليس لست كنا كانوا كنت كانت
يكون تكون تكونوا يكونوا تكونين
مع لدى عند اما أما اذا إذا إن ان أن
الذي التي الذين اللاتي اللواتي اللذان اللتان
كما حيث بينما حين عندما لما منذ
بعد قبل خلال ضد دون غير سوى فقط
لو سمحت لو سمحتم لو سمحتي
عايز عايزه عايزها عايزهم عايزين عاوزه
عاوز عاوزه عاوزين
محتاج محتاجه محتاجين
وقد و قد و لم و لن و لا و ان و إن
""".split())

ENGLISH_STOPWORDS = set("""
i me my you your yours he him his she her hers
we our ours they them their theirs
this that these those here there
is am are was were be been being
a an the
and or but if while for to from of in on at by with
about above below under over into out up down
as than then so such just very really
can could may might should would will do did done
have has had having
want need like
under below less more around about between
""".split())

# -------------------------
# Intent keywords (Query Understanding)
# -------------------------

INTENT_KEYWORDS = {
    "latest": ["latest", "new", "جديد", "احدث"],
    "cheap": ["cheap", "رخيص", "ارخص"],
    "quality": ["quality", "جوده", "كويس", "جيد"],
    "premium": ["premium", "غالي", "فلاجشيب"],
    "around_price": ["around", "approximately", "حوالي"],
}

def extract_intents(text: str):
    """
    استخراج نية المستخدم (cheap, latest, quality, etc.)
    """
    text = text.lower()
    intents = []

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                intents.append(intent)
                break

    return intents

# -------------------------
# 2) Language detection
# -------------------------

def detect_lang(text: str) -> str:
    """بسيطة: لو فيه حروف عربية → ar، غير كده → en"""
    for ch in text:
        if '\u0600' <= ch <= '\u06FF':
            return "ar"
    return "en"

# -------------------------
# 3) Normalization helpers
# -------------------------

ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
ARABIC_TATWEEL = "\u0640"

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ENGLISH_DIGITS = "0123456789"
ARABIC_DIGIT_MAP = str.maketrans(ARABIC_DIGITS, ENGLISH_DIGITS)


def normalize_arabic(text: str) -> str:
    """توحيد الألف/الياء، إزالة التشكيل، إزالة التطويل، وتحويل الأرقام العربية لإنجليزي."""
    text = text.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و")
    text = text.replace("ة", "ه")
    text = ARABIC_DIACRITICS.sub("", text)
    text = text.replace(ARABIC_TATWEEL, "")
    text = text.translate(ARABIC_DIGIT_MAP)
    return text


def normalize_english(text: str) -> str:
    """Lowercase + إزالة تكرار المسافات."""
    text = text.lower()
    return text


def clean_common(text: str) -> str:
    """إزالة URLs, emails, emojis, وأي symbol مالوش لازمة."""
    # URLs
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    # Emails
    text = re.sub(r"\S+@\S+", " ", text)

    # Emojis / رموز غير حروف أو أرقام أو مسافات
    text = re.sub(r"[^\w\s]", " ", text)

    # مسافات زيادة
    text = re.sub(r"\s+", " ", text).strip()

    return text

# -------------------------
# 4) Tokenization
# -------------------------

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

def tokenize(text: str):
    """Tokenization بسيطة على مستوى الكلمات (حروف + أرقام)."""
    return TOKEN_PATTERN.findall(text)


# -------------------------
# 5) Stopwords removal
# -------------------------

def remove_stopwords(tokens, lang: str):
    if lang == "ar":
        return [t for t in tokens if t not in ARABIC_STOPWORDS]
    else:
        return [t for t in tokens if t not in ENGLISH_STOPWORDS]


# -------------------------
# 6) Phrase handling (bigram merge)
# -------------------------

AR_PHRASE_MAP = {
    ("اقل", "من"): "اقل_من",
    ("اقل", "عن"): "اقل_من",
    ("تحت", "حد"): "تحت_حد",
}

EN_PHRASE_MAP = {
    ("less", "than"): "less_than",
    ("under", "than"): "under_than",
    ("up", "to"): "up_to",
}

def merge_phrases(tokens, lang: str):
    """دمج كلمتين شائعتين إلى توكن واحد (زي اقل من → اقل_من)."""
    if lang == "ar":
        phrase_map = AR_PHRASE_MAP
    else:
        phrase_map = EN_PHRASE_MAP

    i = 0
    merged = []
    n = len(tokens)

    while i < n:
        if i < n - 1:
            pair = (tokens[i], tokens[i+1])
            if pair in phrase_map:
                merged.append(phrase_map[pair])
                i += 2
                continue
        merged.append(tokens[i])
        i += 1

    return merged

# -------------------------
# 7) Main preprocessing pipeline
# -------------------------

def preprocess_text(text: str):
    """
    Pipeline رئيسي بيطبق كل الخطوات:
      1) detect_lang
      2) normalization (Arabic / English)
      3) cleaning (URLs/emails/punctuation)
      4) lowercase
      5) tokenization
      6) stopwords removal
      7) phrase merging
    يرجّع: tokens, lang
    """
    if not isinstance(text, str):
        text = str(text)

    lang = detect_lang(text)

    text = text.strip()

    if lang == "ar":
        text = normalize_arabic(text)
        # نخلي الإنجليزي lowercase برضه لو موجود جوّه النص العربي
        text = text.lower()
    else:
        text = normalize_english(text)

    text = clean_common(text)

    tokens = tokenize(text)
    tokens = [t for t in tokens if t]  # شيل التوكنز الفاضية

    # Lowercase final (عشان الأمان)
    tokens = [t.lower() for t in tokens]

    tokens = remove_stopwords(tokens, lang)
    tokens = merge_phrases(tokens, lang)
    intents = extract_intents(text)


    return tokens, lang, intents



# -------------------------
# 8) Quick self-test
# -------------------------

if __name__ == "__main__":
    examples = [
        "عايز موبايل سامسونج تحت 9000 جنيه لو سمحت",
        "I want a Samsung phone under 300 dollars",
        "ارخص كوتش اسود مقاس 46 أقل من 1500",
    ]
    for s in examples:
        toks, lg, intents = preprocess_text(s)
        print(f"INTENTS: {intents}")
        print(f"TEXT: {s}")
        print(f"LANG: {lg} | TOKENS: {toks}")
        print("-" * 60)
