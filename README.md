# Smart Shopping Assistant — Streamlit Frontend

This repository contains a Streamlit front-end for the Smart Shopping Assistant (Arabic NLP → attribute extraction → product ranking).

Quick start

1. Create and activate a Python virtual environment (recommended). In PowerShell:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the Streamlit app from the project root:

```powershell
streamlit run app.py
```

Notes

- The app expects a product CSV at `data/products.csv` by default. You can upload a CSV via the sidebar to override it.
- If you see import errors for `nlp` or `search`, ensure the project root is in your `PYTHONPATH` or run from this repository root so Python can find the packages.
- The original logic (preprocessing, attribute extraction, search) lives under `nlp/` and `search/`. This `app.py` is a polished front-end that uses those modules.

Next improvements (suggested)

- Extract shared code into `nlp/utils.py` and `nlp/predict.py` for easier unit testing.
- Add unit tests for attribute extraction and search ranking.
- Add CI checks and a Dockerfile for reproducible deployment.
# Smart Shopping NLP

A Natural Language Processing project that allows users to search for products using Arabic or English text. The system extracts product attributes (product type, brand, budget, size, color) from user input, matches them against a cleaned product dataset, and returns the best results.

---
## 🚀 Features
- Arabic & English text preprocessing
- Attribute extraction using rule-based NLP
- Brand and product category mapping
- Intelligent product search and filtering
- Ranking results by price
- Streamlit web interface

---
## 🧠 System Architecture
```
User Query (AR/EN)
       │
       ▼
Preprocessing (Cleaning + Tokenization + Stopwords)
       │
       ▼
Attribute Extraction (Product, Brand, Budget, Size, Color)
       │
       ▼
Search Engine (Filtering + Ranking)
       │
       ▼
Streamlit UI (Results Display)
```

---
## 📂 Project Structure
```
nlp-project/
│
├── data/
│   ├── raw_products.csv
│   └── products.csv
│
├── nlp/
│   ├── preprocessing.py
│   ├── attribute_extraction.py
│   └── __init__.py
│
├── search/
│   ├── search_engine.py
│   └── __init__.py
│
├── app.py
├── prepare_dataset.py
├── requirements.txt
└── README.md
```

---
## 🧪 Example Queries
- "عايز موبايل سامسونج تحت 9000"
- "I want a Samsung phone under 300 dollars"
- "ارخص كوتش اسود مقاس 46"

---
## ▶️ Running the Project
```bash
pip install -r requirements.txt
streamlit run app.py
```

---
## 📦 Requirements
```txt
streamlit
pandas
numpy
python-dateutil
```

---
## 👨‍💻 Team
- Abdelrahman and team

