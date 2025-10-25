from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import re

# Configuration
app = FastAPI(
    title="Drugs Data API",
    description="Bilingual (EN/AR) Medicine Information API with smart name matching",
    version="3.0"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and Clean Data
def load_data():
    df = pd.read_csv("Drugs_discription.csv", dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").drop_duplicates().fillna("Unknown")
    return df

df = load_data()

def format_to_bullets(text):
    if not isinstance(text, str):
        return text
    parts = re.split(r'[;,]', text)
    parts = [p.strip() for p in parts if p.strip()]
    return "\n".join(f"• {p}" for p in parts)

# Language dictionaries
EN = {
    "no_match": "No matching drug found.",
    "use": "Use",
    "side": "Side Effects",
    "sub": "Substitutes",
    "tclass": "Therapeutic Class",
    "cclass": "Chemical Class",
    "habit": "Habit Forming",
    "trade": "Trade Name",
    "sci": "Scientific Name",
}

AR = {
    "no_match": "لم يتم العثور على دواء مطابق.",
    "use": "الاستخدام",
    "side": "الأعراض الجانبية",
    "sub": "البدائل",
    "tclass": "الفئة العلاجية",
    "cclass": "الفئة الكيميائية",
    "habit": "قابلية الإدمان",
    "trade": "الاسم التجاري",
    "sci": "الاسم العلمي",
}

def get_text(lang, key):
    return (AR if lang == "arabic" else EN)[key]

# Smart Matching Search
def search_drug(query: str):
    q = query.lower().strip()
    search_columns = [col for col in ["TradeName", "ScientificName"] if col in df.columns]
    mask = pd.Series(False, index=df.index)
    pattern = rf"\b{re.escape(q)}\b"
    for col in search_columns:
        mask |= df[col].astype(str).str.lower().str.contains(pattern, na=False, regex=True)
    return df[mask]

# API Endpoint
@app.get("/search")
def search_drug_api(
    name: str = Query(..., description="Drug name (trade or scientific)"),
    language: str = Query("english", description="Language: english or arabic"),
    use: bool = Query(True),
    side: bool = Query(True),
    sub: bool = Query(True),
    tclass: bool = Query(True),
    cclass: bool = Query(True),
    habit: bool = Query(False)
):
    results = search_drug(name)

    if results.empty:
        return {
            "query": name,
            "count": 0,
            "message": get_text(language, "no_match"),
            "results": []
        }

    data = []
    for _, row in results.head(5).iterrows():
        trade = row.get("TradeName", "Unknown")
        sci = row.get("ScientificName", "Unknown")
        q = name.lower()

        if q in str(sci).lower():
            main = {"main": f"{sci}", "secondary": f"{get_text(language, 'trade')}: {trade}"}
        else:
            main = {"main": f"{trade}", "secondary": f"{get_text(language, 'sci')}: {sci}"}

        item = {**main}

        if use:
            item[get_text(language, "use")] = format_to_bullets(row.get("use", "Unknown"))

        if side:
            item[get_text(language, "side")] = format_to_bullets(row.get("sideEffect", "Unknown"))

        if sub:
            item[get_text(language, "sub")] = format_to_bullets(row.get("substitute", "Unknown"))

        if tclass:
            item[get_text(language, "tclass")] = format_to_bullets(row.get("Therapeutic Class", "Unknown"))

        if cclass:
            item[get_text(language, "cclass")] = format_to_bullets(row.get("Chemical Class", "Unknown"))

        if habit and "Habit Forming" in row:
            item[get_text(language, "habit")] = row["Habit Forming"]

        data.append(item)

    return {
        "query": name,
        "count": len(data),
        "language": language,
        "results": data
    }
