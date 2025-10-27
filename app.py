from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import re
from googletrans import Translator

# Configuration
app = FastAPI(
    title="Drugs Data API",
    description="Bilingual (EN/AR) Medicine Information API",
    version="3.2"
)

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = Translator()

# Load and Clean Data
def load_data():
    df = pd.read_csv("Drugs_discription.csv", dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").drop_duplicates().fillna("Unknown")
    return df

df = load_data()

def format_list(text, lang):
    if not isinstance(text, str):
        return text

    parts = [p.strip() for p in re.split(r'[;,]', text) if p.strip()]
    if lang == "arabic":
        translated = []
        for item in parts:
            try:
                tr = translator.translate(item, dest="ar").text
                translated.append(f"- {tr}")
            except:
                translated.append(f"- {item}")
        return "\n".join(translated)
    
    return "\n".join(f"- {p}" for p in parts)

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

# Smart search
def search_drug(query: str):
    q = query.lower().strip()
    search_columns = ["TradeName", "ScientificName"]
    mask = pd.Series(False, index=df.index)

    for col in search_columns:
        if col in df.columns:
            mask |= df[col].astype(str).str.lower().str.contains(q)

    return df[mask]

# API Endpoint
@app.get("/search")
def search_drug_api(
    name: str = Query(...),
    language: str = Query("english"),
    use: bool = Query(True),
    side: bool = Query(True),
    sub: bool = Query(True),
    tclass: bool = Query(True),
    cclass: bool = Query(True),
    habit: bool = Query(False)
):
    language = language.lower()
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

        # Main title shown depends on search
        if name.lower() in str(sci).lower():
            main = {"main": sci, "secondary": f"{get_text(language, 'trade')}: {trade}"}
        else:
            main = {"main": trade, "secondary": f"{get_text(language, 'sci')}: {sci}"}
        item = {**main}
        if use:
            item[get_text(language, "use")] = format_list(row.get("use", "Unknown"), language)
        if side:
            item[get_text(language, "side")] = format_list(row.get("sideEffect", "Unknown"), language)
        if sub:
            item[get_text(language, "sub")] = format_list(row.get("substitute", "Unknown"), language)
        if tclass:
            item[get_text(language, "tclass")] = row.get("Therapeutic Class", "Unknown")
        if cclass:
            item[get_text(language, "cclass")] = row.get("Chemical Class", "Unknown")
        if habit:
            item[get_text(language, "habit")] = row.get("Habit Forming", "Unknown")
        data.append(item)
    return {
        "query": name,
        "count": len(data),
        "language": language,
        "results": data
    }
