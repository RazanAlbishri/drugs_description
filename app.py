# 💊 Medicine App (FastAPI Version) — Smart Display (EN/AR)
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import re

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}

# -------------------------------------------
# ⚙️ App Configuration
# -------------------------------------------
app = FastAPI(
    title="💊 Drugs Data API",
    description="Bilingual (EN/AR) Medicine Information API with smart name matching",
    version="2.0"
)

# Allow CORS (to allow access from frontend or Streamlit if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------
# 📂 Load and Clean Data
# -------------------------------------------
def load_data():
    df = pd.read_csv("Drugs_discription.csv", dtype=str, low_memory=False)
    df.columns = df.columns.str.strip()
    df = df.dropna(how="all").drop_duplicates().fillna("Unknown")
    return df

df = load_data()

# -------------------------------------------
# 🌐 Language dictionaries
# -------------------------------------------
EN = {
    "title": "💊 Drugs Dataset Dashboard & Chatbot",
    "chat_header": "💬 Smart Drugs Chatbot",
    "no_match": "⚠️ No matching drug found.",
    "use": "Use",
    "side": "⚠️ Side Effects",
    "sub": "💊 Substitutes",
    "tclass": "🏥 Therapeutic Class",
    "cclass": "🧪 Chemical Class",
    "habit": "Habit Forming",
    "trade": "💊 Trade Name",
    "sci": "🧪 Scientific Name",
}

AR = {
    "title": "💊 لوحة بيانات الأدوية وروبوت المحادثة",
    "chat_header": "💬 روبوت الأدوية الذكي",
    "no_match": "⚠️ لم يتم العثور على دواء مطابق.",
    "use": "الاستخدام",
    "side": "⚠️ الأعراض الجانبية",
    "sub": "💊 البدائل",
    "tclass": "🏥 الفئة العلاجية",
    "cclass": "🧪 الفئة الكيميائية",
    "habit": "قابلية الإدمان",
    "trade": "💊 الاسم التجاري",
    "sci": "🧪 الاسم العلمي",
}

def get_text(lang, key):
    return (AR if lang == "arabic" else EN)[key]

# -------------------------------------------
# 🔍 Search Logic
# -------------------------------------------
def search_drug(query: str):
    q = query.lower().strip()
    search_columns = [col for col in ["TradeName", "ScientificName"] if col in df.columns]
    mask = pd.Series(False, index=df.index)
    pattern = rf"\b{re.escape(q)}\b"
    for col in search_columns:
        mask |= df[col].astype(str).str.lower().str.contains(pattern, na=False, regex=True)
    return df[mask]

# -------------------------------------------
# 🚀 API Endpoint: /search
# -------------------------------------------
@app.get("/search")
def search_drug_api(
    name: str = Query(..., description="Drug name (trade or scientific)"),
    language: str = Query("english", description="Language: english or arabic"),
    use: bool = Query(True),
    side: bool = Query(True),
    sub: bool = Query(False),
    tclass: bool = Query(False),
    cclass: bool = Query(False),
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
    for _, row in results.head(3).iterrows():
        trade = row.get("TradeName", "Unknown")
        sci = row.get("ScientificName", "Unknown")
        q = name.lower()

        # Determine which name is main
        if q in str(sci).lower():
            main = {"main": f"🧪 {sci}", "secondary": f"{get_text(language, 'trade')}: {trade}"}
        else:
            main = {"main": f"💊 {trade}", "secondary": f"{get_text(language, 'sci')}: {sci}"}

        item = {**main}

        if use and "use" in row:
            item[get_text(language, "use")] = row["use"]
        if side and "sideEffect" in row:
            item[get_text(language, "side")] = row["sideEffect"]
        if sub and "substitute" in row:
            item[get_text(language, "sub")] = row["substitute"]
        if tclass and "Therapeutic Class" in row:
            item[get_text(language, "tclass")] = row["Therapeutic Class"]
        if cclass and "Chemical Class" in row:
            item[get_text(language, "cclass")] = row["Chemical Class"]
        if habit and "Habit Forming" in row:
            item[get_text(language, "habit")] = row["Habit Forming"]

        data.append(item)

    return {
        "query": name,
        "count": len(data),
        "language": language,
        "results": data
    }

# -------------------------------------------
# 🏠 Root Endpoint
# -------------------------------------------
@app.get("/")
def root():
    return {
        "message": "💊 Welcome to the Drugs Data API (EN/AR)",
        "usage": "Use /search?name=augmentin&language=english to query medicines"
    }
