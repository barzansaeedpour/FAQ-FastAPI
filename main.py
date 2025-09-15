# main.py
import os
from fastapi import FastAPI, Query  # وارد کردن FastAPI و Query برای تعریف API endpoints
from intent_handler import load_intents, load_responses, find_intent, explain_query  # توابع مدیریت intents و پاسخ‌ها
from get_response import process_query  # تابع fallback برای پردازش سوالات خارج از intents
from dotenv import load_dotenv

#  API Key برای اتصال به Google Gemini (یا سایر مدل‌ها)

load_dotenv()
# Access the API key
API_KEY = os.getenv("API_KEY")

#  ایجاد اپلیکیشن FastAPI
app = FastAPI(title="UOK FAQ Bot")

#  لود اولیه intents و responses
# intents_by_file یک دیکشنری است که کل intents هر فایل را نگه می‌دارد
# responses یک دیکشنری از پاسخ‌های متناظر با intents است
intents_by_file = load_intents()
responses = load_responses()

# --- مسیر اصلی پاسخ‌دهی ---
@app.get("/Response/{query}")
async def get_response(query: str):
    """
    endpoint اصلی برای گرفتن پاسخ بر اساس query کاربر.
    ابتدا تلاش می‌کند intent مناسب را پیدا کند، اگر پیدا نشد fallback با process_query انجام می‌شود.
    """
    # 🔹 جستجوی intent مناسب
    answer = find_intent(query, intents_by_file, responses)
    
    # 🔹 اگر intent پیدا شد، پاسخ مربوطه را برگردان
    if answer:
        return {"type": "intent", "answer": answer}
    
    # 🔹 اگر intent پیدا نشد، از fallback استفاده کن
    return {"type": "fallback", "answer": process_query(query, API_KEY)}

# --- مسیر debug برای بررسی intents ---
@app.get("/debug/intents")
def debug_intents():
    """
    بازگشت خلاصه‌ای از intents لود شده.
    شامل: نام فایل، تعداد intents، و پیش‌نمایش سه مثال اول.
    """
    summary = {}
    for fname, items in intents_by_file.items():
        summary[fname] = []
        for it in items[:20]:  # محدود کردن به 20 intent اول هر فایل
            examples = it.get("examples")
            
            # 🔹 نرمال‌سازی پیش‌نمایش مثال‌ها
            if isinstance(examples, str):
                # جدا کردن خطوط و حذف خطوط خالی
                ex_list = [line.strip() for line in examples.split("\n") if line.strip()]
                ex_preview = ex_list[:3]  # فقط 3 مثال اول
            elif isinstance(examples, list):
                ex_preview = examples[:3]
            else:
                ex_preview = []

            # 🔹 اضافه کردن اطلاعات intent و پیش‌نمایش مثال‌ها به summary
            summary[fname].append({"intent": it.get("intent"), "examples_preview": ex_preview})
    
    # 🔹 بازگشت summary و کلیدهای responses برای بررسی
    return {"files": summary, "responses_keys": list(responses.keys())}

# --- مسیر debug برای بررسی نحوه تطبیق query ---
@app.get("/debug/match")
def debug_match(query: str = Query(..., description="query to explain"), cutoff: float = 0.5):
    """
    توضیح روند تطبیق query کاربر با intents.
    cutoff: آستانه تشخیص intent موفق (0 تا 1)
    """
    info = explain_query(query, intents_by_file, responses, cutoff=cutoff)
    return info


from intent_handler import prepare_embeddings
prepare_embeddings(intents_by_file)

@app.get("/debug/{query}")
async def debug_response(query: str):
    return explain_query(query, intents_by_file, responses)
