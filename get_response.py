# get_response.py
import pathlib
from google import genai
from google.genai import types
import fitz
from openai import OpenAI

# ---------- اینجا فایل دسکریپشن import می‌شود ----------
from file_description import file_data  # فرض بر این است که این فایل شامل لیست فایل‌هاست

def extract_text_from_pdf(filepath):
    """Extract text from a PDF using PyMuPDF (fitz)."""
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_query(user_query: str, gemini_api_key='', openai_api_key=''):
    from dotenv import load_dotenv
    import os
    load_dotenv()
    # Access the API key
    gemini_api_key = os.getenv("API_KEY")
    openai_api_key = os.getenv("openai_api_key")
    
    files_dir = pathlib.Path("./files")
    
    if not files_dir.exists():
        return "❌ فولدر فایل‌ها موجود نیست!"

    # مرتب‌سازی فایل‌ها بر اساس priority از بالا به پایین
    sorted_files = sorted(file_data, key=lambda x: -x["priority"])

    gemini_success = False
    gemini_response = ""
    gemini_error = None

    if gemini_api_key:
        try:
            client = genai.Client(api_key=gemini_api_key)
            for fdata in sorted_files:
                filepath = files_dir / fdata["filename"]
                if not filepath.exists():
                    continue
                prompt = f"""
شما یک دستیار هوشمند هستید.
وظیفه شما پاسخ به سوال کاربر است بر اساس محتوای PDF:

عنوان فایل: {fdata['filename']}
توضیح فایل: {fdata['description']}

سوال کاربر: {user_query}

پاسخ را فقط بر اساس متن فایل بده، اگر جواب نبود بگو:
"این فایل پاسخ سوال شما را ندارد."
"""
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[types.Part.from_bytes(data=filepath.read_bytes(), mime_type='application/pdf'), prompt]
                )
                if response.text.strip() and "این فایل پاسخ سوال شما را ندارد." not in response.text:
                    gemini_success = True
                    gemini_response = response.text
                    break
        except Exception as e:
            gemini_error = str(e)

    if gemini_success:
        return gemini_response

    if not openai_api_key:
        return f"❌ خطای Gemini: {gemini_error or 'نامشخص'}، برای fallback به OpenAI API Key نیاز است."

    combined_text = ""
    for fdata in sorted_files:
        filepath = files_dir / fdata["filename"]
        if not filepath.exists():
            continue
        combined_text += f"\n--- {fdata['filename']} ---\n"
        combined_text += f"توضیح: {fdata['description']}\n"
        combined_text += extract_text_from_pdf(filepath)

    combined_prompt = f"""
شما یک دستیار هوشمند هستید.
سوال کاربر: {user_query}

متن و توضیحات PDFها:
{combined_text}

پاسخ را فقط بر اساس متن PDFها بده، اگر جواب نیست بگو:
"این فایل پاسخ سوال شما را ندارد."
"""

    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": combined_prompt}],
        temperature=0.7
    )

    return response.choices[0].message.content
