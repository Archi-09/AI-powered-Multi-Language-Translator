from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Language mapping for mBART50
LANGUAGE_CODES = {
    "English": "en_XX",
    "Hindi": "hi_IN",
    "French": "fr_XX",
    "Spanish": "es_XX",
    "German": "de_DE",
    "Chinese": "zh_CN"
}

# Load model & tokenizer once at startup
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

# Request structure
class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str

@app.post("/translate")
async def translate(req: TranslationRequest):
    try:
        print("\n🔵 Translation requested")
        print("Text:", req.text)
        print("Source:", req.source_lang, "| Target:", req.target_lang)

        source_code = LANGUAGE_CODES.get(req.source_lang)
        target_code = LANGUAGE_CODES.get(req.target_lang)

        if not source_code or not target_code:
            print("❌ Unsupported language")
            return {"error": "Unsupported language pair"}

        # Set source language
        tokenizer.src_lang = source_code
        inputs = tokenizer(req.text, return_tensors="pt")

        # Translate with forced target language
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code)
        )

        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        print("✅ Translation successful:", translated_text)

        return {"translation": translated_text}

    except Exception as e:
        print("❌ Translation failed:", str(e))
        return {"error": f"Translation failed: {str(e)}"}

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_home():
    return FileResponse(os.path.join("static", "index.html"))
