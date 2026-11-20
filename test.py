from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

for name in MODELS:
    try:
        response = client.models.generate_content(
            model=name,
            contents="Say hello in one short sentence.",
        )
        print("WORKED with:", name)
        print(response.text)
        break
    except Exception as e:
        print("FAILED:", name, "->", str(e)[:80])
