import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("GEMINI_API_KEY not found.")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
response = requests.get(url)

if response.status_code == 200:
    models = response.json().get("models", [])
    print("Available Embedding Models:")
    for m in models:
        name = m.get("name")
        if "embed" in name.lower() or "text-embedding" in name.lower():
            print(f" - {name}")
else:
    print(f"Failed to fetch models: {response.status_code} {response.text}")
