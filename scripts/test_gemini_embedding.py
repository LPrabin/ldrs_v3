import asyncio
import litellm
import os
from dotenv import load_dotenv

load_dotenv(override=True)

async def test_specific_model():
    model = "gemini/gemini-embedding-001"
    print(f"Testing Embedding Model: {model}")
    
    try:
        kwargs = {
            "model": model,
            "input": ["hello"],
            "api_key": os.getenv("GEMINI_API_KEY")
        }
        response = await litellm.aembedding(**kwargs)
        print("✅ Configured Embedding Key is VALID for", model)
    except Exception as e:
        print(f"❌ Validation FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_specific_model())