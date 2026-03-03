
import os
import asyncio
import litellm
from dotenv import load_dotenv

load_dotenv()

async def test_gemini_chat():
    model = "gemini/gemini-2.0-flash"
    api_key = os.getenv("GEMINI_API_KEY")
    
    print(f"Testing Gemini Chat Model: {model}")
    
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "api_key": api_key
        }
        response = await litellm.acompletion(**kwargs)
        print("✅ GEMINI_API_KEY is VALID for Chat")
    except Exception as e:
        print(f"❌ GEMINI_API_KEY VALIDATION FAILED for Chat: {e}")

if __name__ == "__main__":
    asyncio.run(test_gemini_chat())
