
import os
import asyncio
import litellm
from dotenv import load_dotenv

load_dotenv()

async def test_chat():
    model = os.getenv("DEFAULT_MODEL", "qwen3-vl")
    # Prefix with openai/ if not already prefixed and we have a custom api_base
    if "/" not in model:
        model = f"openai/{model}"
        
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("API_BASE") or os.getenv("BASE_URL")
    
    print(f"Testing Chat Model: {model}")
    print(f"Base URL: {api_base}")
    
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5
        }
        if api_key:
            kwargs["api_key"] = api_key
        if api_base:
            kwargs["api_base"] = api_base
            
        response = await litellm.acompletion(**kwargs)
        print("✅ Chat API Key is VALID")
    except Exception as e:
        print(f"❌ Chat API Key VALIDATION FAILED: {e}")

async def test_embedding():
    model = os.getenv("EMBEDDING_MODEL", "gemini/embedding-001")
    # For Gemini, the standard embedding model name in LiteLLM is 'gemini/text-embedding-004'
    # but let's try the one from .env first, and then fallback if it fails.
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    
    print(f"\nTesting Embedding Model: {model}")
    
    try:
        kwargs = {
            "model": model,
            "input": ["hi"]
        }
        if api_key:
            kwargs["api_key"] = api_key
            
        response = await litellm.aembedding(**kwargs)
        print("✅ Embedding API Key is VALID")
    except Exception as e:
        print(f"❌ Embedding API Key VALIDATION FAILED for {model}: {e}")
        
        # Fallback to a common gemini embedding model
        if "gemini" in model:
            fallback = "gemini/text-embedding-004"
            print(f"Trying fallback: {fallback}")
            try:
                kwargs["model"] = fallback
                response = await litellm.aembedding(**kwargs)
                print(f"✅ Embedding API Key is VALID (with fallback {fallback})")
            except Exception as e2:
                print(f"❌ Embedding API Key VALIDATION FAILED for fallback {fallback}: {e2}")

if __name__ == "__main__":
    asyncio.run(test_chat())
    asyncio.run(test_embedding())
