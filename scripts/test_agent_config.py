import asyncio
import litellm
from agent.config import AgentConfig

async def test_config():
    config = AgentConfig()
    
    # Test embedding model configuration
    print(f"Testing Embedding Model: {config.embedding_model}")
    
    # Attempt an embedding
    try:
        kwargs = config.litellm_embedding_kwargs
        kwargs["input"] = ["hello"]
        response = await litellm.aembedding(**kwargs)
        print("✅ Configured Embedding Key is VALID!")
    except Exception as e:
        print(f"❌ Configured Embedding Key Validation FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(test_config())
