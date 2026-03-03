import asyncio
import litellm
import os
from agent.config import AgentConfig

async def check_dim():
    config = AgentConfig()
    kwargs = config.litellm_embedding_kwargs
    print(f"Model configured: {kwargs.get('model')}")
    print(f"API Base configured: {kwargs.get('api_base')}")
    kwargs["input"] = ["hello world"]
    
    try:
        response = await litellm.aembedding(**kwargs)
        emb = response.data[0]["embedding"]
        print(f"Returned dimension: {len(emb)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_dim())