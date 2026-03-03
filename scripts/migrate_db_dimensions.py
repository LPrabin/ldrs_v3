import asyncio
import logging
from agent.config import AgentConfig
import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def update_schema():
    config = AgentConfig()
    
    logger.info("Connecting to database...")
    try:
        conn = await asyncpg.connect(config.async_postgres_dsn)
        
        logger.info("Dropping existing vector index...")
        await conn.execute("DROP INDEX IF EXISTS idx_sections_embedding;")
        
        logger.info("Altering embedding column to vector(3072)...")
        await conn.execute("ALTER TABLE sections ALTER COLUMN embedding TYPE vector(3072);")
        
        logger.info("Database schema updated successfully!")
        await conn.close()
    except Exception as e:
        logger.error(f"Failed to update database schema: {e}")

if __name__ == "__main__":
    asyncio.run(update_schema())
