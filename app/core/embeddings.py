"""Embedding generation module using OpenAI embeddings."""

from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """Get cached OpenAI embeddings instance.

    Returns:
        Configured OpenAIEmbeddings instance
    """
    settings = get_settings()
    logger.info(f"Initializing embeddings model: {settings.embedding_model}")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )

    logger.info("Embeddings model initialized successfully")
    return embeddings





    