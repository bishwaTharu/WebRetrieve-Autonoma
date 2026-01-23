import logging
from typing import List
from google import genai
from WebRetrieve_Autonoma.config import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddings:
    """Gemini embedding model implementation."""

    def __init__(self):
        """Initialize Gemini embeddings client."""
        try:
            logger.info(f"Initializing Gemini embeddings with model: {settings.embedding_model_name}")
            self.client = genai.Client(api_key=settings.gemini_api_key)
            self.model_name = settings.embedding_model_name
            logger.info("Successfully initialized Gemini embeddings client")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using Gemini.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            try:
                result = self.client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                embeddings.append(result.embeddings[0].values)
            except Exception as e:
                logger.error(f"Error embedding text: {e}")
                # Fallback to zero vector
                embeddings.append([0.0] * 768)  # Default embedding size
        
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query using Gemini.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        try:
            result = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            # Fallback to zero vector
            return [0.0] * 768  # Default embedding size
