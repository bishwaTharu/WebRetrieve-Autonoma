"""
Google Search tool for Gemini API with grounding capabilities.
"""

import logging
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from WebRetrieve_Autonoma.config import settings

logger = logging.getLogger(__name__)


class GoogleSearchTool:
    """Google Search tool using Gemini API with grounding."""

    def __init__(self):
        """Initialize Google Search tool."""
        try:
            logger.info("Initializing Google Search tool with Gemini API")
            self.client = genai.Client(api_key=settings.gemini_api_key)
            logger.info("Successfully initialized Google Search tool")
        except Exception as e:
            logger.error(f"Failed to initialize Google Search tool: {e}")
            raise

    def search_with_grounding(
        self, 
        query: str, 
        model: str = "gemini-2.0-flash-exp"
    ) -> Dict[str, Any]:
        """
        Perform a search with grounding using Gemini API.
        
        Args:
            query: The search query
            model: The Gemini model to use
            
        Returns:
            Dictionary containing the response and grounding metadata
        """
        try:
            logger.info(f"Performing grounded search for query: {query}")
            
            # Create grounding tool
            grounding_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            # Configure the generation with grounding
            config = types.GenerateContentConfig(
                tools=[grounding_tool]
            )
            
            # Generate content with grounding
            response = self.client.models.generate_content(
                model=model,
                contents=query,
                config=config,
            )
            
            # Extract grounding metadata if available
            grounding_metadata = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    grounding_metadata = candidate.grounding_metadata
            
            result = {
                "text": response.text,
                "grounding_metadata": grounding_metadata,
                "model": model,
                "query": query
            }
            
            logger.info(f"Successfully performed grounded search for: {query}")
            return result
            
        except Exception as e:
            logger.error(f"Error performing grounded search: {e}")
            # Fallback to regular generation without grounding
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=query,
                )
                
                result = {
                    "text": response.text,
                    "grounding_metadata": None,
                    "model": model,
                    "query": query,
                    "fallback": True
                }
                
                logger.warning(f"Used fallback generation for: {query}")
                return result
                
            except Exception as fallback_error:
                logger.error(f"Fallback generation also failed: {fallback_error}")
                raise

    def extract_sources(self, grounding_metadata) -> List[Dict[str, str]]:
        """
        Extract sources from grounding metadata.
        
        Args:
            grounding_metadata: The grounding metadata from the response
            
        Returns:
            List of sources with title and URI
        """
        if not grounding_metadata or not hasattr(grounding_metadata, 'grounding_chunks'):
            return []
        
        sources = []
        for chunk in grounding_metadata.grounding_chunks:
            if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                sources.append({
                    "title": getattr(chunk.web, 'title', 'Unknown'),
                    "uri": chunk.web.uri
                })
        
        return sources

    def get_search_queries(self, grounding_metadata) -> List[str]:
        """
        Extract search queries from grounding metadata.
        
        Args:
            grounding_metadata: The grounding metadata from the response
            
        Returns:
            List of search queries used
        """
        if not grounding_metadata or not hasattr(grounding_metadata, 'web_search_queries'):
            return []
        
        return list(grounding_metadata.web_search_queries)
