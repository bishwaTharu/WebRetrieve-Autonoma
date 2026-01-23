import logging
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from WebRetrieve_Autonoma.config import settings
from WebRetrieve_Autonoma.utils.gemini_embeddings import GeminiEmbeddings
from WebRetrieve_Autonoma.utils.google_search_tool import GoogleSearchTool
import urllib.parse

logger = logging.getLogger(__name__)


class AgentTools:
    """Tools for web search and RAG retrieval using Google Search."""

    def __init__(self):
        """Initialize tools with configuration from settings."""
        logger.info("Initializing AgentTools with Google Search configuration")

        try:
            logger.info(
                f"Initializing Gemini embeddings: {settings.embedding_model_name}"
            )
            self.embeddings = GeminiEmbeddings()

            self.vector_store = SKLearnVectorStore(embedding=self.embeddings)
            self.documents = []
            logger.info(
                f"Initialized vector store with embedding model: {settings.embedding_model_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

        # Initialize Google Search tool
        try:
            self.google_search = GoogleSearchTool()
            logger.info("Successfully initialized Google Search tool")
        except Exception as e:
            logger.error(f"Failed to initialize Google Search tool: {e}")
            raise

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        logger.info(
            f"Initialized text splitter with chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap}"
        )

    @tool
    def web_search(self, query: str) -> str:
        """
        Search the web using Google Search with grounding.
        
        Args:
            query: The search query
            
        Returns:
            Search results with citations and sources
        """
        try:
            logger.info(f"Performing web search for: {query}")
            
            # Use Google Search with grounding
            result = self.google_search.search_with_grounding(query)
            
            # Format the response
            response_text = result["text"]
            
            # Add sources if available
            if result["grounding_metadata"]:
                sources = self.google_search.extract_sources(result["grounding_metadata"])
                if sources:
                    response_text += "\n\nSources:"
                    for i, source in enumerate(sources[:5], 1):  # Limit to top 5 sources
                        response_text += f"\n{i}. {source['title']} - {source['uri']}"
            
            logger.info(f"Successfully completed web search for: {query}")
            return response_text
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            return f"Error performing web search: {str(e)}"

    @tool
    def search_and_analyze(self, query: str) -> str:
        """
        Search the web and provide detailed analysis of the results.
        
        Args:
            query: The search query for analysis
            
        Returns:
            Comprehensive analysis with sources
        """
        try:
            logger.info(f"Performing search and analysis for: {query}")
            
            # Perform grounded search
            result = self.google_search.search_with_grounding(query)
            
            # Extract additional information for analysis
            search_queries = []
            sources = []
            
            if result["grounding_metadata"]:
                search_queries = self.google_search.get_search_queries(result["grounding_metadata"])
                sources = self.google_search.extract_sources(result["grounding_metadata"])
            
            # Format comprehensive response
            analysis = f"**Analysis for: {query}**\n\n"
            analysis += f"{result['text']}\n\n"
            
            if search_queries:
                analysis += "**Search Queries Used:**\n"
                for i, sq in enumerate(search_queries, 1):
                    analysis += f"{i}. {sq}\n"
                analysis += "\n"
            
            if sources:
                analysis += "**Sources:**\n"
                for i, source in enumerate(sources, 1):
                    analysis += f"{i}. **{source['title']}**\n   {source['uri']}\n"
            
            logger.info(f"Successfully completed search and analysis for: {query}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in search and analyze: {e}")
            return f"Error performing search and analysis: {str(e)}"

    def add_documents(self, texts: list[str]) -> None:
        """
        Add documents to the vector store for RAG.
        
        Args:
            texts: List of text documents to add
        """
        try:
            if not texts:
                return
            
            # Split texts into chunks
            chunks = []
            for text in texts:
                chunks.extend(self.text_splitter.split_text(text))
            
            # Create documents
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            self.documents.extend(documents)
            
            logger.info(f"Added {len(documents)} document chunks to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    @tool
    def search_documents(self, query: str) -> str:
        """
        Search through stored documents using RAG.
        
        Args:
            query: The search query
            
        Returns:
            Relevant document excerpts
        """
        try:
            if not self.documents:
                return "No documents available for search. Please add documents first."
            
            # Retrieve relevant documents
            retriever = self.vector_store.as_retriever(search_kwargs={"k": settings.rag_top_k})
            docs = retriever.get_relevant_documents(query)
            
            if not docs:
                return "No relevant documents found for the query."
            
            # Format results
            results = f"Found {len(docs)} relevant document excerpts:\n\n"
            for i, doc in enumerate(docs, 1):
                results += f"**Excerpt {i}:**\n{doc.page_content}\n\n"
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return f"Error searching documents: {str(e)}"

    def get_tools(self):
        """Get all available tools."""
        return [self.web_search, self.search_and_analyze, self.search_documents]
