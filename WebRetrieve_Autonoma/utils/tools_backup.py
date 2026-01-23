import logging
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from WebRetrieve_Autonoma.config import settings
from WebRetrieve_Autonoma.utils.gemini_embeddings import GeminiEmbeddings
import urllib.parse

##
logger = logging.getLogger(__name__)


class AgentTools:
    """Tools for web crawling and RAG retrieval."""

    def __init__(self):
        """Initialize tools with configuration from settings."""
        logger.info("Initializing AgentTools with configuration")

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

        self.crawler_config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
        logger.info(
            f"Initialized text splitter with chunk_size={settings.chunk_size}, chunk_overlap={settings.chunk_overlap}"
        )

    async def _web_crawler_logic(self, url: str) -> str:
        """
        Logic for crawling the given website URL and extracting content.

        Args:
            url: The URL to crawl

        Returns:
            A summary of the crawl operation including indexed content and links
        """
        logger.info(f"Starting web crawl for URL: {url}")

        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=self.crawler_config)

                if not result or not result.success:
                    error_msg = getattr(result, "error_message", "Unknown error")
                    logger.error(f"Failed to crawl {url}: {error_msg}")
                    return f"Failed to crawl {url}. Error: {error_msg}"

                texts = self.text_splitter.split_text(result.markdown)
                logger.info(f"Split content into {len(texts)} chunks for {url}")

                docs = []
                for i, text in enumerate(texts):
                    contextual_header = f"[Source: {url} | Title: {getattr(result, 'metadata', {}).get('title', 'No Title')}]\n"
                    contextual_content = contextual_header + text

                    docs.append(
                        Document(
                            page_content=contextual_content,
                            metadata={
                                "source": url,
                                "title": getattr(result, "metadata", {}).get(
                                    "title", "No Title"
                                ),
                                "chunk": i,
                            },
                        )
                    )

                self.vector_store.add_documents(docs)
                self.documents.extend(docs)
                logger.info(f"Successfully indexed {len(docs)} chunks from {url}")

                links = result.links.get("internal", [])
                links_summary = "\n".join(
                    [
                        f"- {l.get('text', 'No text')}: {l.get('href', 'No href')}"
                        for l in links[:15]
                    ]
                )

                summary = f"Successfully crawled and indexed {url}.\n\n"
                summary += f"Indexed {len(docs)} content chunks.\n\n"
                summary += f"Internal links found:\n{links_summary}"

                return summary

        except Exception as e:
            logger.error(f"Error crawling {url}: {e}")
            return f"Error crawling {url}: {str(e)}"

    @tool
    def web_crawler(self, url: str) -> str:
        """
        Crawl a website and index its content for retrieval.

        Args:
            url: The URL to crawl

        Returns:
            A summary of the crawl operation
        """
        import asyncio

        try:
            return asyncio.run(self._web_crawler_logic(url))
        except Exception as e:
            logger.error(f"Error in web crawler tool: {e}")
            return f"Error in web crawler: {str(e)}"

    @tool
    def rag_search(self, query: str) -> str:
        """
        Search through crawled documents using RAG.

        Args:
            query: The search query

        Returns:
            Relevant document excerpts
        """
        try:
            if not self.documents:
                return "No documents available for search. Please crawl some websites first."

            retriever = self.vector_store.as_retriever(search_kwargs={"k": settings.rag_top_k})
            docs = retriever.get_relevant_documents(query)

            if not docs:
                return "No relevant documents found for the query."

            results = f"Found {len(docs)} relevant document excerpts:\n\n"
            for i, doc in enumerate(docs, 1):
                results += f"**Excerpt {i}:**\n{doc.page_content}\n\n"

            return results

        except Exception as e:
            logger.error(f"Error in RAG search: {e}")
            return f"Error in RAG search: {str(e)}"

    def get_tools(self):
        """Get all available tools."""
        return [self.web_crawler, self.rag_search]
