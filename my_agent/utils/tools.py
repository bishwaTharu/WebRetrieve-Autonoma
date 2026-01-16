import logging
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from my_agent.config import settings
import urllib.parse


logger = logging.getLogger(__name__)


class AgentTools:
    """Tools for web crawling and RAG retrieval."""

    def __init__(self):
        """Initialize tools with configuration from settings."""
        logger.info("Initializing AgentTools with configuration")

        try:
            self.vector_store = SKLearnVectorStore(
                embedding=HuggingFaceEmbeddings(
                    model_name=settings.embedding_model_name,
                    model_kwargs={"device": settings.embedding_device},
                )
            )
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

                docs = [
                    Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "title": getattr(result, "metadata", {}).get(
                                "title", "No Title"
                            ),
                            "chunk": i,
                        },
                    )
                    for i, text in enumerate(texts)
                ]

                self.vector_store.add_documents(docs)
                logger.info(f"Successfully indexed {len(docs)} chunks from {url}")

                links = result.links.get("internal", [])
                links_summary = "\n".join(
                    [
                        f"- {l.get('text', 'No text')}: {l.get('href', 'No href')}"
                        for l in links[:15]
                    ]
                )

                return (
                    f"✓ Successfully crawled and indexed {url}.\n"
                    f"📄 Indexed {len(docs)} content chunks.\n"
                    f"Content snippet: {result.markdown[:300]}...\n\n"
                    f"🔗 Found {len(links)} internal links (showing top 15 for deeper research):\n{links_summary if links_summary else 'No internal links found'}"
                )
        except Exception as e:
            logger.exception(f"Error crawling {url}")
            return f"❌ Error crawling {url}: {str(e)}"

    def _rag_retrieval_logic(self, query: str) -> str:
        """
        Logic for searching the internal knowledge base.

        Args:
            query: The search query

        Returns:
            Retrieved context from the knowledge base
        """
        logger.info(f"Performing RAG retrieval for query: {query[:100]}...")

        try:
            docs = self.vector_store.similarity_search(query, k=settings.rag_top_k)

            if not docs:
                logger.warning("No relevant documents found in vector store")
                return "⚠️ No relevant information found in the knowledge base. Try crawling more URLs first."

            logger.info(f"Retrieved {len(docs)} relevant documents")

            content = "\n\n".join(
                [
                    f"📄 Source: {d.metadata.get('source', 'Unknown')}\n"
                    f"Title: {d.metadata.get('title', 'No Title')}\n"
                    f"Content: {d.page_content}"
                    for d in docs
                ]
            )
            return f"✓ Retrieved {len(docs)} relevant contexts:\n\n{content}"
        except Exception as e:
            logger.exception("Error during RAG retrieval")
            return f"❌ Error during retrieval: {str(e)}"

    async def _google_search_logic(self, query: str) -> str:
        """
        Logic for performing a Google search using crawl4ai.

        Args:
            query: The search query

        Returns:
            A list of relevant links found in the search results
        """
        logger.info(f"Performing Google search for: {query}")
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}"

        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=self.crawler_config)

                if not result or not result.success:
                    error_msg = getattr(result, "error_message", "Unknown error")
                    logger.error(f"Failed to search Google: {error_msg}")
                    return f"Failed to search Google. Error: {error_msg}"

                links = []
                if result.links:
                    all_links = result.links.get("external", [])
                    
                    for link in all_links:
                        href = link.get("href", "")
                        text = link.get("text", "")
                        
                        if (
                            href 
                            and href.startswith("http") 
                            and "google.com" not in href 
                            and "googleusercontent" not in href
                        ):
                            links.append(f"- {text}: {href}")

                unique_links = []
                seen_urls = set()
                for l in links:
                    url_part = l.split(": ")[-1]
                    if url_part not in seen_urls:
                        unique_links.append(l)
                        seen_urls.add(url_part)
                        if len(unique_links) >= 10:
                            break

                formatted_links = "\n".join(unique_links)
                return (
                    f"✓ Google Search Results for '{query}':\n"
                    f"{formatted_links}\n\n"
                    "SUGGESTION: Use 'web_crawler_tool' to crawl the most relevant URLs from the list above."
                )

        except Exception as e:
            logger.exception(f"Error searching Google for {query}")
            return f"❌ Error searching Google: {str(e)}"

    def get_tools(self):
        @tool
        async def google_search_tool(query: str) -> str:
            """
            Performs a Google search to find relevant websites and URLs.
            Use this when you need to find information but don't have a specific URL to crawl.
            """
            return await self._google_search_logic(query)

        @tool
        async def web_crawler_tool(url: str) -> str:
            """
            Crawls the given website URL and extracts content.
            It also extracts internal links which can be used to deep dive into the site.
            """
            return await self._web_crawler_logic(url)

        @tool
        def rag_retrieval_tool(query: str) -> str:
            """
            Searches the internal knowledge base for technical details about previously crawled websites.
            Use this to answer specific questions based on the content gathered.
            """
            return self._rag_retrieval_logic(query)

        return [google_search_tool, web_crawler_tool, rag_retrieval_tool]
