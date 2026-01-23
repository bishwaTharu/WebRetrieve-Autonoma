import logging
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
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
            # Use Google Search tool to crawl the website
            result = self.google_search.crawl_website(url)

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

            return (
                f"✓ Successfully crawled and indexed {url}.\n"
                f"📄 Indexed {len(docs)} content chunks.\n"
                f"Content snippet: {result.markdown[:300]}...\n\n"
                f"🔗 Found {len(links)} internal links (showing top 15 for deeper research):\n{links_summary if links_summary else 'No internal links found'}"
            )
        except Exception as e:
            logger.exception(f"Error crawling {url}")
            return f"❌ Error crawling {url}: {str(e)}"

    def _rag_retrieval_logic(self, query: str) -> list[Document]:
        """
        Logic for searching the internal knowledge base using Hybrid Retrieval.

        Args:
            query: The search query

        Returns:
            List of retrieved documents
        """
        logger.info(f"Performing Hybrid RAG retrieval for query: {query[:100]}...")

        try:
            if not self.documents:
                logger.warning("No documents in knowledge base")
                return []

            available_docs = len(self.documents)
            safe_k = min(available_docs, settings.rag_top_k)
            logger.info(
                f"Retrieving {safe_k} documents (Available: {available_docs}, Requested: {settings.rag_top_k})"
            )

            bm25_retriever = BM25Retriever.from_documents(self.documents)
            bm25_retriever.k = safe_k

            vector_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": safe_k}
            )

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
            )

            try:
                candidates = ensemble_retriever.invoke(query)
            except Exception as ensemble_err:
                logger.warning(
                    f"Ensemble retrieval failed: {ensemble_err}. Falling back to vector-only."
                )
                # Fallback to pure vector search if ensemble fails
                candidates = vector_retriever.invoke(query)

            if not candidates:
                return []

            # ---------------------------------------------------------
            # MLE OPTIMIZATION: LLM-Based Reranking (Cross-Encoder Style)
            # ---------------------------------------------------------
            logger.info(f"Reranking {len(candidates)} candidates using AI...")

            rerank_prompt = (
                "You are an expert technical reranker. Given a user query and a set of search results, "
                "identify the top 5 most highly relevant results that contain precise technical details. "
                "Only return results that directly contribute to answering the query.\n\n"
                "Query: {query}\n\n"
                "Candidates:\n{candidates}\n\n"
                "Return the indices (0, 1, 2...) of the top results as a comma-separated list."
            ).format(
                query=query,
                candidates="\n".join(
                    [
                        f"[{i}] {c.page_content[:200]}..."
                        for i, c in enumerate(candidates)
                    ]
                ),
            )

            logger.info(
                f"Retrieved {len(candidates)} candidates via RAG (Ensemble/Vector)"
            )
            return candidates

        except Exception as e:
            logger.exception("Error during hybrid RAG retrieval")
            return []

    def format_docs(self, docs: list[Document]) -> str:
        """Helper to format documents for display/context."""
        if not docs:
            return "No relevant information found."

        content = "\n\n".join(
            [
                f"📄 Source: {d.metadata.get('source', 'Unknown')}\n"
                f"Title: {d.metadata.get('title', 'No Title')}\n"
                f"Content: {d.page_content}"
                for d in docs
            ]
        )
        return content

    async def _google_search_logic(self, query: str) -> str:
        """
        Logic for performing a Google search using crawl4ai.

        Args:
            query: The search query

        Returns:
            A list of relevant links found in the search results
        """
        browser_config = BrowserConfig(
            headless=True,
            extra_args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",  # Prevents crashes in memory-constrained containers
                "--disable-gpu",
            ],
        )
        logger.info(f"Performing Google search for: {query}")
        encoded_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={encoded_query}"

        try:
            async with AsyncWebCrawler(config=browser_config) as crawler:
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
            logger.info(f"RAG retrieval tool called with query: {query}")
            results = self._rag_retrieval_logic(query)

            if not results or not isinstance(results, list):
                return "No relevant technical details found in the indexed documents."

            formatted_results = []
            for i, doc in enumerate(results):
                source = doc.metadata.get(
                    "url", doc.metadata.get("source", "Unknown source")
                )
                formatted_results.append(
                    f"Source [{i+1}]: {source}\nContent: {doc.page_content}\n---"
                )

            return "\n\n".join(formatted_results)

        return [google_search_tool, web_crawler_tool, rag_retrieval_tool]
