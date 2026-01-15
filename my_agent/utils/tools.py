from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from langchain_text_splitters import RecursiveCharacterTextSplitter


class AgentTools:
    def __init__(self):
        self.vector_store = SKLearnVectorStore(
            embedding=HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
        )
        self.crawler_config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

    async def _web_crawler_logic(self, url: str) -> str:
        """
        Logic for crawling the given website URL and extracting content.
        """
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=self.crawler_config)

                if not result or not result.success:
                    return f"Failed to crawl {url}. Error: {getattr(result, 'error_message', 'Unknown error')}"

                texts = self.text_splitter.split_text(result.markdown)
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
                links = result.links.get("internal", [])
                links_summary = "\n".join(
                    [f"- {l.get('text')}: {l.get('href')}" for l in links[:15]]
                )

                return (
                    f"Successfully crawled and indexed {url}.\n"
                    f"Content snippet: {result.markdown[:300]}...\n\n"
                    f"Found internal links (useful for deeper research):\n{links_summary}"
                )
        except Exception as e:
            return f"Error crawling {url}: {str(e)}"

    def _rag_retrieval_logic(self, query: str) -> str:
        """
        Logic for searching the internal knowledge base.
        """
        try:
            docs = self.vector_store.similarity_search(query, k=5)
            if not docs:
                return "No relevant information found in the knowledge base. Try crawling more URLs."

            content = "\n\n".join(
                [
                    f"Source: {d.metadata.get('source')}\nContent: {d.page_content}"
                    for d in docs
                ]
            )
            return f"Retrieved Context:\n{content}"
        except Exception as e:
            return f"Error during retrieval: {str(e)}"

    def get_tools(self):
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

        return [web_crawler_tool, rag_retrieval_tool]
