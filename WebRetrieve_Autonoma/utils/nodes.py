import logging
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain.chat_models import init_chat_model
from WebRetrieve_Autonoma.utils.state import AgentState
from WebRetrieve_Autonoma.utils.tools import AgentTools
from WebRetrieve_Autonoma.config import settings
from langchain_core.runnables import RunnableConfig


logger = logging.getLogger(__name__)


class AgentNodes:
    """Nodes for the LangGraph agent workflow."""

    def __init__(self, model_name: str = None):
        """
        Initialize agent nodes with LLM and tools.
        """
        self.logger = logger
        logger.info("Initializing AgentNodes")

        try:
            from WebRetrieve_Autonoma.utils.rate_limiter import AsyncTokenBucket

            self.rate_limiter = AsyncTokenBucket(
                rate=settings.llm_rate_limit_rps,
                capacity=settings.llm_rate_limit_bucket_size,
            )
            logger.info(
                f"Initialized Rate Limiter: {settings.llm_rate_limit_rps} RPS, "
                f"Burst: {settings.llm_rate_limit_bucket_size}"
            )

            self.tools_instance = AgentTools()
            self.tools = self.tools_instance.get_tools()
            self.tools_by_name = {tool.name: tool for tool in self.tools}
            logger.info("Initialized AgentTools")
        except Exception as e:
            logger.exception("Failed to initialize tools/rate limiter")
            raise

        try:
            model_name = model_name or settings.llm_model_name
            provider = "openai" if "gpt" in model_name.lower() else "groq"

            logger.info(
                f"Initializing LLM with model: {model_name} | Provider: {provider}"
            )

            init_kwargs = {
                "model": model_name,
                "model_provider": provider,
                "temperature": settings.llm_temperature,
            }

            if provider == "groq":
                init_kwargs["api_key"] = settings.groq_api_key
            elif provider == "openai":
                init_kwargs["api_key"] = settings.github_api_key
                init_kwargs["base_url"] = settings.github_base_url

            self.llm = init_chat_model(**init_kwargs)

            self.fast_llm = self.llm
            self.reasoning_llm = self.llm

            logger.info(f"Initialized LLMs for nodes using provider: {provider}")
        except Exception as e:
            logger.exception(f"Failed to initialize LLM for model {model_name}")
            raise

    async def llm_call(self, state: AgentState, config: RunnableConfig):
        """Dynamic LLM call with tool binding, model switching, and TPM management."""
        logger.info("LLM Call Node: Generating response")
        messages = state["messages"]

        model_name = None
        if config and "configurable" in config:
            model_name = config["configurable"].get("model")
            logger.info(f"Frontend requested model: {model_name}")

        selected_model_name = model_name or settings.llm_model_name
        provider = "openai" if "gpt" in selected_model_name.lower() else "groq"

        logger.info(
            f"Resolved Provider: {provider} | Selected Model: {selected_model_name}"
        )

        try:
            logger.info("Acquiring rate limit token...")
            await self.rate_limiter.acquire()
            logger.info("Token acquired. Proceeding with LLM call.")

            init_kwargs = {
                "model": selected_model_name,
                "model_provider": provider,
                "temperature": settings.llm_temperature,
            }

            if provider == "groq":
                init_kwargs["api_key"] = settings.groq_api_key
            elif provider == "openai":
                init_kwargs["api_key"] = settings.github_api_key
                init_kwargs["base_url"] = settings.github_base_url

            current_llm = init_chat_model(**init_kwargs)

            processed_messages = []
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    if len(msg.content) > 5000:  # Increased limit slightly
                        truncated_content = (
                            msg.content[:5000] + "\n... [TRUNCATED FOR TOKENS] ..."
                        )
                        processed_messages.append(
                            ToolMessage(
                                content=truncated_content, tool_call_id=msg.tool_call_id
                            )
                        )
                    else:
                        processed_messages.append(msg)
                else:
                    processed_messages.append(msg)

            sys_msg = SystemMessage(
                content=(
                    "You are a robust WebRetrieve Autonoma. Your mission is to thoroughly research websites and provide technical answers.\n\n"
                    "STRATEGY:\n"
                    "1. SEARCH: When given a question without a specific URL, use 'google_search_tool' to find relevant sources.\n"
                    "2. SELECT: Review the search results and identify the most promising URLs (top 3-5).\n"
                    "3. CRAWL: Use 'web_crawler_tool' on each promising URL to gather detailed content. Analyze the content and internal links.\n"
                    "4. DEEP DIVE: If the initial pages don't have the full answer but show promising links (e.g., 'Pricing', 'Docs', 'About'), crawl those links too.\n"
                    "5. RETRIEVE: Once you have enough data indexed, use 'rag_retrieval_tool' to find precise details across all crawled pages.\n"
                    "6. SYNTHESIZE: Provide a comprehensive and technical summary. Cite the specific URLs where you found the info.\n\n"
                    "TOKEN MANAGEMENT:\n"
                    "Tool outputs may be truncated to stay within limits. If you need more detail from a specific crawl, use 'rag_retrieval_tool' with specific keywords.\n\n"
                    "RAG OPTIMIZATION (Agentic Few-Shot):\n"
                    "User: 'What are the deployment options for LangGraph?'\n"
                    "Thought: I've already crawled the LangGraph docs, but the initial summary was truncated. I need specific deployment details.\n"
                    'Call: rag_retrieval_tool("langgraph deployment options docker kubernetes")\n'
                    "Observation: [Retrieved specific chunks about Docker containers and Kubernetes helm charts]\n"
                    "Thought: This gives me the specific technicals I was missing. I can now synthesize the full answer.\n"
                    "Answer: LangGraph supports deployment via Docker containers... [Source: langgraph-docs]\n\n"
                    "User: 'Compare the pricing of Groq and OpenAI.'\n"
                    "Thought: I have crawled both pricing pages. Now I need to extract the exact numbers side-by-side.\n"
                    'Call: rag_retrieval_tool("Groq vs OpenAI pricing per 1M tokens")\n'
                    "Observation: [Retrieved chunks with pricing tables]\n"
                    "Answer: Groq charges $0.27/1M tokens for Llama 3 70B, while OpenAI... [Source: pricing-pages]\n\n"
                    "SUGGESTED QUESTIONS:\n"
                    "At the very end of your response, strictly output a JSON block with 3-4 suggested follow-up questions. "
                    'Format: ```json\n{"suggested_questions": ["Question 1?", "Question 2?"]}\n```\n'
                    "This block must be the very last thing in your response."
                )
            )

            llm_with_tools = current_llm.bind_tools(self.tools)
            response = await llm_with_tools.ainvoke([sys_msg] + processed_messages)

            logger.info(f"LLM Response Content: {response.content[:200]}...")
            logger.info(f"LLM Response Tool Calls: {response.tool_calls}")

            return {"messages": [response]}

        except Exception as e:
            logger.exception(f"Error in LLM Call for model {selected_model_name}")
            # Return a fallback message so the agent doesn't just die silently
            return {
                "messages": [
                    AIMessage(
                        content=f"An error occurred while generating the response: {str(e)}"
                    )
                ]
            }

    async def tool_node(self, state: AgentState):
        logger.info("Tool Node: Executing tools")
        last_message = state["messages"][-1]
        results = []

        tasks = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in self.tools_by_name:
                logger.info(f"Scheduling tool: {tool_name}")
                tool_instance = self.tools_by_name[tool_name]
                tasks.append(tool_instance.ainvoke(tool_args))
            else:
                logger.warning(f"Tool {tool_name} not found")

                async def error_coro(t_name=tool_name):
                    return f"Error: Tool '{t_name}' not found. Available tools: {list(self.tools_by_name.keys())}"

                tasks.append(error_coro())

        if tasks:
            import asyncio

            execution_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(execution_results):
                tool_call = last_message.tool_calls[i]
                if isinstance(result, Exception):
                    logger.error(f"Error executing tool {tool_call['name']}: {result}")
                    content = f"Exception during tool execution: {str(result)}"
                else:
                    content = str(result)

                results.append(
                    ToolMessage(content=content, tool_call_id=tool_call["id"])
                )

        return {"messages": results}

    def should_continue(self, state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return "__end__"
