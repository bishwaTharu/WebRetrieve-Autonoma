import logging
from typing import Optional, Tuple
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from WebRetrieve_Autonoma.utils.state import AgentState
from WebRetrieve_Autonoma.utils.tools import AgentTools
from WebRetrieve_Autonoma.config import settings
from langchain_core.runnables import RunnableConfig
import re
import json


logger = logging.getLogger(__name__)


def _clean_response_content(content) -> str:
    """
    Clean and filter response content to prevent validation data and internal content from being shown.
    Specifically handles Gemini's response format with 'text' and 'extras' fields.
    """
    if not content:
        return content
    
    # Handle case where content is a list of dictionaries (Gemini format)
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Only extract the 'text' field, completely ignore 'extras', 'signature', 'index', etc.
                if 'text' in item:
                    text_parts.append(item['text'])
                # Skip any items that don't have a 'text' field
            elif isinstance(item, str):
                # If it's a plain string, include it
                text_parts.append(item)
        content = ''.join(text_parts)
    
    # Ensure content is a string
    if not isinstance(content, str):
        content = str(content)
    
    # Remove any remaining validation arrays with type, text, extras, index fields
    # This handles cases where the list structure might be stringified
    content = re.sub(r'\[.*?["\']type["\'].*?["\']text["\'].*?["\']extras["\'].*?["\']index["\'].*?\]', '', content, flags=re.DOTALL)
    
    # Remove signature patterns (both single and double quotes)
    content = re.sub(r'["\']signature["\']:\s*["\'][^"\']*["\']', '', content)
    
    # Remove arrays that look like validation data
    content = re.sub(r'\[\s*\{[^}]*["\']type["\']:\s*["\']text["\'][^}]*\}\s*\]', '', content, flags=re.DOTALL)
    
    # Remove any remaining JSON-like structures with 'extras' field
    content = re.sub(r'\{[^}]*["\']extras["\']:[^}]*\}', '', content)
    
    # Remove any remaining 'index' fields
    content = re.sub(r'["\']index["\']:\s*\d+', '', content)
    
    # Clean up extra whitespace and newlines
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = content.strip()
    
    # Line-by-line filtering for any remaining problematic patterns
    lines = content.split('\n')
    cleaned_lines = []
    skip_line = False
    
    for line in lines:
        # Check for both single and double quoted patterns
        if any(pattern in line for pattern in [
            "'type':", '"type":',
            "'extras':", '"extras":',
            "'signature':", '"signature":',
            "'index':", '"index":'
        ]):
            skip_line = True
            continue
        if skip_line and line.strip() in [']', '},', '}']:
            skip_line = False
            continue
        if not skip_line:
            cleaned_lines.append(line)
    
    cleaned_content = '\n'.join(cleaned_lines).strip()
    
    # Ensure we don't return empty content
    if not cleaned_content or len(cleaned_content) < 10:
        return "I apologize, but I encountered an issue processing the response. Please try your question again."
    
    return cleaned_content


def _split_provider_model(model_name: str) -> Tuple[Optional[str], str]:
    if not model_name:
        return None, model_name

    provider_prefixes = {"groq", "github", "openrouter", "gemini"}
    if "/" in model_name:
        prefix, rest = model_name.split("/", 1)
        if prefix.lower() in provider_prefixes and rest:
            return prefix.lower(), rest

    return None, model_name


def _resolve_backend(model_name: str) -> Tuple[str, str]:
    provider_prefix, raw_model = _split_provider_model(model_name)

    if provider_prefix in {"groq", "github", "openrouter", "gemini"}:
        return provider_prefix, raw_model

    if raw_model and raw_model.endswith(":free"):
        return "openrouter", raw_model

    github_models = {"gpt-4o", "gpt-4o-mini"}
    if raw_model in github_models:
        return "github", raw_model

    gemini_models = {"gemini-2.5-flash"}
    if raw_model in gemini_models:
        return "gemini", raw_model

    return "groq", raw_model


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
            backend, resolved_model = _resolve_backend(model_name)

            logger.info(
                f"Initializing LLM with model: {resolved_model} | Provider: {backend}"
            )

            if backend == "gemini":
                self.llm = ChatGoogleGenerativeAI(
                    model=resolved_model,
                    temperature=settings.llm_temperature,
                    api_key=settings.gemini_api_key
                )
            else:
                model_provider = "groq" if backend == "groq" else "openai"
                init_kwargs = {
                    "model": resolved_model,
                    "model_provider": model_provider,
                    "temperature": settings.llm_temperature,
                }

                if backend == "groq":
                    init_kwargs["api_key"] = settings.groq_api_key
                elif backend == "github":
                    init_kwargs["api_key"] = settings.github_api_key
                    init_kwargs["base_url"] = settings.github_base_url
                elif backend == "openrouter":
                    init_kwargs["api_key"] = settings.openrouter_api_key
                    init_kwargs["base_url"] = settings.openrouter_base_url

                self.llm = init_chat_model(**init_kwargs)

            self.fast_llm = self.llm
            self.reasoning_llm = self.llm

            logger.info(f"Initialized LLMs for nodes using provider: {backend}")
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
        backend, resolved_model = _resolve_backend(selected_model_name)

        logger.info(
            f"Resolved Provider: {backend} | Selected Model: {resolved_model}"
        )

        try:
            logger.info("Acquiring rate limit token...")
            await self.rate_limiter.acquire()
            logger.info("Token acquired. Proceeding with LLM call.")

            if backend == "gemini":
                current_llm = ChatGoogleGenerativeAI(
                    model=resolved_model,
                    temperature=settings.llm_temperature,
                    api_key=settings.gemini_api_key
                )
            else:
                model_provider = "groq" if backend == "groq" else "openai"
                init_kwargs = {
                    "model": resolved_model,
                    "model_provider": model_provider,
                    "temperature": settings.llm_temperature,
                }

                if backend == "groq":
                    init_kwargs["api_key"] = settings.groq_api_key
                elif backend == "github":
                    init_kwargs["api_key"] = settings.github_api_key
                    init_kwargs["base_url"] = settings.github_base_url
                elif backend == "openrouter":
                    init_kwargs["api_key"] = settings.openrouter_api_key
                    init_kwargs["base_url"] = settings.openrouter_base_url

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
                    "CRITICAL OUTPUT RULES:\n"
                    "- NEVER output validation data, signatures, or internal metadata\n"
                    "- NEVER show arrays with 'type', 'text', 'extras', 'index' fields\n"
                    "- NEVER include source code, JSON objects, or technical internals in your main response\n"
                    "- ONLY provide clean, human-readable answers to user questions\n"
                    "- Your response should be natural language only, not structured data\n\n"
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
            # CRITICAL: Pass config to ensure events are tracked by astream_events
            response = await llm_with_tools.ainvoke([sys_msg] + processed_messages, config=config)

            logger.info(f"LLM Response Content: {response.content[:200]}...")
            logger.info(f"LLM Response Tool Calls: {response.tool_calls}")

            # Clean the response content to remove validation data and internal content
            cleaned_content = _clean_response_content(response.content)
            
            # Create a new response with cleaned content
            cleaned_response = AIMessage(
                content=cleaned_content,
                tool_calls=response.tool_calls if hasattr(response, 'tool_calls') else None
            )

            return {"messages": [cleaned_response]}

        except Exception as e:
            logger.exception(f"Error in LLM Call for model {selected_model_name}")
            error_text = str(e)
            if (
                backend == "openrouter"
                and (
                    "No endpoints found matching your data policy" in error_text
                    or "Free model publication" in error_text
                )
            ):
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "OpenRouter blocked this request due to your privacy/data policy settings for free models. "
                                "Enable the required setting in your OpenRouter account, then retry.\n\n"
                                "Fix:\n"
                                "- Open https://openrouter.ai/settings/privacy\n"
                                "- Enable the option that allows free model access / publication under your data policy\n"
                                "- Retry the request\n\n"
                                "(OpenRouter returned: 'No endpoints found matching your data policy (Free model publication)')"
                            )
                        )
                    ]
                }

            # Return a fallback message so the agent doesn't just die silently
            return {
                "messages": [
                    AIMessage(
                        content=f"An error occurred while generating the response: {error_text}"
                    )
                ]
            }

    async def tool_node(self, state: AgentState, config: RunnableConfig = None):
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
                # Pass config to tool calls for tracing
                tasks.append(tool_instance.ainvoke(tool_args, config=config))
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
