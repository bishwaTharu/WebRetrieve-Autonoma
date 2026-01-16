import logging
from typing import Literal
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from my_agent.utils.state import AgentState
from my_agent.utils.tools import AgentTools
from my_agent.config import settings


logger = logging.getLogger(__name__)


class AgentNodes:
    """Nodes for the LangGraph agent workflow."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize agent nodes with LLM and tools.
        
        Args:
            model_name: Optional model name override, defaults to config
        """
        self.logger = logger
        logger.info("Initializing AgentNodes")
        
        # Initialize tools
        try:
            self.tools_instance = AgentTools()
            self.tools = self.tools_instance.get_tools()
            self.tools_by_name = {tool.name: tool for tool in self.tools}
            logger.info(f"Initialized {len(self.tools)} tools: {list(self.tools_by_name.keys())}")
        except Exception as e:
            logger.exception("Failed to initialize tools")
            raise

        # Initialize LLM with rate limiting
        try:
            model_name = model_name or settings.llm_model_name
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=settings.llm_rate_limit_rps,
                max_bucket_size=settings.llm_rate_limit_bucket_size
            )
            
            self.llm = init_chat_model(
                model=model_name,
                model_provider="groq",
                temperature=settings.llm_temperature,
                api_key=settings.groq_api_key,
                rate_limiter=rate_limiter,
            ).bind_tools(self.tools)
            
            logger.info(f"Initialized LLM: {model_name} with rate limit {settings.llm_rate_limit_rps} RPS")
        except Exception as e:
            logger.exception("Failed to initialize LLM")
            raise

    async def llm_call(self, state: AgentState, config: dict = None):
        """
        Calls the LLM to decide the next action based on the current state.
        Supports dynamic model selection via config.
        """
        # Get model from config or fallback to settings
        # LangGraph passes config as a RunnableConfig which may have .get or not depending on context
        requested_model = None
        if config and "configurable" in config:
            requested_model = config["configurable"].get("model")
            
        model_name = requested_model or settings.llm_model_name
        
        logger.info(f"LLM Call - Requested: {requested_model}, Final: {model_name}")

        try:
            # Re-init model dynamically for this specific call
            llm = init_chat_model(
                model=model_name,
                model_provider="groq",
                temperature=settings.llm_temperature,
                api_key=settings.groq_api_key,
            ).bind_tools(self.tools)
            
            logger.debug(f"Successfully initialized dynamic LLM: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to init dynamic model {model_name}. Error: {e}")
            logger.info(f"Falling back to default model: {settings.llm_model_name}")
            llm = self.llm 

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
                "Tool outputs may be truncated to stay within limits. If you need more detail from a specific crawl, use 'rag_retrieval_tool' with specific keywords."
            )
        )

        # Truncate conversation history if it's getting too large for Groq free tier
        messages = state["messages"]
        if len(messages) > 8: # Reduced from 10 to be safer
            messages = messages[-8:]
            logger.info("Truncated history to last 8 messages")

        try:
            response = await llm.ainvoke([sys_msg] + messages)
            return {"messages": [response]}
        except Exception as e:
            if "413" in str(e) or "rate_limit" in str(e).lower():
                logger.error(f"TPM/Rate limit hit: {e}")
                error_content = (
                    "⚠️ The research data is currently too large for the selected model's limit (TPM). "
                    "I have indexed the information, but I need to provide a more concise summary. "
                    "\n\n**Quick Summary:** I found relevant information but reached a technical limit while processing the full details. "
                    "Please ask a more specific question about the links I've found."
                )
                from langchain_core.messages import AIMessage
                return {"messages": [AIMessage(content=error_content)]}
            raise e

    async def tool_node(self, state: AgentState):
        """
        Executes the tool calls requested by the LLM.
        """
        last_message = state["messages"][-1]
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"Executing tool: {tool_name}")

            try:
                tool_func = self.tools_by_name.get(tool_name)
                
                if not tool_func:
                    observation = f"❌ Error: Tool '{tool_name}' not found"
                else:
                    observation = await tool_func.ainvoke(tool_args)
                    
                    max_chars = 6000 
                    if len(str(observation)) > max_chars:
                        logger.warning(f"Tool {tool_name} output too large ({len(str(observation))} chars), truncating to {max_chars}")
                        observation = str(observation)[:max_chars] + "\n\n... [TRUNCATED DUE TO SIZE LIMITS] ...\nUse rag_retrieval_tool for more specific details."
                    
                    logger.info(f"Tool {tool_name} completed")
                    
            except Exception as e:
                logger.exception(f"Error executing tool {tool_name}")
                observation = f"❌ Error executing {tool_name}: {str(e)}"

            results.append(
                ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
            )

        return {"messages": results}

    def should_continue(self, state: AgentState) -> Literal["tool_node", "__end__"]:
        """
        Determines whether the graph should continue to tool execution or end.
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tool_node"
        return "__end__"
