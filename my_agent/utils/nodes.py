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

    async def llm_call(self, state: AgentState):
        """
        Calls the LLM to decide the next action based on the current state.
        """
        sys_msg = SystemMessage(
            content=(
                "You are a robust Agentic RAG assistant. Your mission is to thoroughly research websites and provide technical answers.\n\n"
                "STRATEGY:\n"
                "1. SEARCH: When given a question without a specific URL, use 'google_search_tool' to find relevant sources.\n"
                "2. SELECT: Review the search results and identify the most promising URLs (top 3-5).\n"
                "3. CRAWL: Use 'web_crawler_tool' on each promising URL to gather detailed content. Analyze the content and internal links.\n"
                "4. DEEP DIVE: If the initial pages don't have the full answer but show promising links (e.g., 'Pricing', 'Docs', 'About'), crawl those links too.\n"
                "5. RETRIEVE: Once you have enough data indexed, use 'rag_retrieval_tool' to find precise details across all crawled pages.\n"
                "6. SYNTHESIZE: Provide a comprehensive and technical summary. Cite the specific URLs where you found the info.\n\n"
                "Always be proactive. If the user asks for 'cost' and you don't have it, look for pricing pages."
            )
        )

        response = await self.llm.ainvoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    async def tool_node(self, state: AgentState):
        """
        Executes the tool calls requested by the LLM.
        
        Args:
            state: The current agent state
            
        Returns:
            Updated state with tool execution results
        """
        last_message = state["messages"][-1]
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            try:
                tool_func = self.tools_by_name.get(tool_name)
                
                if not tool_func:
                    error_msg = f"Tool '{tool_name}' not found in available tools"
                    logger.error(error_msg)
                    observation = f"❌ Error: {error_msg}"
                else:
                    # langchain_core.tools.tool decorator handles self if applied to a method
                    observation = await tool_func.ainvoke(tool_args)
                    logger.info(f"Tool {tool_name} completed successfully")
                    
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
