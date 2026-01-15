import os
import logging
from typing import Literal
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from my_agent.utils.state import AgentState
from my_agent.utils.tools import AgentTools


class AgentNodes:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.tools_instance = AgentTools()
        self.tools = self.tools_instance.get_tools()
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        rate_limiter = InMemoryRateLimiter(requests_per_second=0.1, max_bucket_size=10)
        self.llm = init_chat_model(
            model=model_name,
            model_provider="groq",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
            rate_limiter=rate_limiter,
        ).bind_tools(self.tools)

    async def llm_call(self, state: AgentState):
        """
        Calls the LLM to decide the next action based on the current state.
        """
        sys_msg = SystemMessage(
            content=(
                "You are a robust Agentic RAG assistant. Your mission is to thoroughly research websites and provide technical answers.\n\n"
                "STRATEGY:\n"
                "1. CRAWL: When a URL is provided, use 'web_crawler_tool'. Analyze the content and the internal links provided in the output.\n"
                "2. DEEP DIVE: If the initial page doesn't have the full answer but shows promising links (e.g., 'Pricing', 'Docs', 'About'), crawl those links too.\n"
                "3. RETRIEVE: Once you have enough data indexed, use 'rag_retrieval_tool' to find precise details across all crawled pages.\n"
                "4. SYNTHESIZE: Provide a comprehensive and technical summary. Cite the specific URLs where you found the info.\n\n"
                "Always be proactive. If the user asks for 'cost', look for pricing pages."
            )
        )

        response = await self.llm.ainvoke([sys_msg] + state["messages"])
        return {"messages": [response]}

    async def tool_node(self, state: AgentState):
        """
        Executes the tool calls requested by the LLM.
        """
        last_message = state["messages"][-1]
        results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_func = self.tools_by_name[tool_name]

            self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

            # Since our tools are methods, we need to handle them correctly
            # langchain_core.tools.tool decorator handles self if applied to a method
            observation = await tool_func.ainvoke(tool_args)

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
