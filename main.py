import asyncio
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from my_agent.agent import graph

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def run_example():
    """
    Runs a sample query through the agent.
    """
    query = (
        "Go to https://crawl4ai.com/ and tell me how it handles JavaScript execution."
    )

    inputs = {"messages": [HumanMessage(content=query)]}

    logger.info("----------Starting Agentic Workflow----------")
    async for output in graph.astream(inputs, stream_mode="updates"):
        for node_name, state_update in output.items():
            logger.info(f"\n--- Node: {node_name} ---")
            if "messages" in state_update:
                # Print the last message in the update
                last_msg = state_update["messages"][-1]
                if hasattr(last_msg, "pretty_print"):
                    last_msg.pretty_print()
                else:
                    logger.info(last_msg)


if __name__ == "__main__":
    if asyncio.get_event_loop().is_running():
        import nest_asyncio

        nest_asyncio.apply()
        asyncio.create_task(run_example())
    else:
        asyncio.run(run_example())
