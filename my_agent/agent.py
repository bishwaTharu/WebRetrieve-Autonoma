from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from my_agent.utils.state import AgentState
from my_agent.utils.nodes import AgentNodes


class AgentWorkflow:
    def __init__(self):
        self.nodes = AgentNodes()
        self.workflow = StateGraph(AgentState)
        self._build_graph()
        self.checkpointer = MemorySaver()

    def _build_graph(self):
        self.workflow.add_node("llm_call", self.nodes.llm_call)
        self.workflow.add_node("tool_node", self.nodes.tool_node)

        self.workflow.add_edge(START, "llm_call")

        self.workflow.add_conditional_edges(
            "llm_call",
            self.nodes.should_continue,
            {"tool_node": "tool_node", "__end__": END},
        )

        self.workflow.add_edge("tool_node", "llm_call")

    def compile(self):
        # MLE ROBUSTNESS: Set recursion limit to prevent infinite loops (standard for ReAct)
        return self.workflow.compile(
            checkpointer=self.checkpointer
        )


agent_instance = AgentWorkflow()
graph = agent_instance.compile()
