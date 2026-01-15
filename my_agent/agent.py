from langgraph.graph import StateGraph, START, END
from my_agent.utils.state import AgentState
from my_agent.utils.nodes import AgentNodes


class AgentWorkflow:
    def __init__(self):
        self.nodes = AgentNodes()
        self.workflow = StateGraph(AgentState)
        self._build_graph()

    def _build_graph(self):
        # Add nodes
        self.workflow.add_node("llm_call", self.nodes.llm_call)
        self.workflow.add_node("tool_node", self.nodes.tool_node)

        # Set entry point
        self.workflow.add_edge(START, "llm_call")

        # Set conditional edges
        self.workflow.add_conditional_edges(
            "llm_call",
            self.nodes.should_continue,
            {"tool_node": "tool_node", "__end__": END},
        )

        # Add edge back to llm_call from tool_node
        self.workflow.add_edge("tool_node", "llm_call")

    def compile(self):
        return self.workflow.compile()


agent_instance = AgentWorkflow()
graph = agent_instance.compile()
