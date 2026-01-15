import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    """
    Represents the state of our agent.
    """
    # Annotated with operator.add means messages will be appended rather than overwritten
    messages: Annotated[list[AnyMessage], operator.add]
