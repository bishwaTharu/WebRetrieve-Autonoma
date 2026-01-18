import operator
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langchain_core.documents import Document

class AgentState(TypedDict):
    """
    Represents the state of our agent.
    """
    messages: Annotated[list[AnyMessage], operator.add]
    query: str
    documents: list[Document]
    generation: str
