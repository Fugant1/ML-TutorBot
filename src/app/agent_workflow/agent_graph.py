from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END, START

def create_graph(llm):
    builder = StateGraph.Builder()

    