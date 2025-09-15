from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict
import logging

from src.app.rag_pipelines.general_rag_pípeline import Rag_Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

class ChatState(TypedDict):
    input: str
    output: str
    tool_calls: list[dict]
    retries: int

async def router_node(state: ChatState, possible_tools:list[str]):
    llm = None 
    state['retries'] = 0
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a router that decides the next step in a workflow based on the input, and possible tool calls."),
            ("user", "Input: {input}\n\n Possible Tools: {possible_tools}\n\n Decide the next step to give the best answer possible, return ONLY: 'rag_retriever', 'code_interpreter' or 'final_answer'.")])
        response = await llm.ainvoke(prompt.format(input=state['input'], possible_tools=possible_tools))
        if(response.content not in possible_tools + ['final_answer']):
            logging.error(f"Router couldn't understand the input\nRetries: {state['retries']}/3\nretrying...")
            state['retries'] += 1
            if state['retries'] > 3:
                logging.error("Max retries exceeded and still couldn't understand the input")
                return 'final_answer'
        state['retries'] = 0
        return response.content
    except Exception as e:
        logging.error(f"Error while in router_node: {e}", traceback=True)
        state['retries'] += 1
        if state['retries'] > 3:
            logging.error("Max retries exceeded, something went wrong, try again...")
            return None
    
async def rag_retriever_node(state: ChatState):
    llm = None
    rag = Rag_Pipeline(llm)

async def code_interpreter_node(state: ChatState):
    llm = None

async def final_answer_node(state: ChatState):
    llm = None

def create_graph():
    possible_tools = ['rag_retriever', 'code_interpreter']
    builder = StateGraph.Builder()

    builder.add_node("router", lambda state: router_node(state, possible_tools))
    builder.add_node("rag_retriever", rag_retriever_node)
    builder.add_node("code_interpreter", code_interpreter_node)
    builder.add_node("final_answer", final_answer_node)
    builder.set_entry_point("router")
    builder.add_conditional_edge("router", router_node, {
            "rag_retriever": "rag_retriever",
            "code_interpreter": "code_interpreter",
            "final_answer": "final_answer_node"
        })
    builder.add_edge("rag_retriever", END)
    builder.add_edge("code_interpreter", END)
    builder.add_edge("final_answer", END)

    return builder.compile()