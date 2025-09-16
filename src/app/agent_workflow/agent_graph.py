from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from typing import TypedDict
import logging

from src.app.rag_pipelines.general_rag_pípeline import Rag_Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

#Chatstate to help maintaining some data while navigating the nodes of the graph
class ChatState(TypedDict):
    input: str
    output: str
    tool_calls: list[dict]
    tool_results: list[dict]
    retries: int

async def router_node(state: ChatState, possible_tools:list[str]):
    #this node is the main node, it starts here and defines all the next steps
    #the router will decide which tool to use, the code interpreter or the RAG to help the user
    llm = None 
    #added a retry system to handle unpredicted behavior of the model
    state['retries'] = 0
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a router that decides the next step in a workflow based on the input, and possible tool calls."),
            ("user", "Input: {input}\n\n Possible Tools: {possible_tools}\n\n Decide the next step to give the best answer possible, return ONLY: 'rag_retriever', 'code_interpreter' or 'final_answer'.")])
        #the options 'rag_retriever', 'code_interpreter' or 'final_answer' are simple, the final_answer is called as a final step in every call
        #if tools are not needed, just skips them and goes throw the final answer directly
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
    #throw all to the RAG pipeline to get the most relevant data
    rag = Rag_Pipeline()
    rag_data = rag.query_docs(state['input'])
    #add the proper info to the state to the final_answer model be able to use it
    state['tool_calls'].append({'tool': 'rag_retriever', 'input': state['input']})
    state['tool_results'].append({'tool': 'rag_retriever', 'output': rag_data})
    return state

async def code_interpreter_node(state: ChatState):
    #will think what to do here, maybe run it in a sandboxed env via docker?? Idk, acept suggestions
    llm1 = None
    llm2 = None
    prompt1 = ChatPromptTemplate.from_messages([
        ("system", """You are a code interpreter that is experts in python, what you will do:
            - Analyze the input and split in text and code
            - If the code have an error, explain it
            - Return only the code followed by a short explanation of what the code does or if it has an error dividing them by a /
            Example of output: print("Hello World") / This code prints Hello World to the console.
            """),
        ("user", "Input: {input}"), 
    ])
    formatted_prompt1 = prompt1.format(input=state['input'])
    response = await llm1.ainvoke(formatted_prompt1)
    response_splited = response.content.split(' / ')
    code = response_splited[0]
    explanation_or_error = response_splited[1] if len(response) > 1 else "No explanation or error provided."
    tool = PythonAstREPLTool(description=code)
    code_run_return = await tool.arun(code)
    prompt2 = ChatPromptTemplate.from_messages([
        ("system", """You are a code interpreter that is experts in python, what you will do:
            - You will be given the input, the code that was run, a brief description and the output of the code
            - If the code have an error, explain it and point where the error occurs
            - If the code runs correctly, explain what the code does"""),
        ("user", "Input: {input}\n\nCode: {code}\n\nDescription: {explanation_or_error}\n\nOutput: {code_run_return}"),
        ])
    formatted_prompt2 = prompt2.format(input=state['input'], code=code, explanation_or_error=explanation_or_error, code_run_return=code_run_return)
    response2 = await llm2.ainvoke(formatted_prompt2)
    state['tool_calls'].append({'tool': 'code_interpreter', 'input': state['input']})
    state['tool_results'].append({'tool': 'code_interpreter', 'output': response2.content})
    return state

async def final_answer_node(state: ChatState):
    #as simple as it looks, just give all the info to the model and let it answer
    #it will have the input, and the tool_results with the info of the RAG or the code interpreter if used
    llm = None
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in python and machine learning, what you will do:
            - You will be given the input and the results of some tools that were used to help you give the best answer possible
            - If the input is a question, answer it using the tool results if needed
            - If the input has no tool results, just answer it with your own knowledge if you can
         Rules:
            - If the input is not related to python or machine learning, just answer that you can't help with that
            - If the input is related to python or machine learning, but you don't know the answer, just say that you don't know
            """),
        ("user", "Input: {input}, tool results: {tool_results}"), 
    ])
    formatted_prompt = prompt.format(input=state['input'], tool_results=state['tool_results'])
    response = await llm.ainvoke(formatted_prompt)
    state['output'] = response.content
    return state

def create_graph():
    #will add here the option of not use some tool, or adding other tools
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
            "final_answer": "final_answer"
        })
    builder.add_edge("rag_retriever", "final_answer")
    builder.add_edge("code_interpreter", "final_answer")
    builder.add_edge("final_answer", END)

    return builder.compile()