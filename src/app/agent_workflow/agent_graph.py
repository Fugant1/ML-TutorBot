from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from typing import TypedDict
import logging
import os

from src.app.rag_pipelines.general_rag_pipeline import Rag_Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

#Chatstate to help maintaining some data while navigating the nodes of the graph
class ChatState(TypedDict):
    input: str
    output: str
    tool_calls: list[dict]
    tool_results: list[dict]
    next_step: str
    retries: int

async def router_node(state: ChatState, google_api_key, possible_tools:list[str]):
    #this node is the main node, it starts here and defines all the next steps
    #the router will decide which tool to use, the code interpreter or the RAG to help the user
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=google_api_key)
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
        state['tool_calls'] = []
        state['tool_results'] = []
        state['retries'] = 0
        state['next_step'] = response.content
        return state
    except Exception as e:
        logging.error(f"Error while in router_node: {e}", traceback=True)
        state['retries'] += 1
        if state['retries'] > 3:
            logging.error("Max retries exceeded, something went wrong, try again...")
            return None
    
async def rag_retriever_node(state: ChatState):
    #throw all to the RAG pipeline to get the most relevant data
    rag = Rag_Pipeline()
    rag_data = await rag.query_docs(state['input'])
    #add the proper info to the state to the final_answer model be able to use it
    state['tool_calls'].append({'tool': 'rag_retriever', 'input': state['input']})
    state['tool_results'].append({'tool': 'rag_retriever', 'output': rag_data})
    return state

async def code_interpreter_node(state: ChatState, google_api_key):
    #this node is a bit more complex, we have a code spliter and interpreter and a final explainer and code builder
    #the first LLM will split the input in code and text and will return a brief description of what the code does or what error it has
    #the second LLM will be given the input, the code, the description and the output of the code, and will explain what the code does or what error it has and why
    llm1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=google_api_key)
    llm2 = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=google_api_key)
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
    explanation_or_error = response_splited[1] if len(response_splited) > 1 else "No explanation or error provided."
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

async def final_answer_node(state: ChatState, google_api_key):
    #as simple as it looks, just give all the info to the model and let it answer
    #it will have the input, and the tool_results with the info of the RAG or the code interpreter if used
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.7, api_key=google_api_key)
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

def create_graph(google_api_key: str):
    possible_tools = ['rag_retriever', 'code_interpreter']
    builder = StateGraph(ChatState)
    


    async def router_node_async(state: ChatState): return await router_node(state, google_api_key, possible_tools) 
    async def rag_retriever_node_async(state: ChatState): return await rag_retriever_node(state) 
    async def code_interpreter_node_async(state: ChatState): return await code_interpreter_node(state, google_api_key) 
    async def final_answer_node_async(state: ChatState): return await final_answer_node(state, google_api_key) 
    async def router_condition(state: ChatState):
        next_step = state.get('next_step', 'final_answer')
        if next_step not in ["rag_retriever", "code_interpreter", "final_answer"]:
            next_step = "final_answer"
        return next_step
    builder.add_node("router", router_node_async) 
    builder.add_node("rag_retriever", rag_retriever_node_async) 
    builder.add_node("code_interpreter", code_interpreter_node_async) 
    builder.add_node("final_answer", final_answer_node_async) 
    builder.set_entry_point("router") 
    builder.add_conditional_edges( "router", router_condition, 
                                  { "rag_retriever": "rag_retriever", 
                                   "code_interpreter": "code_interpreter", 
                                   "final_answer": "final_answer", }, 
                                   ) 
    builder.add_edge("rag_retriever", "final_answer") 
    builder.add_edge("code_interpreter", "final_answer") 
    builder.add_edge("final_answer", END)

    return builder.compile()