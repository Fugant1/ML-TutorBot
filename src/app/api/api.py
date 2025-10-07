from fastapi import FastAPI, Query, Request 
from contextlib import asynccontextmanager

from agent_workflow.agent_graph import ChatState, create_graph
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY") #You need to set your Google API key here to this whole app run properly

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent_graph = create_graph(google_api_key)
    yield

chat_router = FastAPI(lifespan=lifespan)

@chat_router.get("/chat")
def create_app(request: Request, input: str = Query(...)):
    agent_graph = request.app.state.agent_graph
    state = ChatState(input=input)
    state = agent_graph.invoke(state)
    return state