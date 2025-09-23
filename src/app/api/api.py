from fastapi import FastAPI, Query, Request 
from contextlib import asynccontextmanager

from agent_workflow.agent_graph import ChatState, create_graph

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent_graph = create_graph()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/chat")
def create_app(request: Request, input: str = Query(...)):
    agent_graph = request.app.state.agent_graph
    state = ChatState(input=input)
    agent_graph.invoke(state)
    return state