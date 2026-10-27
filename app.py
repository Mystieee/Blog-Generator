import uvicorn
from fastapi import FastAPI, Request
from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

import os
from dotenv import load_dotenv
load_dotenv()

app=FastAPI()

#to track in langsmith
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "message": "Blog Generator API is running successfully!"
    }

#CREATE API
@app.post("/blogs")
async def create_blogs(request: Request):
    data = await request.json()
    topic = data.get("topic", "")

    #get the llm object
    groqllm= GroqLLM()

    llm = groqllm.get_llm()

    #get the graph
    graph_builder = GraphBuilder(llm)

    if topic:
        graph = graph_builder.setup_graph(usecase="topic")
        state = graph.invoke({"topic": topic})

    return {"data": state}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port = 8000, reload=True)