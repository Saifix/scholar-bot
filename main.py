from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llm import ScholarBot


app = FastAPI(openapi_url="/openapi.json", docs_url="/docs")
users = {}


class User:
    def __init__(self):
        self.agent = ScholarBot()
        self.llm = self.agent.create_llm()


class ChatInput(BaseModel):
    text: str


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)


@app.post('/start/{user_id}')
async def start_chat(user_id: str):
    if user_id not in users:
        users[user_id] = User()
        return {"message": "Chat started", "user_id": user_id}
    else:
        raise HTTPException(status_code=400, detail="User already exists")


@app.post('/chat/{user_id}')
async def chat(user_id: str, input_data: ChatInput):
    if user_id in users:
        user = users[user_id]
        response = user.llm.run(input_data.text)
        print("Person Input: ", input_data.text)
        print("GPT Response: ", response)
        return {"response": response}
    else:
        raise HTTPException(status_code=404, detail="User not found")


@app.post('/stop/{user_id}')
async def stop_chat(user_id: str):
    if user_id in users:
        del users[user_id]
        return {"message": "Chat stopped", "user_id": user_id}
    else:
        raise HTTPException(status_code=404, detail="User not found")

