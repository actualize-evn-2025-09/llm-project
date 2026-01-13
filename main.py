from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Configure CORS (so your frontend can talk to this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
llm = OpenAI()

# store conversation histories
conversations = {}

# Define request body structure (like Rails strong params)
class ChatMessage(BaseModel):
    message: str
    conversation_id: str = "default"


# Root endpoint
@app.get("/")
def index():
    return {"message": "Chatbot API is running! TEST"}


# Chat endpoint
@app.post("/chat")
def create(chat_message: ChatMessage):
    conversation_id = chat_message.conversation_id
    user_message = chat_message.message
    
    if conversation_id not in conversations:
        conversations[conversation_id] = [
            {"role": "developer", "content": "You are a helpful AI assistant."}
        ]
    
    conversations[conversation_id].append({"role": "user", "content": user_message})
    
    response = llm.responses.create(
        model="gpt-4.1-mini",
        temperature=1,
        input=conversations[conversation_id]
    )
    
    assistant_message = response.output_text
    conversations[conversation_id].append({"role": "assistant", "content": assistant_message})
    
    return {
        "message": assistant_message,
        "conversation_id": conversation_id
    }