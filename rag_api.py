import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from pinecone import Pinecone

load_dotenv()

app = FastAPI()
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

llm = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
dense_index = pc.Index("llm-project")  # Use YOUR index name

conversations = {}
conversation_chunks = {}  # NEW: Track chunks per conversation

# What Conversation Chunks will look like:
# conversation_chunks = {
#    "default": {
#        "chunk_id_123": "Documentation text about Flamehamster...",
#        "chunk_id_456": "Documentation text about Rumblechirp...",
#    },
#    "user_abc": {
#        "chunk_id_789": "Some other documentation...",
#    }
# } 


class ChatMessage(BaseModel):
   message: str
   conversation_id: str = "default"

# user_input = "It keeps crashing"
# rag_chunks = "default": {}
def rag(user_input, rag_chunks):
   """Search Pinecone and ADD chunks to the dictionary"""
   results = dense_index.search(
       namespace="all-gross",
       query={
           "top_k": 3,
           "inputs": {"text": user_input}
       }
   )
  
   for hit in results['result']['hits']:
       fields = hit.get('fields')
       chunk_text = fields.get('chunk_text')
       rag_chunks[hit['_id']] = chunk_text  # Store with ID as key
      #  "default": {"rumblechirp-chunk-39": "Having problems with Rumberchirp", "emrgency-chunk-17": "Emergency room server", "flamehamster-chunk-3": "Entirely free software"}

def system_prompt(rag_chunks=None):
   return {
       "role": "developer",
       "content": f"""You are an AI customer support technician who is
       knowledgeable about software products created by the company called GROSS.
       The products are:
       * Flamehamster, a web browser.
       * Rumblechirp, an email client.
       * GuineaPigment, a drawing tool for creating/editing SVGs
       * EMRgency, an electronic medical record system
       * Verbiage++, a content management system.


       You represent GROSS, and you are having a conversation with a human user
       who needs technical support with at least one of these GROSS products.
      
       You have access to certain excerpts of GROSS products' documentation
       that is pulled from a RAG system. Use this info (and no other info)
       to advise the user. Here are the documentation excerpts: {rag_chunks}


       When helping troubleshoot a user's issue, ask a proactive question to
       help determine what exactly the issue is. When asking proactive follow-up
       questions, ask exactly one question at a time."""
   }

@app.get("/")
def index():
   return {"message": "GROSS Support Chatbot API"}

@app.post("/chat")
def create(chat_message: ChatMessage):
   conversation_id = chat_message.conversation_id
   user_message = chat_message.message
  
   # Initialize if new conversation and conversation chunk
   if conversation_id not in conversations:
       conversations[conversation_id] = [
           system_prompt(),
           {"role": "assistant", "content": "How can I help you today?"}
       ]
       conversation_chunks[conversation_id] = {}  # Empty dict for this conversation's chunks
      #  conversation_chunk["default"] = {}
  
   # Get RAG chunks (adds to the dictionary via pass-by-reference!)
   rag(user_message, conversation_chunks[conversation_id])
  #  rag("Hello, tell me about Flamehamster", "default": {})
  
   # REWRITE HISTORY: Update system prompt with accumulated chunks
   conversations[conversation_id][0] = system_prompt(conversation_chunks[conversation_id])
   #  conversations["default"][0] = system_prompt("default": {"rumblechirp-chunk-39": "Having problems with Rumberchirp", "emrgency-chunk-17": "Emergency room server", "flamehamster-chunk-3": "Entirely free software"})
  
   # Add user message (just the message, no RAG in user prompt!)
   conversations[conversation_id].append({"role": "user", "content": user_message})
  
   # Get response
   response = llm.responses.create(
       model="gpt-4.1-mini",
       temperature=0,
       input=conversations[conversation_id]
   )
  
   assistant_message = response.output_text
   conversations[conversation_id].append({"role": "assistant", "content": assistant_message})
  
   return {
       "message": assistant_message,
       "conversation_id": conversation_id
   }

@app.get("/conversations/{conversation_id}")
def show(conversation_id: str):
   if conversation_id not in conversations:
       return {"error": "Conversation not found"}
   return {"conversation_id": conversation_id, "history": conversations[conversation_id]}

@app.delete("/conversations/{conversation_id}")
def destroy(conversation_id: str):
   if conversation_id in conversations:
       del conversations[conversation_id]
   if conversation_id in conversation_chunks:
       del conversation_chunks[conversation_id]
   return {"message": "Conversation deleted"}