# bot_06.py - Chatbot with Langfuse Automated Scoring (FIXED)

import os
from dotenv import load_dotenv
from langfuse.openai import OpenAI
from langfuse import get_client
from pinecone import Pinecone

load_dotenv()
llm = OpenAI()
langfuse = get_client()  # Changed from Langfuse()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
dense_index = pc.Index("llm-project")


# ============================================
# Automated check functions
# ============================================

def check_no_excerpts(response):
    if "excerpt" in response.lower():
        return 0
    return 1


def check_no_external_links(response):
    if "http://" in response or "https://" in response:
        return 0
    return 1


# ============================================
# Your existing chatbot code
# ============================================

def rag(user_input):
    results = dense_index.search(
        namespace="all-gross",
        query={"top_k": 3, "inputs": {"text": user_input}}
    )
    documentation = ""
    for hit in results['result']['hits']:
        fields = hit.get('fields')
        chunk_text = fields.get('chunk_text')
        documentation += chunk_text
    return documentation


def system_prompt():
    return {"role": "developer", "content": """You are an AI customer support
    technician for GROSS software products."""}


def user_prompt(user_input, documentation):
    return {"role": "user",
            "content": f"""Here are excerpts from the official GROSS product
            documentation: {documentation}. Use this info to answer: {user_input}"""}


if __name__ == "__main__":
    print(f"Assistant: How can I help you today?\n")
    user_input = input("User: ")
    history = [system_prompt()]

    while user_input != "exit":
        # Wrap the whole interaction in a Langfuse span
        with langfuse.start_as_current_span(name="chat-turn") as span:
            documentation = rag(user_input)
            history += [user_prompt(user_input, documentation)]
            
            response = llm.responses.create(
                model="gpt-4.1-mini",
                temperature=0,
                input=history
            )
            
            bot_response = response.output_text
            print(f"\nAssistant: {bot_response}\n")
            
            # Score this trace
            span.score_trace(
                name="no_excerpts",
                value=check_no_excerpts(bot_response),
                data_type="NUMERIC"
            )
            span.score_trace(
                name="no_external_links",
                value=check_no_external_links(bot_response),
                data_type="NUMERIC"
            )
        
        history += [{"role": "assistant", "content": bot_response}]
        user_input = input("User: ")
    
    print("Done! Check Langfuse for your scored traces.")