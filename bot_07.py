from dotenv import load_dotenv
from openai import OpenAI

load_dotenv('.env')

llm = OpenAI()

assistant_message = "How can I help you today?"
print(f"Assistant: {assistant_message}\n")
user_prompt = input("User: ")
history = [
    {"role": "developer", "content": "You are a helpful AI assistant."},
    {"role": "assistant", "content": assistant_message},
    {"role": "user", "content": user_prompt},
]

while user_prompt != "exit":
  response = llm.responses.create(
    model="gpt-4.1-mini",
    temperature=0,
    input=history
  )

  print(f"\nAssistant: {response.output_text}")

  user_prompt = input("\nUser: ")

  history += [
    {"role": "assistant", "content": response.output_text},
    {"role": "user", "content": user_prompt}
  ]