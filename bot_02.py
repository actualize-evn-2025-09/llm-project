from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

llm = OpenAI()

# Memory System

assistant_message = "Arrgh, how can I help you matey?"
user_input = input(f"Assistant: {assistant_message}\n")
history = [
  {"role": "developer", "content": "You are a helpful AI assistant who always talks like a pirate."},
  {"role": "assistant", "content": assistant_message},
  {"role": "user", "content": user_input}
]

while user_input != "exit":
  response = llm.responses.create(
    model="gpt-4.1-mini",
    temperature=1,
    input=history
  )

  print(f"\nAssistant: {response.output_text}")

  user_input = input("\nUser: ")

  history += [
    {"role": "assistant", "content": response.output_text},
    {"role": "user", "content": user_input}
  ]

  # print("-----------")
  # print(history)
  # print("-----------")
