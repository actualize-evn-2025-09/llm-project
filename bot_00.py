from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

llm = OpenAI()
# creates an instance of the OpenAI client class so that way we can interact with their API
# person = Person.new


user_input = input("I'm a chatbot! Ask me anything: \n")


response = llm.responses.create(
  model="gpt-4.1-mini",
  temperature=1.5,
  # augmenting the prompt
  input=f"Respond to the following like a pirate: {user_input}"
)

print(response.output_text)