from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

llm = OpenAI()
# creates an instance of the OpenAI client class so that way we can interact with their API
# person = Person.new

response = llm.responses.create(
  model="gpt-4.1-mini",
  temperature=0,
  input="Who was the first president?"
)

print(response.output_text)