from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def translate_to_french(text):
  llm = OpenAI()
  response = llm.responses.create(
    model="gpt-4.1-mini",
    temperature=1,
    input=f"Translate the following into French, and only include the translation itself with no extra introductory text: {text}"
  )

  return response.output_text

user_input = input("Tell me something and I'll translate it into French: \n")

print(translate_to_french(user_input))