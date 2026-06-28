from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print("Key loaded:", api_key is not None)
print("Model:", model)

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Say only: OpenAI API works."}
        ],
    )

    print("SUCCESS:")
    print(response.choices[0].message.content)

except Exception as e:
    print("OPENAI API ERROR:")
    print(e)