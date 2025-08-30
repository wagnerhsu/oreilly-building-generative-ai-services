from fastapi import FastAPI
from openai import OpenAI

app = FastAPI()
# LM Studio doesn't require an API key, but OpenAI client still expects one
openai_client = OpenAI(api_key="lm-studio", base_url="http://localhost:1234/v1")


@app.get("/")
def root_controller():
    return {"status": "healthy"}


@app.get("/chat")
def chat_controller(prompt: str = "Inspire me"):
    try:
        response = openai_client.chat.completions.create(
            model="google/gemma-3n-e4b",  # LM Studio uses a generic model identifier
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        statement = response.choices[0].message.content.strip()
        return {"statement": statement}
    except Exception as e:
        return {"error": f"Failed to get response from LM Studio: {str(e)}"}
