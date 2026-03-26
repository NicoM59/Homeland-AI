import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "Missing OPENROUTER_API_KEY. Add it to your .env file or environment variables."
        )

    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-OpenRouter-Title": "final_project_jedha",
        },
    )


def get_default_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")


def ask_llm(prompt: str, system_prompt: str | None = None) -> str:
    client = get_openrouter_client()
    model_name = get_default_model()

    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.2,
        max_tokens=25000
    )

    return response.choices[0].message.content or ""