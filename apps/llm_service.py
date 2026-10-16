
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5-coder:3b"

SYSTEM_PROMPT = """
You are Debug Dojo, a programming debugging coach.
Give concise, progressive debugging hints.
Never reveal complete corrected solutions or hidden tests.
Use only the code and evidence provided.
Never claim code was executed unless execution was verified.
If evidence is insufficient, say so.
"""


def ask_qwen(user_message: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "options": {
            "temperature": 0.2,
            "num_predict": 120,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=90,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Check that Ollama is running."
        ) from exc
