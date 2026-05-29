import requests
import json
import sys

MODEL = "llama3.2"

def start_chat():
    print("Starting chat with model:", MODEL)
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'bye':
            print("Ending chat.")
            break

        payload = {
            "model": "llama3.2:3b",
            "prompt": user_input,
            "stream": False,
        }

        response = requests.post("http://localhost:11434/api/generate", json=payload)

        data = response.json()
        print("\nResponse from LLaMA 3.2:")
        print(data["response"])


if __name__ == "__main__":
    start_chat()          
