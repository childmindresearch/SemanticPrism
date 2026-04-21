import os
import sys

# Ensure src is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.llm_client import SemanticLLMClient
from pydantic import BaseModel

class TestResponse(BaseModel):
    message: str

def test_inference():
    print("Initializing client...")
    client = SemanticLLMClient("config.yaml")
    
    print(f"Base URL: {client.base_url}")
    print(f"Model Name: {client.model_name}")
    print(f"Token length: {len(client.api_key)}")
    
    print("\nSending inference request...")
    res = client.safe_api_call_sync(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say 'Hello, World!' and confirm you can hear me.",
        response_model=TestResponse
    )
    
    if res:
        print("\nSuccess! Received response:")
        print(res.model_dump_json(indent=2))
    else:
        print("\nFailed to get a response. Check error logs.")
        print("Error history:", client.error_history)

if __name__ == "__main__":
    test_inference()
