import os
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.cloud import aiplatform
from openai import OpenAI
import warnings

# Suppress google API FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

# --- Explicit Configuration ---
ENDPOINT_ID = "mg-endpoint-57d42eb6-4be2-4509-bfe3-73e0dd3d0d6f"
PROJECT_ID = "607968988315"
INPUT_DATA_FILE = "keys/knowledgeontology-1c9b2932ef2d.json" 
LOCATION = "us-central1"

def get_vertex_token(cred_file: str) -> str:
    SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
    credentials = service_account.Credentials.from_service_account_file(
        cred_file, 
        scopes=SCOPES
    )
    credentials.refresh(Request())
    return credentials.token

def test_openai_compatible_endpoint():
    print(f"--- Generating Vertex Bearer Token ---")
    
    # Resolve absolute path for credential file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cred_path = os.path.join(script_dir, "..", INPUT_DATA_FILE)
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
    token = get_vertex_token(cred_path)
    
    print(f"--- Resolving Dedicated DNS for Vertex AI Endpoint ---")
    endpoint_path = f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}"
    endpoint = aiplatform.Endpoint(endpoint_path)
    
    if not endpoint.dedicated_endpoint_dns:
        print("ERROR: No dedicated endpoint DNS found for this endpoint. OpenAI compatibility requires a dedicated DNS.")
        return
        
    dedicated_dns = endpoint.dedicated_endpoint_dns
    print(f"Resolved Dedicated DNS: {dedicated_dns}")
    
    # Construct standard Vertex Native Base URL
    base_url = f"https://{dedicated_dns}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"
    print(f"\n--- Connecting to Native Vertex Endpoint ---")
    print(f"Endpoint: {base_url}")
    
    print("\n--- Sending Predict Request ---")
    
    import urllib.request
    import json
    
    payload = {
        "instances": [
            {"prompt": "Write a short poem about a cat sitting on a fence."}
        ],
        "parameters": {
            "temperature": 0.7,
            "maxOutputTokens": 512
        }
    }
    
    data = json.dumps(payload).encode('utf-8')
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {token}'
    }
    
    req = urllib.request.Request(base_url, data=data, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            print("\n--- Success! ---")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"\nRequest failed: {e}")

if __name__ == "__main__":
    test_openai_compatible_endpoint()