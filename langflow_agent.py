import requests
import os
import uuid

api_key = 'sk-t3z-YZP1ub96i_zKh1QDhPWUZMjmUY3SWtZskudL_5Q'
url = "http://localhost:3000/api/v1/run/8381aa52-84d3-43f0-99b0-b72e1d331324"  # The complete API endpoint URL for this flow

# Request payload configuration
payload = {
    "output_type": "chat",
    "input_type": "chat",
    "input_value": "How to make a sandwich?"
}
payload["session_id"] = str(uuid.uuid4())

headers = {"x-api-key": api_key}

try:
    # Send API request
    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes

    # Print response
    print(response.json()['outputs'][0]['outputs'][0]['outputs']['message']['message'])

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")