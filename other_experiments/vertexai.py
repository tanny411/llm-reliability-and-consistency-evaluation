## NOT TESTED YET ##

import requests
import google.auth
import google.auth.transport.requests
from google.oauth2 import service_account

# Define the required SCOPES
SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
# Replace 'NAME_OF_FILE' with your service account JSON file name that you downloaded
SERVICE_ACCOUNT_FILE = 'NAME_OF_FILE.json'
# Load credentials from the service account file with the specified SCOPES
cred = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
# Create an authentication request
auth_req = google.auth.transport.requests.Request()
# Refresh the credentials
cred.refresh(auth_req)
# Obtain the bearer token
bearer_token = cred.token

# Define the base URL for your specific region (us-central1 in this example)
base_url = "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/us-central1/endpoints/{endpoint_id}:predict"
# Replace 'awesome-dogfish-399811' with your GCP project ID
project_id = "awesome-dogfish-399811"
endpoint_id = "8471563369176891392"


# Define the request body for your specific prompt and parameters
request_body = {
    "instances": [
        {
            "prompt": "Write a poem about Valencia.",
            "max_length": 10,
            "top_k": 10
        }
    ]
}

# Create the full URL using the project and endpoint IDs
full_url = base_url.format(project_id=project_id, endpoint_id=endpoint_id)

headers = {
    "Authorization": "Bearer {bearer_token}".format(bearer_token=bearer_token),
    "Content-Type": "application/json"
}

# Send a POST request to the model endpoint
resp = requests.post(full_url, json=request_body, headers=headers)

# Print the response from the model
print(resp.json())