import pytest
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.openai_client import OpenAIRealtimeClient, HEADERS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix the typo in the variable name
TRASCRIPTION_URL = os.getenv("TRASCRIPTION_URL")
if not TRASCRIPTION_URL:
    raise ValueError("Environment variable TRASCRIPTION_URL is not set")

# A pytest fixture to provide a client instance and ensure cleanup
@pytest.fixture(scope="module")
def openai_realtime_client():
    """
    Fixture to provide an OpenAIRealtimeClient instance for tests.
    Ensures the connection is closed after tests in the module are done.
    """
    client = OpenAIRealtimeClient(TRASCRIPTION_URL, HEADERS)
    yield client
    # Teardown: Close the connection after the test(s) complete
    print("\n--- Fixture Teardown: Closing WebSocket connection ---")
    client.close_connection()

def test_transcription_session_flow(openai_realtime_client):
    """
    Tests the sequence of WebSocket connection, transcription session creation,
    sending an update, and validating the updated response.
    """
    client = openai_realtime_client

    # Step 1: Connect to WebSocket - using the transcription-specific method
    print("\n--- Test Step 1: Connecting to WebSocket ---")
    connected = client.connect_and_wait_for_transcription_session_created(timeout=15)
    assert connected, "Failed to connect to WebSocket"
    assert client.is_connected, "Client is not connected"

    # Step 2: Validate the response (which should already be in latest_received_message)
    print("\n--- Test Step 2: Validating transcription_session.created details ---")
    validation_result = client.validate_transcription_session_created()
    assert validation_result, "Failed to validate transcription_session.created response"
    
    # Print the validation results
    print("\n=== Validation Results ===")
    print(f"Type: {validation_result['type']}")
    print(f"Event ID: {validation_result['event_id']}")
    print(f"Session ID: {validation_result['session_id']}")