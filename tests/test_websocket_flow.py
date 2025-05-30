# test_websocket_flow.py
import pytest
import sys
import os
import aifc
import sunau
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.openai_client import OpenAIRealtimeClient, URL, HEADERS, get_audio_base64_from_data_folder
import time

# A pytest fixture to provide a client instance and ensure cleanup
@pytest.fixture(scope="module")
def openai_realtime_client():
    """
    Fixture to provide an OpenAIRealtimeClient instance for tests.
    Ensures the connection is closed after tests in the module are done.
    """
    client = OpenAIRealtimeClient(URL, HEADERS)
    yield client
    # # Teardown: Close the connection after the test(s) complete
    # print("\n--- Fixture Teardown: Closing WebSocket connection ---")
    # client.close_connection()

def test_websocket_session_flow(openai_realtime_client):
    """
    Tests the sequence of WebSocket connection, session creation,
    sending an update, and receiving a session update.
    """
    client = openai_realtime_client

    # Step 1: Connect to WebSocket and wait for session.created
    print("\n--- Test Step 1: Connecting and waiting for session.created ---")
    connected_and_created = client.connect_and_wait_for_session_created(timeout=15)
    assert connected_and_created, "Failed to connect or receive session.created event."
    assert client.is_connected, "Client is not connected after session.created."
    assert client.global_session_id is not None, "session.id was not captured after session.created."
    print(f"Captured Session ID: {client.global_session_id}")

    # Give a small buffer time if needed, though the events should be fast
    time.sleep(1) 

    # Step 2: Send session.update event and wait for session.updated
    print("\n--- Test Step 2: Sending session.update and waiting for session.updated ---")
    updated_success = client.send_session_update_and_wait_for_updated(timeout=10)
    assert updated_success, "Failed to send session.update or receive session.updated event."
    assert client.is_connected, "Client disconnected after session.updated."
    # You can add more specific assertions here, e.g., checking instructions in the received session object
    # For example, if you wanted to verify the instructions were updated:
    # (This would require storing the full session object in the client)
    # assert "You are a helpful assistant." in client.last_received_session_instructions 
    
    print("\n--- Test Passed: WebSocket session flow completed successfully ---")

    # Step 3: Generate base64 audio from a file
    audio_filename = "answer_5_20250324_192921_pcm16.wav"  # Place your file in data/audio/
    base64_audio = client.get_audio_base64_from_data_folder(audio_filename, save_processed_files=True)
    assert base64_audio is not None, "Failed to generate base64 string from audio file"
    print("Base64 audio string (first 100 chars):", base64_audio[:100])


