# test_websocket_flow.py
import pytest
import sys
import os
import aifc
import sunau
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.openai_client import OpenAIRealtimeClient, URL, HEADERS
import time
import traceback

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
    event_id = client.send_session_update_and_wait_for_updated(timeout=10)
    assert event_id, "Failed to send session.update or receive session.updated event."
    assert client.is_connected, "Client disconnected after session.updated."
    print(f"Fetched event_id: {event_id}")

    # #Step 3: Generate base64 audio from a file
    audio_filename = "largest_planet.wav"  # Place your file in data/audio/
    base64_audio = client.get_audio_base64_from_data_folder(audio_filename, save_processed_files=True)
    assert base64_audio is not None, "Failed to generate base64 string from audio file"
    print("Base64 audio string (first 100 chars):", base64_audio[:100])


    # Step 4: Send input_audio_buffer.append event and wait for input_audio_buffer.speech_started
    result = client.send_audio_buffer_and_validate_speech_started(event_id, base64_audio)
    if not result:
        print("Latest received message:", getattr(client, "latest_received_message", None))
        time.sleep(5)  # Give the server a moment to respond, if needed
        print("Latest received message after wait:", getattr(client, "latest_received_message", None))

#    # Step 5: send input_audio_buffer.commit_and_validate conversation.item.input_audio_transcription.completed event
#     # ...existing code...
#     commit_responses = client.send_audio_buffer_commit_and_validate(event_id, timeout=10)
#     if not commit_responses:
#         print("Latest received message:", getattr(client, "latest_received_message", None))
#         time.sleep(5)  # Give the server a moment to respond, if needed
#         print("Latest received message after wait:", getattr(client, "latest_received_message", None))
#     else:
#         # Store the transcript from the completed transcription event
#         transcript = commit_responses.get("conversation.item.input_audio_transcription.completed", {}).get("transcript")
#         print("Final transcript:",  )
# # ...existing code...

#       # Step 6: send response.create and validate the response events
#     response_data = client.send_response_create_and_validate(event_id, transcript)
#     if response_data:
#      audio_base64 = response_data["combined_audio_delta"]
#     # Do something with audio_base64

    commit_responses = client.send_audio_buffer_commit_and_validate(event_id, timeout=10)
    if not commit_responses:
        print("Latest received message:", getattr(client, "latest_received_message", None))
        time.sleep(5)
        print("Latest received message after wait:", getattr(client, "latest_received_message", None))
        return  # Exit if we don't get commit responses

    # Log the transcript completion
    transcript = commit_responses.get("conversation.item.input_audio_transcription.completed", {}).get("transcript")
    if transcript:
        print(f"\n--- Transcript received successfully ---")
        print(f"Transcript: {transcript}")

    # Step 6: Send response.create and log each response as it arrives
    print("\n--- Sending response.create ---")
    create_response_event_id = f"event_{int(time.time())}"
    response_data = client.send_response_create_and_validate(create_response_event_id, transcript)  # Add debug flag
        
    if response_data:
        print("\n--- Response.create Summary ---")
        print(f"Total delta chunks received: {len(response_data.get('response.audio.delta', []))}")
        print(f"Combined audio length: {len(response_data.get('combined_audio_delta', ''))}")
                
                # Save the audio for further processing
        audio_base64 = response_data.get("combined_audio_delta")
        if audio_base64:
            print("Successfully received complete audio response")
        else:
            print("No audio data received in response")
