# tests/test_websocket_flow.py
import pytest
import os
import json
import numpy as np
import soundfile as sf # Make sure sf is imported if used in setup_audio_and_json_data

from src.openai_client import OpenAIRealtimeClient, URL, HEADERS
from src.audio_processor import get_audio_base64_from_data_folder, update_json_with_audio_data


@pytest.fixture(scope="module")
def openai_realtime_client_instance():
    """Provides an OpenAIRealtimeClient instance for the test module."""
    print("\n--- Fixture Setup: Initializing OpenAIRealtimeClient ---")
    client = OpenAIRealtimeClient(URL, HEADERS)
    yield client
    print("\n--- Fixture Teardown: Closing WebSocket connection ---")
    client.close_connection()
    print("Module-level cleanup complete.")


@pytest.fixture(scope="function")
def connected_openai_client(openai_realtime_client_instance):
    """Connects the OpenAIRealtimeClient and waits for session.created."""
    client = openai_realtime_client_instance
    print("\n--- Fixture Setup: Connecting to WebSocket ---")
    
    connected_and_created = client.connect_and_wait_for_session_created(timeout=15)
    
    assert connected_and_created, "Fixture Failed: Failed to connect or receive session.created."
    assert client.is_connected, "Fixture Failed: Client is not connected after session.created."
    assert client.global_session_id is not None, "Fixture Failed: session.id was not captured."
    
    print(f"Fixture: Client connected. Session ID: {client.global_session_id}")
    
    yield client 

    # Resetting events for safety for next test if any
    client.connection_open_event.clear()
    client.session_created_event.clear()
    client.session_updated_event.clear()
    client.speech_started_event.clear()
    client.close_event.clear()


@pytest.fixture(scope="module")
def setup_audio_and_json_data():
    """Sets up 'data' folder with dummy audio and session_events.json for testing."""
    data_folder = "data"
    processed_audio_folder = os.path.join(data_folder, "processed_audio")
    session_events_json_path = os.path.join(data_folder, "session_events.json")
    dummy_audio_filename = "test_audio_for_append.wav"
    dummy_wav_path = os.path.join(data_folder, dummy_audio_filename)

    os.makedirs(data_folder, exist_ok=True)

    # Create dummy WAV file
    try:
        samplerate = 16000 # Hz
        duration = 2 # seconds
        frequency = 440 # Hz
        t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
        amplitude = 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        sf.write(dummy_wav_path, data, samplerate, subtype='PCM_16')
        print(f"\nFixture Setup: Dummy audio created: {dummy_wav_path}")
    except ImportError:
        pytest.skip("NumPy not found, skipping audio setup. Please install numpy or provide a real audio file.")
    except Exception as e:
        pytest.fail(f"Fixture Setup: Failed to create dummy audio file: {e}")

    # Create initial session_events.json
    initial_json_content = {
      "event_id": "initial_event_id_placeholder",
      "type": "input_audio_buffer.append",
      "audio": ""
    }
    with open(session_events_json_path, 'w') as f:
        json.dump(initial_json_content, f, indent=2)
    print(f"Fixture Setup: Initial JSON created: {session_events_json_path}")

    yield dummy_audio_filename # Provide filename to the test

    # Teardown: Clean up created files and folders
    print("\n--- Fixture Teardown: Cleaning up audio and JSON data ---")
    if os.path.exists(dummy_wav_path):
        os.remove(dummy_wav_path)
    if os.path.exists(session_events_json_path):
        os.remove(session_events_json_path)
    if os.path.exists(processed_audio_folder):
        for f in os.listdir(processed_audio_folder):
            os.remove(os.path.join(processed_audio_folder, f))
        os.rmdir(processed_audio_folder)
    if not os.listdir(data_folder): 
        os.rmdir(data_folder)
    print("Audio and JSON data cleanup complete.")


def test_full_websocket_audio_flow(connected_openai_client, setup_audio_and_json_data):
    """
    Tests the full sequence:
    1. Connect and receive session.created (handled by fixture)
    2. Send session.update and receive session.updated
    3. Prepare audio data & update local JSON with event_id from session.updated
    4. Send input_audio_buffer.append event
    5. Validate input_audio_buffer.speech_started response
    """
    client = connected_openai_client 
    audio_filename = setup_audio_and_json_data 

    print("\n--- Test Phase: Sending session.update ---")
    updated_success = client.send_session_update_and_wait_for_updated(timeout=10)
    assert updated_success, "Failed to send session.update or receive session.updated."
    
    # event_id is updated by the on_message handler in OpenAIRealtimeClient
    event_id_for_audio_append = client.global_event_id 
    print(f"Session updated. Event ID captured: {event_id_for_audio_append}")

    print("\n--- Test Phase: Preparing audio and updating local JSON ---")
    json_updated_successfully = update_json_with_audio_data(audio_filename, event_id_for_audio_append)
    assert json_updated_successfully, "Failed to update local JSON file."

    # Read the updated JSON to get the full payload for sending
    project_root = os.getcwd()
    data_folder_path = os.path.join(project_root, "data")
    json_file_path = os.path.join(data_folder_path, "session_events.json")
    with open(json_file_path, 'r') as f:
        audio_append_payload_from_json = json.load(f)
    
    base64_audio_from_json = audio_append_payload_from_json["audio"]
    event_id_to_send_with_audio = audio_append_payload_from_json["event_id"] 

    print("\n--- Test Phase: Sending audio append event ---")
    audio_append_success = client.send_audio_append_event(
        event_id_to_send_with_audio, 
        base64_audio_from_json, 
        timeout=15 
    )
    assert audio_append_success, "Failed to send audio append or receive speech_started event."

    print("\n--- Test Completed Successfully ---")