# openai_client.py
import websocket
import json
import os
from dotenv import load_dotenv
import threading
import librosa
import soundfile as sf
import io
import sunau
import aifc
import time

import base64

def validate_transcription_session_created(self):
    """
    Checks and validates if the latest received message is a 'transcription_session.created' event.
    
    Returns:
        dict: Dictionary containing validation results and session data if successful, None otherwise
    """
    created_event = self.latest_received_message
    
    if not created_event or created_event.get("type") != "transcription_session.created":
        print("Error: No valid transcription_session.created event found in latest message")
        print(f"Latest message type: {created_event.get('type') if created_event else 'None'}")
        return None
    
    print("\n--- Validating transcription_session.created event ---")
    
    # Validate type
    event_type = created_event.get("type")
    print(f"Type: {event_type}")
    if event_type != "transcription_session.created":
        print(f"Validation failed: Expected type to be 'transcription_session.created', got '{event_type}'")
        return None
    
    # Validate event_id
    event_id = created_event.get("event_id")
    print(f"Event ID: {event_id}")
    if not event_id:
        print("Validation failed: 'event_id' is missing")
        return None
    
    # Validate session
    session = created_event.get("session", {})
    session_id = session.get("id")
    print(f"Session ID: {session_id}")
    if not session_id:
        print("Validation failed: 'session.id' is missing")
        return None
    
    # Save the session ID
    self.global_session_id = session_id
    
    # Output session details
    print("\nTranscription Session Details:")
    print(f"- Object: {session.get('object')}")
    print(f"- Expires at: {session.get('expires_at')}")
    print(f"- Audio format: {session.get('input_audio_format')}")
    print(f"- Turn detection: {session.get('turn_detection')}")
    
    return {
        "type": event_type,
        "event_id": event_id,
        "session_id": session_id,
        "expires_at": session.get("expires_at"),
        "session": session
    }

def send_transcription_session_update_and_validate(self, timeout=10, custom_payload=None):
    """
    Sends a 'transcription_session.update' event to the WebSocket exactly as provided
    and validates the 'transcription_session.updated' response.
    
    Args:
        timeout (int): Maximum time to wait for response in seconds
        custom_payload (dict, optional): Custom payload to send. If None, uses payload from file.
        
    Returns:
        dict: Dictionary with event_id and session_id on success, None on failure
    """
    if not self.is_connected:
        print("Error: Not connected to WebSocket. Cannot send update.")
        return None

    # Step 1: Use custom payload if provided, otherwise load from file
    if custom_payload:
        update_payload = custom_payload
        print("Using provided custom payload")
    else:
        # Load from file in project root directory
        project_root = os.getcwd()
        data_file_path = os.path.join(project_root, "data", "session_events", "transcription_session_update.json")
        
        try:
            with open(data_file_path, 'r') as f:
                update_payload = json.load(f)
                print(f"Successfully loaded payload from {data_file_path}")
                print("Using payload EXACTLY as defined in file with NO MODIFICATIONS")
        except FileNotFoundError:
            print(f"Error: Transcription session update file not found at {data_file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in transcription session update file at {data_file_path}")
            return None

    # Step 2: Send payload without modifications
    try:
        json_payload = json.dumps(update_payload)
        self.ws.send(json_payload)
        print("\n--- Sent 'transcription_session.update' event ---")
        print(json_payload)
    except Exception as e:
        print(f"Failed to send 'transcription_session.update' event: {e}")
        return None
    
    # Step 3: Wait for and validate response
    print(f"Waiting for 'transcription_session.updated' event (timeout: {timeout}s)...")
    
    start_time = time.time()
    response_data = None
    
    while (time.time() - start_time) < timeout:
        if self.latest_received_message:
            msg = self.latest_received_message
            self.latest_received_message = None  # Clear message
            
            if msg.get("type") == "transcription_session.updated":
                response_data = msg
                break
                
            print(f"Received message type '{msg.get('type')}', waiting for 'transcription_session.updated'")
        time.sleep(0.1)  # Short delay between checks
    
    # Step 4: Validate the response
    if not response_data:
        print(f"Error: Did not receive 'transcription_session.updated' response within {timeout} seconds")
        return None
    
    print("\n--- Validating transcription_session.updated response ---")
    
    # Validate type
    response_type = response_data.get("type")
    print(f"Type: {response_type}")
    if response_type != "transcription_session.updated":
        print(f"Validation failed: Expected type to be 'transcription_session.updated', got '{response_type}'")
        return None
    
    # Validate event_id
    response_event_id = response_data.get("event_id")
    print(f"Event ID: {response_event_id}")
    if not response_event_id:
        print("Validation failed: 'event_id' is missing.")
        return None
    
    # Validate session
    session = response_data.get("session", {})
    response_session_id = session.get("id")
    print(f"Session ID: {response_session_id}")
    if not response_session_id:
        print("Validation failed: 'session.id' is missing.")
        return None
    
    # Validate session consistency
    if hasattr(self, 'global_session_id') and self.global_session_id:
        if response_session_id != self.global_session_id:
            print(f"Warning: Session ID changed from {self.global_session_id} to {response_session_id}")
        else:
            print(f"Validation successful: Session ID remained consistent: {response_session_id}")
    
    # Update the global session ID
    self.global_session_id = response_session_id
    
    # Output updated session settings
    print("\nTranscription Session Updated Successfully!")
    print("Updated settings:")
    print(f"- Input audio format: {session.get('input_audio_format')}")
    print(f"- Transcription model: {session.get('input_audio_transcription', {}).get('model')}")
    print(f"- Noise reduction: {session.get('input_audio_noise_reduction', {}).get('type')}")
    print(f"- Turn detection threshold: {session.get('turn_detection', {}).get('threshold')}")
    print(f"- Turn detection silence duration: {session.get('turn_detection', {}).get('silence_duration_ms')} ms")
    print(f"- Session expires at: {session.get('expires_at')}")
    
    # Return the key information
    return {
        "event_id": response_event_id,
        "session_id": response_session_id,
        "type": response_type,
        "expires_at": session.get("expires_at"),
        "session": session
    }