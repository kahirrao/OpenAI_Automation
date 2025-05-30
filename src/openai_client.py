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

import base64

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it or replace the placeholder.")

URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
HEADERS = [
    f"Authorization: Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta: realtime=v1"
]

class OpenAIRealtimeClient:
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
        self.ws = None
        self.global_event_id = None       # Stores event_id from session.created
        self.global_session_id = None     # Stores session.id from session.created/updated
        self.is_connected = False
        
        # Events for synchronization
        self.connection_open_event = threading.Event()
        self.session_created_event = threading.Event()
        self.session_updated_event = threading.Event()
        self.close_event = threading.Event() # For signaling final closure

        self._ws_thread = None # To hold the WebSocket thread
        self.latest_received_message = None

    def on_message(self, ws, message):
        """
        Callback function to handle incoming WebSocket messages.
        """
        print("\n--- Received message ---")
        print(message)

        try:
            data = json.loads(message)
            self.latest_received_message = data  # <-- Add this line
            msg_type = data.get("type")
            
            if msg_type == "session.created":
                print("Processing 'session.created' event.")
                expected_type = "session.created"
                if data.get("type") == expected_type:
                    print(f"Validation successful: 'type' is '{expected_type}'")
                else:
                    print(f"Validation failed: Expected 'type' to be '{expected_type}', but got '{data.get('type')}'")

                event_id_from_msg = data.get("event_id")
                if event_id_from_msg:
                    self.global_event_id = event_id_from_msg
                    print(f"Validation successful: 'event_id' is present and assigned to self.global_event_id: '{self.global_event_id}'")
                else:
                    print("Validation failed: 'event_id' is missing.")

                session_id_from_msg = data.get("session", {}).get("id")
                if session_id_from_msg:
                    self.global_session_id = session_id_from_msg
                    print(f"Validation successful: 'session.id' is present and assigned to self.global_session_id: '{self.global_session_id}'")
                else:
                    print("Validation failed: 'session.id' is missing.")
                
                self.session_created_event.set() # Signal that session.created was received
                
            elif msg_type == "session.updated":
                print("Processing 'session.updated' event.")
                expected_type = "session.updated"
                if data.get("type") == expected_type:
                    print(f"Validation successful: 'type' is '{expected_type}'")
                else:
                    print(f"Validation failed: Expected 'type' to be '{expected_type}', but got '{data.get('type')}'")

                event_id_from_msg = data.get("event_id")
                if event_id_from_msg:
                    print(f"Validation successful: 'event_id' is present: '{event_id_from_msg}'")
                else:
                    print("Validation failed: 'event_id' is missing for session.updated.")

                session_id_from_msg = data.get("session", {}).get("id")
                if session_id_from_msg:
                    if self.global_session_id and self.global_session_id != session_id_from_msg:
                        print(f"Warning: session.id in session.updated ({session_id_from_msg}) differs from initial session.id ({self.global_session_id})")
                    else:
                        print(f"Validation successful: 'session.id' is present and consistent: '{session_id_from_msg}'")
                    self.global_session_id = session_id_from_msg 
                else:
                    print("Validation failed: 'session.id' is missing for session.updated.")
                
                self.session_updated_event.set() # Signal that session.updated was received

            else:
                print(f"Received unhandled message type: {msg_type}")

            print(f"\nCurrent self.global_event_id: {self.global_event_id}")
            print(f"Current self.global_session_id: {self.global_session_id}\n")

        except json.JSONDecodeError:
            print("Error: Could not decode JSON from message.")
        except Exception as e:
            print(f"An unexpected error occurred during message processing: {e}")

    def on_error(self, ws, error):
        """
        Callback function to handle WebSocket errors.
        """
        print(f"Error: {error}")
        self.is_connected = False
        self.connection_open_event.set() # Unblock if waiting for connection
        self.session_created_event.set() # Unblock if waiting for session.created
        self.session_updated_event.set() # Unblock if waiting for session.updated
        self.close_event.set() # Signal final closure on error

    def on_close(self, ws, close_status_code, close_msg):
        """
        Callback function when the WebSocket connection is closed.
        """
        print(f"Connection closed. Status code: {close_status_code}, Message: {close_msg}")
        self.is_connected = False
        self.connection_open_event.set() # Unblock if waiting for connection
        self.session_created_event.set() # Unblock if waiting for session.created
        self.session_updated_event.set() # Unblock if waiting for session.updated
        self.close_event.set() # Signal final closure
        print(f"Final self.global_event_id at close: {self.global_event_id}")
        print(f"Final self.global_session_id at close: {self.global_session_id}")

    def on_open(self, ws):
        """
        Callback function when the WebSocket connection is opened.
        """
        print("Connection opened.")
        self.is_connected = True
        self.connection_open_event.set() # Signal that the connection is open
        print(f"Instance variables at open: event_id={self.global_event_id}, session_id={self.global_session_id}")

    def _run_websocket(self):
        """Internal method to run the WebSocket in a separate thread."""
        self.ws.run_forever()
        print("WebSocket thread finished.")

    def connect_and_wait_for_session_created(self, timeout=10):
        """
        Establishes the WebSocket connection and waits for the session.created event.
        Returns True on success, False on failure or timeout.
        """
        print(f"Attempting to connect to: {self.url}")
        self.ws = websocket.WebSocketApp(
            self.url,
            header=self.headers,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        
        # Start the WebSocket in a separate daemon thread
        # A daemon thread will exit automatically when the main program exits
        self._ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self._ws_thread.start()

        print(f"Waiting for connection to open (timeout: {timeout}s)...")
        if not self.connection_open_event.wait(timeout=timeout):
            print("Error: Connection did not open in time.")
            self.close_connection()
            return False

        print(f"Waiting for 'session.created' event (timeout: {timeout}s)...")
        if not self.session_created_event.wait(timeout=timeout):
            print("Error: 'session.created' event not received in time.")
            self.close_connection()
            return False
        
        if not self.is_connected: # Check if connection is still active after receiving event
            print("Error: Connection closed after receiving 'session.created'.")
            return False

        print("Successfully connected and received 'session.created'.")
        return True

    def send_session_update_and_wait_for_updated(self, timeout=10):
            """
            Constructs and sends the session.update event to the WebSocket,
            then waits for the session.updated response.
            Returns event_id on success, None on failure or timeout.
            """
            if not self.is_connected:
                print("Error: Not connected to WebSocket. Cannot send update.")
                return None

            script_dir = os.path.dirname(os.path.abspath(__file__))
            data_file_path = os.path.join(script_dir, '..', 'data', 'session_events', 'session_update.json')

            update_payload = {}
            try:
                with open(data_file_path, 'r') as f:
                    update_payload = json.load(f)
                json_payload = json.dumps(update_payload)
                self.ws.send(json_payload)
                print(f"\n--- Sent 'session.update' event ---")
                print(json_payload)
            except Exception as e:
                print(f"Failed to send 'session.update' event: {e}")
                return None

            print(f"Waiting for 'session.updated' event (timeout: {timeout}s)...")
            if not self.session_updated_event.wait(timeout=timeout):
                print("Error: 'session.updated' event not received in time.")
                return None
            if not self.is_connected:
                print("Error: Connection closed after receiving 'session.updated'.")
                return None
            # Extract event_id from the latest received message
            event_id = None
            if hasattr(self, "latest_received_message") and self.latest_received_message:
                event_id = self.latest_received_message.get("event_id")
            if event_id:
                print(f"Successfully sent 'session.update' and received 'session.updated'. Event ID: {event_id}")
                return event_id
            else:
                print("session.updated received but event_id not found in latest_received_message.")
                return None

    def close_connection(self):
        """
        Closes the WebSocket connection and waits for the thread to finish.
        """
        if self.ws and self.is_connected:
            print("Closing WebSocket connection...")
            self.ws.close()
            # Wait for on_close to execute and set close_event
            if not self.close_event.wait(timeout=5):
                print("Warning: WebSocket close may not have completed gracefully.")
        elif self.ws:
            print("WebSocket already closed or not connected.")
        else:
            print("No WebSocket object to close.")

        if self._ws_thread and self._ws_thread.is_alive():
            print("Waiting for WebSocket thread to join...")
            self._ws_thread.join(timeout=5) # Wait for the thread to finish
            if self._ws_thread.is_alive():
                print("Warning: WebSocket thread did not terminate gracefully.")
        print("Connection cleanup complete.")

    def process_audio_to_base64(self, input_file, output_dir=None):
    # ... (same code as before) ...
        try:
            # --- Step 1: Convert to PCM INT16 WAV in memory ---
            audio, sr = librosa.load(input_file, sr=16000, mono=True)
            base_filename = os.path.splitext(os.path.basename(input_file))[0]  
            if output_dir:
                os.makedirs(output_dir, exist_ok=True) 
                output_pcm16_path = os.path.join(output_dir, f"{base_filename}_pcm16.wav")
                output_base64_path = os.path.join(output_dir, f"{base_filename}_base64.txt")
            else:
                input_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else os.getcwd()
                output_pcm16_path = os.path.join(input_dir, f"{base_filename}_pcm16.wav")
                output_base64_path = os.path.join(input_dir, f"{base_filename}_base64.txt")
            sf.write(output_pcm16_path, audio, sr, subtype='PCM_16')

            # --- Step 2: Convert PCM16 WAV file to Base64 encoded string ---
            with open(output_pcm16_path, 'rb') as file:
                wav_bytes = file.read()
                base64_encoded = base64.b64encode(wav_bytes).decode('utf-8')
            # --- Step 3: Save Base64 to file (optional) ---
            if output_dir: 
                with open(output_base64_path, 'w') as f:
                    f.write(base64_encoded)
            return base64_encoded

        except FileNotFoundError:
                print(f"Error: Input file not found at {input_file}")
                return None
        except Exception as e:
                print(f"An unexpected error occurred during audio processing: {str(e)}")
                return None

# --- Existing function: get_audio_base64_from_data_folder ---
    def get_audio_base64_from_data_folder(self, audio_filename, save_processed_files=False):
        # ... (same code as before) ...
        project_root = os.getcwd()
        data_folder_path = os.path.join(project_root, "data", "audio")
        input_audio_path = os.path.join(data_folder_path, audio_filename)

        output_sub_dir = None
        if save_processed_files:
            output_sub_dir = os.path.join(data_folder_path, "audio")
            os.makedirs(output_sub_dir, exist_ok=True) 

        base64_string = self.process_audio_to_base64(input_audio_path, output_dir=output_sub_dir if save_processed_files else None)
        
        if base64_string:
            print(f"Successfully obtained Base64 for {audio_filename}.")
        else:
            print(f"Failed to obtain Base64 for {audio_filename}.")
        return base64_string


    def send_audio_buffer_and_validate_speech_started(self, event_id, audio_data_base64):    
        """
        Constructs and sends an 'input_audio_buffer.append' event to the WebSocket.
        Immediately attempts to validate the last received message as 'speech_started'
        and stores key data globally and in the instance.

        Returns True on success, False on failure.
        """
        if not self.is_connected:
            print("Error: Not connected to WebSocket. Cannot send audio buffer.")
            return False

        request_body = {
            "event_id": event_id,
            "type": "input_audio_buffer.append",
            "audio": audio_data_base64
        }

        try:
            json_payload = json.dumps(request_body)
            self.ws.send(json_payload)
            print(f"\n--- Sent 'input_audio_buffer.append' event ---")
            # For display, truncate the audio data
            display_payload = request_body.copy()
            display_payload["audio"] = display_payload["audio"][:100] + "..." if len(display_payload["audio"]) > 100 else display_payload["audio"]
            print(json.dumps(display_payload, indent=2))
        except Exception as e:
            print(f"Failed to send 'input_audio_buffer.append' event: {e}")
            return False

        # Directly attempt to validate the 'latest_received_message'
        # This assumes your _simulate_server_response (or real on_message)
        # has already processed and stored the incoming message.
        if self.latest_received_message and self.latest_received_message.get("type") == "input_audio_buffer.speech_started":
            response_data = self.latest_received_message
            self.latest_received_message = None # Clear after processing

            required_keys = ["type", "event_id", "item_id"]

            # 1. Validate that all required keys are present
            for key in required_keys:
                if key not in response_data:
                    print(f"Validation Error: Missing key '{key}' in response.")
                    return False

            # 2. Validate data types
            if not isinstance(response_data["type"], str):
                print(f"Validation Error: 'type' is not a string.")
                return False
            if not isinstance(response_data["event_id"], str):
                print(f"Validation Error: 'event_id' is not a string.")
                return False
            if not isinstance(response_data["item_id"], str):
                print(f"Validation Error: 'item_id' is not a string.")
                return False

            # 3. Store in global variables
            global GLOBAL_RESPONSE_TYPE
            global GLOBAL_RESPONSE_EVENT_ID
            global GLOBAL_RESPONSE_ITEM_ID
            GLOBAL_RESPONSE_TYPE = response_data["type"]
            GLOBAL_RESPONSE_EVENT_ID = response_data["event_id"]
            GLOBAL_RESPONSE_ITEM_ID = response_data["item_id"]

            # 4. Store in class instance (recommended)
            self.audio_buffer_response_data = response_data

            print(f"Successfully validated and stored 'input_audio_buffer.speech_started' data.")
            print(f"  GLOBAL_RESPONSE_TYPE: {GLOBAL_RESPONSE_TYPE}")
            print(f"  GLOBAL_RESPONSE_EVENT_ID: {GLOBAL_RESPONSE_EVENT_ID}")
            print(f"  GLOBAL_RESPONSE_ITEM_ID: {GLOBAL_RESPONSE_ITEM_ID}")
            return True
        else:
            print("Error: Expected 'input_audio_buffer.speech_started' response not found or invalid.")
            return False

# Example of how you might run it directly (for testing without pytest)
if __name__ == "__main__":
    client = OpenAIRealtimeClient(URL, HEADERS)
    try:
        if client.connect_and_wait_for_session_created():
            if client.send_session_update_and_wait_for_updated():
                print("\nFull flow completed successfully!")
            else:
                print("\nSession update failed.")
        else:
            print("\nInitial connection failed.")
    finally:
        client.close_connection()
    print("Script finished.")