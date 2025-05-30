# openai_client.py
import websocket
import json
import os
from dotenv import load_dotenv
import threading
import time
import librosa
import soundfile as sf
import os
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

    def on_message(self, ws, message):
        """
        Callback function to handle incoming WebSocket messages.
        """
        print("\n--- Received message ---")
        print(message)

        try:
            data = json.loads(message)
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
        Returns True on success, False on failure or timeout.
        """
        if not self.is_connected:
            print("Error: Not connected to WebSocket. Cannot send update.")
            return False

        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Updated path to include 'session_events' subdirectory
        data_file_path = os.path.join(script_dir, '..', 'data', 'session_events', 'session_update.json')

        update_payload = {}
        try:
            # Load the payload from the JSON file
            with open(data_file_path, 'r') as f:
                update_payload = json.load(f)
            json_payload = json.dumps(update_payload)
            self.ws.send(json_payload)
            print(f"\n--- Sent 'session.update' event ---")
            print(json_payload)
        except Exception as e:
            print(f"Failed to send 'session.update' event: {e}")
            return False

        print(f"Waiting for 'session.updated' event (timeout: {timeout}s)...")
        if not self.session_updated_event.wait(timeout=timeout):
            print("Error: 'session.updated' event not received in time.")
            return False
        
        if not self.is_connected: # Check if connection is still active after receiving event
            print("Error: Connection closed after receiving 'session.updated'.")
            return False

        print("Successfully sent 'session.update' and received 'session.updated'.")
        return True

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

def process_audio_to_base64(input_file, output_dir=None):
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
def get_audio_base64_from_data_folder(audio_filename, save_processed_files=False):
    # ... (same code as before) ...
    project_root = os.getcwd()
    data_folder_path = os.path.join(project_root, "data", "audio")
    input_audio_path = os.path.join(data_folder_path, audio_filename)

    output_sub_dir = None
    if save_processed_files:
        output_sub_dir = os.path.join(data_folder_path, "audio")
        os.makedirs(output_sub_dir, exist_ok=True) 

    base64_string = process_audio_to_base64(input_audio_path, output_dir=output_sub_dir if save_processed_files else None)
    
    if base64_string:
        print(f"Successfully obtained Base64 for {audio_filename}.")
    else:
        print(f"Failed to obtain Base64 for {audio_filename}.")
        
    return base64_string

def get_audio_base64_from_file(input_file_path):
    """
    Converts any audio file to PCM INT16 format (16kHz, mono) in memory,
    and then returns the Base64 encoded string of this PCM INT16 audio.

    Args:
        input_file_path (str): Path to the input audio file (e.g., .mp3, .wav, .ogg).

    Returns:
        str or None: The Base64 encoded string of the PCM INT16 audio,
                     or None if an error occurs (e.g., file not found, audio processing error).
    """
    try:
        # Step 1: Load the audio file and convert to 16kHz, mono
        print(f"Loading and resampling '{os.path.basename(input_file_path)}' to 16kHz, mono...")
        audio_data, sample_rate = librosa.load(input_file_path, sr=16000, mono=True)
        print("Audio loaded and resampled.")

        # Step 2: Convert audio data to PCM INT16 bytes in memory
        # Use a BytesIO object to simulate a file in memory
        buffer = io.BytesIO()
        # Write the audio data to the in-memory buffer as PCM_16 WAV
        sf.write(buffer, audio_data, sample_rate, subtype='PCM_16', format='WAV')
        buffer.seek(0) # Rewind the buffer to the beginning

        # Step 3: Read the bytes from the buffer and Base64 encode them
        wav_bytes = buffer.read()
        base64_encoded_string = base64.b64encode(wav_bytes).decode('utf-8')
        
        print(f"Successfully converted '{os.path.basename(input_file_path)}' to Base64 (PCM INT16).")
        return base64_encoded_string

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file_path}'.")
        return None
    except Exception as e:
        print(f"An error occurred during audio processing or Base64 encoding: {str(e)}")
        return None

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