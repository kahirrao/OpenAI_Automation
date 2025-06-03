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
        if self.latest_received_message and self.latest_received_message.get("type") == "input_audio_buffer.speech_started":
            response_data = self.latest_received_message
            self.latest_received_message = None # Clear after processing

            print("Processing 'input_audio_buffer.speech_started' event.")
            expected_type = "input_audio_buffer.speech_started"
            # Validate 'type'
            if response_data.get("type") == expected_type:
                print(f"Validation successful: 'type' is '{expected_type}'")
            else:
                print(f"Validation failed: Expected 'type' to be '{expected_type}', but got '{response_data.get('type')}'")
                return False

            # Validate 'event_id'
            event_id_from_msg = response_data.get("event_id")
            if event_id_from_msg:
                print(f"Validation successful: 'event_id' is present: '{event_id_from_msg}'")
            else:
                print("Validation failed: 'event_id' is missing for input_audio_buffer.speech_started.")
                return False

            # Validate 'item_id'
            item_id_from_msg = response_data.get("item_id")
            if item_id_from_msg:
                print(f"Validation successful: 'item_id' is present: '{item_id_from_msg}'")
            else:
                print("Validation failed: 'item_id' is missing for input_audio_buffer.speech_started.")
                return False

            # Store in global variables
            global GLOBAL_RESPONSE_TYPE
            global GLOBAL_RESPONSE_EVENT_ID
            global GLOBAL_RESPONSE_ITEM_ID
            GLOBAL_RESPONSE_TYPE = response_data["type"]
            GLOBAL_RESPONSE_EVENT_ID = response_data["event_id"]
            GLOBAL_RESPONSE_ITEM_ID = response_data["item_id"]

            # Store in class instance
            self.audio_buffer_response_data = response_data

            print(f"\nCurrent self.global_event_id: {self.global_event_id}")
            print(f"Current self.global_session_id: {self.global_session_id}\n")

            print(f"Successfully validated and stored 'input_audio_buffer.speech_started' data.")
            print(f"  GLOBAL_RESPONSE_TYPE: {GLOBAL_RESPONSE_TYPE}")
            print(f"  GLOBAL_RESPONSE_EVENT_ID: {GLOBAL_RESPONSE_EVENT_ID}")
            print(f"  GLOBAL_RESPONSE_ITEM_ID: {GLOBAL_RESPONSE_ITEM_ID}")
            return True
        else:
            print("Error: Expected 'input_audio_buffer.speech_started' response not found or invalid.")
            return False

    def send_audio_buffer_commit_and_validate(self, event_id, expected_item_id=None, timeout=10):
        """
        Sends 'input_audio_buffer.commit' and waits for 4 responses:
        - input_audio_buffer.committed
        - conversation.item.created
        - conversation.item.input_audio_transcription.delta
        - conversation.item.input_audio_transcription.completed

        Stores event_type, event_id, item_id, and transcript for each.
        Returns a dict with all captured data, or None on failure.
        """
        if not self.is_connected:
            print("Error: Not connected to WebSocket. Cannot send commit.")
            return None

        commit_payload = {
            "event_id": event_id,
            "type": "input_audio_buffer.commit"
        }

        try:
            self.ws.send(json.dumps(commit_payload))
            print("\n--- Sent 'input_audio_buffer.commit' event ---")
            print(json.dumps(commit_payload, indent=2))
        except Exception as e:
            print(f"Failed to send 'input_audio_buffer.commit' event: {e}")
            return None

        # Prepare to collect responses
        responses = {
            "input_audio_buffer.committed": {},
            "conversation.item.created": {},
            "conversation.item.input_audio_transcription.delta": {},
            "conversation.item.input_audio_transcription.completed": {}
        }
        received_types = set()
        transcript = ""
        start_time = time.time()

        while len(received_types) < 4 and (time.time() - start_time) < timeout:
            if self.latest_received_message:
                msg = self.latest_received_message
                msg_type = msg.get("type")
                self.latest_received_message = None  # Clear for next

                if msg_type in responses:
                    print(f"Processing '{msg_type}' event.")
                    responses[msg_type]["event_id"] = msg.get("event_id")
                    responses[msg_type]["item_id"] = msg.get("item_id") or msg.get("item", {}).get("id")
                    responses[msg_type]["type"] = msg_type

                    # For delta/completed, capture transcript
                    if msg_type == "conversation.item.input_audio_transcription.delta":
                        delta = msg.get("delta")
                        if delta:
                            transcript += delta
                        responses[msg_type]["transcript"] = transcript
                        print(f"Delta transcript: {delta}")
                    elif msg_type == "conversation.item.input_audio_transcription.completed":
                        responses[msg_type]["transcript"] = msg.get("transcript")
                        print(f"Final transcript: {msg.get('transcript')}")
                    elif msg_type == "conversation.item.created":
                        # transcript may be null here
                        content = msg.get("item", {}).get("content", [])
                        if content and isinstance(content, list):
                            responses[msg_type]["transcript"] = content[0].get("transcript")
                    received_types.add(msg_type)
                    print(f"Validation successful: '{msg_type}' received with event_id: '{responses[msg_type]['event_id']}', item_id: '{responses[msg_type]['item_id']}'")
                else:
                    print(f"Received unhandled message type: {msg_type}")

            else:
                time.sleep(0.1)  # Wait briefly for next message


        # Store for later use
        self.commit_response_data = responses
        print("\n--- Commit Response Summary ---")
        for k, v in responses.items():
            print(f"{k}: {v}")

        return responses
    
    def send_audio_buffer_clear_and_validate(self, event_id, timeout=10):
        """
        Sends an 'input_audio_buffer.clear' event and validates the 'input_audio_buffer.cleared' response.       
        Args:
            event_id (str): The event ID to use in the request
            timeout (int): Maximum time to wait for response in seconds
            
        Returns:
            dict: Response data containing event_id and type, or None on failure
        """
        if not self.is_connected:
            print("Error: Not connected to WebSocket. Cannot send clear command.")
            return None

        clear_payload = {
            "event_id": event_id,
            "type": "input_audio_buffer.clear"
        }

        try:
            json_payload = json.dumps(clear_payload)
            self.ws.send(json_payload)
            print("\n--- Sent 'input_audio_buffer.clear' event ---")
            print(json_payload)
        except Exception as e:
            print(f"Failed to send 'input_audio_buffer.clear' event: {e}")
            return None

        # Wait for and validate response
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.latest_received_message:
                response_data = self.latest_received_message
                self.latest_received_message = None  # Clear after processing
                
                if response_data.get("type") == "input_audio_buffer.cleared":
                    print("Processing 'input_audio_buffer.cleared' event.")
                    
                    # Validate event_id
                    response_event_id = response_data.get("event_id")
                    if response_event_id:
                        print(f"Validation successful: 'event_id' is present: '{response_event_id}'")
                    else:
                        print("Validation failed: 'event_id' is missing.")
                        return None
                    
                    # Store the response data
                    self.buffer_cleared_response = response_data
                    
                    print(f"Successfully received and validated 'input_audio_buffer.cleared'")
                    return {
                        "type": response_data.get("type"),
                        "event_id": response_event_id
                    }
                else:
                    print(f"Received message with type '{response_data.get('type')}', waiting for 'input_audio_buffer.cleared'")
            
            time.sleep(0.1)  # Short delay between checks
        
        print(f"Error: Did not receive 'input_audio_buffer.cleared' response within {timeout} seconds")
        return None
    
    def send_conversation_item_retrieve_and_validate(self, event_id, item_id, timeout=10):
        """
        Sends a 'conversation.item.retrieve' event and validates the 'conversation.item.retrieved' response.
        
        Args:
            event_id (str): The event ID to use in the request
            item_id (str): The item ID to retrieve
            timeout (int): Maximum time to wait for response in seconds
            
        Returns:
            dict: Response data containing event_id, item details, transcript and audio, or None on failure
        """
        if not self.is_connected:
            print("Error: Not connected to WebSocket. Cannot retrieve item.")
            return None

        retrieve_payload = {
            "event_id": event_id,
            "type": "conversation.item.retrieve",
            "item_id": item_id
        }

        try:
            json_payload = json.dumps(retrieve_payload)
            self.ws.send(json_payload)
            print("\n--- Sent 'conversation.item.retrieve' event ---")
            print(json_payload)
        except Exception as e:
            print(f"Failed to send 'conversation.item.retrieve' event: {e}")
            return None

        # Wait for and validate response
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.latest_received_message:
                response_data = self.latest_received_message
                self.latest_received_message = None  # Clear after processing
                
                if response_data.get("type") == "conversation.item.retrieved":
                    print("Processing 'conversation.item.retrieved' event.")
                    
                    # Validate event_id
                    response_event_id = response_data.get("event_id")
                    if response_event_id:
                        print(f"Validation successful: 'event_id' is present: '{response_event_id}'")
                    else:
                        print("Validation failed: 'event_id' is missing.")
                        return None
                    
                    # Validate item and item_id
                    item = response_data.get("item", {})
                    response_item_id = item.get("id")
                    if response_item_id:
                        if response_item_id == item_id:
                            print(f"Validation successful: Item ID matches: '{response_item_id}'")
                        else:
                            print(f"Validation failed: Item ID mismatch. Expected '{item_id}', got '{response_item_id}'")
                            return None
                    else:
                        print("Validation failed: Item ID is missing in response.")
                        return None
                    
                    # Extract transcript and audio if present
                    content = item.get("content", [])
                    transcript = None
                    audio_data = None
                    audio_format = None
                    
                    if content and len(content) > 0:
                        for content_item in content:
                            if content_item.get("type") == "input_audio":
                                transcript = content_item.get("transcript")
                                audio_data = content_item.get("audio")
                                audio_format = content_item.get("format")
                                break
                    
                    # Store the response data
                    self.item_retrieved_response = response_data
                    
                    result = {
                        "type": response_data.get("type"),
                        "event_id": response_event_id,
                        "item_id": response_item_id,
                        "transcript": transcript,
                        "has_audio": audio_data is not None,
                        "audio_format": audio_format
                    }
                    
                    print(f"Successfully received and validated 'conversation.item.retrieved'")
                    print(f"Transcript: {transcript[:100]}..." if transcript and len(transcript) > 100 else f"Transcript: {transcript}")
                    print(f"Audio data: {'Present' if audio_data else 'Not present'}")
                    
                    return result
                else:
                    print(f"Received message with type '{response_data.get('type')}', waiting for 'conversation.item.retrieved'")
            
            time.sleep(0.1)  # Short delay between checks
        
        print(f"Error: Did not receive 'conversation.item.retrieved' response within {timeout} seconds")
        return None

    def send_conversation_item_delete_and_validate(self, event_id, item_id, timeout=10):
        """
        Sends a 'conversation.item.delete' event and validates the 'conversation.item.deleted' response.
        
        Args:
            event_id (str): The event ID to use in the request
            item_id (str): The item ID to delete
            timeout (int): Maximum time to wait for response in seconds
            
        Returns:
            dict: Response data containing event_id, type and item_id, or None on failure
        """
        if not self.is_connected:
            print("Error: Not connected to WebSocket. Cannot delete item.")
            return None

        delete_payload = {
            "event_id": event_id,
            "type": "conversation.item.delete",
            "item_id": item_id
        }

        try:
            json_payload = json.dumps(delete_payload)
            self.ws.send(json_payload)
            print("\n--- Sent 'conversation.item.delete' event ---")
            print(json_payload)
        except Exception as e:
            print(f"Failed to send 'conversation.item.delete' event: {e}")
            return None

        # Wait for and validate response
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.latest_received_message:
                response_data = self.latest_received_message
                self.latest_received_message = None  # Clear after processing
                
                if response_data.get("type") == "conversation.item.deleted":
                    print("Processing 'conversation.item.deleted' event.")
                    
                    # Validate event_id
                    response_event_id = response_data.get("event_id")
                    if response_event_id:
                        print(f"Validation successful: 'event_id' is present: '{response_event_id}'")
                    else:
                        print("Validation failed: 'event_id' is missing.")
                        return None
                    
                    # Validate item_id
                    response_item_id = response_data.get("item_id")
                    if response_item_id:
                        if response_item_id == item_id:
                            print(f"Validation successful: Item ID matches: '{response_item_id}'")
                        else:
                            print(f"Validation failed: Item ID mismatch. Expected '{item_id}', got '{response_item_id}'")
                            return None
                    else:
                        print("Validation failed: Item ID is missing in response.")
                        return None
                    
                    # Store the response data
                    self.item_deleted_response = response_data
                    
                    print(f"Successfully received and validated 'conversation.item.deleted'")
                    return {
                        "type": response_data.get("type"),
                        "event_id": response_event_id,
                        "item_id": response_item_id
                    }
                else:
                    print(f"Received message with type '{response_data.get('type')}', waiting for 'conversation.item.deleted'")
            
            time.sleep(0.1)  # Short delay between checks
        
        print(f"Error: Did not receive 'conversation.item.deleted' response within {timeout} seconds")
        return None
    
    def send_response_create_and_validate(self, event_id, transcript, timeout=60):
        """
        Sends response.create and validates response events.
        Returns dict with responses or None on failure.
        """
        if not self.is_connected:
            print("Error: WebSocket not connected")
            return None

        # Construct and send payload
        payload = {
        "event_id": event_id,
        "type": "response.create",
        "response": {
            "modalities": ["text", "audio"],
            "instructions": transcript,
            "voice": "sage",
            "output_audio_format": "pcm16",
            "tool_choice": "none",
            "temperature": 0.8,
            "max_output_tokens": 1024	
        }
    }
        print(f"Event ID: {event_id}")
        print(f"Type: response.create")
        try:
            self.ws.send(json.dumps(payload))
            print("\n=== Sending response.create ===")
            print(json.dumps(payload, indent=2))
        except Exception as e:
            print(f"Failed to send response.create: {e}")
            return None

        # Track responses
        responses = {
            "response.created": None,
            "response.audio.delta": [],
            "response.audio.completed": None,
            "combined_audio_delta": ""
        }
        
        
        start_time = time.time()
        complete = False
        
        while not complete and (time.time() - start_time) < timeout:
            if self.latest_received_message:
                msg = self.latest_received_message
                msg_type = msg.get("type")
                self.latest_received_message = None  # Clear for next message
                
                print(f"\nReceived message type: {msg_type}")
                
                if msg_type == "response.created":
                    responses["response.created"] = msg
                    print("✓ Captured response.created")
                elif msg_type == "response.audio.delta":
                    delta = msg.get("delta", "")
                    responses["response.audio.delta"].append(delta)
                    responses["combined_audio_delta"] += delta
                    print(f"✓ Captured audio delta chunk ({len(delta)} bytes)")
                elif msg_type == "response.audio.completed":
                    responses["response.audio.completed"] = msg
                    print("✓ Captured response.audio.completed")
                    complete = True  # Exit after completion
                    break
            
            time.sleep(10)  # Small delay between checks

        # Validate responses
        if not responses["response.created"]:
            print("❌ Missing response.created event")
            return None
        if not responses["response.audio.completed"]:
            print("❌ Missing response.audio.completed event")
            return None
        if not responses["response.audio.delta"]:
            print("⚠️ Warning: No audio delta chunks received")

        print("\n=== Response Summary ===")
        print(f"Total delta chunks: {len(responses['response.audio.delta'])}")
        print(f"Combined audio size: {len(responses['combined_audio_delta'])} bytes")
        
        return responses

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