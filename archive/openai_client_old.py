import websocket
import json
import os
from dotenv import load_dotenv
import threading
import time # Import time for potential delays
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
        self.close_event = threading.Event()
        self.session_updated_received = threading.Event() # New event to signal session.updated receipt

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
                
            elif msg_type == "session.updated":
                print("Processing 'session.updated' event.")
                expected_type = "session.updated"
                if data.get("type") == expected_type:
                    print(f"Validation successful: 'type' is '{expected_type}'")
                else:
                    print(f"Validation failed: Expected 'type' to be '{expected_type}', but got '{data.get('type')}'")

                event_id_from_msg = data.get("event_id")
                if event_id_from_msg:
                    # You might want to store this new event_id if subsequent events use it
                    print(f"Validation successful: 'event_id' is present: '{event_id_from_msg}'")
                else:
                    print("Validation failed: 'event_id' is missing for session.updated.")

                session_id_from_msg = data.get("session", {}).get("id")
                if session_id_from_msg:
                    # This should ideally match the session_id from session.created
                    if self.global_session_id and self.global_session_id != session_id_from_msg:
                        print(f"Warning: session.id in session.updated ({session_id_from_msg}) differs from initial session.id ({self.global_session_id})")
                    else:
                        print(f"Validation successful: 'session.id' is present and consistent: '{session_id_from_msg}'")
                    # Optionally update the global_session_id if it's meant to change or be re-confirmed
                    self.global_session_id = session_id_from_msg 
                else:
                    print("Validation failed: 'session.id' is missing for session.updated.")
                
                self.session_updated_received.set() # Signal that session.updated was received

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
        self.close_event.set() # Signal closure on error
        self.session_updated_received.set() # Also unblock if waiting for update

    def on_close(self, ws, close_status_code, close_msg):
        """
        Callback function when the WebSocket connection is closed.
        """
        print(f"Connection closed. Status code: {close_status_code}, Message: {close_msg}")
        self.is_connected = False
        self.close_event.set() # Signal closure
        self.session_updated_received.set() # Also unblock if waiting for update
        print(f"Final self.global_event_id at close: {self.global_event_id}")
        print(f"Final self.global_session_id at close: {self.global_session_id}")

    def on_open(self, ws):
        """
        Callback function when the WebSocket connection is opened.
        """
        print("Connection opened.")
        self.is_connected = True
        print(f"Instance variables at open: event_id={self.global_event_id}, session_id={self.global_session_id}")
        
        # Immediately send the session.update event after connection is open
        self.send_session_update()

    def connect(self):
        """
        Establishes and runs the WebSocket connection.
        This method will block until the connection closes.
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
        
        # Run in a separate thread to allow main thread to potentially wait
        # for session_updated_received or do other work if needed later.
        # For simple sequential execution, ws.run_forever() without threading.Thread works too.
        # self.ws.run_forever(dispatcher=threading.Event().wait) 
        self.ws.run_forever()       
        print("WebSocket connection finished.")

    def run(self):
        """
        A wrapper method to start the connection and wait for key events.
        """
        connect_thread = threading.Thread(target=self.connect)
        connect_thread.start()
        # Wait for the session.created event (implied by connection open)
        # and then the session.updated event
        print("Waiting for session.updated confirmation...")
        # Give a reasonable timeout for the session.updated response
        if self.session_updated_received.wait(timeout=10): 
            print("Successfully received 'session.updated' response.")
            # At this point, you could send audio or text events
            # For demonstration, we'll close the connection after a short delay
            print("Keeping connection open for 5 seconds after update for demonstration...")
            time.sleep(5)
            self.close()
        else:
            print("Timed out waiting for 'session.updated' response or connection closed prematurely.")
            self.close() # Ensure connection is closed on timeout

        connect_thread.join() # Wait for the WebSocket thread to finish

    def close(self):
        """
        Closes the WebSocket connection.
        """
        if self.ws:
            print("Closing WebSocket connection...")
            self.ws.close()
            self.close_event.wait(timeout=5) # Wait for on_close to execute
            if self.is_connected: 
                print("Warning: WebSocket close may not have completed gracefully.")
        else:
            print("No active WebSocket connection to close.")

    def _run_websocket(self):
        """Internal method to run the WebSocket in a separate thread."""
        self.ws.run_forever(dispatcher=threading.Event().wait)
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
        
        if not self.is_connected: 
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
        
        if not self.is_connected: 
            print("Error: Connection closed after receiving 'session.updated'.")
            return False

        print("Successfully sent 'session.update' and received 'session.updated'.")
        return True

    def send_audio_append_event(self, event_id_from_prev_response, base64_audio, timeout=10):
        """
        Sends an 'input_audio_buffer.append' event with the given Base64 audio data
        and event_id, then waits for the 'input_audio_buffer.speech_started' response.

        Args:
            event_id_from_prev_response (str): The event_id to include in the payload,
                                               typically from a previous server response
                                               like session.updated.
            base64_audio (str): The Base64 encoded PCM16 audio string.
            timeout (int): How long to wait for the speech_started response.

        Returns:
            bool: True if the event was sent and speech_started response received, False otherwise.
        """
        if not self.is_connected:
            print("Error: Not connected to WebSocket. Cannot send audio append event.")
            return False

        # Reset the event before sending to ensure we wait for a *new* signal
        self.speech_started_event.clear() 

        audio_append_payload = {
            "event_id": event_id_from_prev_response, # Use the event_id from the session.updated response
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }
        try:
            json_payload = json.dumps(audio_append_payload)
            self.ws.send(json_payload)
            print(f"\n--- Sent 'input_audio_buffer.append' event ---")
            print(json_payload[:200] + "..." if len(json_payload) > 200 else json_payload) # Print truncated audio for brevity
        except Exception as e:
            print(f"Failed to send 'input_audio_buffer.append' event: {e}")
            return False

        print(f"Waiting for 'input_audio_buffer.speech_started' event (timeout: {timeout}s)...")
        if not self.speech_started_event.wait(timeout=timeout):
            print("Error: 'input_audio_buffer.speech_started' event not received in time.")
            return False
        
        if not self.is_connected: 
            print("Error: Connection closed after sending audio append.")
            return False

        print("Successfully sent 'input_audio_buffer.append' and received 'input_audio_buffer.speech_started'.")
        return True

    def close_connection(self):
        """
        Closes the WebSocket connection and waits for the thread to finish.
        """
        if self.ws and self.is_connected:
            print("Closing WebSocket connection...")
            self.ws.close()
            if not self.close_event.wait(timeout=5):
                print("Warning: WebSocket close may not have completed gracefully.")
        elif self.ws:
            print("WebSocket already closed or not connected.")
        else:
            print("No WebSocket object to close.")

        if self._ws_thread and self._ws_thread.is_alive():
            print("Waiting for WebSocket thread to join...")
            self._ws_thread.join(timeout=5)
            if self._ws_thread.is_alive():
                print("Warning: WebSocket thread did not terminate gracefully.")
        print("Connection cleanup complete.")




# --- Main execution block ---
if __name__ == "__main__":
    client = OpenAIRealtimeClient(URL, HEADERS)
    client.run() # Use the new run method to manage the flow

    print("\nScript execution completed.")
    print(f"Final client.global_event_id: {client.global_event_id}")
    print(f"Final client.global_session_id: {client.global_session_id}")

