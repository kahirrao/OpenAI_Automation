# websocket_manager.py
import websocket
import json
import threading
import time

class WebSocketManager:
    """
    Manages WebSocket connections and provides basic callbacks for WebSocket events.
    This class handles the low-level WebSocket operations separate from business logic.
    """
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
        self.ws = None
        self.is_connected = False
        
        # Events for synchronization
        self.connection_open_event = threading.Event()
        self.close_event = threading.Event()
        
        self._ws_thread = None
        self.latest_received_message = None
        
        # Callback function that can be set by the client
        self.message_callback = None
    
    def on_message(self, ws, message):
        """
        Callback function to handle incoming WebSocket messages.
        """
        print("\n--- Received message ---")
        print(message)

        try:
            data = json.loads(message)
            self.latest_received_message = data
            
            # Call the client's callback if it exists
            if self.message_callback:
                self.message_callback(data)
                
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
        self.connection_open_event.set()  # Unblock if waiting for connection
        self.close_event.set()  # Signal final closure on error

    def on_close(self, ws, close_status_code, close_msg):
        """
        Callback function when the WebSocket connection is closed.
        """
        print(f"Connection closed. Status code: {close_status_code}, Message: {close_msg}")
        self.is_connected = False
        self.connection_open_event.set()  # Unblock if waiting for connection
        self.close_event.set()  # Signal final closure

    def on_open(self, ws):
        """
        Callback function when the WebSocket connection is opened.
        """
        print("Connection opened.")
        self.is_connected = True
        self.connection_open_event.set()  # Signal that the connection is open

    def _run_websocket(self):
        """Internal method to run the WebSocket in a separate thread."""
        self.ws.run_forever()
        print("WebSocket thread finished.")

    def connect(self, timeout=10):
        """
        Establishes the WebSocket connection.
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
        self._ws_thread = threading.Thread(target=self._run_websocket, daemon=True)
        self._ws_thread.start()

        print(f"Waiting for connection to open (timeout: {timeout}s)...")
        if not self.connection_open_event.wait(timeout=timeout):
            print("Error: Connection did not open in time.")
            self.close_connection()
            return False
            
        return self.is_connected

    def send_message(self, message):
        """
        Sends a message through the WebSocket.
        
        Args:
            message: Either a JSON string or a dictionary to be converted to JSON
            
        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.is_connected:
            print("Error: Not connected to WebSocket. Cannot send message.")
            return False
            
        try:
            if isinstance(message, dict):
                json_payload = json.dumps(message)
            else:
                json_payload = message
                
            self.ws.send(json_payload)
            return True
        except Exception as e:
            print(f"Failed to send message: {e}")
            return False

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
            self._ws_thread.join(timeout=5)  # Wait for the thread to finish
            if self._ws_thread.is_alive():
                print("Warning: WebSocket thread did not terminate gracefully.")
        print("Connection cleanup complete.")
        
    def wait_for_message_type(self, expected_type, timeout=10):
        """
        Waits for a message with the specified type.
        
        Args:
            expected_type (str): The type of message to wait for
            timeout (int): Maximum time to wait in seconds
            
        Returns:
            dict: The message data if received, None otherwise
        """
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            if self.latest_received_message and self.latest_received_message.get("type") == expected_type:
                message = self.latest_received_message
                self.latest_received_message = None  # Clear after processing
                return message
            time.sleep(0.1)  # Short delay between checks
        
        print(f"Error: Did not receive message of type '{expected_type}' within {timeout} seconds")
        return None