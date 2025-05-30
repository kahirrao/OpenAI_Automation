# src/audio_processor.py
# (Contents as provided in the previous turn, without example usage)
import librosa
import soundfile as sf
import os
import base64
import json # New import for JSON operations

# --- Existing function: process_audio_to_base64 ---
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
    data_folder_path = os.path.join(project_root, "data")
    input_audio_path = os.path.join(data_folder_path, audio_filename)

    output_sub_dir = None
    if save_processed_files:
        output_sub_dir = os.path.join(data_folder_path, "processed_audio")
        os.makedirs(output_sub_dir, exist_ok=True) 

    base64_string = process_audio_to_base64(input_audio_path, output_dir=output_sub_dir if save_processed_files else None)
    
    if base64_string:
        print(f"Successfully obtained Base64 for {audio_filename}.")
    else:
        print(f"Failed to obtain Base64 for {audio_filename}.")
        
    return base64_string

# --- Existing function: update_json_with_audio_data ---
def update_json_with_audio_data(audio_filename, event_id_from_response, json_template_name="session_events.json"):
    # ... (same code as before) ...
    project_root = os.getcwd()
    data_folder_path = os.path.join(project_root, "data")
    json_file_path = os.path.join(data_folder_path, json_template_name)

    print(f"\n--- Updating JSON file: {json_file_path} ---")

    base64_audio = get_audio_base64_from_data_folder(audio_filename, save_processed_files=True)
    
    if base64_audio is None:
        print("Error: Could not get Base64 audio string. JSON update aborted.")
        return False

    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        print("JSON file read successfully.")
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_file_path}. Please create it with the specified structure.")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_file_path}. Check file format.")
        return False
    except Exception as e:
        print(f"An error occurred while reading the JSON file: {e}")
        return False

    if isinstance(json_data, dict):
        json_data["event_id"] = event_id_from_response
        json_data["type"] = "input_audio_buffer.append"
        json_data["audio"] = base64_audio
        print(f"JSON data updated with event_id: '{event_id_from_response}' and new audio data.")
    else:
        print(f"Error: JSON file content is not a dictionary. Cannot update.")
        return False

    try:
        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON file successfully updated at {json_file_path}")
        return True
    except Exception as e:
        print(f"An error occurred while writing the updated JSON file: {e}")
        return False