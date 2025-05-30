import librosa
import soundfile as sf
import os
import base64

def convert_to_pcm16(input_file, output_file=None):
    """
    Convert any audio file to PCM INT16 format
    Args:
        input_file: Path to input audio file
        output_file: Path for output file (optional)
    """
    try:
        # Load the audio file
        audio, sr = librosa.load(input_file, sr=16000, mono=True)
        
        # Generate output filename if not provided
        if output_file is None:
            filename, _ = os.path.splitext(input_file)
            output_file = f"{filename}_pcm16.wav"
        
        # Write as WAV with PCM16 format
        sf.write(output_file, audio, sr, subtype='PCM_16')
        print(f"Conversion successful! Saved as: {output_file}")
        
    except Exception as e:
        print(f"Error converting file: {str(e)}")

def convert_to_base64(wav_file):
    """
    Convert WAV file to base64 encoded string
    Args:
        wav_file: Path to WAV file
    Returns:
        base64 encoded string
    """
    try:
        with open(wav_file, 'rb') as file:
            wav_bytes = file.read()
            base64_encoded = base64.b64encode(wav_bytes).decode('utf-8')
            return base64_encoded
    except Exception as e:
        print(f"Error encoding file to base64: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    # Directory containing audio files
    audio_dir = os.path.join(os.getcwd(), "data_audio")
    
    # Process all audio files in the directory
    for filename in os.listdir(audio_dir):
        if filename.endswith(('.mp3', '.wav', '.ogg')):
            # Convert to PCM16
            input_path = os.path.join(audio_dir, filename)
            print(f"Converting to PCM16: {filename}")
            convert_to_pcm16(input_path)
            
            # Convert PCM16 to base64
            pcm16_file = os.path.splitext(input_path)[0] + "_pcm16.wav"
            if os.path.exists(pcm16_file):
                print(f"Converting to base64: {os.path.basename(pcm16_file)}")
                base64_encoded = convert_to_base64(pcm16_file)
                
                # Save base64 to file
                base64_file = os.path.splitext(pcm16_file)[0] + "_base64.txt"
                with open(base64_file, 'w') as f:
                    f.write(base64_encoded)
                print(f"Base64 saved as: {os.path.basename(base64_file)}")
                print("-" * 50)