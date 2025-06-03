import base64
import pyttsx3
import wave
import array

def base64_to_text(base64_input):
    """
    Convert base64 to text
    Args:
        base64_input: base64 encoded string or file path
    Returns:
        decoded text
    """
    try:
        # Check if input is a file path
        if isinstance(base64_input, str) and base64_input.endswith('.txt'):
            with open(base64_input, 'r') as file:
                base64_input = file.read().strip()

        # Decode base64
        decoded_bytes = base64.b64decode(base64_input)
        return decoded_bytes.decode('utf-8')

    except Exception as e:
        return f"Error decoding base64: {str(e)}"

def text_to_base64(text_input):
    """
    Convert text to base64
    Args:
        text_input: text string or file path to convert
    Returns:
        base64 encoded string
    """
    try:
        # Check if input is a file path
        if isinstance(text_input, str) and text_input.endswith('.txt'):
            with open(text_input, 'r') as file:
                text_input = file.read()

        # Encode text to base64
        encoded_bytes = text_input.encode('utf-8')
        base64_output = base64.b64encode(encoded_bytes)
        return base64_output.decode('utf-8')

    except Exception as e:
        return f"Error encoding to base64: {str(e)}"

def text_to_wave(text, output_file="output.wav", voice_speed=150):
    """
    Convert text to WAV audio file
    Args:
        text: Text to convert to speech
        output_file: Output WAV file path (default: output.wav)
        voice_speed: Speed of speech (default: 150)
    Returns:
        Path to created WAV file
    """
    try:
        # Initialize the text-to-speech engine
        engine = pyttsx3.init()

        # Set properties
        engine.setProperty('rate', voice_speed)  # Speed of speech

        # Save to file
        engine.save_to_file(text, output_file)
        engine.runAndWait()

        return output_file

    except Exception as e:
        return f"Error creating WAV file: {str(e)}"

if __name__ == "__main__":
    # Test text to base64 conversion
    test_text = "Hello"
    encoded_result = text_to_base64(test_text)
    print(f"\nText to base64: {encoded_result}")

    # Verify by decoding back
    decoded_result = base64_to_text(encoded_result)
    print(f"Verified decode: {decoded_result}")

    # Test with file input
    # Uncomment and modify path to use file input
    # file_path = "path_to_your_text_file.txt"
    # encoded_result = text_to_base64(file_path)
    # print(f"File encode result: {encoded_result}")

    # Test the conversion
    sample_text = "Hello"
    output_path = text_to_wave(sample_text)
    print(f"WAV file created at: {output_path}")

    # Test with different speed
    fast_output = text_to_wave(
        sample_text, 
        output_file="fast_output.wav", 
        voice_speed=200
    )
    print(f"Fast WAV file created at: {fast_output}")