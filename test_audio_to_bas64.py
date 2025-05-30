import os
import pytest
from audio_to_bas64 import convert_to_pcm16, convert_to_base64

@pytest.fixture
def sample_wav(tmp_path):
    # Create a small dummy WAV file using soundfile
    import numpy as np
    import soundfile as sf
    wav_path = tmp_path / "test.wav"
    data = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
    sf.write(wav_path, data, 16000)
    return str(wav_path)

def test_convert_to_pcm16_creates_file(sample_wav, tmp_path):
    output_file = str(tmp_path / "output_pcm16.wav")
    convert_to_pcm16(sample_wav, output_file)
    assert os.path.exists(output_file)
    # Check file is not empty
    assert os.path.getsize(output_file) > 0

def test_convert_to_base64_returns_string(sample_wav, tmp_path):
    # First, convert to PCM16
    output_file = str(tmp_path / "output_pcm16.wav")
    convert_to_pcm16(sample_wav, output_file)
    # Now test base64 conversion
    base64_str = convert_to_base64(output_file)
    assert isinstance(base64_str, str)
    assert len(base64_str) > 0