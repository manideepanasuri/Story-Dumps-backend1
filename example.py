import io
import re

import numpy as np
from onnxruntime import InferenceSession
from kokoro_onnx import Kokoro
import soundfile as sf

MODEL_PATH = "kokoro-v1.0.onnx" # These paths are relative to /app inside the container
VOICES_PATH = "voices-v1.0.bin"

# Configure ONNX Runtime to use CUDA, with a fallback to CPU
# The order matters: it tries CUDA first.
providers = ['CPUExecutionProvider']

try:
    session = InferenceSession(MODEL_PATH, providers=providers)
    print("ONNX Runtime session initialized with GPU (CUDAExecutionProvider).")
except Exception as e:
    print(f"Failed to initialize with GPU: {e}. Falling back to CPU.")
    session = InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("ONNX Runtime session initialized with CPU (CPUExecutionProvider).")

kokoro = Kokoro.from_session(session, VOICES_PATH)

# ... rest of your text-to-speech logic
# Your speech generation code:
# samples, sample_rate = kokoro.create("Hello, this is a test from Intel Iris Xe!", voice="en_US_Bella", speed=1.0, lang="en-us")
# ... (then convert to bytes and save/use)
temp="Hello, This is my voice"

sentences = re.split(r'(?<=[.!?])\s+', temp) # Use your splitting logic
audio_segments_list = []
audio_sampling_rate = 22050
for sentence in sentences:
    if sentence.strip():
        samples, sample_rate = kokoro.create(sentence.strip(), voice="af_bella",speed=1)
        audio_segments_list.append(samples)
        audio_sampling_rate = sample_rate

if not audio_segments_list:
    print("No audio to process.")

concatenated_samples = np.concatenate(audio_segments_list)
common_sample_rate = audio_sampling_rate # Or whatever sample_rate kokoro.create() returns

audio_byte_stream = io.BytesIO()
sf.write("audio_af_bella.mp3", concatenated_samples, samplerate=common_sample_rate)
concatenated_audio_bytes = audio_byte_stream.getvalue()

from fastapi import FastAPI

app = FastAPI()

