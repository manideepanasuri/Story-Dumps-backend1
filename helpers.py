import os
import tempfile

import boto3
import requests
from botocore.client import Config
import ffmpeg
from io import BytesIO
from faster_whisper import WhisperModel
import torch
from dotenv import load_dotenv
load_dotenv()

import io
import re

import numpy as np
from onnxruntime import InferenceSession
from kokoro_onnx import Kokoro
import soundfile as sf

MODEL_PATH = "kokoro-v1.0.onnx"
VOICES_PATH = "voices-v1.0.bin"

# Configure ONNX Runtime to use CUDA, with a fallback to CPU
# The order matters: it tries CUDA first.
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']

try:
    session = InferenceSession(MODEL_PATH, providers=providers)
    print(f"ONNX Runtime session initialized with {providers[0]}.")
except Exception as e:
    print(f"Failed to initialize with GPU: {e}. Falling back to CPU.")
    session = InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    print("ONNX Runtime session initialized with CPU (CPUExecutionProvider).")

kokoro = Kokoro.from_session(session, VOICES_PATH)


MINIO_STORAGE = {
    "ENDPOINT":os.getenv('MINIO_ENDPOINT'),
    "ACCESS_KEY": os.getenv('MINIO_ACCESS_KEY'),
    "SECRET_KEY": os.getenv('MINIO_SECRET_KEY'),
    "BUCKET_NAME": os.getenv('MINIO_BUCKET_NAME'),
    "SECURE": bool(os.getenv('MINIO_SECURE')),  # False means HTTP, True means HTTPS
}
s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_STORAGE['ENDPOINT'],
        aws_access_key_id=MINIO_STORAGE['ACCESS_KEY'],
        aws_secret_access_key=MINIO_STORAGE['SECRET_KEY'],
        config=Config(signature_version='s3v4'),
        region_name='ap-south-1',  # MinIO uses 'us-east-1' by default, can be anything
        verify=True,  # For self-signed certs, set True if using real SSL
    )

# --- Starting Whisper Transcription ---
print("--- Starting Whisper Transcription ---")
model_name = "base" # Or "medium", "large-v3", etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Faster Whisper model: '{model_name}' on device: {device}...")
compute="float16" if device == "cuda" else "int8"

whisper_model = WhisperModel(model_name, device=device, compute_type=compute) # Use "float16" for GPU for better performance

print("Model loaded")

def upload_file_to_minio(file_bytes, filename)->str:
    if isinstance(file_bytes, BytesIO):
        file_obj = file_bytes
    else:
        file_obj = BytesIO(file_bytes)
    # Upload file_obj (a file-like object) to bucket with filename/key
    url=s3_client.upload_fileobj(file_obj, MINIO_STORAGE['BUCKET_NAME'], filename)
    print(url)
    # Return the public URL of the uploaded file if bucket is public
    url = f"{MINIO_STORAGE['ENDPOINT']}{MINIO_STORAGE['BUCKET_NAME']}/{filename}"
    return url

def get_audio_from_text(text:str,useduuid:str,speed:float,voice:str="af_bella")->str:
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Use your splitting logic
    audio_segments_list = []
    audio_sampling_rate = 22050
    for sentence in sentences:
        if sentence.strip():
            samples, sample_rate = kokoro.create(sentence.strip(), voice=voice, speed=speed)
            audio_segments_list.append(samples)
            audio_sampling_rate = sample_rate

    if not audio_segments_list:
        print("No audio to process.")

    concatenated_samples = np.concatenate(audio_segments_list)
    common_sample_rate = audio_sampling_rate  # Or whatever sample_rate kokoro.create() returns

    audio_byte_stream = io.BytesIO()
    sf.write(audio_byte_stream, concatenated_samples,format='MP3',
        subtype='MPEG_LAYER_III', samplerate=common_sample_rate)
    concatenated_audio_bytes = audio_byte_stream.getvalue()

    filename = f"audio/{useduuid}.mp3"
    fileurl =  upload_file_to_minio(concatenated_audio_bytes, filename)
    return fileurl

def generate_ass_file(subtitle_array, useduuid:str, video_width=1920, video_height=1080):

    # ASS file header with styling
    ass_header = f"""[Script Info]
Title: Animated Word Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {video_width}
PlayResY: {video_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,72,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,1,0,0,100,100,0,0,1,4,0,2,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def seconds_to_ass_time(seconds):
        """Convert seconds to ASS time format (H:MM:SS.CC)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def create_animation_tags(start_time, end_time, text):
        """Create ASS animation tags for grow and vanish effect"""
        duration = end_time - start_time

        # Animation timings (in milliseconds relative to subtitle start)
        grow_duration = min(200, duration * 1000 * 0.3)  # 30% of duration or 200ms max
        vanish_start = max(duration * 1000 - 200, duration * 1000 * 0.7)  # Start vanish at 70% or 200ms before end
        vanish_duration = duration * 1000 - vanish_start

        # Grow animation: scale from 0 to 1.2 to 1.0 with slight bounce
        grow_tags = f"{{\\t(0,{grow_duration:.0f},\\fscx50\\fscy50)}}{{\\t(0,{grow_duration:.0f},\\fscx120\\fscy120)}}{{\\t({grow_duration:.0f},{grow_duration * 1.2:.0f},\\fscx100\\fscy100)}}"

        # Vanish animation: fade out and scale down
        vanish_tags = f"{{\\t({vanish_start:.0f},{duration * 1000:.0f},\\alpha&HFF&\\fscx70\\fscy70)}}"

        # Combine with text styling
        styled_text = f"{{\\an5\\pos({video_width // 2},{video_height // 2})\\bord4\\shad0\\fs72\\b1\\i1}}{grow_tags}{vanish_tags}{text}"

        return styled_text

    # Generate subtitle events
    events = []
    for subtitle in subtitle_array:
        start_time = subtitle['start']
        end_time = subtitle['end']
        text = subtitle['text']

        start_ass = seconds_to_ass_time(start_time)
        end_ass = seconds_to_ass_time(end_time)

        animated_text = create_animation_tags(start_time, end_time, text)

        event_line = f"Dialogue: 0,{start_ass},{end_ass},Default,,0,0,0,,{animated_text}"
        events.append(event_line)

    filedata=ass_header+"\n".join(events)
    file_bytes=BytesIO(filedata.encode('utf-8'))
    url=upload_file_to_minio(file_bytes,"ass/"+useduuid+".ass")
    return url

def transcribe_audio_with_whisper(audio_path:str):
    """
    Transcribes an audio file using OpenAI Whisper and returns a list of subtitle dictionaries.
    Checks for GPU availability and uses it if present.
    """
    subtitles = []
    try:
        response = requests.get(audio_path)
        response.raise_for_status()  # ensure it succeeded

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(response.content)
            tmp_path=tmp.name
            tmp.flush()  # ensure data is written
            segments,info = whisper_model.transcribe(tmp.name,word_timestamps=True)
        os.remove(tmp_path)
        for each in segments:
            words = each.words
            for word in words:
                start = word.start
                end = word.end
                text = word.word.strip()
                if text:
                    subtitles.append({
                        'start': start,
                        'end': end,
                        'text': text,
                    })
        print("Transcription complete.")
        return subtitles
    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        return []

def generate_video_url(audio_url:str,ass_url:str,useduuid:str,video_url:str)->str:

    response = requests.get(ass_url)
    response.raise_for_status()  # Ensure download succeeded
    # Create a unique temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".ass") as temp_ass_file:
        temp_ass_file.write(response.content)
        ass_path = temp_ass_file.name
    try:
        audio_info=ffmpeg.probe(audio_url,select_streams='a')
        audio_duration = float(audio_info['streams'][0]['duration'])
    except ffmpeg.Error as e:
        print(e.stderr.decode())
        raise Exception(f"Error during Whisper transcription: {e}")


    print(ass_path)
    video = (
        ffmpeg
        .input(video_url, stream_loop=-1)  # loop video infinitely
        .filter('trim', duration=audio_duration)  # cut video to match audio
        .filter('setpts', 'PTS-STARTPTS')  # reset timestamps
        .filter('ass',filename=ass_path)  # burn subtitles
    )
    audio = ffmpeg.input(audio_url)
    try:
        process = (
            ffmpeg
            .output(video, audio.audio, 'pipe:',
                    vcodec='libx264', acodec='aac',
                    t=audio_duration,
                    format='mp4',
                    movflags='frag_keyframe+empty_moov',)
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        out, err = process.communicate()
        if process.returncode != 0:
            print("error generating video url")
            raise RuntimeError(f"FFmpeg error: {err.decode()}")
        file_name = f"videos/{useduuid}.mp4"
        output_url = upload_file_to_minio(out, file_name)
        print(f"Video URL: {output_url}")
        os.remove(ass_path)
        return output_url
    except Exception as e:
        print(e.stderr.decode())
        os.remove(ass_path)
        return ""

def generate_thumbnail_url(remote_url: str,useduuid:str) :
    # Use ffmpeg-python to extract a single frame
    process = (
        ffmpeg
        .input(remote_url, ss=1)  # seek to 1 second
        .output('pipe:', vframes=1, format='image2', vcodec='mjpeg')
        .overwrite_output()
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    out, err = process.communicate()
    if process.returncode != 0:
        print("error making image")
        raise RuntimeError(f"FFmpeg error: {err.decode()}")
    file_name = f"thumbnail/{useduuid}.jpg"
    out_url=upload_file_to_minio(out, file_name)
    return out_url



