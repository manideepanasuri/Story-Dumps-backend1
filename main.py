
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from helpers import *
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
load_dotenv()
app=FastAPI()

class Data(BaseModel):
    text: str
    voice: str
    speed: float
    bg_video_url: str
    useduuid: str


@app.post("/")
def read_root(data: Data,access_token: str = Header(..., alias="access_token")):
    print("hi")
    if access_token!=os.environ.get("GENERATE_VIDEO_SECRET"):

        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        audio_url = get_audio_from_text(data.text, data.useduuid,data.speed,data.voice)

        subtitles_array = transcribe_audio_with_whisper(audio_url)
        ass_url = generate_ass_file(subtitles_array, data.useduuid)

        video_url = generate_video_url(audio_url, ass_url, data.useduuid,data.bg_video_url)
        thumbnail_url = generate_thumbnail_url(video_url, data.useduuid)
        data={
            "audio_url": audio_url,
            "video_url": video_url,
            "ass_url":ass_url,
            "thumbnail_url": thumbnail_url,
        }
        ret_data={
            "success": True,
            "data": data,
        }
        return ret_data
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"{e}")