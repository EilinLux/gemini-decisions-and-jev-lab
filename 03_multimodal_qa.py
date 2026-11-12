"""
03_multimodal_qa.py — Native Multimodal Decision Making (Audio & Video)
=======================================================================
What we are testing:
  1. Direct Audio & Video Evaluation: Passing raw MP3 audio and MP4 video URIs
     directly into `gemini-decision-1` without an intermediate Whisper/ASR
     transcription or frame-captioning pipeline.
  2. Media Resolution Cost Control: Setting `resolution="low"` on video blocks
     to minimize visual input tokens while maintaining calibrated classification
     accuracy and 0 output tokens.
"""

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="google.genai")

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(http_options=types.HttpOptions(api_version="v1beta"))


def test_audio_decision():
    result = client.decisions.create(
        model="gemini-decision-1",
        input=[
            {
                "type": "audio",
                "uri": "https://storage.googleapis.com/generativeai-downloads/data/Apollo-11_Day-01-Highlights-10s.mp3",
                "mime_type": "audio/mp3",
            }
        ],
        questions={
            "mentions_spaceflight": {
                "type": "noul",
                "instructions": "Is the speaker talking about spaceflight or a rocket launch?",
            }
        },
    )
    ans = result.answers["mentions_spaceflight"]
    print(f"[Audio] Spaceflight P(Yes): {ans.noul:.3f} | Output Tokens: {result.usage.total_output_tokens}")


def test_video_decision():
    result = client.decisions.create(
        model="gemini-decision-1",
        input=[
            {
                "type": "video",
                "uri": "https://storage.googleapis.com/generativeai-downloads/videos/ForBiggerBlazes.mp4",
                "mime_type": "video/mp4",
                "resolution": "low",
            }
        ],
        questions={
            "shows_fire": {
                "type": "noul",
                "instructions": "Does this video clip show fire or flames?",
            }
        },
    )
    ans = result.answers["shows_fire"]
    print(f"[Video] Shows Fire P(Yes):  {ans.noul:.3f} | Output Tokens: {result.usage.total_output_tokens}")


if __name__ == "__main__":
    test_audio_decision()
    test_video_decision()
