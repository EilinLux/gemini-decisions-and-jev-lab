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

if **name** == "**main**":
test_audio_decision()
test_video_decision()
