from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(http_options=types.HttpOptions(api_version="v1beta"))

def main():
input_text = "Customer returned the sweater because they preferred green over blue."

```
result = client.decisions.create(
    model="gemini-decision-1",
    input=input_text,
    questions={
        "is_defect": {
            "type": "noul",
            "instructions": "Is the customer reporting a physical product defect?",
            "criteria": {
                "true_description": "The item arrived broken, torn, stained, or malfunctioning.",
                "false_description": "Preference, color choice, sizing, shipping delay, or pricing.",
            },
        }
    },
)

ans = result.answers["is_defect"]
print(f"Input:         {input_text}")
print(f"P(Yes):        {ans.noul:.4f}")
print(f"Confidence:    {ans.confidence:.4f}")
print(f"Input Tokens:  {result.usage.total_input_tokens}")
print(f"Output Tokens: {result.usage.total_output_tokens}")

```

if **name** == "**main**":
main()
