"""
01_basic_noul.py — Binary Predicate Evaluation (`noul` Primitive)
==================================================================
What we are testing:
  1. Zero-Output-Token Inference: Unlike autoregressive LLMs that generate JSON
     strings token by token, `gemini-decision-1` evaluates a yes/no predicate
     (`noul`) in a single forward pass with `total_output_tokens == 0`.
  2. Explicit Boundary Criteria: Using `true_description` and `false_description`
     to separate genuine product defects from customer color/size preferences.
  3. Calibrated Probability Bounds: Verifying that binary probabilities remain
     strictly clamped inside [0.01, 0.99] alongside a mathematical `confidence`
     score (`max(noul, 1 - noul)`).
"""

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="google.genai")

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

client = genai.Client(http_options=types.HttpOptions(api_version="v1beta"))


def main():
    input_text = "Customer returned the sweater because they preferred green over blue."

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


if __name__ == "__main__":
    main()