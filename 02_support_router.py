"""
02_support_router.py — Multi-Primitive Parallel Evaluation & 3-Zone Routing
===========================================================================
What we are testing:
  1. Parallel, Isolated Scoring: Combining all three decision primitives
     (`noul`, `choice`, and `score`) in one request against a shared JSON state.
     Shared input tokens are counted once, and questions cannot see each other.
  2. Normalized Choice Distributions: Ensuring `choice` probabilities across
     `billing`, `technical`, `shipping`, and `other` sum cleanly to 1.0.
  3. Three-Zone Calibrated Thresholding: Using calibrated probabilities as
     deterministic control-flow levers in native Python:
       - > 0.90: Auto-execute workflow
       - < 0.10: Safely bypass workflow
       - 0.10 to 0.90: Escalate edge cases to human-in-the-loop review
"""
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="google.genai")
import json
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(http_options={"api_version": "v1beta"})


def route_ticket(order_state: dict, customer_message: str) -> dict:
    payload = [
        {"type": "text", "text": json.dumps(order_state) + "\n"},
        {"type": "text", "text": json.dumps({"message": customer_message})},
    ]

    start = time.perf_counter()
    result = client.decisions.create(
        model="gemini-decision-1",
        input=payload,
        questions={
            "is_refund": {
                "type": "noul",
                "instructions": "Is the customer requesting a refund or duplicate charge reversal?",
            },
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this support ticket?",
                "criteria": {
                    "billing": "Double charges, invoices, payment methods, and refunds.",
                    "technical": "Hardware defects, software bugs, or product setup issues.",
                    "shipping": "Missing packages, tracking updates, and delivery delays.",
                    "other": "General inquiries outside billing, technical, or shipping.",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket based on financial impact and churn risk?",
                "criteria": [
                    "Low: General question with no immediate impact.",
                    "Medium: Standard issue, customer is calm.",
                    "High: Active billing error or broken product.",
                    "Critical: Severe outage or explicit threat to cancel subscription.",
                ],
            },
        },
    )
    latency_ms = (time.perf_counter() - start) * 1000

    a = result.answers
    refund_prob = a["is_refund"].noul
    dept = a["department"]
    urgency = a["urgency"]

    # Three-zone calibrated decision logic
    if refund_prob > 0.90 and dept.confidence > 0.75:
        routing_decision = f"AUTO_QUEUE -> {dept.choice.upper()}"
    elif refund_prob < 0.10 and dept.confidence > 0.75:
        routing_decision = f"STANDARD_QUEUE -> {dept.choice.upper()}"
    else:
        routing_decision = "HUMAN_ESCALATION (Low Confidence / Edge Case)"

    return {
        "latency_ms": round(latency_ms, 1),
        "routing_decision": routing_decision,
        "is_refund_prob": refund_prob,
        "department": dept.choice,
        "department_probs": dept.probabilities,
        "urgency_score": urgency.score,
        "output_tokens": result.usage.total_output_tokens,
    }


if __name__ == "__main__":
    sample_order = {"order_id": "8821", "tier": "Gold", "recent_charges_usd": [49.99, 49.99]}
    sample_msg = "I was charged twice this month. Fix it today or I cancel my membership."

    decision = route_ticket(sample_order, sample_msg)
    print(json.dumps(decision, indent=2))