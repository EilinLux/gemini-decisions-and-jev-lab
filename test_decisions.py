import pytest
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

@pytest.fixture(scope="module")
def client():
return genai.Client(http_options=types.HttpOptions(api_version="v1beta"))

def test_zero_output_tokens_and_noul_bounds(client):
"""Verify output tokens are strictly 0 and noul stays inside [0.01, 0.99]."""
res = client.decisions.create(
model="gemini-decision-1",
input="The glass screen arrived completely shattered into pieces.",
questions={
"damaged": {
"type": "noul",
"instructions": "Did the item arrive physically damaged?",
}
},
)
ans = res.answers["damaged"]
assert res.usage.total_output_tokens == 0
assert 0.01 <= ans.noul <= 0.99
assert ans.noul > 0.90

def test_choice_probabilities_sum_to_one(client):
"""Verify choice probabilities form a normalized distribution summing to ~1.0."""
res = client.decisions.create(
model="gemini-decision-1",
input="My credit card was billed twice for invoice #402.",
questions={
"team": {
"type": "choice",
"instructions": "Which team owns this ticket?",
"criteria": {
"billing": "Payments and invoices",
"shipping": "Delivery and tracking",
"other": "Anything else",
},
}
},
)
ans = res.answers["team"]
assert ans.choice == "billing"
assert abs(sum(ans.probabilities.values()) - 1.0) < 1e-3

def test_score_rubric_ordering(client):
"""Verify score returns expected weighted float and legend mapping."""
res = client.decisions.create(
model="gemini-decision-1",
input="Production database is completely unreachable for all customers.",
questions={
"severity": {
"type": "score",
"instructions": "Rate the incident severity.",
"criteria": [
"Minor UI typo",
"Degraded non-critical feature",
"Total production outage",
],
}
},
)
ans = res.answers["severity"]
assert ans.score > 1.75
assert "2" in ans.legend
