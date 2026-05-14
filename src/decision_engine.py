import json

# Load files
with open("outputs/if_inference.json", "r") as f:
    if_result = json.load(f)

with open("outputs/rf_inference.json", "r") as f:
    rf_result = json.load(f)

# Extract IF score correctly (IMPORTANT FIX)
if_score = if_result["results"][0]["anomaly_score"]

# RF score (safe fallback)
rf_score = rf_result.get("risk_score", 0)

# Combine
risk_score = (if_score + rf_score) / 2

# Decision logic
if risk_score > 0.7:
    decision = "BLOCK"
elif risk_score > 0.4:
    decision = "REVIEW"
else:
    decision = "ALLOW"

# Output
output = {
    "if_score": if_score,
    "rf_score": rf_score,
    "final_risk_score": risk_score,
    "decision": decision
}

print(json.dumps(output, indent=4))