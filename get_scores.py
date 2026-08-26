import json
import os

# Using the exact absolute path from your example
out_dir = "/home/thaddus/GateMem/outputs"

print(f"{'Baseline':<40} | {'Judge':<10} | {'U (%)':<8} | {'A (%)':<8} | {'F (%)':<8} | {'MGS (%)':<8}")
print("-" * 98)

if not os.path.exists(out_dir):
    print(f"Cannot find directory: {out_dir}")

# os.walk will reliably find all subfolders, no matter how they are nested
for root, dirs, files in sorted(os.walk(out_dir)):
    if "summary.json" not in files:
        continue

    summary_path = os.path.join(root, "summary.json")
    baseline = os.path.basename(root)

    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, ValueError):
        print(f"{baseline:<40} | Error parsing JSON")
        continue

    if not isinstance(data, dict) or not data:
        print(f"{baseline:<40} | File is empty")
        continue

    # Judged runs carry an llm_judge block; otherwise the top-level numbers
    # are still the rule-based scorer's.
    judge = data.get("judge_key") or ("llm" if "llm_judge" in data else "rule-based")

    u_score = data.get("utility_accuracy", 0) * 100
    a_score = data.get("privacy_leakage_rate", 0) * 100
    f_score = data.get("deletion_leakage_rate", 0) * 100
    mgs_score = data.get("compliance_utility_score", 0) * 100

    print(
        f"{baseline:<40} | {judge:<10} | {u_score:<8.2f} | {a_score:<8.2f} | "
        f"{f_score:<8.2f} | {mgs_score:<8.2f}"
    )
