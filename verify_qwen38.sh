#!/usr/bin/env bash
set -euo pipefail

# Smoke-test the reasoning-parser setup before starting a long sweep.
# PASS requires: reasoning split into reasoning_content, `content` free of
# <think> tags, and the harness's loose JSON parser accepting the content.

source "$HOME/gatemem/.venv-nb/bin/activate"
cd "$HOME/GateMem"

python - <<'PY'
import json, sys, requests
sys.path.insert(0, '/home/thaddus/GateMem')
from bench.agents.a_mem import AMemAgent

prompt = (
    "You are extracting agentic memory metadata for a single multi-party medical interaction turn.\n"
    "Return STRICT JSON only with keys: summary (string), keywords (array of strings), "
    "entities (array of principal_ids or names), categories (array of short tags).\n"
    "Keep summary <= 30 words.\n\nturn_id: t032\nspeaker_principal_id: patient_lila_chen\n"
    "speaker_role: patient\nrecord_refs: []\nmemory_ops: []\n"
    "text: Please delete my old address from the record, I moved to Harbor House last month.\n"
)

r = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer dummy"},
    json={"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": prompt}],
          "temperature": 0.2, "max_tokens": 16384},
    timeout=1800,
)
msg = r.json()["choices"][0]["message"]
content = msg.get("content") or ""
# vLLM 0.26 returns the split-off trace as "reasoning"; older builds and some
# other OpenAI-compatible servers call it "reasoning_content". Accept either.
reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""

checks = {
    "reasoning split out of content": bool(reasoning),
    "content has no <think> tags": "<think>" not in content and "</think>" not in content,
    "content is non-empty": bool(content.strip()),
    "harness JSON parser accepts content": isinstance(AMemAgent._parse_json_loose(content), dict),
}
for name, ok in checks.items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

print(f"\nreasoning_chars={len(reasoning)} content_preview={content[:100]!r}")
print("\nRESULT:", "PASS - safe to start the sweep" if all(checks.values()) else "FAIL - do not start the sweep")
sys.exit(0 if all(checks.values()) else 1)
PY
