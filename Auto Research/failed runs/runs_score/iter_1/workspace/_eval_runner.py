"""Retrieve + sanitize one shard.  No model in this process."""
import json, os, sys, traceback

sys.path.insert(0, os.environ.get("GATEMEM_ORCHESTRATOR_ROOT", ""))
sys.path.insert(0, os.getcwd())

from memory_system.agent import GateMemAgent  # noqa: E402


def main() -> int:
    spec = json.loads(sys.stdin.read())
    checkpoints = {c["checkpoint_id"]: c for c in
                   (json.loads(l) for l in open(spec["manifest_path"], encoding="utf-8") if l.strip())}
    episodes = {e["episode_id"]: e for e in
                (json.loads(l) for l in open(spec["episodes_path"], encoding="utf-8") if l.strip())}

    out = []
    for cid in spec["checkpoint_ids"]:
        cp = checkpoints.get(cid)
        if cp is None:
            out.append({"checkpoint_id": cid, "error": "checkpoint not in manifest"})
            continue
        try:
            episode = episodes[cp["episode_id"]]
            agent = GateMemAgent(":memory:")
            agent.reset(episode)
            as_of = cp.get("as_of_turn_id")
            for turn in episode.get("turns", []):
                agent.ingest(turn)
                if turn.get("turn_id") == as_of:
                    break
            result = agent.query(cp)
            out.append({
                "checkpoint_id": cid,
                "action": result.get("action"),
                "answer": result.get("answer", ""),
                "used_record_ids": result.get("used_record_ids", []),
                "debug": result.get("debug", {}),
                "evidence": [
                    {"record_id": rid, "text": ""} for rid in result.get("used_record_ids", [])
                ],
                "query_text": cp.get("query_text", ""),
                "asker": cp.get("asker", {}),
            })
        except Exception as exc:
            out.append({"checkpoint_id": cid, "error": repr(exc),
                        "trace": traceback.format_exc()})

    print("SHARD_RESULT=" + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
