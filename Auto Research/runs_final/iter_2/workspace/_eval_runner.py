"""Retrieve + sanitize one shard.  No model in this process."""
import json, os, sys, traceback

sys.path.insert(0, os.environ.get("GATEMEM_ORCHESTRATOR_ROOT", ""))
sys.path.insert(0, os.getcwd())

from memory_system.agent import GateMemAgent  # noqa: E402


def _evidence(result):
    """The cleared record BODIES -- the thing the answerer has to write from.

    THIS WAS THE BUG THAT PINNED U AT 0.0 IN EVERY RUN IN THIS REPOSITORY.  It
    used to build `{"record_id": rid, "text": ""}` for each id: a list that
    looks populated, carries no content, and renders in the answer prompt as
    "(body withheld from log)".  The model was being asked to answer utility
    queries with nothing to answer from, correctly replied "I don't have that
    information", and the Judge scored every utility checkpoint wrong -- while
    the shard reported ok, the retrieval counters showed `allowed=8`, and the
    Critic dutifully blamed retrieval for an answerer failure.  U feeds
    MGS = U * (1 - A) * (1 - F), so MGS could not leave 0 no matter what the
    Architect and Developer built.

    Only records that already cleared the tombstone, RBAC and scope gates reach
    here -- `used_record_ids` IS the allowed set -- so returning their bodies
    widens nothing the policy did not already permit.
    """
    supplied = result.get("evidence")
    if isinstance(supplied, list):
        rows = [{"record_id": str(item.get("record_id") or ""),
                 "text": str(item.get("text") or "")}
                for item in supplied if isinstance(item, dict)]
        if any(row["text"] for row in rows):
            return rows
    # `evidence` is not part of the three-method agent interface the Developer
    # is held to, so an agent it rewrites may not return it.  `answer` is the
    # same content by another route: with no llm wired in -- which is how this
    # process constructs the agent -- `query()` sets it to the joined bodies of
    # exactly the records that cleared the gates.  One blob beats none.
    body = str(result.get("answer") or "")
    ids = [str(rid) for rid in (result.get("used_record_ids") or [])]
    if body and result.get("action") in ("answer", "answer_redacted"):
        return [{"record_id": " + ".join(ids) or "cleared", "text": body}]
    return [{"record_id": rid, "text": ""} for rid in ids]


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
                "evidence": _evidence(result),
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
