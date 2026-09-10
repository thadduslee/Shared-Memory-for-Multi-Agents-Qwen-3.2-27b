"""Smoke test: run the freshly written agent on sample medical checkpoints."""
import json, os, sys, traceback

sys.path.insert(0, os.environ.get("GATEMEM_ORCHESTRATOR_ROOT", ""))
sys.path.insert(0, os.getcwd())


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    try:
        from memory_system.agent import GateMemAgent
    except Exception:
        print("SMOKE_RESULT=" + json.dumps({"ok": False, "error": "import failed"}))
        traceback.print_exc()
        return 1

    try:
        import config as orch_config
        from gatemem_adapter import (VALID_ACTIONS, assert_no_hidden_fields,
                                     load_medical_dataset, strip_hidden_fields)
        if orch_config.MOCK_MODE:
            from mocks.dataset import build_mock_dataset
            dataset = build_mock_dataset()
        else:
            dataset = load_medical_dataset(orch_config.GATEMEM_DATA_DIR)
    except Exception:
        print("SMOKE_RESULT=" + json.dumps({"ok": False, "error": "dataset load failed"}))
        traceback.print_exc()
        return 1

    checkpoints = sorted(dataset.checkpoints, key=lambda c: c["checkpoint_id"])[:n]
    results, failures = [], []
    for cp in checkpoints:
        try:
            episode = dataset.episodes_by_id[cp["episode_id"]]
            agent = GateMemAgent(":memory:")
            agent.reset(episode)
            for turn in dataset.turns_up_to(cp["episode_id"], cp["as_of_turn_id"]):
                agent.ingest(turn)
            safe = strip_hidden_fields(cp)
            assert_no_hidden_fields(safe, where="smoke checkpoint")
            out = agent.query(safe)
            assert out.get("action") in VALID_ACTIONS, "invalid action: %r" % out.get("action")
            assert isinstance(out.get("answer"), str), "answer must be a string"
            results.append({"checkpoint_id": cp["checkpoint_id"], "action": out["action"]})
        except Exception as exc:
            failures.append({"checkpoint_id": cp["checkpoint_id"], "error": repr(exc)})
            traceback.print_exc()

    payload = {"ok": not failures, "n": len(checkpoints),
               "results": results, "failures": failures}
    print("SMOKE_RESULT=" + json.dumps(payload))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
