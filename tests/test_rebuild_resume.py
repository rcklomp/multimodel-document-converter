"""v2.11 Phase 1 step 0 — rebuild script resume + retry pins.

Locks the behavior added in 2026-05-17 after the v2.10 Phase 8 cycle
crashed mid-rebuild on `[Errno 61] Connection refused` when Ollama
became unreachable. The new flags + helper functions in
`scripts/rebuild_mmrag_v2_8_for_rc1.py` exist precisely to make the
5-7 h rebuild survive transient backend hiccups without losing prior
progress.

Tested invariants:

1. CLI surface — `--collection`, `--provider`, `--model`, `--api-key`,
   `--resume`, `--no-recreate`, `--log-path` are all present.
2. `_doc_chunk_ids()` correctly extracts chunk_ids from an
   ingestion.jsonl, skipping the metadata record on line 1.
3. `_qdrant_count_chunks_for_doc()` returns 0 when Qdrant is
   unreachable (conservative behavior — forces re-ingest rather
   than silently skipping).
4. `_doc_is_fully_ingested()` applies the 90 % threshold for "fully
   ingested" (matches the ~0.44 % ingest_to_qdrant filter rate
   observed in v2.10).
5. The script imports cleanly and exposes the canonical 34-doc list.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REBUILD_SCRIPT = REPO_ROOT / "scripts" / "rebuild_mmrag_v2_8_for_rc1.py"


def _load_rebuild_module():
    spec = importlib.util.spec_from_file_location(
        "rebuild_mmrag_v2_8_for_rc1", REBUILD_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("rebuild_mmrag_v2_8_for_rc1", module)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_rebuild_script_help_lists_new_flags() -> None:
    """All v2.11 flags must be visible in --help so operators know
    they exist."""
    proc = subprocess.run(
        [sys.executable, str(REBUILD_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    help_text = proc.stdout + proc.stderr
    for flag in (
        "--collection",
        "--provider",
        "--model",
        "--api-key",
        "--resume",
        "--no-recreate",
        "--log-path",
    ):
        assert flag in help_text, f"{flag} missing from rebuild --help"


def test_canonical_34_list_present_and_complete() -> None:
    mod = _load_rebuild_module()
    assert hasattr(mod, "CANONICAL_34")
    assert len(mod.CANONICAL_34) == 34
    # Sanity: each canonical doc name corresponds to a directory under
    # output/ (or did at v2.10 rebuild time). We can't assert the
    # directory exists here (it's gitignored), but we can assert the
    # list is a sequence of plausible identifiers.
    assert all(isinstance(name, str) and name for name in mod.CANONICAL_34)
    # First doc is the one historically ingested with --recreate.
    assert mod.CANONICAL_34[0] == "HarryPotter_and_the_Sorcerers_Stone"


def test_doc_chunk_ids_skips_metadata_record(tmp_path: Path) -> None:
    mod = _load_rebuild_module()
    jsonl = tmp_path / "ingestion.jsonl"
    jsonl.write_text(
        "\n".join([
            json.dumps({"object_type": "ingestion_metadata", "doc_id": "abc"}),
            json.dumps({"chunk_id": "abc_001_text_aaaa", "doc_id": "abc"}),
            json.dumps({"chunk_id": "abc_002_text_bbbb", "doc_id": "abc"}),
            json.dumps({"chunk_id": "abc_003_image_cccc", "doc_id": "abc"}),
        ]),
        encoding="utf-8",
    )
    ids = mod._doc_chunk_ids(jsonl)
    assert ids == {"abc_001_text_aaaa", "abc_002_text_bbbb", "abc_003_image_cccc"}


def test_doc_chunk_ids_returns_empty_on_missing_file(tmp_path: Path) -> None:
    mod = _load_rebuild_module()
    missing = tmp_path / "does_not_exist.jsonl"
    assert mod._doc_chunk_ids(missing) == set()


def test_qdrant_count_chunks_returns_zero_on_unreachable() -> None:
    """The whole point of --resume is to survive Qdrant or backend
    flakes; if Qdrant is unreachable we must NOT silently treat the
    doc as 'present' (would skip an unfinished doc). Conservative
    behavior: return 0 => force re-ingest."""
    mod = _load_rebuild_module()
    # Use a localhost port that is virtually guaranteed not to host Qdrant
    # to simulate an unreachable backend.
    result = mod._qdrant_count_chunks_for_doc(
        "http://127.0.0.1:1",
        "any_collection",
        {"chunk_a", "chunk_b"},
    )
    assert result == 0


def test_qdrant_count_chunks_returns_zero_on_empty_input() -> None:
    mod = _load_rebuild_module()
    # Should short-circuit without making a network call.
    result = mod._qdrant_count_chunks_for_doc(
        "http://127.0.0.1:1", "any_collection", set()
    )
    assert result == 0


def test_doc_is_fully_ingested_threshold_at_90_percent(tmp_path: Path, monkeypatch) -> None:
    """A doc is 'fully ingested' when ≥ 90 % of its chunk_ids are
    present in the target collection (accounts for the ~0.44 %
    ingest_to_qdrant filter rate observed in v2.10)."""
    mod = _load_rebuild_module()
    # Stub the network call so we control "how many are present".
    chunk_ids = {f"c_{i:03d}" for i in range(100)}
    jsonl = tmp_path / "ingestion.jsonl"
    lines = [json.dumps({"object_type": "ingestion_metadata"})]
    lines += [json.dumps({"chunk_id": cid}) for cid in sorted(chunk_ids)]
    jsonl.write_text("\n".join(lines), encoding="utf-8")

    # Case 1: 100/100 present → fully ingested.
    monkeypatch.setattr(mod, "_qdrant_count_chunks_for_doc",
                        lambda *_a, **_kw: 100)
    done, present, expected = mod._doc_is_fully_ingested(jsonl, "url", "c")
    assert done is True
    assert (present, expected) == (100, 100)

    # Case 2: 90/100 present → fully ingested (threshold).
    monkeypatch.setattr(mod, "_qdrant_count_chunks_for_doc",
                        lambda *_a, **_kw: 90)
    done, present, expected = mod._doc_is_fully_ingested(jsonl, "url", "c")
    assert done is True

    # Case 3: 89/100 present → NOT fully ingested.
    monkeypatch.setattr(mod, "_qdrant_count_chunks_for_doc",
                        lambda *_a, **_kw: 89)
    done, present, expected = mod._doc_is_fully_ingested(jsonl, "url", "c")
    assert done is False

    # Case 4: 0/100 present → NOT fully ingested.
    monkeypatch.setattr(mod, "_qdrant_count_chunks_for_doc",
                        lambda *_a, **_kw: 0)
    done, present, expected = mod._doc_is_fully_ingested(jsonl, "url", "c")
    assert done is False


def test_resume_implies_no_recreate(monkeypatch, tmp_path: Path) -> None:
    """When --resume is set, the script must NOT pass --recreate even
    on doc 1 (re-creating the collection would defeat resume)."""
    mod = _load_rebuild_module()
    # Reach inside _ingest_one to check the cmd it builds.
    captured_cmds: list[list[str]] = []

    def fake_subprocess_run(cmd, *a, **kw):
        captured_cmds.append(list(cmd))
        class _Proc:
            returncode = 0
        return _Proc()

    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)
    # Touch a minimal jsonl so the existence check passes.
    fake_jsonl = tmp_path / "ingestion.jsonl"
    fake_jsonl.write_text(json.dumps({"object_type": "ingestion_metadata"}) + "\n",
                          encoding="utf-8")

    # Without --resume, doc 1 gets --recreate.
    mod._ingest_one(
        jsonl_path=fake_jsonl, collection="test_coll", provider="ollama",
        model=None, api_key=None, qdrant_url="http://localhost:6333",
        recreate=True,
    )
    assert "--recreate" in captured_cmds[0]

    captured_cmds.clear()
    # With recreate=False, --recreate is NOT passed.
    mod._ingest_one(
        jsonl_path=fake_jsonl, collection="test_coll", provider="ollama",
        model=None, api_key=None, qdrant_url="http://localhost:6333",
        recreate=False,
    )
    assert "--recreate" not in captured_cmds[0]


def test_provider_passthrough(monkeypatch, tmp_path: Path) -> None:
    """The rebuild script must thread --provider/--model/--api-key to
    the per-doc ingest_to_qdrant.py invocation."""
    mod = _load_rebuild_module()
    captured: list[list[str]] = []

    def fake_subprocess_run(cmd, *a, **kw):
        captured.append(list(cmd))
        class _Proc:
            returncode = 0
        return _Proc()

    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)
    fake_jsonl = tmp_path / "ingestion.jsonl"
    fake_jsonl.write_text(json.dumps({"object_type": "ingestion_metadata"}) + "\n",
                          encoding="utf-8")

    mod._ingest_one(
        jsonl_path=fake_jsonl,
        collection="mmrag_v2_8__qwen3_dashscope",
        provider="dashscope",
        model="text-embedding-v4",
        api_key="sk-fake-key-for-test",
        qdrant_url="http://localhost:6333",
        recreate=False,
    )
    cmd = captured[0]
    assert "--provider" in cmd and cmd[cmd.index("--provider") + 1] == "dashscope"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == "text-embedding-v4"
    assert "--api-key" in cmd and cmd[cmd.index("--api-key") + 1] == "sk-fake-key-for-test"
    assert "--collection" in cmd and cmd[cmd.index("--collection") + 1] == "mmrag_v2_8__qwen3_dashscope"


def test_retry_on_transient_failure(monkeypatch, tmp_path: Path) -> None:
    """A single transient failure should not abort the rebuild — the
    wrapper retries with exponential backoff. After exhausting
    retries, it returns the last non-zero rc."""
    mod = _load_rebuild_module()
    attempts: list[int] = []

    class _ProcFail:
        returncode = 1

    class _ProcOK:
        returncode = 0

    def fake_subprocess_run(cmd, *a, **kw):
        attempts.append(len(attempts) + 1)
        # Fail twice, succeed on attempt 3.
        return _ProcFail() if len(attempts) < 3 else _ProcOK()

    # Make sleeps instant so the test runs fast.
    monkeypatch.setattr(mod.subprocess, "run", fake_subprocess_run)
    monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

    fake_jsonl = tmp_path / "ingestion.jsonl"
    fake_jsonl.write_text("{}", encoding="utf-8")

    rc = mod._ingest_one(
        jsonl_path=fake_jsonl, collection="c", provider="ollama",
        model=None, api_key=None, qdrant_url="u", recreate=False,
        retries=3,
    )
    assert rc == 0
    assert len(attempts) == 3
