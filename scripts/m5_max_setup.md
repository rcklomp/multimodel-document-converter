# M5 Max (128 GB) → VLM extraction node — setup + measurement runbook

Goal: turn the fresh MacBook Pro M5 Max into a local Qwen3-VL serving node, then
**measure** whether it makes vision-native extraction viable (the open question
since both the Mac Mini — too little RAM — and the GX10 — bandwidth-starved at
11.3 tok/s decode — failed the throughput bar). One probe run decides it.

This is operational tooling, not governance. Paste/run on the M5 once it's up.

---

## 0. Why this machine (the hypothesis we're testing)
Decode is **memory-bandwidth-bound**; Apple Max chips have ~2× the GB10's
bandwidth, *and* their GPU prefills far faster (a 2021 M1 Max already beat the
2025 GB10 on both axes — omlx.ai: PP 162 / TG 15.1 tok/s at 8-bit). 128 GB kills
the Mac Mini's RAM-starvation failure. So the M5 Max *should* be the node that
works. **But measure, don't extrapolate** — that's the whole point of this kit.

## 1. macOS base
```bash
xcode-select --install                      # Command Line Tools
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install miniforge git
```

## 2. Two SEPARATE envs — do not merge them
The M5 runs (a) a standalone **VLM server** and (b) optionally the **pipeline/probe
client**. They communicate only over HTTP (`VLM_NATIVE_ENDPOINT`), share no Python
process, and have **conflicting dependency sets** — so they get **independent envs**:

- **Project env (`mmrag-v2`, Python 3.10):** the pipeline + probe client. Pinned to
  3.10 only because `docling==2.86.0` caps `transformers`. ⚠️ **Do NOT `pip install
  mlx-vlm` into this env** — mlx-vlm needs a newer `transformers` than Docling allows;
  that collision is what bit the first M5 attempt. (The 3.10 cap itself is an
  unverified inheritance from v1.0.0 — see backlog item, it may be liftable.)
- **VLM-server env (separate, §3):** whatever Python/transformers mlx-vlm wants.

```bash
git clone <gitea-or-github-remote> MM-Converter && cd MM-Converter
# Project/client env — ONLY needed if this box also runs the pipeline or probe:
conda env create -f environment.yml && conda activate mmrag-v2
pip install -e ".[dev]"                      # gives the probe its deps (pymupdf, etc.)
```

## 3. Serve Qwen3-VL on an OpenAI-compatible endpoint (:8000)
**PRIMARY: `mlx-vlm` (Blaizzy) — its own server.** Decision 2026-05-31 after an
inference-server survey (see `docs/paper/FINDINGS_LOG.md`): oMLX is tuned for
*coding agents* — its headline two-tier hot-RAM/cold-SSD KV cache pays off only
on long, **reused** prefixes. Our workload is the opposite (single-shot pages,
one big image, no shared prefix), so that cache never hits and its SSD tier
*thrashed* under the Mac Mini's RAM pressure ("SSD cache write queue full",
throughput collapse 57→331 s/page). `mlx-vlm` is the VLM-native **upstream** that
oMLX/LM Studio both build on — Qwen3-VL lands there first, and its cache is the
*vision-feature* kind that actually fits us. It also removes the SSD-tier failure
mode entirely. Leanest, most directly debuggable path.

```bash
# Download both — 128 GB fits either trivially (8B 8-bit ≈ 8 GB, 4-bit ≈ 4 GB).
huggingface-cli download mlx-community/Qwen3-VL-8B-Instruct-8bit
huggingface-cli download mlx-community/Qwen3-VL-8B-Instruct-4bit

export MLX_API_KEY="<any token — set so the probe's bearer header is exercised>"

# Dedicated env for the server — NOT mmrag-v2 (see §2). Let pip pull the
# transformers/mlx versions mlx-vlm actually requires; check its own
# `requires-python` / pinned deps first rather than assuming 3.10 works:
conda create -y -n vlm-server python=3.12 && conda activate vlm-server
pip install mlx-vlm                          # pulls a recent transformers — fine, it's isolated
python -m mlx_vlm.server --model mlx-community/Qwen3-VL-8B-Instruct-8bit --port 8000
```

### Operating the server — explicit load/unload (NOT a boot daemon)
Once installed, drive it with `scripts/vlm_serve.sh` rather than the raw command.
**Deliberately not a `launchd` LaunchAgent:** a KeepAlive/RunAtLoad agent would
silently park ~9 GB of model in RAM at every login, and you'd only discover it
under memory pressure. The wrapper instead loads the model **only when you ask**,
detaches so it outlives your shell/SSH session (the M1 can keep hitting the
endpoint), and `status` always prints the live RAM footprint so a resident model
is never invisible.

```bash
./scripts/vlm_serve.sh start     # load (~9GB), detach, wait until /v1/models is ready
./scripts/vlm_serve.sh status    # ● running  pid=…  RAM=9.5GB  model=…  + endpoint URL
./scripts/vlm_serve.sh stop      # free the model from RAM
./scripts/vlm_serve.sh restart
```
Reads the key from `~/.mlx_api_key`, binds `0.0.0.0:8000` (set `VLM_HOST=127.0.0.1`
for loopback-only), logs to `~/Library/Logs/mlx-vlm.log`. Override target with
`VLM_MODEL` / `VLM_PORT` / `VLM_HOST`. A browser chat client for this endpoint
(uses the *server's* loaded model, not a second copy) ships at
`scripts/m5_vlm_chat.html` — open it locally or `scp` it to the client box.

**FALLBACK / A-B comparand: oMLX** (the incumbent; the Mac Mini at `10.0.10.246`
runs it). Worth keeping because 128 GB removes the RAM pressure that made its SSD
tier thrash, so on the M5 it *may* be fine — but that's a hypothesis to test, not
a default. Confirm the exact load/serve flags from the Mac Mini
(`ssh ronmeijer@10.0.10.246` — it already runs this; CLI flag names are the one
thing this runbook can't verify blind), then serve the same model on a different
port and probe both (§4). **Do not silently reinstate oMLX as primary without the
probe numbers — the survey already rejected it on architecture grounds.**

**Text judge (separate concern):** LM Studio is the better long-term judge host
(mature, headless `llmster`, Apple-tuned for M5) — but its unified MLX engine has
**not** migrated Qwen3-VL vision yet (only Gemma 3 + Pixtral), so it is **not** a
vision-extraction option today.

Health: `curl -s -H "Authorization: Bearer $MLX_API_KEY" http://localhost:8000/v1/models`

## 4. Measure (the decisive step) — run the profiler
```bash
# 8-bit (quality-leaning) — with the fidelity check on the known invoice page
python scripts/vlm_profile_probe.py \
  --endpoint http://localhost:8000/v1 --model Qwen3-VL-8B-Instruct-8bit \
  --api-key "$MLX_API_KEY" \
  --pdf data/business_form/0013_140302111325_001.pdf --page 0 \
  --expect "Level Automotive,Castrol,1.949,60"

# repeat for 4-bit (--model ...-4bit) and a figure page
python scripts/vlm_profile_probe.py --endpoint http://localhost:8000/v1 \
  --model Qwen3-VL-8B-Instruct-4bit --api-key "$MLX_API_KEY" \
  --pdf data/academic_journal/IRJET_Modeling_of_Solar_PV_system_under.pdf --page 3
```

## 5. Interpret (the go/no-go)
The probe prints prefill, decode tok/s, s/page, and a fidelity verdict. Compare:

| | prefill | decode | s/page | verdict |
|---|--:|--:|--:|---|
| GX10 BF16 (rejected) | ~57 | 11.3 | ~132 | too slow |
| M1 Max 8-bit (bench) | 162 | 15.1 | — | reference |
| **M5 Max + mlx-vlm (primary)** | ? | ? | ? | **fill in** |
| **M5 Max + oMLX (A-B)** | ? | ? | ? | **fill in** |

Run the **same probe against both servers** (mlx-vlm on :8000, oMLX on another
port) on the large-context magazine page that thrashed the Mac Mini — that page
is the real stability test, not the invoice. Pick the winner on s/page **and**
the fidelity check, the way the FP8 fraud was caught. If oMLX matches mlx-vlm on
the M5 (the 128 GB may neutralise its SSD-tier thrash), either is fine; if it
regresses or stalls on the large page, mlx-vlm stands as primary.

- **s/page is the decision number.** Canonical set ≈ 720 VLM pages → hours =
  720 × s-page / 3600. If ~35–50 s/page (extrapolated), that's ~7–10 h — an
  overnight job, viable. If it's GX10-class (~130 s), local stays non-viable.
- **8-bit vs 4-bit:** prefer 8-bit if fidelity holds (the probe's `--expect`
  must pass — MLX 4-bit was *quality-fine* on the Mac Mini, unlike vLLM-FP8 which
  hallucinated blank pages; verify, don't assume). 128 GB means you never quantise
  to fit — only for speed.
- **Thermals:** it's a laptop. A 10-h grind will throttle somewhat; a multi-day
  full-corpus run on a portable is still dubious — size the job accordingly.

## 6. If viable → wire it in
- Point extraction at the M5: `export VLM_NATIVE_ENDPOINT=http://<m5-host>:8000/v1`
  `VLM_NATIVE_MODEL=Qwen3-VL-8B-Instruct-8bit` (production reads these).
- The GX10 reverts to **judge-only** (it's stable + fine for the text judge).
- Add a dated entry to `docs/paper/FINDINGS_LOG.md` with the measured numbers —
  it confirms or corrects the F7 bandwidth extrapolation (the backlog item).

## 7. Backlog — investigate the project's Python 3.10 cap (separate task)
`requires-python = ">=3.10,<3.11"` has been in place since the v1.0.0 initial
commit (2026-01-05) and was never re-examined at V3 kickoff. No dependency in
`pyproject.toml` visibly demands `<3.11`; the real runtime ceiling is
`docling==2.86.0`'s `transformers` pin (Surya is **not** a dependency — OCR is
`python-doctr`). To lift it safely: in a throwaway env, resolve the project deps
on 3.11 then 3.12, run `pytest tests/` green, and only then relax the cap (likely
to `>=3.10,<3.13`) + bump `environment.yml`. This is a deliberate project-wide
change, not a casual flip — but it does **not** block the M5 server, which runs in
its own env (§2) regardless of what the project pins.
