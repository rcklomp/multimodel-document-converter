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

## 2. Project + env (for the probe + pipeline)
```bash
git clone <gitea-or-github-remote> MM-Converter && cd MM-Converter
conda env create -f environment.yml && conda activate mmrag-v2
pip install -e ".[dev]"                      # gives the probe its deps (pymupdf, etc.)
```

## 3. Serve Qwen3-VL on an OpenAI-compatible endpoint (:8000)
The existing stack uses **omlx-server** (the Mac Mini at `10.0.10.246` runs it,
and the omlx.ai benchmark used it). Install it the same way the Mac Mini did
(omlx.ai ships pip/Homebrew + a DMG; use pip/Homebrew on a server node so
structured-output works), then load both precisions and serve:

```bash
# Download both — 128 GB fits either trivially (8B 8-bit ≈ 8 GB, 4-bit ≈ 4 GB).
huggingface-cli download mlx-community/Qwen3-VL-8B-Instruct-8bit
huggingface-cli download mlx-community/Qwen3-VL-8B-Instruct-4bit

# Serve on :8000 with an API key. ⚠️ CONFIRM the exact omlx load/serve invocation
# from the Mac Mini (`ssh ronmeijer@10.0.10.246` ... it already runs this) — the
# CLI flag names are the one thing this runbook can't verify blind.
export MLX_API_KEY="<same key as Mac Mini>"
# omlx serve ... --model mlx-community/Qwen3-VL-8B-Instruct-8bit --port 8000   # confirm flags
```
**Alternative if omlx is awkward to reproduce:** `mlx-vlm` ships its own
OpenAI-compatible server —
`pip install mlx-vlm && python -m mlx_vlm.server --model mlx-community/Qwen3-VL-8B-Instruct-8bit --port 8000`.

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
| **M5 Max — measured** | ? | ? | ? | **fill in** |

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
