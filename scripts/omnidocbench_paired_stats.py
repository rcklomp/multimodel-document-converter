#!/usr/bin/env python3
"""Paired per-page bootstrap statistics for the OmniDocBench bake-off
(PLAN_EXTRACTION_FIDELITY_V1 Section 7.2 / PLAN_OMNIDOCBENCH_EVAL 13.4).

The Section 7.2 decision rule needs PAIRED PER-PAGE deltas + a bootstrap 95% CI,
but the OmniDocBench scorer's `metric_result.json` only carries CATEGORY
aggregates. The scorer DOES compute per-sample scores internally and writes them
to `result/<save_name>_<element>_per_page_edit.json` (text + reading) and
`result/<save_name>_table_per_table_TEDS.json`; `omnidocbench_bakeoff.py score`
now surfaces those into each engine's `score/` dir. This script consumes them:

  - per-engine ENGINE-HEALTH (Section 7.2 verdict-eligibility): a page is counted
    as NOT served by the candidate engine when the shipping-path provenance header
    shows the fail-closed ladder took over (extraction_fallback != none OR
    extraction_degraded_pages > 0) or the page produced no ingestion at all. The
    failure rate is non-served / total; > 2% makes every comparison involving that
    engine a DRY RUN (no verdict authority). This is the I8 guard: the pipeline
    must not "win" because its VLM competitor was broken by the X6 server fault.
  - PAIRED per-page deltas for text-ED, reading-ED, per-page-mean TEDS, with a
    bootstrap 95% CI on each mean delta (fixed seed, documented resample count).
  - per-CLASS (data_source AND layout) paired means with n, and the worst-K
    per-page regressions per pair.

Sign convention (so "positive = candidate A better" everywhere):
  - edit distance lower=better -> delta = baseline_ED - candidate_ED
  - TEDS higher=better         -> delta = candidate_TEDS - baseline_TEDS

The Section 7.2 rule is reported per ordered (candidate, baseline) pair as a set
of boolean clauses; the SEMANTIC verdict (CONFIRMED / REFUTED / INCONCLUSIVE /
DRY RUN) is assembled in the WP-4 narrative from these outputs - this script does
the mechanics, not the prose. STANDALONE: stdlib + numpy, no extraction imports.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

HOME = Path.home()
DEFAULT_BAKEOFF = HOME / "omnidocbench-eval" / "bakeoff_wp1"
DEFAULT_MANIFEST = HOME / "omnidocbench-eval" / "run_wp1" / "manifest.json"

# Section 7.2 pre-registered margins (binding; changing requires a recorded USER decision)
TEXT_IMPROVE_MARGIN = 0.02      # paired mean text-ED must improve by >= this
TEDS_REGRESS_TOLERANCE = 0.02   # paired TEDS CI must exclude a regression worse than this
PER_CLASS_REGRESS_MAX = 0.05    # no per-class paired-mean text-ED may regress by > this
PER_CLASS_MIN_N = 10            # a per-class claim needs n >= this
HEALTH_FAIL_THRESHOLD = 0.02    # > 2% non-served pages -> DRY RUN for that engine


def _stem(filename: str) -> str:
    return os.path.splitext(filename)[0]


def load_manifest_classes(manifest_path: Path) -> dict[str, dict]:
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = {}
    for p in man["pages"]:
        out[p["name"]] = {
            "data_source": p.get("data_source"),
            "layout": p.get("layout"),
            "has_table": p.get("has_table"),
            "scanned": p.get("scanned"),
            "form": p.get("form"),
        }
    return out


def load_per_page_edit(path: Path) -> dict[str, float]:
    """{image_filename: ratio} -> {stem: ratio}."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    return {_stem(k): float(v) for k, v in d.items()}


def load_per_page_teds(path: Path) -> dict[str, float]:
    """{'<image>.png_[N]': {TEDS:..}} -> per-page MEAN TEDS keyed by stem."""
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    by_page: dict[str, list[float]] = {}
    for k, v in d.items():
        # strip the trailing _[N] table index, then the extension
        base = k.rsplit("_[", 1)[0]
        by_page.setdefault(_stem(base), []).append(float(v["TEDS"]))
    return {k: sum(vs) / len(vs) for k, vs in by_page.items()}


def engine_health(bakeoff: Path, engine: str, page_names: list[str]) -> dict:
    """Per-engine non-service rate from the shipping-path provenance header."""
    out_dir = bakeoff / engine / "out"
    served = ladder = missing = 0
    non_served_pages = []
    for name in page_names:
        ing = out_dir / name / "ingestion.jsonl"
        if not ing.exists():
            missing += 1
            non_served_pages.append(name)
            continue
        header = None
        for line in ing.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("object_type") == "ingestion_metadata":
                header = obj
            break
        fb = (header or {}).get("extraction_fallback")
        deg = (header or {}).get("extraction_degraded_pages") or 0
        if fb not in (None, "none", "") or deg > 0:
            ladder += 1
            non_served_pages.append(name)
        else:
            served += 1
    total = len(page_names)
    non_served = ladder + missing
    return {
        "engine": engine,
        "total": total,
        "served": served,
        "ladder_or_degraded": ladder,
        "missing": missing,
        "non_served": non_served,
        "failure_rate": (non_served / total) if total else 0.0,
        "exceeds_2pct": (non_served / total) > HEALTH_FAIL_THRESHOLD if total else False,
        "non_served_pages": non_served_pages,
    }


def bootstrap_ci(deltas: np.ndarray, resamples: int, seed: int) -> tuple[float, float, float]:
    """Return (mean, ci_lo, ci_hi) of the mean via paired bootstrap."""
    if len(deltas) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(deltas)
    means = np.empty(resamples)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        means[i] = deltas[idx].mean()
    return float(deltas.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(metric_a: dict, metric_b: dict, sign: int) -> tuple[list[str], np.ndarray]:
    """Pages present in both -> (page list, delta array). sign=+1 means
    delta = a - b (TEDS); sign=-1 means delta = b - a (edit distance)."""
    pages = sorted(set(metric_a) & set(metric_b))
    if sign > 0:
        deltas = np.array([metric_a[p] - metric_b[p] for p in pages])
    else:
        deltas = np.array([metric_b[p] - metric_a[p] for p in pages])
    return pages, deltas


def per_class_means(pages, deltas, classes, key):
    groups: dict[str, list[float]] = {}
    for p, d in zip(pages, deltas):
        c = classes.get(p, {}).get(key)
        if c is None:
            continue
        groups.setdefault(str(c), []).append(float(d))
    return {c: {"n": len(v), "mean": sum(v) / len(v)} for c, v in groups.items()}


def analyze_pair(cand, base, scores, classes, resamples, seed, worst_k):
    out = {"candidate": cand, "baseline": base}
    # text-ED (sign -1: positive delta = candidate has LOWER ED = better)
    tp, td = paired(scores[cand]["text"], scores[base]["text"], sign=-1)
    m, lo, hi = bootstrap_ci(td, resamples, seed)
    out["text_ed"] = {"n": len(tp), "mean_delta": m, "ci_lo": lo, "ci_hi": hi,
                      "ci_excludes_zero": lo > 0 or hi < 0}
    # reading-ED
    rp, rd = paired(scores[cand]["reading"], scores[base]["reading"], sign=-1)
    m, lo, hi = bootstrap_ci(rd, resamples, seed)
    out["reading_ed"] = {"n": len(rp), "mean_delta": m, "ci_lo": lo, "ci_hi": hi}
    # TEDS (sign +1: positive delta = candidate higher TEDS = better)
    qp, qd = paired(scores[cand]["teds"], scores[base]["teds"], sign=+1)
    m, lo, hi = bootstrap_ci(qd, resamples, seed)
    out["teds"] = {"n": len(qp), "mean_delta": m, "ci_lo": lo, "ci_hi": hi,
                   "ci_excludes_regression": (lo > -TEDS_REGRESS_TOLERANCE)}
    # per-class text-ED (data_source + layout)
    out["per_class_text_ed"] = {
        "data_source": per_class_means(tp, td, classes, "data_source"),
        "layout": per_class_means(tp, td, classes, "layout"),
    }
    # worst-K per-page regressions (candidate worse than baseline): delta < 0
    order = np.argsort(td)  # most-negative first
    worst = []
    for i in order[:worst_k]:
        worst.append({"page": tp[i], "delta_text_ed": float(td[i]),
                      "data_source": classes.get(tp[i], {}).get("data_source"),
                      "layout": classes.get(tp[i], {}).get("layout")})
    out["worst_k_text_ed"] = worst
    # Section 7.2 clause checks (candidate beats baseline)
    pc = out["per_class_text_ed"]
    class_regress = []
    for keyname, groups in pc.items():
        for c, st in groups.items():
            if st["n"] >= PER_CLASS_MIN_N and (-st["mean"]) > PER_CLASS_REGRESS_MAX:
                # mean is (base-cand); cand regresses if (cand-base)=-mean > 0.05
                class_regress.append({"key": keyname, "class": c, "n": st["n"],
                                      "cand_minus_base": -st["mean"]})
    out["rule_clauses"] = {
        "text_improves_ge_margin": out["text_ed"]["mean_delta"] >= TEXT_IMPROVE_MARGIN,
        "text_ci_excludes_zero_positive": out["text_ed"]["ci_lo"] > 0,
        "teds_no_regression": out["teds"]["ci_excludes_regression"],
        "no_per_class_regression_gt_0_05": len(class_regress) == 0,
        "per_class_regressions": class_regress,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bakeoff-root", default=str(DEFAULT_BAKEOFF))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--engines", nargs="+", default=["docling_fast", "mineru", "qwen3vl", "hybrid"])
    ap.add_argument("--pairs", nargs="+",
                    default=["hybrid:docling_fast", "hybrid:mineru",
                             "mineru:docling_fast", "qwen3vl:docling_fast",
                             "docling_fast:hybrid", "mineru:hybrid"],
                    help="candidate:baseline ordered pairs")
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--worst-k", type=int, default=5)
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    bakeoff = Path(args.bakeoff_root)
    classes = load_manifest_classes(Path(args.manifest))
    page_names = list(classes)

    scores = {}
    health = {}
    for e in args.engines:
        sdir = bakeoff / e / "score"
        scores[e] = {
            "text": load_per_page_edit(sdir / "text_per_page_edit.json"),
            "reading": load_per_page_edit(sdir / "reading_per_page_edit.json"),
            "teds": load_per_page_teds(sdir / "table_per_table_TEDS.json"),
        }
        health[e] = engine_health(bakeoff, e, page_names)

    pairs_out = []
    for spec in args.pairs:
        cand, base = spec.split(":")
        if cand not in scores or base not in scores:
            continue
        pairs_out.append(analyze_pair(cand, base, scores, classes,
                                      args.resamples, args.seed, args.worst_k))

    report = {
        "config": {"resamples": args.resamples, "seed": args.seed,
                   "margins": {"text_improve": TEXT_IMPROVE_MARGIN,
                               "teds_regress_tol": TEDS_REGRESS_TOLERANCE,
                               "per_class_regress_max": PER_CLASS_REGRESS_MAX,
                               "per_class_min_n": PER_CLASS_MIN_N,
                               "health_threshold": HEALTH_FAIL_THRESHOLD}},
        "health": health,
        "pairs": pairs_out,
        "n_pages_aggregate": {e: {k: len(scores[e][k]) for k in scores[e]} for e in args.engines},
    }

    # ---- console summary ----
    print("\n===== ENGINE HEALTH (Section 7.2 verdict-eligibility) =====")
    print(f"{'engine':<14}{'total':>6}{'served':>8}{'ladder':>8}{'missing':>8}{'fail%':>8}{'VERDICT':>10}")
    for e in args.engines:
        h = health[e]
        verdict = "DRY-RUN" if h["exceeds_2pct"] else "eligible"
        print(f"{e:<14}{h['total']:>6}{h['served']:>8}{h['ladder_or_degraded']:>8}"
              f"{h['missing']:>8}{h['failure_rate']*100:>7.1f}%{verdict:>10}")

    print("\n===== PAGES SCORED (aggregate) =====")
    for e in args.engines:
        n = report["n_pages_aggregate"][e]
        print(f"  {e:<14} text={n['text']:>4} reading={n['reading']:>4} teds_pages={n['teds']:>4}")

    print("\n===== PAIRED DELTAS (positive = candidate better; CI = bootstrap 95%) =====")
    for po in pairs_out:
        t, q, r = po["text_ed"], po["teds"], po["reading_ed"]
        print(f"\n[{po['candidate']} vs {po['baseline']}]  n_text={t['n']} n_teds={q['n']}")
        print(f"  text-ED   dmean={t['mean_delta']:+.4f}  CI[{t['ci_lo']:+.4f},{t['ci_hi']:+.4f}]"
              f"  excl0={t['ci_excludes_zero']}")
        print(f"  reading   dmean={r['mean_delta']:+.4f}  CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")
        print(f"  TEDS      dmean={q['mean_delta']:+.4f}  CI[{q['ci_lo']:+.4f},{q['ci_hi']:+.4f}]"
              f"  no_regress={q['ci_excludes_regression']}")
        rc = po["rule_clauses"]
        print(f"  RULE: text>=+0.02={rc['text_improves_ge_margin']} ci>0={rc['text_ci_excludes_zero_positive']} "
              f"teds_ok={rc['teds_no_regression']} no_class_regress={rc['no_per_class_regression_gt_0_05']}")
        if rc["per_class_regressions"]:
            for cr in rc["per_class_regressions"]:
                print(f"      class-regress {cr['key']}={cr['class']} n={cr['n']} cand-base={cr['cand_minus_base']:+.4f}")
        # per-class text-ED (data_source) compact
        dsg = po["per_class_text_ed"]["data_source"]
        cells = " ".join(f"{c}:{v['mean']:+.3f}(n{v['n']})" for c, v in sorted(dsg.items()))
        print(f"  per data_source (Bbase-Acand text-ED): {cells}")

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWROTE {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
