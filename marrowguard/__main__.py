"""marrow-guard command line.

    python3 -m marrowguard make-data     # deterministic synthetic recordings
    python3 -m marrowguard calibrate     # fit profile + fix the threshold (train split only)
    python3 -m marrowguard evaluate      # held-out rates + swap test -> report.json
    python3 -m marrowguard score <file>  # score recordings against the fixed profile
    python3 -m marrowguard selftest      # format + determinism checks
    python3 -m marrowguard run           # all of the above in order (what ./run.sh calls)
"""
import argparse
import json
import os
import sys

from . import format as fmt
from .features import extract, vector
from . import profile as prof
from .synth import PERSONAS, STYLES, build_dataset, simulate

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "model")


def _paths():
    return {
        "train": os.path.join(DATA, "train.jsonl"),
        "eval": os.path.join(DATA, "eval.jsonl"),
        "model": os.path.join(MODEL_DIR, "profile.json"),
        "config": os.path.join(MODEL_DIR, "config.json"),
        "report": os.path.join(ROOT, "report.json"),
    }


def _records(path):
    return list(fmt.load_jsonl(path))


def cmd_make_data(args):
    os.makedirs(DATA, exist_ok=True)
    pairs = build_dataset(sessions_per_cell=args.sessions_per_cell)
    train = [r for split, r in pairs if split == "train"]
    evals = [r for split, r in pairs if split == "eval"]
    p = _paths()
    fmt.dump_jsonl(p["train"], train)
    fmt.dump_jsonl(p["eval"], evals)

    sample_dir = os.path.join(DATA, "sample")
    os.makedirs(sample_dir, exist_ok=True)
    for style in STYLES:
        for persona in PERSONAS:
            rec = simulate(style, persona, f"{style}-{persona}-sample", seed=7)
            rec["meta"] = {"style": style, "persona": persona, "split": "sample"}
            fmt.dump_jsonl(os.path.join(sample_dir, f"{style}-{persona}.jsonl"), [rec])
    total_events = sum(len(r["events"]) for r in train + evals)
    print(f"make-data: train={len(train)} eval={len(evals)} sessions, {total_events} events, "
          f"3 styles x 4 personas, sample/ has one file per cell")
    return 0


def cmd_calibrate(args):
    p = _paths()
    if not os.path.exists(p["train"]):
        print("calibrate: run make-data first", file=sys.stderr)
        return 2
    train = _records(p["train"])

    enroll = [r for r in train if r["meta"]["persona"] in ("owner", "owner_tired")]
    labels = {}
    for r in train:
        labels[r["session_id"]] = "owner" if r["meta"]["persona"] in ("owner", "owner_tired") else "other"
    threshold, model, diag = prof.calibrate(enroll, train, labels, k=args.k)

    os.makedirs(MODEL_DIR, exist_ok=True)
    model["threshold"] = threshold
    model["calibrated_on"] = "train split only; eval split never read"
    model["enrol_set"] = "every real-user session in the train split (ordinary and tired days), k routines"
    model["enroll_sessions"] = len(enroll)
    with open(p["model"], "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2, sort_keys=True)
    with open(p["config"], "w", encoding="utf-8") as fh:
        json.dump({
            "accept_threshold": threshold,
            "cap": prof.CAP,
            "k": args.k,
            "fixed_on": "train split before any evaluation session was scored",
            "train_hash": model["train_hash"],
            "limits": prof.LIMITS,
        }, fh, indent=2, sort_keys=True)

    best = max(diag, key=lambda d: (d["balanced_accuracy"], -abs(d["threshold"] - threshold)))
    print(f"calibrate: enrol={len(enroll)} sessions, threshold={threshold:.2f} "
          f"(train balanced accuracy {best['balanced_accuracy']:.3f}, "
          f"train owner accept {best['train_owner_accept']:.2f}, train other accept {best['train_other_accept']:.2f})")
    return 0


def cmd_evaluate(args):
    p = _paths()
    for need in ("eval", "model", "config"):
        if not os.path.exists(p[need]):
            print(f"evaluate: missing {p[need]}", file=sys.stderr)
            return 2
    with open(p["model"], encoding="utf-8") as fh:
        model = json.load(fh)
    with open(p["config"], encoding="utf-8") as fh:
        config = json.load(fh)
    if config["train_hash"] != model["train_hash"]:
        print("evaluate: config and profile were calibrated on different train sets", file=sys.stderr)
        return 2
    evals = _records(p["eval"])
    train = _records(p["train"])
    result = prof.evaluate(model, config["accept_threshold"], evals, train, k=config.get("k", 3))
    result["threshold"] = config["accept_threshold"]
    result["threshold_source"] = config["fixed_on"]
    result["data"] = {"eval_sessions": len(evals), "train_sessions": len(train),
                      "synthetic": True,
                      "caveat": "all recordings are synthetic; these rates describe the pipeline "
                                "on the shipped fake data, not accuracy on real users"}
    with open(p["report"], "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(prof.fmt_table(result))
    print(f"\nreport: {p['report']}")
    return 0 if result["passed"] else 1


def cmd_score(args):
    p = _paths()
    with open(p["model"], encoding="utf-8") as fh:
        model = json.load(fh)
    with open(p["config"], encoding="utf-8") as fh:
        config = json.load(fh)
    thr = config["accept_threshold"]
    for rec in fmt.load_jsonl(args.path):
        iv, sv = prof.session_vectors(rec)
        sc, ok = prof.decide(model, thr, iv, sv)
        print(f"{rec['session_id']:28s} score={sc:.3f} threshold={thr:.2f} "
              f"{'OPEN' if ok else 'DOUBT'}")
    return 0


def cmd_selftest(args):
    checks = []
    rec = simulate("standard", "owner", "selftest-1", seed=11)
    fmt.validate(rec)
    checks.append(("recording validates against inputtrace/1", True))
    again = simulate("standard", "owner", "selftest-1", seed=11)
    checks.append(("synthetic recorder is deterministic for a seed", json.dumps(rec, sort_keys=True) == json.dumps(again, sort_keys=True)))
    feats = extract(rec)
    checks.append(("features cover every declared name", set(feats) == set(__import__("marrowguard.features", fromlist=["FEATURES"]).FEATURES)))
    checks.append(("pointer efficiency is a fraction", 0.0 <= feats["ptr_eff"] <= 1.0))
    checks.append(("feature extraction ignores the declared source",
                   extract(dict(rec, source="unknown", session_id="x")) == feats))
    bad = json.loads(json.dumps(rec))
    bad["events"][0]["x"] = 1.5
    try:
        fmt.validate(bad)
        checks.append(("invalid coordinates are refused", False))
    except fmt.FormatError:
        checks.append(("invalid coordinates are refused", True))
    for name, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return 0 if all(ok for _, ok in checks) else 1


def cmd_run(args):
    for step in (cmd_make_data, cmd_calibrate, cmd_evaluate):
        rc = step(args)
        if rc:
            return rc
        print()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="marrowguard", description="Marrow prototype: is the person at the computer the enrolled user?")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-data", help="write deterministic synthetic recordings")
    m.add_argument("--sessions-per-cell", type=int, default=16)
    m.set_defaults(fn=cmd_make_data)
    for name, fn in (("calibrate", cmd_calibrate), ("evaluate", cmd_evaluate), ("run", cmd_run)):
        s = sub.add_parser(name)
        s.add_argument("--sessions-per-cell", type=int, default=16)
        s.add_argument("--k", type=int, default=5, help="enrolment routines kept")
        s.set_defaults(fn=fn)
    s = sub.add_parser("score", help="score recordings in a .jsonl file")
    s.add_argument("path")
    s.set_defaults(fn=cmd_score)
    s = sub.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())