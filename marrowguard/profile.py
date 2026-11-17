"""Profile training, scoring and the calibration/evaluation split.

The method is a two-part matcher, not a classifier over people:

* STATE (tempo: pointer speed, sample spacing, rests, text arrival rate, session
  length) says which routine the user is in — a mouse session, a slow-pointer
  dictation session, a switch-access session. Tempo also moves when someone is
  tired or on a different device, so it is used only to route a session, never to
  judge identity.
* IDENTITY (shape and rhythm: turn angle, path efficiency, jitter, click rate,
  rhythm regularity, chunking, correction habits, channel mix) is the habit a
  person carries between devices and moods. Only this decides open/doubt.

Enrolment therefore stores a small bank of routines: a mode centre in state space
and an identity centre plus spread in identity space. The accept threshold is fixed
once on the TRAIN sessions only and then locked; the evaluation sessions never
inform the threshold, the centres or the spreads.
"""
import hashlib
import json
import math

from .features import IDENTITY_FEATURES, STATE_FEATURES, extract, state_vector, vector

CAP = 3.0  # a deviation beyond CAP robust scales counts as fully "not the same habit"
ACCEPT_THRESHOLD_SWEEP = [round(0.10 + 0.01 * i, 2) for i in range(81)]  # 0.10 .. 0.90

# Declared before the evaluation runs; the report judges the method against them.
LIMITS = {
    "max_stranger_accept_rate": 0.20,   # how often a stranger is let in
    "max_owner_doubt_rate": 0.25,       # how often a real user session is doubted (ordinary + tired pooled)
    "max_mimic_accept_rate": 0.35,      # how often someone copying the user's tempo is let in
    "max_swap_open_rate": 0.15,         # a profile trained on someone else must not open for the owner
}


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return float(s[m]) if n % 2 else 0.5 * (s[m - 1] + s[m])


def _mad(xs):
    m = _median(xs)
    return _median([abs(x - m) for x in xs])


def _robust_scale(col, fallback=1.0):
    s = 1.4826 * _mad(col)
    if s < 1e-6:
        s = (max(col) - min(col)) / 2.0 if max(col) > min(col) else fallback
    return max(s, 1e-3)


def session_vectors(rec):
    """(identity vector, state vector) for one recording; never reads meta/source."""
    feats = extract(rec)
    return vector(feats), state_vector(feats)


def _kmeans(vectors, k, iterations=50):
    """Deterministic k-means over enrolment vectors (no labels, no eval data)."""
    if len(vectors) <= k:
        return [list(v) for v in vectors], list(range(len(vectors)))
    order = sorted(range(len(vectors)), key=lambda i: sum(vectors[i]))
    seeds = [vectors[order[min(len(order) - 1, int((j + 0.5) * len(order) / k))]] for j in range(k)]
    centroids = [list(s) for s in seeds]
    assign = [0] * len(vectors)
    for _ in range(iterations):
        groups = [[] for _ in range(k)]
        for i, v in enumerate(vectors):
            best, best_d = 0, float("inf")
            for j, c in enumerate(centroids):
                d = sum((a - b) ** 2 for a, b in zip(v, c))
                if d < best_d:
                    best, best_d = j, d
            groups[best].append(v)
            assign[i] = best
        moved = False
        for j, g in enumerate(groups):
            if not g:
                continue
            new = [sum(col) / len(g) for col in zip(*g)]
            if max(abs(a - b) for a, b in zip(new, centroids[j])) > 1e-9:
                moved = True
            centroids[j] = new
        if not moved:
            break
    return centroids, assign


def fit(identity_vectors, state_vectors, session_ids, k=3):
    """Mode centres in state space; identity centre and spread per mode."""
    if len(identity_vectors) != len(state_vectors):
        raise ValueError("identity and state vectors must line up")
    n_id = len(IDENTITY_FEATURES)
    k = min(k, max(1, len(identity_vectors)))
    mode_centers, assign = _kmeans(state_vectors, k)
    global_scale = [_robust_scale([v[i] for v in identity_vectors]) for i in range(n_id)]

    profiles = []
    for j, mode in enumerate(mode_centers):
        members = [v for v, a in zip(identity_vectors, assign) if a == j]
        if not members:
            continue
        center = [sum(col) / len(members) for col in zip(*members)]
        if len(members) > 2:
            scale = [_robust_scale([v[i] for v in members], global_scale[i]) for i in range(n_id)]
        else:
            scale = list(global_scale)
        profiles.append({
            "mode_center": [round(float(x), 6) for x in mode],
            "center": [round(float(c), 6) for c in center],
            "scale": [round(float(s), 6) for s in scale],
            "members": len(members),
        })
    return {
        "features": list(IDENTITY_FEATURES),
        "state_features": list(STATE_FEATURES),
        "cap": CAP,
        "scale": [round(float(s), 6) for s in global_scale],
        "profiles": profiles,
        "center": profiles[0]["center"],
        "train_sessions": list(session_ids),
        "train_hash": session_hash(session_ids),
    }


def session_hash(ids):
    h = hashlib.sha256()
    for sid in sorted(ids):
        h.update(sid.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _mode_distance(mode_center, state_vec, global_scale):
    acc = 0.0
    for a, b, s in zip(state_vec, mode_center, global_scale):
        acc += ((a - b) / max(s, 1e-9)) ** 2
    return math.sqrt(acc / len(mode_center))


def score(model, ident_vec, state_vec):
    """Similarity in (0,1] between a session and the enrolled routine it belongs to.

    1.0 means 'this is exactly this user's habit in this mode'. The state vector only
    chooses the routine; the identity vector produces the number.
    """
    best = 0.0
    sscale = _state_scale(model)
    # state only routes: keep the routines whose tempo is closest to this session
    ranked = sorted(model["profiles"],
                    key=lambda p: _mode_distance(p["mode_center"], state_vec, sscale)
                    if p.get("mode_center") else float("inf"))
    for prof_ in ranked[:2]:
        acc = 0.0
        for v, c, s in zip(ident_vec, prof_["center"], prof_.get("scale") or model["scale"]):
            z = min(model["cap"], abs(v - c) / max(s, 1e-9))
            acc += z * z
        best = max(best, 1.0 / (1.0 + math.sqrt(acc / len(prof_["center"]))))
    return best


def _state_scale(model):
    return model.get("scale_state") or [1.0] * len(STATE_FEATURES)


def decide(model, threshold, ident_vec, state_vec):
    sc = score(model, ident_vec, state_vec)
    return sc, sc >= threshold


def calibrate(enroll_recs, label_recs, labels, k=3):
    """Fit the profile bank on enrolment sessions and fix the threshold on train only.

    labels: dict session_id -> 'owner' | 'other'. Tired-owner sessions count as the
    real user, so the sweep cannot win by throwing the tired day away.
    """
    ident = [session_vectors(r)[0] for r in enroll_recs]
    state = [session_vectors(r)[1] for r in enroll_recs]
    model = fit(ident, state, [r["session_id"] for r in enroll_recs], k=k)
    model["scale_state"] = [_robust_scale([v[i] for v in state]) for i in range(len(STATE_FEATURES))]

    rows = []
    for r in label_recs:
        iv, sv = session_vectors(r)
        rows.append((labels[r["session_id"]], score(model, iv, sv)))

    diag = []
    best = None
    for thr in ACCEPT_THRESHOLD_SWEEP:
        tp = sum(1 for lab, s in rows if lab == "owner" and s >= thr)
        fn = sum(1 for lab, s in rows if lab == "owner" and s < thr)
        fp = sum(1 for lab, s in rows if lab == "other" and s >= thr)
        tn = sum(1 for lab, s in rows if lab == "other" and s < thr)
        tpr = tp / (tp + fn) if tp + fn else 0.0
        tnr = tn / (tn + fp) if tn + fp else 0.0
        bal = 0.5 * (tpr + tnr)
        diag.append({"threshold": thr, "balanced_accuracy": round(bal, 4),
                     "train_owner_accept": round(tpr, 4), "train_other_accept": round(1.0 - tnr, 4)})
        key = (round(bal, 6), thr)  # tie-break towards the stricter threshold
        if best is None or key > best[0]:
            best = (key, thr)
    return best[1], model, diag


def evaluate(model, threshold, eval_recs, train_recs, k=3):
    """Acceptance rates per style/persona plus the swap test, on held-out sessions."""
    results = {}
    for rec in eval_recs:
        meta = rec.get("meta", {})
        style, persona = meta.get("style", "?"), meta.get("persona", "?")
        iv, sv = session_vectors(rec)
        sc, ok = decide(model, threshold, iv, sv)
        results.setdefault(style, {}).setdefault(persona, []).append((ok, round(sc, 4)))

    def rate(xs):
        return round(sum(1 for ok, _ in xs if ok) / len(xs), 4) if xs else None

    table = {style: {p: {"n": len(v), "accept_rate": rate(v),
                         "mean_score": round(sum(s for _, s in v) / len(v), 4)}
                     for p, v in per.items()}
             for style, per in results.items()}

    def _pooled(personas):
        ok = [ok for per in results.values() for persona, rows in per.items()
              if persona in personas for ok, _ in rows]
        return (sum(1 for o in ok if o) / len(ok)) if ok else None

    real_user_accept = _pooled({"owner", "owner_tired"})
    stranger_accept = _pooled({"stranger"})
    mimic_accept = _pooled({"mimic"})

    # ---- swap test: a profile trained on someone else must not open for the owner ----
    train_by_style = {}
    for rec in train_recs:
        meta = rec.get("meta", {})
        train_by_style.setdefault(meta.get("style", "?"), []).append(rec)
    swap = {}
    same_user = {}
    for style, recs in train_by_style.items():
        owner_eval = [r for r in eval_recs
                      if r.get("meta", {}).get("style") == style and r.get("meta", {}).get("persona") == "owner"]
        per_persona = {}
        for persona in ("stranger", "mimic", "owner_tired"):
            sub = [r for r in recs if r.get("meta", {}).get("persona") == persona]
            if not sub or not owner_eval:
                continue
            fi = [session_vectors(r)[0] for r in sub]
            fs = [session_vectors(r)[1] for r in sub]
            foreign = fit(fi, fs, [r["session_id"] for r in sub], k=k)
            foreign["scale_state"] = [_robust_scale([v[i] for v in fs]) for i in range(len(STATE_FEATURES))]
            opens = [decide(foreign, threshold, *session_vectors(r))[1] for r in owner_eval]
            rate_ = round(sum(1 for o in opens if o) / len(opens), 4)
            # a profile trained on the same person on a tired day is not a swap:
            # opening for them is the behaviour we want, so it is reported apart
            if persona == "owner_tired":
                same_user[style] = rate_
            else:
                per_persona[persona] = rate_
        swap[style] = per_persona
    swap_all = [v for per in swap.values() for v in per.values()]

    summary = {
        "stranger_accept_rate": round(stranger_accept, 4) if stranger_accept is not None else None,
        "mimic_accept_rate": round(mimic_accept, 4) if mimic_accept is not None else None,
        "real_user_accept_rate": round(real_user_accept, 4) if real_user_accept is not None else None,
        "real_user_doubt_rate": round(1 - real_user_accept, 4) if real_user_accept is not None else None,
        "swap_open_rate_max": max(swap_all) if swap_all else None,
        "swap_open_rate_mean": round(sum(swap_all) / len(swap_all), 4) if swap_all else None,
    }

    # ---- ablation: everything in one score, no state/identity split, no routines ----
    def _flat_model(recs):
        iv = [session_vectors(r)[0] + session_vectors(r)[1] for r in recs]
        center = [sum(col) / len(iv) for col in zip(*iv)]
        scale = [_robust_scale([v[i] for v in iv]) for i in range(len(center))]
        return {"profile": {"center": center, "scale": scale}, "cap": CAP}

    def _flat_score(m, vec):
        acc = 0.0
        for v, c, s in zip(vec, m["profile"]["center"], m["profile"]["scale"]):
            z = min(m["cap"], abs(v - c) / max(s, 1e-9))
            acc += z * z
        return 1.0 / (1.0 + math.sqrt(acc / len(m["profile"]["center"])))

    enroll_ordinary = [r for r in train_recs if r.get("meta", {}).get("persona") == "owner"]
    ablation = {}
    if enroll_ordinary:
        flat = _flat_model(enroll_ordinary)
        flat_rows = [(_flat_score(flat, session_vectors(r)[0] + session_vectors(r)[1]), r) for r in train_recs]
        best_thr, best_bal = None, -1
        for thr in ACCEPT_THRESHOLD_SWEEP:
            tpr = sum(1 for s, r in flat_rows if r["meta"]["persona"] in ("owner", "owner_tired") and s >= thr) / \
                max(1, sum(1 for _, r in flat_rows if r["meta"]["persona"] in ("owner", "owner_tired")))
            tnr = sum(1 for s, r in flat_rows if r["meta"]["persona"] not in ("owner", "owner_tired") and s < thr) / \
                max(1, sum(1 for _, r in flat_rows if r["meta"]["persona"] not in ("owner", "owner_tired")))
            if 0.5 * (tpr + tnr) > best_bal:
                best_thr, best_bal = thr, 0.5 * (tpr + tnr)
        def flat_rate(persona):
            hits = [s >= best_thr for s, r in flat_rows if r["meta"]["persona"] == persona]
            return round(sum(1 for h in hits if h) / len(hits), 4) if hits else None
        eval_hits = {}
        for persona in ("owner", "owner_tired", "stranger", "mimic"):
            rows = [r for r in eval_recs if r.get("meta", {}).get("persona") == persona]
            hits = [_flat_score(flat, session_vectors(r)[0] + session_vectors(r)[1]) >= best_thr for r in rows]
            eval_hits[persona] = round(sum(1 for h in hits if h) / len(hits), 4) if hits else None
        ablation["one_score_no_split"] = {
            "threshold_fixed_on_train": best_thr,
            "train_balanced_accuracy": round(best_bal, 4),
            "owner_accept_rate": eval_hits["owner"],
            "tired_owner_accept_rate": eval_hits["owner_tired"],
            "stranger_accept_rate": eval_hits["stranger"],
            "mimic_accept_rate": eval_hits["mimic"],
        }

    checks = {
        "stranger_accept_within_limit": summary["stranger_accept_rate"] is not None
        and summary["stranger_accept_rate"] <= LIMITS["max_stranger_accept_rate"],
        "mimic_accept_within_limit": summary["mimic_accept_rate"] is not None
        and summary["mimic_accept_rate"] <= LIMITS["max_mimic_accept_rate"],
        "owner_doubt_within_limit": summary["real_user_doubt_rate"] is not None
        and summary["real_user_doubt_rate"] <= LIMITS["max_owner_doubt_rate"],
        "swap_test_passes": summary["swap_open_rate_max"] is not None
        and summary["swap_open_rate_max"] <= LIMITS["max_swap_open_rate"],
    }
    return {"per_style": table, "summary": summary, "limits": LIMITS, "checks": checks,
            "swap_test": swap, "same_user_transfer": same_user, "ablation": ablation,
            "passed": all(checks.values())}


def fmt_table(evaluation):
    lines = []
    header = f"{'style':10s} {'persona':12s} {'n':>3s} {'accept':>7s} {'score':>7s}"
    lines.append(header)
    lines.append("-" * len(header))
    for style, per in evaluation["per_style"].items():
        for persona, row in per.items():
            lines.append(f"{style:10s} {persona:12s} {row['n']:3d} {row['accept_rate']:7.2f} {row['mean_score']:7.3f}")
    s = evaluation["summary"]
    lines.append("")
    lines.append(f"stranger let in (FAR)   : {s['stranger_accept_rate']:.2f}")
    lines.append(f"mimic let in            : {s['mimic_accept_rate']:.2f}")
    lines.append(f"real user doubted (FRR) : {s['real_user_doubt_rate']:.2f}   "
                 f"(real-user accept {s['real_user_accept_rate']:.2f})")
    lines.append(f"swap test, foreign profile opens for the owner (max): {s['swap_open_rate_max']:.2f}")
    ab = evaluation.get("ablation", {}).get("one_score_no_split")
    if ab:
        lines.append("")
        lines.append(f"ablation, one score with no state/identity split (threshold {ab['threshold_fixed_on_train']:.2f}):")
        lines.append(f"  owner {ab['owner_accept_rate']:.2f} | tired owner {ab['tired_owner_accept_rate']:.2f}"
                     f" | stranger {ab['stranger_accept_rate']:.2f} | mimic {ab['mimic_accept_rate']:.2f}")
    lines.append("")
    for name, ok in evaluation["checks"].items():
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return "\n".join(lines)