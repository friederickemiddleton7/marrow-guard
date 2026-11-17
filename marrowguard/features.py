"""Feature extraction.

Features are computed from `events` only. A recording's declared source, file
name, session id and meta are never read, so nothing here can depend on where or
how the recording was captured. Pointer geometry is in screen fractions, so a
phone, an eye tracker and a desktop mouse all live on the same scale.
"""
import math

FEATURES = (
    "ptr_speed_med", "ptr_dt_med", "ptr_pause_frac", "ptr_turn_med", "ptr_eff",
    "ptr_rel_jitter", "ptr_click_per_move", "ptr_moves_log",
    "txt_rate", "txt_ici_med", "txt_ici_cv", "txt_chunk_med", "txt_corr_per_100",
    "txt_burst_frac", "txt_via_typed", "txt_via_dictation", "txt_via_onscreen",
    "sess_minutes",
)

# The split below is a mechanism claim, declared before any evaluation ran:
# tempo features move when a person is tired, unwell, or on a different device, so
# they describe STATE; shape and rhythm features are the habits a person carries
# between devices and moods, so they describe IDENTITY. The identity score uses
# only the second group, and state is reported separately instead of locking the
# real user out on a bad day.
STATE_FEATURES = (
    "ptr_speed_med", "ptr_dt_med", "ptr_pause_frac", "ptr_moves_log",
    "txt_rate", "txt_ici_med", "sess_minutes",
)
IDENTITY_FEATURES = (
    "ptr_turn_med", "ptr_eff", "ptr_rel_jitter", "ptr_click_per_move",
    "txt_ici_cv", "txt_chunk_med", "txt_corr_per_100", "txt_burst_frac",
    "txt_via_typed", "txt_via_dictation", "txt_via_onscreen",
)

# Which features are log-scaled before modelling (heavy right tails).
LOG_FEATURES = {"ptr_speed_med", "ptr_dt_med", "ptr_moves_log", "txt_ici_med",
                "txt_chunk_med", "sess_minutes"}

BACKSPACE_KEYS = {"backspace", "delete", "Backspace", "Delete"}

EPS = 1e-9


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return float(s[mid]) if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def extract(rec):
    """Return a dict of features for one recording (keys == FEATURES)."""
    events = rec["events"]
    moves = [e for e in events if e["type"] == "move"]
    clicks = [e for e in events if e["type"] == "click"]
    texts = [e for e in events if e["type"] == "text"]
    keys = [e for e in events if e["type"] == "key"]

    duration_s = max(0.001, (events[-1]["t"] - events[0]["t"]) / 1000.0)

    # ---- pointer ----
    dts, speeds, turns = [], [], []
    path = 0.0
    prev = None
    prev_vec = None
    for e in moves:
        if prev is not None:
            dx, dy = e["x"] - prev["x"], e["y"] - prev["y"]
            dist = math.hypot(dx, dy)
            dt = max(0.001, (e["t"] - prev["t"]) / 1000.0)
            dts.append(dt)
            speeds.append(dist / dt)
            path += dist
            if prev_vec is not None and dist > EPS:
                a_len = math.hypot(*prev_vec)
                b_len = dist
                if a_len > EPS and b_len > EPS:
                    cos = max(-1.0, min(1.0, (prev_vec[0] * dx + prev_vec[1] * dy) / (a_len * b_len)))
                    turns.append(math.acos(cos) / math.pi)
                prev_vec = (dx, dy)
            elif dist > EPS:
                prev_vec = (dx, dy)
        prev = e

    if moves:
        net = math.hypot(moves[-1]["x"] - moves[0]["x"], moves[-1]["y"] - moves[0]["y"])
    else:
        net = 0.0
    # efficiency per burst of movement (a burst ends at a rest longer than 250 ms),
    # so a session with many trips is not scored as if it were one long detour
    burst_eff = []
    seg_start = None
    seg_path = 0.0
    for i in range(1, len(moves)):
        step = math.hypot(moves[i]["x"] - moves[i - 1]["x"], moves[i]["y"] - moves[i - 1]["y"])
        gap = moves[i]["t"] - moves[i - 1]["t"]
        if seg_start is None:
            seg_start = moves[i - 1]
            seg_path = 0.0
        seg_path += step
        if gap > 250:
            disp = math.hypot(moves[i - 1]["x"] - seg_start["x"], moves[i - 1]["y"] - seg_start["y"])
            if seg_path > 0.01:
                burst_eff.append(min(1.0, disp / seg_path))
            seg_start = moves[i]
            seg_path = 0.0
    if seg_start is not None and seg_path > 0.01:
        disp = math.hypot(moves[-1]["x"] - seg_start["x"], moves[-1]["y"] - seg_start["y"])
        burst_eff.append(min(1.0, disp / seg_path))
    eff = _median(burst_eff) if burst_eff else ((net / path) if path > EPS else 0.0)
    jitter = 0.0
    if len(speeds) > 1:
        deltas = [abs(speeds[i + 1] - speeds[i]) for i in range(len(speeds) - 1)]
        base = _mean(speeds)
        jitter = (_mean(deltas) / base) if base > EPS else 0.0

    # ---- text ----
    chars = sum(len(e["text"]) for e in texts)
    via_chars = {"typed": 0, "onscreen": 0, "dictation": 0}
    for e in texts:
        via_chars[e["via"]] += len(e["text"])
    text_gaps = [(texts[i + 1]["t"] - texts[i]["t"]) / 1000.0 for i in range(len(texts) - 1)]
    ici_mean = _mean(text_gaps)
    ici_sd = 0.0
    if len(text_gaps) > 1:
        mu = ici_mean
        ici_sd = math.sqrt(sum((g - mu) ** 2 for g in text_gaps) / (len(text_gaps) - 1))
    bursts = sum(1 for g in text_gaps if g < 0.120)
    corrections = sum(1 for k in keys if k["key"] in BACKSPACE_KEYS) + \
        sum(1 for e in texts if e.get("correction") is True)

    feats = {
        "ptr_speed_med": _median(speeds) if speeds else 0.05,
        "ptr_dt_med": _median(dts) if dts else 0.5,
        "ptr_pause_frac": (sum(1 for d in dts if d > 0.5) / len(dts)) if dts else 0.0,
        "ptr_turn_med": _median(turns) if turns else 0.5,
        "ptr_eff": eff,
        "ptr_rel_jitter": jitter,
        "ptr_click_per_move": (len(clicks) / len(moves)) if moves else 0.0,
        "ptr_moves_log": math.log1p(len(moves)),
        "txt_rate": len(texts) / duration_s,
        "txt_ici_med": _median(text_gaps) if text_gaps else 1.0,
        "txt_ici_cv": (ici_sd / ici_mean) if ici_mean > EPS else 0.0,
        "txt_chunk_med": _median([len(e["text"]) for e in texts]) if texts else 1.0,
        "txt_corr_per_100": (100.0 * corrections / chars) if chars else 0.0,
        "txt_burst_frac": (bursts / len(text_gaps)) if text_gaps else 0.0,
        "txt_via_typed": (via_chars["typed"] / chars) if chars else 0.0,
        "txt_via_dictation": (via_chars["dictation"] / chars) if chars else 0.0,
        "txt_via_onscreen": (via_chars["onscreen"] / chars) if chars else 0.0,
        "sess_minutes": duration_s / 60.0,
    }
    return {k: float(feats[k]) for k in FEATURES}


def vector(feat_dict, names=IDENTITY_FEATURES):
    """Order a feature dict as a list, applying the declared log transforms."""
    out = []
    for name in names:
        v = feat_dict[name]
        if name in LOG_FEATURES:
            v = math.log(max(v, 1e-4))
        out.append(v)
    return out


def state_vector(feat_dict):
    """The tempo/state side, reported separately and never used to lock anyone out."""
    return vector(feat_dict, STATE_FEATURES)