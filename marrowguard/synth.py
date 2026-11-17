"""Synthetic recorders for inputtrace/1.

Everything here is fake and declared as such. Three input styles are simulated,
each with four personas: the owner, the owner on a tired day, a stranger, and a
mimic who copies the owner's headlines (speed, pauses, session length) without
copying the details (path shape, jitter, correction habits).
"""
import math
import random
import zlib

STYLES = ("standard", "dictation", "switch")
PERSONAS = ("owner", "owner_tired", "stranger", "mimic")
SEED = 20261117

# Declared priors: what each style/persona does, not what the model learns.
# speed  = normalised pointer speed (screen fractions per second)
# dt     = median milliseconds between pointer samples
# turn   = mean heading change per sample, in units of pi radians
# eff    = fraction of path length that is net displacement (higher = straighter)
# jitter = multiplicative noise on sample speed
# pause  = fraction of pointer gaps longer than 500 ms
# click  = clicks per pointer sample
# via    = share of text channels
# chunk  = median characters per text event
# ici    = median milliseconds between text events
# corr   = corrections per 100 characters
# burst  = fraction of text gaps under 120 ms
# dur    = median session length in seconds
PRIORS = {
    "standard": {  # mouse + keyboard
        "owner": dict(speed=0.55, dt=70, turn=0.30, eff=0.78, jitter=0.018, pause=0.16,
                      click=0.05, via={"typed": 1.0}, chunk=1.6, ici=95, corr=5, burst=0.25, dur=55),
        # tired: tempo drops, corrections rise a little; the motor habit (turn, eff,
        # jitter, chunking, channel mix) is the same person's
        "owner_tired": dict(speed=0.34, dt=95, turn=0.30, eff=0.78, jitter=0.020, pause=0.30,
                            click=0.05, via={"typed": 1.0}, chunk=1.6, ici=150, corr=7, burst=0.25, dur=75),
        "stranger": dict(speed=0.95, dt=45, turn=0.55, eff=0.52, jitter=0.055, pause=0.09,
                         click=0.02, via={"typed": 1.0}, chunk=1.1, ici=60, corr=2, burst=0.45, dur=45),
        # mimic: copies the owner's tempo headlines, cannot copy the habit
        "mimic": dict(speed=0.55, dt=70, turn=0.52, eff=0.58, jitter=0.040, pause=0.16,
                      click=0.05, via={"typed": 1.0}, chunk=1.3, ici=95, corr=11, burst=0.30, dur=55),
    },
    "dictation": {  # head/eye pointer + dictated text
        "owner": dict(speed=0.22, dt=90, turn=0.40, eff=0.55, jitter=0.035, pause=0.35,
                      click=0.03, via={"dictation": 0.9, "onscreen": 0.1}, chunk=6.0, ici=1100, corr=3, burst=0.10, dur=70),
        "owner_tired": dict(speed=0.15, dt=120, turn=0.40, eff=0.55, jitter=0.037, pause=0.45,
                            click=0.03, via={"dictation": 0.9, "onscreen": 0.1}, chunk=6.0, ici=1700, corr=4, burst=0.10, dur=95),
        "stranger": dict(speed=0.38, dt=60, turn=0.58, eff=0.38, jitter=0.070, pause=0.22,
                         click=0.02, via={"dictation": 1.0}, chunk=11.0, ici=750, corr=1, burst=0.22, dur=55),
        "mimic": dict(speed=0.22, dt=90, turn=0.52, eff=0.47, jitter=0.055, pause=0.35,
                      click=0.03, via={"dictation": 0.9, "onscreen": 0.1}, chunk=6.5, ici=1100, corr=8, burst=0.10, dur=70),
    },
    "switch": {  # switch access: discrete jumps + on-screen keyboard
        "owner": dict(speed=0.10, dt=900, turn=0.20, eff=0.90, jitter=0.010, pause=0.80,
                      click=0.12, via={"onscreen": 1.0}, chunk=1.0, ici=620, corr=8, burst=0.0, dur=80),
        "owner_tired": dict(speed=0.08, dt=1200, turn=0.20, eff=0.90, jitter=0.011, pause=0.85,
                            click=0.12, via={"onscreen": 1.0}, chunk=1.0, ici=950, corr=10, burst=0.0, dur=110),
        "stranger": dict(speed=0.18, dt=600, turn=0.35, eff=0.75, jitter=0.020, pause=0.65,
                         click=0.08, via={"onscreen": 1.0}, chunk=1.0, ici=380, corr=4, burst=0.0, dur=65),
        "mimic": dict(speed=0.10, dt=900, turn=0.33, eff=0.80, jitter=0.016, pause=0.80,
                      click=0.12, via={"onscreen": 1.0}, chunk=1.0, ici=620, corr=15, burst=0.0, dur=80),
    },
}

WORDS = ("the quick brown fox jumps over a lazy dog while the room stays quiet "
         "and the cursor keeps moving between the panels of the page").split()


def _lognorm(rng, median, sigma):
    return median * math.exp(rng.gauss(0.0, sigma))


def _pick_via(rng, via):
    r = rng.random()
    acc = 0.0
    for name in sorted(via):
        acc += via[name]
        if r <= acc:
            return name
    return sorted(via)[-1]


def simulate(style, persona, session_id, seed=None):
    """Return one inputtrace/1 recording produced from the declared priors."""
    if style not in STYLES or persona not in PERSONAS:
        raise ValueError(f"unknown style/persona: {style}/{persona}")
    if seed is None:
        seed = zlib.crc32(f"{style}|{persona}|{session_id}".encode())
    rng = random.Random(seed)
    p = PRIORS[style][persona]

    dur_s = max(12.0, _lognorm(rng, p["dur"], 0.10))
    n_moves = max(6, int(dur_s * 1000.0 / p["dt"]))
    events = []

    # pointer: bursts of movement toward a target, with rests between them
    elapsed = 0
    x, y = rng.random(), rng.random()
    pointer_events = []
    while elapsed < dur_s * 1000.0:
        tx, ty = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
        dist = math.hypot(tx - x, ty - y)
        dt_ms = _lognorm(rng, p["dt"], 0.30)
        step = max(1e-3, p["speed"] * dt_ms / 1000.0)
        n = max(2, min(80, int(dist / step) + 1))
        bow = p["turn"] * dist * rng.uniform(0.4, 1.0)
        px_perp, py_perp = -(ty - y), (tx - x)
        norm = math.hypot(px_perp, py_perp) or 1.0
        cx = (x + tx) / 2 + bow * px_perp / norm
        cy = (y + ty) / 2 + bow * py_perp / norm
        for i in range(1, n + 1):
            u = i / n
            sx = (1 - u) ** 2 * x + 2 * (1 - u) * u * cx + u * u * tx
            sy = (1 - u) ** 2 * y + 2 * (1 - u) * u * cy + u * u * ty
            sx += rng.gauss(0.0, p["jitter"] * 0.06)
            sy += rng.gauss(0.0, p["jitter"] * 0.06)
            elapsed += int(max(4, _lognorm(rng, dt_ms, 0.25)))
            pointer_events.append({"t": elapsed, "type": "move",
                                   "x": round(min(0.999, max(0.001, sx)), 4),
                                   "y": round(min(0.999, max(0.001, sy)), 4)})
            if rng.random() < p["click"]:
                elapsed += int(rng.uniform(40, 180))
                pointer_events.append({"t": elapsed, "type": "click",
                                       "x": round(min(0.999, max(0.001, sx)), 4),
                                       "y": round(min(0.999, max(0.001, sy)), 4),
                                       "button": "left"})
        x, y = tx, ty
        if rng.random() < p["pause"]:
            elapsed += int(rng.uniform(400, 2500))
        else:
            elapsed += int(rng.uniform(60, 350))
    events = pointer_events

    # text
    chars_per_s = max(0.2, p["chunk"] / (p["ici"] / 1000.0))
    budget = min(900, int(dur_s * chars_per_s))
    written = 0
    t_text = int(rng.uniform(20, 400))
    corrections = 0
    while written < budget:
        t_text += int(max(15, _lognorm(rng, p["ici"], 0.45)))
        if rng.random() < p["burst"]:
            t_text += int(rng.uniform(10, 90))
        n = max(1, int(round(_lognorm(rng, p["chunk"], 0.45))))
        via = _pick_via(rng, p["via"])
        if via == "dictation":
            text = " ".join(rng.choice(WORDS) for _ in range(max(1, n // 5)))[: max(1, n)]
        else:
            text = "".join(rng.choice("etaoinshrdlucmfwyp") for _ in range(n))
        written += len(text)
        events.append({"t": t_text, "type": "text", "via": via, "text": text})
        # corrections: backspaces for keyed text, a replace event for dictation
        if rng.random() < (p["corr"] / 100.0) * max(1, len(text)):
            corrections += 1
            t_text += int(rng.uniform(120, 900))
            if via == "dictation":
                events.append({"t": t_text, "type": "text", "via": via,
                               "text": rng.choice(WORDS)[:2], "correction": True})
            else:
                for _ in range(rng.randint(1, 2)):
                    t_text += int(rng.uniform(90, 260))
                    events.append({"t": t_text, "type": "key", "key": "backspace"})
    if corrections == 0:  # every real session has at least one slip
        t_text += int(rng.uniform(100, 400))
        events.append({"t": t_text, "type": "key", "key": "backspace"})

    events.sort(key=lambda e: e["t"])
    return {
        "format": "inputtrace/1",
        "session_id": session_id,
        "source": style,
        "events": events,
    }


def build_dataset(sessions_per_cell=8, seed=SEED):
    """Yield (split, recording) pairs; split is 'train' or 'eval', by session index."""
    out = []
    n = 0
    for style in STYLES:
        for persona in PERSONAS:
            for i in range(sessions_per_cell):
                sid = f"{style}-{persona}-{i:02d}"
                split = "train" if i % 2 == 0 else "eval"
                n += 1
                rec = simulate(style, persona, sid, seed=seed + n * 7919)
                rec["meta"] = {"style": style, "persona": persona, "split": split}
                out.append((split, rec))
    return out