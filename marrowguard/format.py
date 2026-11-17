"""inputtrace/1 — an operating-system-neutral recording format for input events.

One recording per line of JSONL. Nothing in a recording says where it was captured;
x/y are fractions of the screen, never pixels, and `t` is milliseconds from the
first event, never a wall clock.
"""
import json

FORMAT = "inputtrace/1"
EVENT_TYPES = ("move", "click", "key", "text")
TEXT_VIA = ("typed", "onscreen", "dictation")


class FormatError(ValueError):
    """A recording that does not satisfy inputtrace/1."""


def validate(rec, where="<record>"):
    if not isinstance(rec, dict):
        raise FormatError(f"{where}: recording must be an object")
    if rec.get("format") != FORMAT:
        raise FormatError(f"{where}: format must be {FORMAT!r}")
    for key in ("session_id", "source", "events"):
        if key not in rec:
            raise FormatError(f"{where}: missing {key!r}")
    events = rec["events"]
    if not isinstance(events, list) or not events:
        raise FormatError(f"{where}: events must be a non-empty list")
    last_t = -1
    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            raise FormatError(f"{where}: event {i} is not an object")
        kind = ev.get("type")
        if kind not in EVENT_TYPES:
            raise FormatError(f"{where}: event {i} has unknown type {kind!r}")
        t = ev.get("t")
        if not isinstance(t, int) or isinstance(t, bool) or t < 0:
            raise FormatError(f"{where}: event {i} t must be a non-negative int (ms)")
        if t < last_t:
            raise FormatError(f"{where}: event {i} goes back in time ({t} < {last_t})")
        last_t = t
        if kind in ("move", "click"):
            for axis in ("x", "y"):
                v = ev.get(axis)
                if not isinstance(v, (int, float)) or not (0.0 <= float(v) <= 1.0):
                    raise FormatError(f"{where}: event {i} {axis} must be a fraction in [0,1]")
        if kind == "key" and not isinstance(ev.get("key"), str):
            raise FormatError(f"{where}: event {i} key must be a string")
        if kind == "text":
            if ev.get("via") not in TEXT_VIA:
                raise FormatError(f"{where}: event {i} via must be one of {TEXT_VIA}")
            if not isinstance(ev.get("text"), str):
                raise FormatError(f"{where}: event {i} text must be a string")
    return rec


def load_jsonl(path):
    """Yield validated recordings from a .jsonl file."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FormatError(f"{path}:{lineno}: invalid JSON: {exc}") from None
            yield validate(rec, where=f"{path}:{lineno}")


def dump_jsonl(path, recs):
    with open(path, "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(validate(rec), separators=(",", ":"), sort_keys=True))
            fh.write("\n")