# inputtrace/1 — a plain, operating-system-neutral recording format

A recording is one JSON object per line (JSONL). Nothing in it says where it was
captured. Two properties make that true:

* positions are **fractions of the screen** (`0.0`–`1.0`), never pixels, so a phone,
  an eye tracker, a head mouse and a desktop mouse share one scale;
* time is **milliseconds since the first event** (`t`), never a wall clock and never a
  sample index, so a 30 Hz switch stream and a 1000 Hz mouse stream compare directly.

No capture software is needed to make one. Any tool that can write these events can
produce a recording; a person can even type one by hand.

## Shape

```json
{"format":"inputtrace/1","session_id":"2026-11-17-a","source":"free text",
 "events":[{"t":0,"type":"move","x":0.51,"y":0.33},
           {"t":120,"type":"click","x":0.51,"y":0.33,"button":"left"},
           {"t":780,"type":"text","via":"typed","text":"h"},
           {"t":1120,"type":"text","via":"dictation","text":"hello there"},
           {"t":1400,"type":"key","key":"backspace"},
           {"t":1510,"type":"text","via":"onscreen","text":"k","correction":true}]}
```

| field | required | meaning |
|-------|----------|---------|
| `format` | yes | exactly `inputtrace/1` |
| `session_id` | yes | any stable string; used for reference, never as a feature |
| `source` | yes | free text (e.g. `standard`, `dictation`, `switch`). **Advisory only.** The scorer never reads it, so a wrong or hostile value changes nothing |
| `events` | yes | non-empty list, `t` non-decreasing |
| `meta` | no | anything a producer wants to keep (the shipped synthetic data carries the style/persona/split there). Ignored by feature extraction |

## Events

| `type` | fields | notes |
|--------|--------|-------|
| `move` | `t`, `x`, `y` | pointer position at that moment |
| `click` | `t`, `x`, `y`, `button` | a press at that position |
| `key` | `t`, `key` | a key by name (`"backspace"`, `"enter"`, `"a"`). Optional: a recording with no key events is valid |
| `text` | `t`, `via`, `text` | text as it arrives. `via` is `"typed"`, `"onscreen"` or `"dictation"`. That distinction is required, because inserting a phrase by voice is a different act from typing it |
| | `correction` (optional, `text` only) | `true` when this text was produced to fix earlier text (replacing a word, re-speaking a phrase) |

Rules the loader enforces: `0.0 <= x,y <= 1.0`; `t` a non-negative integer in
milliseconds; events in non-decreasing time order; `via` one of the three names.
Anything else is refused with a message naming the record and the event.

## What a recorder has to do

1. Emit a `move` per pointer sample (or per discrete jump for switch access). Downsample
   or upsample freely: features are medians and ratios, not per-sample absolutes.
2. Emit `text` in the chunks the user actually produced: single characters for a
   keyboard or an on-screen keyboard, whole phrases for dictation.
3. Set `via` honestly. Do not invent `correction` events that did not happen.
4. Never record what was typed into password fields, and never record audio or camera
   input: the format has no fields for either, on purpose.

## Privacy

The format contains what a person typed, so a recording is personal data. Ship
synthetic recordings; keep real ones local; strip `text` (leave `via` and lengths) if a
recording ever has to leave the machine it was made on.