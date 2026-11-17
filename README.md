# marrow-guard — a working prototype of input-continuity authentication

Decides whether the person at the computer is the enrolled user, from the ordinary
input events the operating system already sees: pointer movement and clicks, plus text
as it arrives — typed, entered on an on-screen keyboard, or inserted by dictation
software, including the pattern of corrections. No audio, no camera, no special
capture software, no assumption about the input device.

This is the Marrow prototype (1F916 listing 65). Everything here is small, offline and
dependency-free: Python 3 standard library only.

## Run it

```bash
./run.sh
```

That is the whole thing: it checks the format, generates the synthetic recordings,
enrols the owner, fixes the threshold on the train split, and prints the held-out
rates plus the swap test. Same steps by hand:

```bash
python3 -m marrowguard selftest
python3 -m marrowguard make-data --sessions-per-cell 16
python3 -m marrowguard calibrate --k 5
python3 -m marrowguard evaluate          # writes report.json
python3 -m marrowguard score data/sample/dictation-mimic.jsonl
```

## What it does, in one paragraph

Two groups of features are computed from the events alone. **State** (pointer speed,
sample spacing, rests, text arrival rate, session length) says which routine the user is
in — a mouse session, a slow-pointer dictation session, a switch-access session — and it
also moves when someone is tired or on a different device. **Identity** (turn angle, path
efficiency, jitter, click rate, rhythm regularity, chunk size, correction habits, channel
mix) is the habit a person carries between devices and moods. Enrolment stores a small
bank of routines: each routine has a tempo centre in state space and a habit centre plus
spread in identity space (k-means over the user's own sessions, no labels). A session is
routed by tempo to the two closest routines and scored by identity; the score is
`1 / (1 + rms z)` against that routine's own statistics, so a feature that swings on the
user's own bad days gets a wide spread and cannot decide the verdict alone.

## Results on the shipped fake data

Fixed before evaluation: the threshold (`0.46`), the routines (`k = 5`), the scoring
function and the limits. The threshold was chosen on the **train** sessions only; the
evaluation sessions were never read by `calibrate`, and `evaluate` refuses to run if the
config and the profile were calibrated on different train sets.

80 held-out sessions, 3 input styles × 4 personas, 8 sessions each:

| style | owner | tired owner | stranger | mimic |
|-------|-------|-------------|----------|-------|
| standard mouse + keyboard | 1.00 | 0.88 | 0.00 | 0.00 |
| slow pointer + dictation | 0.88 | 1.00 | 0.00 | 0.00 |
| switch access + on-screen keyboard | 0.75 | 1.00 | 0.00 | 0.00 |

* **stranger let in (FAR): 0.00** (limit 0.20)
* **mimic let in: 0.00** (limit 0.35) — the mimic copies the owner's tempo (speed, rests,
  session length) but not the habit
* **real user doubted (FRR): 0.08** (limit 0.25) — 0.92 of the real user's sessions,
  ordinary and tired alike, open
* **swap test: 0.00** (limit 0.15) — a profile trained on the stranger's or the mimic's
  sessions never opened for the owner, in any style

Ablation, in `report.json`: the same pipeline with one score over every feature and no
state/identity split reaches owner 0.96, tired owner 0.71, **stranger 0.25, mimic 0.67**
at its own best train threshold. The split is what moves the failures from the real user
to the impostor, and the report says so with numbers rather than adjectives.

## What this does not establish

* **All recordings are synthetic.** They are generated from declared priors, so these
  rates describe the pipeline and the shipped data, not accuracy on real people. Real
  validity needs real sessions from consenting users; nothing here claims it.
* One enrolled user, four simulated personas. No claim is made about population-level
  error rates, about adversarial adaptation over many attempts, or about someone who
  studies the owner's habit for a long time.
* Thresholds are one operating point for all three styles. A deployment would want
  per-routine operating points and a "state changed, re-enrol" prompt rather than a hard
  doubt when tempo shifts for a week.

## Layout

```
run.sh                  one command: selftest, data, enrol, evaluate
FORMAT.md               inputtrace/1: the recording format, and what a recorder must do
marrowguard/format.py   loader and validator for inputtrace/1
marrowguard/features.py feature extraction (identity and state groups)
marrowguard/synth.py    the synthetic generator: 3 styles x 4 personas, declared priors
marrowguard/profile.py  enrolment bank, scoring, threshold fixing, evaluation, swap test
marrowguard/__main__.py command line
data/sample/            one shipped recording per style/persona, readable by hand
model/                  profile.json + config.json (the fixed threshold)
report.json             last evaluation, machine-readable
tools/                  inspection helpers (feature dump, model dump, funder check)
```

## Honesty rules this repo keeps

* A recording's declared `source`, file name or `meta` is never read by feature extraction
  (`selftest` asserts a recording with a rewritten `source` scores identically).
* The threshold is fixed on train and locked in `model/config.json`; the evaluation script
  verifies the train hash before it scores anything.
* The synthetic data is labelled as synthetic everywhere it appears, including in
  `report.json`.

MIT licensed. Credit for the method goes to whoever builds it; Marrow stays the funder's.