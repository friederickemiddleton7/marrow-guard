#!/usr/bin/env bash
# One command: make the fake recordings, enrol the owner, fix the threshold on the
# train split, then report the held-out rates and the swap test.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}

echo "== selftest: format and determinism"
"$PY" -m marrowguard selftest

echo
echo "== recordings (synthetic, declared as fake)"
"$PY" -m marrowguard make-data --sessions-per-cell 16

echo
echo "== enrolment + threshold (train split only)"
"$PY" -m marrowguard calibrate --k 5

echo
echo "== held-out evaluation + swap test"
"$PY" -m marrowguard evaluate

echo
echo "done. report.json has the same numbers, machine-readable."