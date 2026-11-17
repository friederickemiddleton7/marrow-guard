import json, sys
sys.path.insert(0, "/root/hermes-work/marrow-guard")
from marrowguard import format as fmt, profile as prof
from marrowguard.features import extract, vector, IDENTITY_FEATURES

train = list(fmt.load_jsonl("/root/hermes-work/marrow-guard/data/train.jsonl"))
evals = list(fmt.load_jsonl("/root/hermes-work/marrow-guard/data/eval.jsonl"))
cfg = json.load(open("/root/hermes-work/marrow-guard/model/config.json"))
thr = cfg["accept_threshold"]
print("threshold", thr, "k", cfg.get("k"))

for style in ("standard", "dictation", "switch"):
    owner_eval = [r for r in evals if r["meta"]["style"] == style and r["meta"]["persona"] == "owner"]
    for persona in ("owner", "owner_tired", "stranger", "mimic"):
        sub = [r for r in train if r["meta"]["style"] == style and r["meta"]["persona"] == persona]
        m = prof.fit([vector(extract(r)) for r in sub], [r["session_id"] for r in sub], k=cfg.get("k", 3))
        scs = [prof.score(m, vector(extract(r))) for r in owner_eval]
        print(f"{style:10s} foreign={persona:12s} owner-eval scores={[round(s,3) for s in scs]} "
              f"opens={sum(1 for s in scs if s >= thr)}/{len(scs)}")

# what does the score look like for the real model?
model = json.load(open("/root/hermes-work/marrow-guard/model/profile.json"))
for style in ("standard",):
    for persona in ("owner", "owner_tired", "stranger", "mimic"):
        rows = [r for r in evals if r["meta"]["style"] == style and r["meta"]["persona"] == persona]
        print(style, persona, [round(prof.score(model, vector(extract(r))), 3) for r in rows], "thr", thr)