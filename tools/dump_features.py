import json, sys
sys.path.insert(0, "/root/hermes-work/marrow-guard")
from marrowguard import format as fmt
from marrowguard.features import extract, FEATURES

rows = list(fmt.load_jsonl("/root/hermes-work/marrow-guard/data/train.jsonl"))
for style in ("standard", "dictation", "switch"):
    print("==", style)
    print("persona      " + " ".join(f"{f.replace('ptr_','p_').replace('txt_','t_')[:9]:>9s}" for f in FEATURES))
    for r in rows:
        if r["meta"]["style"] != style:
            continue
        f = extract(r)
        print(f"{r['meta']['persona']:12s} " + " ".join(f"{f[k]:9.3f}" for k in FEATURES))