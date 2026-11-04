import re, hashlib
def funcs(p):
    src = open(p, encoding="utf-8").read()
    out = {}
    for name in ["s_measure_folder", "weighted_f_measure_folder", "e_measure_folder", "mae_folder"]:
        i = src.index("def " + name)
        m = re.search(r"\ndef (?!%s)" % name, src[i:])
        body = src[i:i + m.start()] if m else src[i:]
        out[name] = hashlib.sha256(body.strip().encode()).hexdigest()[:12]
    return out
a = funcs("evaluate.py"); b = funcs("Ours/m1_ablation/evaluate.py")
for k in a:
    print(k, a[k], b[k], "same" if a[k] == b[k] else "DIFF")
