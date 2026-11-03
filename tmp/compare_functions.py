import re, hashlib
def funcs(path):
    src = open(path, encoding="utf-8").read()
    out = {}
    for name in ["s_measure_folder", "weighted_f_measure_folder", "e_measure_folder", "mae_folder"]:
        start = src.index("def " + name)
        # find next top-level def
        nxt = re.search(r"\ndef (?!%s)" % name, src[start:])
        body = src[start:start + nxt.start()] if nxt else src[start:]
        out[name] = hashlib.sha256(body.strip().encode()).hexdigest()
    return out
a = funcs(r"F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\evaluate.py")
b = funcs(r"F:\Code_Repositories_2\CursorCode\Seal_Model_Reproduction\Ours\evaluate.py")
for k in a:
    print(k, "same" if a[k] == b[k] else "DIFFERENT", a[k][:12], b[k][:12])
