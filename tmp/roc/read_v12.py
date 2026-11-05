import json
d = json.load(open("Ours/roc_pr/v12_20261105/pd_fa_curves.json"))
print(list(d)[:4])
o = d["Ours"] if "Ours" in d else d["methods"][0]
print({k: (v if not isinstance(v, list) else len(v)) for k, v in o.items()} if isinstance(o, dict) else o)