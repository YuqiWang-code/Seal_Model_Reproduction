import json, sys
p = sys.argv[1] if len(sys.argv) > 1 else "tmp/roc/target/Ours.json"
d = json.load(open(p))
th = d["thresholds"]
P, F = d["Pd"], d["Fa"]
i = min(range(len(th)), key=lambda k: abs(th[k] - 128.0))
print("label", d["label"], "points", len(th), "threshold@", th[i])
print("Pd@128 %.5f  Fa@128 %.4e" % (P[i], F[i]))
print("Pd max %.5f at t=%g, Fa there %.4e" % (max(P), th[P.index(max(P))], F[P.index(max(P))]))
# knee = first (ascending Fa) point with Pd >= 0.98 * platform
plat = P[-1] if P[-1] >= max(P) * 0.999 else max(P)
order = sorted(range(len(th)), key=lambda k: F[k])
for k in order:
    if P[k] >= 0.98 * plat:
        print("plat(Pd at largest Fa) %.5f ; knee Fa %.4e Pd %.5f" % (P[-1], F[k], P[k]))
        break