from pathlib import Path
a = Path("AGENTS.md").read_bytes(); r = Path("README.md").read_bytes()
print("mirror ok:", r.endswith(a), "prefix", len(r) - len(a))
