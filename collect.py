import json
from pathlib import Path
from pprint import pprint

out = {}

for p in sorted(Path("results").glob("*_under_*.json")):
    run, recipe = p.stem.rsplit("_under_", 1)
    r = json.load(open(p))["Attack Results"]
    out.setdefault(run, {})["-"] = r["Original accuracy:"]
    out[run][recipe] = r["Accuracy under attack:"]

pprint(out, sort_dicts=False)
