import json
from pathlib import Path
m=Path("data/manifest.json")
d=json.loads(m.read_text(encoding="utf-8"))
if "lexical_coverage.json" not in d["training_files"]:d["training_files"].append("lexical_coverage.json")
d["topics"]["lexical_coverage"]=len(json.loads(Path("data/train/lexical_coverage.json").read_text(encoding="utf-8"))["examples"])
d["total_examples"]=sum(len(json.loads((Path("data/train")/f).read_text(encoding="utf-8"))["examples"]) for f in d["training_files"])
m.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("updated total",d["total_examples"])
