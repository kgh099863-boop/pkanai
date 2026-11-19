import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
for p in (root/"data"/"evaluation").glob("*.json"):
 d=json.loads(p.read_text(encoding="utf-8"))
 d["metadata"].pop("purpose",None)
 p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("removed non-standard metadata field from evaluation shards")
