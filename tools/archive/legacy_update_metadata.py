import json
from pathlib import Path
for p in Path("data/train").glob("*.json"):
    obj=json.loads(p.read_text(encoding="utf-8"))
    meta=obj.get("metadata")
    if isinstance(meta,dict):
        meta["format"]="BPE subword sequence-to-sequence"
        meta["tokenizer"]="shared Chinese-Thai BPE with character fallback"
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("Updated training shard metadata to BPE")
