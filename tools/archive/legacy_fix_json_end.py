from pathlib import Path
for name in ["data/evaluation/generalization.json","data/evaluation/compositional_v2.json"]:
    p=Path(name)
    s=p.read_text(encoding="utf-8")
    s=s[:s.rfind("}")+1]+"\n"
    p.write_text(s,encoding="utf-8")
