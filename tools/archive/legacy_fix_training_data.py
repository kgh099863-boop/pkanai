import json
from pathlib import Path
p=Path("data/train/compositional_augmentation.json")
obj=json.loads(p.read_text(encoding="utf-8"))
fixed=0
for x in obj["examples"]:
    if x.get("frame")!="time_no_have":continue
    old=x["th"]
    if "换钱" in x["zh"]:
        x["th"]=x["th"].replace("ฉันไม่มีแลกเงิน","ฉันไม่ได้แลกเงิน")
    if "更多的练习" in x["zh"]:
        x["th"]=x["th"].replace("ฉันไม่มีฝึกฝนเพิ่มเติม","ฉันไม่มีการฝึกฝนเพิ่มเติม")
    if x["th"]!=old:fixed+=1
p.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
m=Path("data/manifest.json")
manifest=json.loads(m.read_text(encoding="utf-8"))
if "natural_patterns.json" not in manifest["training_files"]:
    manifest["training_files"].append("natural_patterns.json")
manifest["total_examples"]=sum(len(json.loads((Path("data/train")/f).read_text(encoding="utf-8"))["examples"]) for f in manifest["training_files"])
manifest["topics"]["natural_patterns"]=len(json.loads(Path("data/train/natural_patterns.json").read_text(encoding="utf-8"))["examples"])
m.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("fixed flawed negation translations:",fixed)
print("manifest total:",manifest["total_examples"])
