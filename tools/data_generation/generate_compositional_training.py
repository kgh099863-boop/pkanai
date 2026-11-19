import json
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"
manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
lex=json.loads((DATA/"compositional_lexicon.json").read_text(encoding="utf-8"))
train_dir=DATA/"train"
seen=set()
for f in manifest["training_files"]:
    o=json.loads((train_dir/f).read_text(encoding="utf-8"))
    seen.update(x["zh"] for x in o["examples"])
held=set()
for p in (DATA/"evaluation").glob("*.json"):
    o=json.loads(p.read_text(encoding="utf-8"))
    held.update(x["zh"] for x in o["examples"])

items={k:v for k,v in lex["items"].items() if k not in {"更多","泰语","中文","学习","休息","东西"}}
places=lex["places"]; foods=lex["foods"]; drinks=lex["drinks"]
adjs=lex["adjectives"]; emotions=lex["emotions"]; times=lex["times"]
rows=[]; skipped=Counter()
def add(zh,th,frame):
    if zh in held:
        skipped["heldout"]+=1; return
    if zh in seen:
        skipped["duplicate"]+=1; return
    seen.add(zh);rows.append({"zh":zh,"th":th,"frame":frame})

for t,tt in times.items():
    for z,th in items.items():
        add(f"我{t}需要{z}。",f"{tt}ฉันต้องการ{th}","time_need")
        add(f"我{t}没有{z}。",f"{tt}ฉันไม่มี{th}","time_no_have")
    for z,th in drinks.items():
        add(f"我{t}想喝{z}。",f"{tt}ฉันอยากดื่ม{th}","time_drink")
    for z,th in foods.items():
        add(f"我{t}想吃{z}。",f"{tt}ฉันอยากกิน{th}","time_eat")
    for z,th in places.items():
        add(f"我{t}想去{z}。",f"{tt}ฉันอยากไป{th}","time_go")
        add(f"我们{t}去{z}。",f"{tt}พวกเราไป{th}","time_group_go")
    add(f"你{t}有时间吗？",f"{tt}คุณมีเวลาไหม","time_have_time")
for z,th in items.items():
    add(f"你今天需要{z}吗？",f"วันนี้คุณต้องการ{th}ไหม","you_need")
    add(f"我明天正在找{z}。",f"พรุ่งนี้ฉันกำลังหา{th}","find_tomorrow")
    add(f"请给我看{z}。",f"กรุณาให้ฉันดู{th}","show_item")
    add(f"请帮我检查一下{z}。",f"กรุณาช่วยตรวจสอบ{th}ให้ฉันหน่อย","check_item")
    add(f"我今天不喜欢{z}。",f"วันนี้ฉันไม่ชอบ{th}","dislike_item")
for z,th in places.items():
    add(f"你明天要去{z}吗？",f"พรุ่งนี้คุณจะไป{th}ไหม","you_go_tomorrow")
    add(f"我们明天在{z}见面。",f"เราพบกันที่{th}พรุ่งนี้","meet_place")
for z,th in adjs.items():
    add(f"这个地方非常{z}。",f"สถานที่นี้{th}มาก","place_very_adj")
    add(f"我觉得这个地方很{z}。",f"ฉันคิดว่าสถานที่นี้{th}มาก","think_place_adj")
for z,th in emotions.items():
    add(f"我今天有点{z}。",f"วันนี้ฉัน{th}นิดหน่อย","today_emotion")

out=train_dir/"compositional_augmentation.json"
grouped=[{"id":i,"section":"ประโยคผสมและการขยายรูปแบบ","category":row["frame"],
          "zh":row["zh"],"th1":row["th"],"th2":None,"th3":None,"frame":row["frame"]}
         for i,row in enumerate(rows,1)]
payload={"schema_version":1,"topic":"compositional_augmentation",
         "description":"Template-generated augmentation from the project lexicon. Held-out evaluation strings are excluded.",
         "metadata":{"source_language":"zh","target_language":"th","task":"translation","format":"json_multi_reference_object","tokenizer":"shared_bpe_250_with_character_fallback","generation_method":"template_generated","review_status":"needs_human_review"},
         "examples":grouped}
out.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
if out.name not in manifest["training_files"]:
    manifest["training_files"].append(out.name)
def ref_count(shard):
    return sum(1 for x in shard["examples"] for k in ("th1","th2","th3") if isinstance(x.get(k),str) and x[k].strip())
shards=[json.loads((train_dir/f).read_text(encoding="utf-8")) for f in manifest["training_files"]]
manifest["total_examples"]=sum(ref_count(shard) for shard in shards)
manifest["total_source_groups"]=sum(len(shard["examples"]) for shard in shards)
manifest["topics"]["compositional_augmentation"]=len(rows)
manifest["training_files"]=sorted(manifest["training_files"])
(DATA/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
overlap=sorted({x["zh"] for x in rows}&held)
assert not overlap,overlap
print("ADDED",len(rows),"TOTAL",manifest["total_examples"],"HELDOUT",len(held),"OVERLAP",len(overlap))
print("FRAMES",dict(Counter(x["frame"] for x in rows)))
print("SKIPPED",dict(skipped))
