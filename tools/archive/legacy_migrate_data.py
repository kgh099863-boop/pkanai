import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
source=DATA/"train.json"
obj=json.loads(source.read_text(encoding="utf-8"))
train_dir=DATA/"train"
train_dir.mkdir(parents=True,exist_ok=True)
counts=Counter()
for cat in obj["categories"]:
    topic=cat["topic"]
    payload={
        "metadata": obj.get("metadata",{}),
        "topic": topic,
        "description": cat.get("description",""),
        "examples": cat.get("examples",[])
    }
    (train_dir/f"{topic}.json").write_text(
        json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    counts[topic]=len(payload["examples"])

# Generalization examples are deliberately separate from training files.
generalization=[
("我今天想喝咖啡。","วันนี้ฉันอยากดื่มกาแฟ"),
("你明天要去医院吗？","พรุ่งนี้คุณจะไปโรงพยาบาลไหม"),
("这个苹果多少钱？","แอปเปิลลูกนี้ราคาเท่าไหร่"),
("我们下午在学校见面。","เราพบกันที่โรงเรียนตอนบ่าย"),
("今天晚上我们一起吃饭。","คืนนี้พวกเราไปกินข้าวด้วยกัน"),
("这个地方非常安静。","สถานที่นี้เงียบสงบมาก"),
("我现在需要帮助。","ตอนนี้ฉันต้องการความช่วยเหลือ"),
("我喜欢喝茶，也喜欢喝咖啡。","ฉันชอบดื่มชา และชอบดื่มกาแฟด้วย"),
("请给我看这个文件。","กรุณาให้ฉันดูไฟล์นี้"),
("你今天有时间吗？","วันนี้คุณมีเวลาไหม"),
("我不想买这个东西。","ฉันไม่อยากซื้อสิ่งนี้"),
("请慢一点说，我听不清楚。","กรุณาพูดช้าลงหน่อย ฉันฟังไม่ชัด"),
("我们明天早上去机场。","พรุ่งนี้เช้าพวกเราไปสนามบิน"),
("这个房间很干净，也很舒服。","ห้องนี้สะอาดมากและสบายมาก"),
("我已经收到你的消息了。","ฉันได้รับข้อความของคุณแล้ว"),
("请帮我检查一下手机。","กรุณาช่วยตรวจสอบโทรศัพท์มือถือให้ฉันหน่อย"),
("我想学习更多泰语。","ฉันอยากเรียนภาษาไทยเพิ่มเติม"),
("如果你有问题，请告诉我。","ถ้าคุณมีปัญหา กรุณาบอกฉัน"),
("现在外面正在下雨。","ตอนนี้ข้างนอกกำลังฝนตก"),
("我们可以在这里休息一下。","พวกเราพักที่นี่สักหน่อยได้"),
("这个价格太贵了，可以便宜一点吗？","ราคานี้แพงเกินไป ลดลงอีกหน่อยได้ไหม"),
("请不要忘记保存文件。","กรุณาอย่าลืมบันทึกไฟล์"),
("我想找一家附近的餐厅。","ฉันอยากหาร้านอาหารใกล้ ๆ"),
("你可以再解释一次吗？","คุณช่วยอธิบายอีกครั้งได้ไหม"),
("我今天很忙，明天再联系你。","วันนี้ฉันยุ่งมาก พรุ่งนี้ค่อยติดต่อคุณ"),
]
# Keep the challenge set genuinely unseen and composed from known CJK characters.
all_examples=[x for p in train_dir.glob("*.json") for x in json.loads(p.read_text(encoding="utf-8"))["examples"]]
seen={x["zh"] for x in all_examples}
known_chars=set("".join(x["zh"] for x in all_examples))
def is_han(c):
    n=ord(c)
    return (0x3400<=n<=0x4DBF or 0x4E00<=n<=0x9FFF or
            0xF900<=n<=0xFAFF or 0x20000<=n<=0x2FA1F)
kept=[]
for zh,th in generalization:
    missing=sorted({c for c in zh if is_han(c) and c not in known_chars})
    if zh not in seen and not missing:
        kept.append({"zh":zh,"th":th})
    else:
        print("SKIP_TEST",zh,"duplicate" if zh in seen else "unknown="+ "".join(missing))
eval_dir=DATA/"evaluation"
eval_dir.mkdir(parents=True,exist_ok=True)
(eval_dir/"generalization.json").write_text(
    json.dumps({"metadata":{"purpose":"held-out compositional generalization test",
                            "not_for_training":True},
                "examples":kept},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
manifest={"format":"pkanai-topic-shards-v1","training_files":[f"{k}.json" for k in sorted(counts)],
          "total_examples":sum(counts.values()),"topics":dict(counts),
          "evaluation_files":["../evaluation/generalization.json"]}
(DATA/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
source.unlink()
print("SPLIT_DONE",sum(counts.values()),"topics",len(counts),"heldout",len(kept))
print("FILES",",".join(f"{k}.json:{v}" for k,v in sorted(counts.items())))
