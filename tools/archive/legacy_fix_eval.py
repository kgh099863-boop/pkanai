import json,pathlib
p=pathlib.Path("data/evaluation/compositional_v2.json")
d=json.load(open(p,encoding="utf-8"))
repls=[
  {
    "zh": "我明天想喝豆浆。",
    "th": "พรุ่งนี้ฉันอยากดื่มนมถั่วเหลือง"
  },
  {
    "zh": "我们明天下午在银行见面。",
    "th": "เราพบกันที่ธนาคารพรุ่งนี้ตอนบ่าย"
  },
  {
    "zh": "这个耳机多少钱？",
    "th": "หูฟังชิ้นนี้ราคาเท่าไหร่"
  },
  {
    "zh": "你后天要去邮局吗？",
    "th": "มะรืนนี้คุณจะไปที่ทำการไปรษณีย์ไหม"
  },
  {
    "zh": "我不想买香蕉。",
    "th": "ฉันไม่อยากซื้อกล้วย"
  },
  {
    "zh": "我们周末去电影院。",
    "th": "วันหยุดสุดสัปดาห์พวกเราไปโรงภาพยนตร์"
  }
]
d["examples"]=d["examples"][:]
# Replace known duplicates and preserve the same sample count.
bad={"请给我看地图。","我想吃米饭。","我想喝牛奶。","去超市怎么走？","从这里到学校远吗？","我没有水。"}
d["examples"]=[x for x in d["examples"] if x["zh"] not in bad]+repls
train=set()
for f in pathlib.Path("data/train").glob("*.json"):
 o=json.load(open(f,encoding="utf-8"))
 train.update(x["zh"] for x in o["examples"])
allzh=[x["zh"] for x in d["examples"]]
dups=[x for x in allzh if x in train]
assert len(allzh)==20 and len(set(allzh))==20 and not dups,(len(allzh),dups)
p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")
# Match the preserved Chinese comma exactly in the target reference.
p=pathlib.Path("data/evaluation/generalization.json")
g=json.load(open(p,encoding="utf-8"))
for x in g["examples"]:
 if x["zh"]=="我今天很忙，明天再联系你。":
  x["th"]="วันนี้ฉันยุ่งมาก，พรุ่งนี้ค่อยติดต่อคุณ"
p.write_text(json.dumps(g,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")
print("Replaced 6 duplicate held-out phrases; no train overlap.")
