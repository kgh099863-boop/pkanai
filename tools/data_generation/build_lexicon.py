import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"

def pairs():
    m=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
    for filename in m["training_files"]:
        obj=json.loads((DATA/"train"/filename).read_text(encoding="utf-8"))
        for x in obj["examples"]:
            for field in ("th1","th2","th3"):
                value=x.get(field)
                if isinstance(value,str) and value.strip():
                    yield x["zh"],value

rows=list(pairs())
lex={"items":{},"places":{},"foods":{},"drinks":{},"adjectives":{},"emotions":{},"times":{
"今天":"วันนี้","明天":"พรุ่งนี้","后天":"มะรืนนี้","昨天":"เมื่อวาน",
"现在":"ตอนนี้","早上":"ตอนเช้า","中午":"ตอนเที่ยง","下午":"ตอนบ่าย",
"晚上":"ตอนเย็น","今晚":"คืนนี้","明天早上":"พรุ่งนี้เช้า",
"明天下午":"พรุ่งนี้ตอนบ่าย","后天下午":"มะรืนนี้ตอนบ่าย",
"周末":"วันหยุดสุดสัปดาห์","下周":"สัปดาห์หน้า",
"下个月":"เดือนหน้า","这个星期":"สัปดาห์นี้"
}}
def extract(src_pre,src_suf,th_pre,th_suf,dest):
    for zh,th in rows:
        if not (zh.startswith(src_pre) and zh.endswith(src_suf)):continue
        if not (th.startswith(th_pre) and th.endswith(th_suf)):continue
        a=zh[len(src_pre):len(zh)-len(src_suf) if src_suf else None]
        b=th[len(th_pre):len(th)-len(th_suf) if th_suf else None]
        if a and b and len(a)<12 and len(b)<40:
            dest.setdefault(a,b)

# Extract item mappings from multiple different sentence frames.
for sp,ss,tp,ts in [
("我需要","。","ฉันต้องการ",""),
("我喜欢","。","ฉันชอบ",""),
("我不喜欢","。","ฉันไม่ชอบ",""),
("我没有","。","ฉันไม่มี",""),
("我正在找","。","ฉันกำลังหา",""),
("我想要","。","ฉันต้องการ",""),
("你有","吗？","คุณมี","ไหม"),
("我想吃","。","ฉันอยากกิน",""),
("我想喝","。","ฉันอยากดื่ม",""),
]:
    target=lex["foods"] if sp=="我想吃" else lex["drinks"] if sp=="我想喝" else lex["items"]
    extract(sp,ss,tp,ts,target)
for sp,tp in [("我想去","ฉันอยากไป"),("我现在在","ตอนนี้ฉันอยู่ที่")]:
    extract(sp,"。",tp,"",lex["places"])
for sp,tp in [("这个地方很","สถานที่นี้"),("这里非常","ที่นี่"),("我觉得这里很","ฉันคิดว่าที่นี่")]:
    extract(sp,"。",tp,"มาก",lex["adjectives"])
for sp,tp,ss,ts in [
("我今天很","วันนี้ฉัน","。","มาก"),
("我现在有点","ตอนนี้ฉัน","。","นิดหน่อย")
]:
    extract(sp,ss,tp,ts,lex["emotions"])
for k,v in {
"帮助":"ความช่วยเหลือ","东西":"สิ่ง","更多":"เพิ่มเติม","泰语":"ภาษาไทย",
"休息":"พักผ่อน","学习":"เรียน","手机":"โทรศัพท์มือถือ",
"文件":"ไฟล์","钥匙":"กุญแจ","钱包":"กระเป๋าสตางค์",
}.items():
    lex["items"][k]=v
# Known fixed lexical items and common terms missing from limited synthetic frames.
for k,v in {
"帮助":"ความช่วยเหลือ","消息":"ข้อความ","泰语":"ภาษาไทย",
"中文":"ภาษาจีน","东西":"สิ่ง"
}.items(): lex["items"][k]=v
out=DATA/"compositional_lexicon.json"
out.write_text(json.dumps(lex,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
print("LEXICON", {k:len(v) for k,v in lex.items()})
