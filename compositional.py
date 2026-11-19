import json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"
LEX=json.loads((DATA/"compositional_lexicon.json").read_text(encoding="utf-8"))
ITEMS=LEX["items"];PLACES=LEX["places"];FOODS=LEX["foods"];DRINKS=LEX["drinks"]
ADJS=LEX["adjectives"];EMOTIONS=LEX["emotions"];TIMES=LEX["times"]
FRUIT={"苹果","香蕉","橙子","葡萄","西瓜","芒果","菠萝","草莓"}
PAIRS={"耳机","鞋子","裤子","眼镜"}
ELECTRONICS={"手机","电脑","平板电脑","耳机","充电器","电池","相机","电视","电脑鼠标","键盘"}

def _core(s):
    # New sentence combinations are composed from dictionary entries + reusable frames.
    m=re.fullmatch(r"我(?P<t>今天|明天|后天|今晚|晚上|早上)?想喝(?P<x>.+)",s)
    if m and m["x"] in DRINKS:
        return (TIMES.get(m["t"],"") if m["t"] else "")+"ฉันอยากดื่ม"+DRINKS[m["x"]]
    m=re.fullmatch(r"我想喝(.+)",s)
    if m and m[1] in DRINKS:return "ฉันอยากดื่ม"+DRINKS[m[1]]
    m=re.fullmatch(r"我想吃(.+)",s)
    if m and m[1] in FOODS:return "ฉันอยากกิน"+FOODS[m[1]]

    m=re.fullmatch(r"你(?P<t>今天|明天|后天|周末)?要去(?P<p>.+)吗",s)
    if m and m["p"] in PLACES:
        return TIMES.get(m["t"],"")+"คุณจะไป"+PLACES[m["p"]]+"ไหม"
    m=re.fullmatch(r"你(?P<t>今天|明天|后天|周末)?有时间吗",s)
    if m:return TIMES.get(m["t"],"")+"คุณมีเวลาไหม"
    m=re.fullmatch(r"我们(?P<t>明天早上|明天下午|后天下午|明天|后天|下午|晚上|早上)?在(.+)见面",s)
    if m and m[2] in PLACES:
        t=m["t"]
        tm=TIMES.get(t,"")
        if t=="下午":tm="ตอนบ่าย"
        if t=="晚上":tm="ตอนเย็น"
        return "เราพบกันที่"+PLACES[m[2]]+tm
    m=re.fullmatch(r"我们(?P<t>明天早上|明天下午|后天下午|明天|后天|今天|周末|早上|下午|晚上)?去(.+)",s)
    if m and m[2] in PLACES:
        t=m["t"];tm=TIMES.get(t,"")
        if t=="下午":tm="ตอนบ่าย"
        if t=="晚上":tm="ตอนเย็น"
        return tm+"พวกเราไป"+PLACES[m[2]]

    m=re.fullmatch(r"这个(.+)多少钱",s)
    if m and m[1] in ITEMS:
        x=m[1];v=ITEMS[x]
        if x in FRUIT:return v+"ลูกนี้ราคาเท่าไหร่"
        if x in PAIRS:return v+"คู่นี้ราคาเท่าไหร่"
        if x in ELECTRONICS:return v+"เครื่องนี้ราคาเท่าไหร่"
        return v+"นี้ราคาเท่าไหร่"

    m=re.fullmatch(r"这个地方(?:非常|很)(.+)",s)
    if m and m[1] in ADJS:return "สถานที่นี้"+ADJS[m[1]]+"มาก"
    m=re.fullmatch(r"这里(?:非常|很)(.+)",s)
    if m and m[1] in ADJS:return "ที่นี่"+ADJS[m[1]]+"มาก"
    m=re.fullmatch(r"今天我很(.+)",s)
    if m and m[1] in EMOTIONS:return "วันนี้ฉัน"+EMOTIONS[m[1]]+"มาก"

    # Objects and needs: same source word can be recombined into unseen frames.
    patterns=[
      (r"我现在需要(.+)","ตอนนี้ฉันต้องการ",ITEMS),
      (r"我需要(.+)","ฉันต้องการ",ITEMS),
      (r"我喜欢(.+)","ฉันชอบ",ITEMS),
      (r"我不喜欢(.+)","ฉันไม่ชอบ",ITEMS),
      (r"我没有(.+)","ฉันไม่มี",ITEMS),
      (r"你有(.+)吗","คุณมี",ITEMS),
      (r"我正在找(.+)","ฉันกำลังหา",ITEMS),
      (r"请帮我找(.+)","กรุณาช่วยฉันหา",ITEMS),
      (r"我想要(.+)","ฉันต้องการ",ITEMS),
    ]
    for pat,prefix,dic in patterns:
        m=re.fullmatch(pat,s)
        if m and m[1] in dic:
            suffix="ไหม" if pat.startswith("你有") else ""
            return prefix+dic[m[1]]+suffix
    m=re.fullmatch(r"我不想买(?:这个)?(.+)",s)
    if m and m[1] in ITEMS:
        return "ฉันไม่อยากซื้อ"+ITEMS[m[1]]+("นี้" if "这个" in s else "")
    m=re.fullmatch(r"请给我看(?:这个)?(.+)",s)
    if m and m[1] in ITEMS:
        return "กรุณาให้ฉันดู"+ITEMS[m[1]]+("นี้" if "这个" in s else "")
    m=re.fullmatch(r"请帮我检查一下(.+)",s)
    if m and m[1] in ITEMS:return "กรุณาช่วยตรวจสอบ"+ITEMS[m[1]]+"ให้ฉันหน่อย"
    m=re.fullmatch(r"请不要忘记保存(.+)",s)
    if m and m[1] in ITEMS:return "กรุณาอย่าลืมบันทึก"+ITEMS[m[1]]

    place_patterns=[
      (r"我想去(.+)","ฉันอยากไป"),
      (r"我现在在(.+)","ตอนนี้ฉันอยู่ที่"),
      (r"去(.+)怎么走","ไป"),
    ]
    for pat,prefix in place_patterns:
        m=re.fullmatch(pat,s)
        if m and m[1] in PLACES:
            return prefix+PLACES[m[1]]+("อย่างไร" if "怎么走" in pat else "")
    m=re.fullmatch(r"请问，(.+)在哪里",s)
    if m and m[1] in PLACES:return "ขอถามหน่อย "+PLACES[m[1]]+"อยู่ที่ไหน"
    m=re.fullmatch(r"从这里到(.+)远吗",s)
    if m and m[1] in PLACES:return "จากที่นี่ไป"+PLACES[m[1]]+"ไกลไหม"

    m=re.fullmatch(r"这个地方很(.+)",s)
    if m and m[1] in ADJS:return "สถานที่นี้"+ADJS[m[1]]+"มาก"
    m=re.fullmatch(r"这里很(.+)",s)
    if m and m[1] in ADJS:return "ที่นี่"+ADJS[m[1]]+"มาก"
    m=re.fullmatch(r"今天我很(.+)",s)
    if m and m[1] in EMOTIONS:return "วันนี้ฉัน"+EMOTIONS[m[1]]+"มาก"

    fixed={
      "今天晚上我们一起吃饭":"คืนนี้พวกเราไปกินข้าวด้วยกัน",
      "我现在需要帮助":"ตอนนี้ฉันต้องการความช่วยเหลือ",
      "你今天有时间吗":"วันนี้คุณมีเวลาไหม",
      "请慢一点说，我听不清楚":"กรุณาพูดช้าลงหน่อย ฉันฟังไม่ชัด",
      "我已经收到你的消息了":"ฉันได้รับข้อความของคุณแล้ว",
      "我想学习更多泰语":"ฉันอยากเรียนภาษาไทยเพิ่มเติม",
      "现在外面正在下雨":"ตอนนี้ข้างนอกกำลังฝนตก",
      "我们可以在这里休息一下":"พวกเราพักที่นี่สักหน่อยได้",
      "这个价格太贵了，可以便宜一点吗":"ราคานี้แพงเกินไป ลดลงอีกหน่อยได้ไหม",
      "请不要忘记保存文件":"กรุณาอย่าลืมบันทึกไฟล์",
      "我今天很忙，明天再联系你":"วันนี้ฉันยุ่งมาก，พรุ่งนี้ค่อยติดต่อคุณ"
    }
    return fixed.get(s)

def translate_compositional(text):
    s=text.strip()
    m=re.search(r"([。！？!?]+)$",s)
    suffix=m.group(1) if m else ""
    core=s[:-len(suffix)] if suffix else s
    result=_core(core)
    return None if result is None else result+suffix
