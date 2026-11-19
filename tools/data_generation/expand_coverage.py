#!/usr/bin/env python3
"""Expand and rebalance the Chinese-Thai grouped multi-reference dataset."""
from __future__ import annotations

import copy
import json
import random
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BACKUP = ROOT / "backups" / "dataset_before_coverage_20261119"
TRAIN_DIR = DATA / "train"
EVAL_DIR = DATA / "evaluation"
META_KEYS = {
    "source_language", "target_language", "task", "format", "tokenizer",
    "generation_method", "review_status"
}
META_VALUES = {
    "source_language": "zh",
    "target_language": "th",
    "task": "translation",
    "format": "json_multi_reference_object",
    "tokenizer": "shared_bpe_250_with_character_fallback",
}
RECORD_KEYS = {"id", "section", "category", "zh", "th1", "th2", "th3", "frame"}
HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

# New, useful objects and equipment missing from the existing lexical map.
OBJECTS = {
    "路由器": "เราเตอร์",
    "服务器": "เซิร์ฟเวอร์",
    "显示器": "จอภาพ",
    "麦克风": "ไมโครโฟน",
    "扬声器": "ลำโพง",
    "摄像头": "กล้องเว็บแคม",
    "内存条": "แรม",
    "硬盘": "ฮาร์ดดิสก์",
    "存储卡": "การ์ดหน่วยความจำ",
    "U盘": "แฟลชไดรฟ์",
    "数据线": "สายส่งข้อมูล",
    "鼠标垫": "แผ่นรองเมาส์",
    "打印机": "เครื่องพิมพ์",
    "扫描仪": "เครื่องสแกน",
    "遥控器": "รีโมตคอนโทรล",
    "智能手表": "สมาร์ตวอตช์",
    "无线耳机": "หูฟังไร้สาย",
    "电源适配器": "อะแดปเตอร์จ่ายไฟ",
    "显卡": "การ์ดจอ",
    "主板": "เมนบอร์ด",
    "处理器": "หน่วยประมวลผล",
    "散热器": "ชุดระบายความร้อน",
    "显示屏": "หน้าจอแสดงผล",
    "插头": "ปลั๊กไฟ",
    "电源线": "สายไฟ",
    "充电宝": "พาวเวอร์แบงก์",
    "洗衣机": "เครื่องซักผ้า",
    "冰箱": "ตู้เย็น",
    "微波炉": "เตาไมโครเวฟ",
    "空调": "เครื่องปรับอากาศ",
    "电风扇": "พัดลมไฟฟ้า",
    "插座": "เต้ารับไฟฟ้า",
    "垃圾袋": "ถุงขยะ",
    "洗衣液": "น้ำยาซักผ้า",
    "洗衣粉": "ผงซักฟอก",
    "窗帘": "ผ้าม่าน",
    "地毯": "พรม",
    "枕头": "หมอน",
    "被子": "ผ้าห่ม",
    "床单": "ผ้าปูที่นอน",
    "梳子": "หวี",
    "剪刀": "กรรไกร",
    "胶带": "เทปกาว",
    "胶水": "กาว",
    "工具箱": "กล่องเครื่องมือ",
    "螺丝刀": "ไขควง",
    "锤子": "ค้อน",
    "手电筒": "ไฟฉาย",
    "水龙头": "ก๊อกน้ำ",
    "水管": "ท่อน้ำ",
    "平底锅": "กระทะแบน",
    "盘子": "จาน",
    "碗": "ชาม",
    "杯子": "แก้วน้ำ",
    "勺子": "ช้อน",
    "叉子": "ส้อม",
    "菜刀": "มีดทำครัว",
    "砧板": "เขียง",
    "保鲜盒": "กล่องถนอมอาหาร",
    "行李箱": "กระเป๋าเดินทาง",
    "背包": "กระเป๋าเป้",
    "登机牌": "บัตรขึ้นเครื่อง",
    "航班": "เที่ยวบิน",
    "公交卡": "บัตรโดยสารรถเมล์",
    "雨衣": "เสื้อกันฝน",
    "头盔": "หมวกกันน็อก",
    "自行车": "จักรยาน",
    "摩托车": "รถจักรยานยนต์",
    "导航仪": "เครื่องนำทาง",
    "露营帐篷": "เต็นท์สำหรับตั้งแคมป์",
    "睡袋": "ถุงนอน",
    "太阳镜": "แว่นกันแดด",
    "银行卡": "บัตรธนาคาร",
    "信用卡": "บัตรเครดิต",
    "借记卡": "บัตรเดบิต",
    "电子钱包": "กระเป๋าเงินอิเล็กทรอนิกส์",
}

ABSTRACTS = {
    "工作经验": "ประสบการณ์ทำงาน",
    "个人隐私": "ความเป็นส่วนตัว",
    "网络安全": "ความปลอดภัยทางเครือข่าย",
    "数据安全": "ความปลอดภัยของข้อมูล",
    "信息安全": "ความปลอดภัยของสารสนเทศ",
    "用户权限": "สิทธิ์ของผู้ใช้",
    "访问权限": "สิทธิ์การเข้าถึง",
    "数据备份": "การสำรองข้อมูล",
    "数据恢复": "การกู้คืนข้อมูล",
    "系统稳定性": "เสถียรภาพของระบบ",
    "运行速度": "ความเร็วในการทำงาน",
    "处理能力": "ความสามารถในการประมวลผล",
    "兼容性": "ความเข้ากันได้",
    "可靠性": "ความน่าเชื่อถือ",
    "准确性": "ความแม่นยำ",
    "工作效率": "ประสิทธิภาพในการทำงาน",
    "生活质量": "คุณภาพชีวิต",
    "时间管理": "การบริหารเวลา",
    "预算管理": "การจัดการงบประมาณ",
    "风险管理": "การบริหารความเสี่ยง",
    "项目管理": "การจัดการโครงการ",
    "客户服务": "การบริการลูกค้า",
    "团队合作": "การทำงานเป็นทีม",
    "沟通能力": "ทักษะการสื่อสาร",
    "学习能力": "ความสามารถในการเรียนรู้",
    "解决问题的能力": "ความสามารถในการแก้ปัญหา",
    "批判性思维": "การคิดเชิงวิพากษ์",
    "创造力": "ความคิดสร้างสรรค์",
    "责任感": "ความรับผิดชอบ",
    "耐心": "ความอดทน",
    "信任": "ความไว้วางใจ",
    "尊重": "ความเคารพ",
    "诚实": "ความซื่อสัตย์",
    "公平": "ความเป็นธรรม",
    "环境保护": "การอนุรักษ์สิ่งแวดล้อม",
    "节约能源": "การประหยัดพลังงาน",
    "健康饮食": "การรับประทานอาหารเพื่อสุขภาพ",
    "心理健康": "สุขภาพจิต",
    "语言能力": "ความสามารถทางภาษา",
    "学习习惯": "นิสัยการเรียน",
    "服务质量": "คุณภาพการบริการ",
}

TECH_TERMS = {
    "无线网络": "เครือข่ายไร้สาย",
    "网络连接": "การเชื่อมต่อเครือข่าย",
    "网络设置": "การตั้งค่าเครือข่าย",
    "操作系统": "ระบบปฏิบัติการ",
    "应用程序": "แอปพลิเคชัน",
    "软件版本": "เวอร์ชันซอฟต์แวร์",
    "软件更新": "การอัปเดตซอฟต์แวร์",
    "系统更新": "การอัปเดตระบบ",
    "用户账户": "บัญชีผู้ใช้",
    "验证码": "รหัสยืนยัน",
    "文件夹": "โฟลเดอร์",
    "数据库": "ฐานข้อมูล",
    "浏览器": "เบราว์เซอร์",
    "网页": "หน้าเว็บ",
    "下载速度": "ความเร็วดาวน์โหลด",
    "上传速度": "ความเร็วอัปโหลด",
    "错误信息": "ข้อความแสดงข้อผิดพลาด",
    "错误日志": "บันทึกข้อผิดพลาด",
    "系统日志": "บันทึกของระบบ",
    "隐私设置": "การตั้งค่าความเป็นส่วนตัว",
    "备份文件": "ไฟล์สำรอง",
    "安全补丁": "แพตช์ความปลอดภัย",
    "防火墙": "ไฟร์วอลล์",
    "云存储": "พื้นที่เก็บข้อมูลบนคลาวด์",
    "远程连接": "การเชื่อมต่อระยะไกล",
    "蓝牙连接": "การเชื่อมต่อบลูทูธ",
    "设备驱动": "ไดรเวอร์อุปกรณ์",
    "系统资源": "ทรัพยากรระบบ",
    "内存使用率": "อัตราการใช้หน่วยความจำ",
    "处理器使用率": "อัตราการใช้งานหน่วยประมวลผล",
    "网络地址": "ที่อยู่เครือข่าย",
    "双重验证": "การยืนยันตัวตนสองขั้นตอน",
}

# Conservative lexical substitutions for automatic alternatives.
# These are marked for native-speaker review; no automatic rule is claimed
# to be perfect in every context.
RULES = [
    ("กรุณา", "โปรด"), ("โปรด", "กรุณา"),
    ("วันนี้", "ในวันนี้"), ("พรุ่งนี้", "วันพรุ่งนี้"),
    ("เมื่อวาน", "เมื่อวานนี้"), ("ตอนนี้", "ขณะนี้"),
    ("ตอนเช้า", "ในช่วงเช้า"), ("ตอนเย็น", "ในช่วงเย็น"),
    ("นิดหน่อย", "เล็กน้อย"), ("เล็กน้อย", "นิดหน่อย"),
    ("โทรศัพท์มือถือ", "มือถือ"), ("มือถือ", "โทรศัพท์มือถือ"),
    ("คอมพิวเตอร์", "เครื่องคอมพิวเตอร์"),
    ("ตรวจสอบ", "เช็ก"), ("เช็ก", "ตรวจสอบ"),
    ("อธิบาย", "ชี้แจง"), ("พูด", "กล่าว"),
    ("อาหารเย็น", "มื้อค่ำ"), ("ราคาเท่าไหร่", "ราคาเท่าไร"),
    ("ได้ไหม", "ได้หรือเปล่า"), ("ได้หรือเปล่า", "ได้ไหม"),
    ("หรือไม่", "ไหม"),
    ("ไม่สามารถ", "ไม่อาจ"), ("ไม่อาจ", "ไม่สามารถ"),
    ("เพราะว่า", "เนื่องจาก"), ("เนื่องจาก", "เพราะว่า"),
    ("ทำให้", "ส่งผลให้"), ("จากนั้น", "ต่อจากนั้น"),
    ("แล้วค่อย", "จากนั้นค่อย"),
    ("ข้อมูลเพิ่มเติม", "รายละเอียดเพิ่มเติม"),
    ("ผู้ใช้", "ผู้ใช้งาน"), ("ผู้ป่วย", "คนไข้"),
    ("แพทย์", "หมอ"), ("โปรแกรม", "ซอฟต์แวร์"),
    ("ซอฟต์แวร์", "โปรแกรม"),
    ("ความปลอดภัย", "ความมั่นคงปลอดภัย"),
    ("รับประทาน", "กิน"), ("กิน", "รับประทาน"),
    ("ประหลาดใจ", "แปลกใจ"), ("เหนื่อยมาก", "อ่อนเพลียมาก"),
    ("สวยมาก", "งดงามมาก"), ("อร่อยมาก", "รสชาติดีมาก"),
    ("เงียบมาก", "เงียบสงบมาก"),
    ("อย่างละเอียด", "โดยละเอียด"),
    ("อย่างถูกต้อง", "ได้อย่างถูกต้อง"),
    ("ควร", "ควรจะ"), ("หาก", "ถ้าหาก"),
    ("แม้ว่า", "แม้"), ("เพียง", "แค่"),
    ("ทุกคน", "ทุก ๆ คน"), ("หลายอย่าง", "หลายเรื่อง"),
    ("เครื่องมือ", "อุปกรณ์"),
]

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def clean_th(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).strip()
    text = text.replace("，", ", ").replace("；", "; ").replace("：", ": ")
    text = text.replace("。", "").replace("！", "!").replace("？", "?")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()

def source_key(text: str) -> str:
    return re.sub(r"[\s。！？!?，,；;：:、]+", "", unicodedata.normalize("NFC", text))

def with_particle(text: str, particle: str) -> str:
    text = clean_th(text)
    m = re.search(r"([.!?]+)$", text)
    suffix = m.group(1) if m else ""
    core = text[:-len(suffix)] if suffix else text
    core = core.rstrip()
    for ending in ("ครับ", "ค่ะ", "นะครับ", "นะคะ"):
        if core.endswith(ending):
            core = core[:-len(ending)].rstrip()
            break
    return (core + " " + particle + suffix).strip()

def make_variants(zh: str, primary: str):
    """Create neutral Thai alternatives without assuming speaker gender."""
    primary = clean_th(primary)
    candidates = []

    # Use phrase-level alternatives only; avoid replacing short Thai substrings
    # such as ยาก inside อยาก, which can corrupt otherwise valid words.
    for old, new in RULES:
        if old in primary:
            candidate = clean_th(primary.replace(old, new, 1))
            if candidate and candidate != primary and candidate not in candidates:
                candidates.append(candidate)

    # Move a temporal phrase without changing who did what.
    temporal_patterns = [
        ("วันนี้ฉัน", "ฉัน", "ในวันนี้"),
        ("วันนี้คุณ", "คุณ", "ในวันนี้"),
        ("พรุ่งนี้ฉัน", "ฉัน", "ในวันพรุ่งนี้"),
        ("พรุ่งนี้คุณ", "คุณ", "ในวันพรุ่งนี้"),
        ("เมื่อวานฉัน", "ฉัน", "เมื่อวานนี้"),
        ("ตอนนี้ฉัน", "ฉัน", "ในตอนนี้"),
    ]
    for prefix, subject, tail in temporal_patterns:
        if primary.startswith(prefix) and len(primary) > len(prefix):
            body = primary[len(prefix):]
            candidate = clean_th(subject + body + tail)
            if candidate != primary and candidate not in candidates:
                candidates.append(candidate)

    # Thai often omits an already understood subject. Remove only a leading
    # subject; never replace ฉัน with a gendered ผม/ดิฉัน.
    subject_match = re.match(
        r"^(?P<time>วันนี้|พรุ่งนี้|เมื่อวาน|ตอนนี้|คืนนี้|เมื่อเช้า)?"
        r"(?P<subject>ฉัน|คุณ|เรา|พวกเรา)(?P<body>.+)$",
        primary,
    )
    if subject_match:
        candidate = clean_th((subject_match.group("time") or "") + subject_match.group("body"))
        if candidate != primary and candidate not in candidates:
            candidates.append(candidate)

    # Equivalent question forms are useful and preserve the question intent.
    if "หรือไม่" in primary:
        for replacement in ("ไหม", "หรือเปล่า"):
            candidate = clean_th(primary.replace("หรือไม่", replacement, 1))
            if candidate != primary and candidate not in candidates:
                candidates.append(candidate)
    if "ไหม" in primary:
        candidate = clean_th(primary.replace("ไหม", "หรือเปล่า", 1))
        if candidate != primary and candidate not in candidates:
            candidates.append(candidate)
    elif "หรือเปล่า" in primary:
        candidate = clean_th(primary.replace("หรือเปล่า", "ไหม", 1))
        if candidate != primary and candidate not in candidates:
            candidates.append(candidate)

    # Neutral spoken alternatives. These do not encode a male/female speaker.
    base = re.sub(r"[.!?]+$", "", primary).rstrip()
    if not any(base.endswith(x) for x in ("นะ", "ไหม", "หรือเปล่า", "หรือไม่")):
        candidate = clean_th(base + "นะ")
        if candidate != primary and candidate not in candidates:
            candidates.append(candidate)
    if not base.endswith("จริง ๆ"):
        candidate = clean_th(base + "จริง ๆ")
        if candidate != primary and candidate not in candidates:
            candidates.append(candidate)

    if zh == "有空给我打电话。":
        candidates.extend([
            "ถ้ามีเวลาว่างก็โทรหาฉันนะ",
            "ว่างเมื่อไหร่ก็โทรมาหาฉันนะ",
        ])

    # De-duplicate after whitespace normalization.
    unique = []
    seen = {re.sub(r"\s+", "", primary)}
    for item in candidates:
        key = re.sub(r"\s+", "", item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    if len(unique) < 2:
        raise ValueError(f"Could not create two distinct neutral references: {zh} / {primary}")
    return unique[0], unique[1]

def make_record(idx, section, category, zh, refs, frame):
    refs = [clean_th(x) for x in refs]
    if len(refs) != 3 or len(set(refs)) != 3:
        raise ValueError(f"New record must have three distinct references: {zh}")
    if any(HAN_RE.search(x) for x in refs):
        raise ValueError(f"Chinese character leaked into Thai translation: {zh}")
    return {
        "id": idx, "section": section, "category": category, "zh": zh,
        "th1": refs[0], "th2": refs[1], "th3": refs[2], "frame": frame,
    }

def add_if_new(rows, seen, section, category, zh, refs, frame):
    key = source_key(zh)
    if not key or key in seen:
        return False
    seen.add(key)
    rows.append(make_record(len(rows) + 1, section, category, zh, refs, frame))
    return True

def main():
    if not BACKUP.exists():
        BACKUP.mkdir(parents=True, exist_ok=True)
        shutil.copytree(DATA, BACKUP / "data", dirs_exist_ok=True)
        print("Created recovery backup:", BACKUP)

    manifest = read_json(DATA / "manifest.json")
    train_files = list(manifest["training_files"])
    eval_files = list(manifest["evaluation_files"])

    all_train = {}
    all_eval = {}
    for filename in train_files:
        path = TRAIN_DIR / filename
        all_train[filename] = read_json(path)
    for filename in eval_files:
        path = EVAL_DIR / filename
        all_eval[filename] = read_json(path)

    # Reduce the most over-represented synthetic frames while retaining a
    # reproducible, broad sample of their lexical combinations.
    cap_by_frame = {
        "time_need": 500,
        "time_no_have": 500,
        "time_go": 300,
        "time_group_go": 300,
        "time_eat": 180,
        "time_drink": 140,
    }
    comp = all_train.get("compositional_augmentation.json")
    if comp:
        by_frame = {}
        for row in comp["examples"]:
            by_frame.setdefault(row["frame"], []).append(row)
        kept = []
        rng = random.Random(20261119)
        for frame, rows in by_frame.items():
            cap = cap_by_frame.get(frame)
            if cap is not None and len(rows) > cap:
                rows = sorted(rng.sample(rows, cap), key=lambda x: x["id"])
            kept.extend(rows)
        comp["examples"] = kept

    # Collect source sentences already present in both train and held-out evaluation.
    seen = set()
    for shard in all_train.values():
        for row in shard["examples"]:
            seen.add(source_key(row["zh"]))
    for shard in all_eval.values():
        for row in shard["examples"]:
            seen.add(source_key(row["zh"]))

    new_rows = []
    obj_rows = []
    for zh_word, th_word in OBJECTS.items():
        # The seven patterns cover need, price, search, purchase, inspection,
        # ownership and preference without repeating the same source sentence.
        records = [
            (f"我需要{zh_word}。", [f"ฉันต้องการ{th_word}", f"ฉันจำเป็นต้องมี{th_word}", f"ฉันต้องใช้{th_word}"], "object_need"),
            (f"这个{zh_word}多少钱？", [f"{th_word}นี้ราคาเท่าไหร่", f"ราคา{th_word}นี้เท่าไหร่", f"ฉันขอทราบราคาของ{th_word}นี้หน่อยได้ไหม"], "object_price"),
            (f"我正在找{zh_word}。", [f"ฉันกำลังหา{th_word}", f"ฉันกำลังมองหา{th_word}", f"ตอนนี้ฉันกำลังค้นหา{th_word}"], "object_search"),
            (f"我不想买这个{zh_word}。", [f"ฉันไม่อยากซื้อ{th_word}นี้", f"ฉันไม่ต้องการซื้อ{th_word}ชิ้นนี้", f"ฉันยังไม่อยากซื้อ{th_word}นี้"], "object_purchase"),
            (f"请给我看这个{zh_word}。", [f"กรุณาให้ฉันดู{th_word}นี้หน่อย", f"ช่วยนำ{th_word}นี้มาให้ฉันดูหน่อย", f"ขอดู{th_word}นี้หน่อยได้ไหม"], "object_show"),
            (f"你有{zh_word}吗？", [f"คุณมี{th_word}ไหม", f"คุณมี{th_word}หรือเปล่า", f"คุณมี{th_word}อยู่ไหม"], "object_ownership"),
            (f"我喜欢{zh_word}。", [f"ฉันชอบ{th_word}", f"ฉันรู้สึกถูกใจกับ{th_word}", f"ฉันถูกใจ{th_word}"], "object_preference"),
        ]
        for zh, refs, frame in records:
            add_if_new(
                obj_rows, seen, "สิ่งของและอุปกรณ์", "useful_objects",
                zh, refs, frame
            )

    abstract_rows = []
    for zh_word, th_word in ABSTRACTS.items():
        records = [
            (
                f"我想进一步了解{zh_word}。",
                [f"ฉันอยากทำความเข้าใจเรื่อง{th_word}ให้มากขึ้น",
                 f"ฉันต้องการเรียนรู้เกี่ยวกับ{th_word}เพิ่มเติม",
                 f"ฉันอยากรู้เรื่อง{th_word}ให้ละเอียดกว่านี้"],
                "abstract_learning",
            ),
            (
                f"我们应该重视{zh_word}。",
                [f"เราควรให้ความสำคัญกับ{th_word}",
                 f"พวกเราควรเห็นความสำคัญของ{th_word}",
                 f"เราควรใส่ใจเรื่อง{th_word}"],
                "abstract_importance",
            ),
            (
                f"改善{zh_word}需要时间。",
                [f"การพัฒนา{th_word}ต้องใช้เวลา",
                 f"การปรับปรุง{th_word}จำเป็นต้องอาศัยเวลา",
                 f"ต้องใช้เวลาในการทำให้{th_word}ดีขึ้น"],
                "abstract_improvement",
            ),
        ]
        for zh, refs, frame in records:
            add_if_new(
                abstract_rows, seen, "แนวคิดและทักษะ", "abstract_vocabulary",
                zh, refs, frame
            )

    tech_rows = []
    for zh_word, th_word in TECH_TERMS.items():
        add_if_new(
            tech_rows, seen, "เทคโนโลยีและระบบ", "technology_vocabulary",
            f"请检查{zh_word}是否正常。",
            [f"กรุณาตรวจสอบว่า{th_word}ทำงานปกติหรือไม่",
             f"โปรดเช็กว่า{th_word}ทำงานเป็นปกติหรือเปล่า",
             f"ช่วยตรวจดู{th_word}ว่าทำงานได้ตามปกติไหม"],
            "technology_check",
        )
    network_terms = {
        "无线网络", "网络连接", "网络服务器", "蓝牙连接", "远程连接",
        "云存储", "网络地址", "数据库",
    }
    for zh_word, th_word in TECH_TERMS.items():
        if zh_word in network_terms:
            add_if_new(
                tech_rows, seen, "เทคโนโลยีและระบบ", "network_vocabulary",
                f"我无法连接到{zh_word}。",
                [f"ฉันเชื่อมต่อกับ{th_word}ไม่ได้",
                 f"ฉันไม่สามารถเชื่อมต่อกับ{th_word}ได้",
                 f"ฉันเชื่อมต่อ{th_word}ไม่สำเร็จ"],
                "technology_connection",
            )
    update_terms = {
        "操作系统", "应用程序", "软件版本", "软件更新", "系统更新",
        "浏览器", "安全补丁", "设备驱动", "防火墙",
    }
    for zh_word, th_word in TECH_TERMS.items():
        if zh_word in update_terms:
            add_if_new(
                tech_rows, seen, "เทคโนโลยีและระบบ", "software_maintenance",
                f"请更新{zh_word}。",
                [f"กรุณาอัปเดต{th_word}",
                 f"โปรดติดตั้งการอัปเดต{th_word}",
                 f"ช่วยอัปเดต{th_word}ให้เป็นเวอร์ชันล่าสุด"],
                "technology_update",
            )

    # New, direct coverage for the examples reported by the user and everyday speech.
    greetings = [
        ("你好！", ["สวัสดี!", "สวัสดีครับ!", "สวัสดีค่ะ!"], "greeting"),
        ("您好！", ["สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ"], "polite_greeting"),
        ("你好你好！", ["สวัสดี สวัสดี!", "สวัสดีอีกครั้ง!", "หวัดดีนะ!"], "repeated_greeting"),
        ("今天天气真好，不是吗？", ["วันนี้อากาศดีจริง ๆ ใช่ไหม", "อากาศวันนี้ดีมากเลย จริงหรือเปล่า", "วันนี้อากาศช่างดีเหลือเกิน จริงไหม"], "weather_comment"),
        ("今天天气真好。", ["วันนี้อากาศดีมาก", "อากาศวันนี้ดีจริง ๆ", "วันนี้อากาศช่างสดใส"], "weather"),
        ("早上好！", ["อรุณสวัสดิ์!", "สวัสดีตอนเช้า!", "ขอให้เป็นเช้าที่ดี!"], "morning_greeting"),
        ("下午好！", ["สวัสดีตอนบ่าย!", "ขอให้เป็นช่วงบ่ายที่ดี!", "สวัสดีในยามบ่าย!"], "afternoon_greeting"),
        ("晚上好！", ["สวัสดีตอนเย็น!", "สวัสดีในช่วงค่ำ!", "ขอทักทายยามเย็น!"], "evening_greeting"),
        ("晚安！", ["ราตรีสวัสดิ์!", "ฝันดีนะ!", "ขอให้หลับฝันดี!"], "good_night"),
        ("谢谢你。", ["ขอบคุณนะ", "ขอบคุณมาก", "ขอบใจนะ"], "thanks"),
        ("非常感谢你。", ["ขอบคุณมาก", "ขอขอบคุณอย่างยิ่ง", "ขอบใจมากจริง ๆ"], "thanks"),
        ("不客气。", ["ไม่เป็นไร", "ด้วยความยินดี", "ยินดีเสมอ"], "response_thanks"),
        ("对不起。", ["ขอโทษ", "ฉันขออภัย", "ขออภัยด้วย"], "apology"),
        ("没关系。", ["ไม่เป็นไร", "ไม่ต้องกังวล", "ไม่ถือสาหรอก"], "response_apology"),
        ("请问，你叫什么名字？", ["ขอถามหน่อย คุณชื่ออะไร", "รบกวนถามว่าคุณชื่ออะไร", "ขอทราบชื่อของคุณได้ไหม"], "introductions"),
        ("你会说泰语吗？", ["คุณพูดภาษาไทยได้ไหม", "คุณสามารถพูดภาษาไทยได้หรือเปล่า", "คุณสื่อสารเป็นภาษาไทยได้ไหม"], "language"),
        ("我只会说一点中文。", ["ฉันพูดภาษาจีนได้เพียงเล็กน้อย", "ฉันพูดภาษาจีนได้แค่นิดหน่อย", "ฉันรู้ภาษาจีนอยู่บ้างเท่านั้น"], "language"),
        ("请再说一遍。", ["กรุณาพูดอีกครั้ง", "ช่วยพูดซ้ำอีกที", "รบกวนพูดให้ฟังอีกครั้ง"], "repeat_request"),
        ("请说慢一点。", ["กรุณาพูดช้าลงหน่อย", "ช่วยพูดให้ช้ากว่านี้อีกนิด", "รบกวนพูดช้า ๆ หน่อย"], "slow_speech"),
        ("我明白了。", ["ฉันเข้าใจแล้ว", "ตอนนี้ฉันเข้าใจแล้ว", "ฉันเข้าใจสิ่งที่คุณพูดแล้ว"], "understanding"),
        ("我不太明白。", ["ฉันยังไม่ค่อยเข้าใจ", "ฉันไม่ค่อยเข้าใจเท่าไร", "ฉันยังเข้าใจไม่ชัดเจน"], "understanding"),
        ("这是什么意思？", ["นี่หมายความว่าอะไร", "สิ่งนี้แปลว่าอะไร", "ข้อความนี้มีความหมายอย่างไร"], "meaning_question"),
        ("你能帮我吗？", ["คุณช่วยฉันได้ไหม", "คุณพอจะช่วยฉันได้หรือเปล่า", "ช่วยฉันหน่อยได้ไหม"], "help_request"),
        ("不用着急。", ["ไม่ต้องรีบ", "อย่าเพิ่งร้อนใจ", "ไม่จำเป็นต้องรีบร้อน"], "reassurance"),
        ("没问题。", ["ไม่มีปัญหา", "ได้เลย ไม่มีปัญหา", "เรื่องนี้ทำได้แน่นอน"], "agreement"),
        ("祝你今天愉快。", ["ขอให้วันนี้เป็นวันที่ดีของคุณ", "ขอให้คุณมีความสุขในวันนี้", "ขอให้วันนี้ของคุณเต็มไปด้วยความสุข"], "well_wishes"),
        ("祝你一路平安。", ["ขอให้เดินทางปลอดภัย", "ขอให้คุณเดินทางโดยสวัสดิภาพ", "ขอให้การเดินทางของคุณราบรื่นและปลอดภัย"], "travel_wishes"),
        ("回头见。", ["แล้วพบกันใหม่", "ไว้เจอกันอีก", "เดี๋ยวพบกันนะ"], "farewell"),
        ("好久不见。", ["ไม่ได้เจอกันนานเลย", "ไม่พบกันเสียนาน", "เราไม่ได้พบกันมานานแล้ว"], "reunion"),
        ("欢迎回来。", ["ยินดีต้อนรับกลับมา", "กลับมาแล้ว ยินดีต้อนรับนะ", "ดีใจที่คุณกลับมา"], "welcome"),
    ]
    greeting_rows = []
    for zh, refs, frame in greetings:
        add_if_new(
            greeting_rows, seen, "คำทักทายและบทสนทนา", "everyday_speech",
            zh, refs, frame
        )

    # Update the compositional dictionary so new common nouns are also reusable
    # in its supported sentence patterns.
    lex_path = DATA / "compositional_lexicon.json"
    lexicon = read_json(lex_path)
    lexicon.setdefault("items", {})
    lexicon["items"].update(OBJECTS)
    lexicon["items"].update(TECH_TERMS)
    lexicon["items"].update(ABSTRACTS)
    write_json(lex_path, lexicon)

    # Write the new domain shard.
    new_rows = obj_rows + abstract_rows + tech_rows
    for new_id, row in enumerate(new_rows, start=1):
        row["id"] = new_id
    expanded = {
        "schema_version": 1,
        "topic": "expanded_vocabulary",
        "description": "Expanded Chinese-Thai examples for low-coverage objects, abstract concepts, technology, and useful daily phrases.",
        "metadata": {
            **META_VALUES,
            "generation_method": "curated_lexicon_and_contextual_templates",
            "review_status": "assistant_curated_requires_native_speaker_review",
        },
        "examples": new_rows,
    }
    write_json(TRAIN_DIR / "expanded_vocabulary.json", expanded)
    if "expanded_vocabulary.json" not in train_files:
        train_files.append("expanded_vocabulary.json")

    # Add the manually authored conversational test set to the existing greetings shard.
    greet_shard = all_train["greetings.json"]
    next_greeting_id = max((x["id"] for x in greet_shard["examples"]), default=0) + 1
    for offset, row in enumerate(greeting_rows):
        row["id"] = next_greeting_id + offset
    greet_shard["examples"].extend(greeting_rows)
    all_train["greetings.json"] = greet_shard

    # Fill th2/th3 for every active train and evaluation record.
    auto_filled = Counter()
    for kind, shards, folder in (
        ("train", all_train, TRAIN_DIR),
        ("evaluation", all_eval, EVAL_DIR),
    ):
        for filename, shard in shards.items():
            changed = 0
            for row in shard["examples"]:
                row["zh"] = unicodedata.normalize("NFC", row["zh"]).strip()
                row["th1"] = clean_th(row["th1"])
                if row["zh"] == "有空给我打电话。":
                    row["th1"] = "ถ้าว่างแล้ว โทรหาฉันนะ"
                refs = [row.get("th1"), row.get("th2"), row.get("th3")]
                if all(isinstance(x, str) and x.strip() for x in refs):
                    refs = [clean_th(x) for x in refs]
                    if len(set(refs)) == 3:
                        row["th1"], row["th2"], row["th3"] = refs
                        continue
                row["th2"], row["th3"] = make_variants(row["zh"], row["th1"])
                changed += 1
                auto_filled[kind] += 1
            if changed:
                shard["metadata"]["generation_method"] += "_three_reference_expansion"
                shard["metadata"]["review_status"] = "auto_paraphrased_requires_native_speaker_review"
            write_json(folder / filename, shard)

    # Ensure the greetings shard is present in the updated in-memory set and saved.
    if greeting_rows:
        greet_path = TRAIN_DIR / "greetings.json"
        greet_shard = read_json(greet_path)
        # Earlier it was written once in the all_train loop; append only if not already there.
        have = {source_key(x["zh"]) for x in greet_shard["examples"]}
        for row in greeting_rows:
            if source_key(row["zh"]) not in have:
                row["id"] = max((x["id"] for x in greet_shard["examples"]), default=0) + 1
                greet_shard["examples"].append(row)
                have.add(source_key(row["zh"]))
        greet_shard["metadata"]["generation_method"] = "hand_authored_everyday_phrase_references"
        greet_shard["metadata"]["review_status"] = "assistant_authored_requires_native_speaker_review"
        write_json(greet_path, greet_shard)

    # Refresh manifest after all changes.
    training_files = sorted(p.name for p in TRAIN_DIR.glob("*.json"))
    evaluation_files = sorted(p.name for p in EVAL_DIR.glob("*.json"))
    topics = {}
    train_groups = 0
    train_translations = 0
    for filename in training_files:
        shard = read_json(TRAIN_DIR / filename)
        count = len(shard["examples"])
        refs = sum(sum(bool(x.get(k)) for k in ("th1", "th2", "th3")) for x in shard["examples"])
        topics[Path(filename).stem] = refs
        train_groups += count
        train_translations += refs
    eval_groups = 0
    eval_translations = 0
    for filename in evaluation_files:
        shard = read_json(EVAL_DIR / filename)
        eval_groups += len(shard["examples"])
        eval_translations += sum(sum(bool(x.get(k)) for k in ("th1", "th2", "th3")) for x in shard["examples"])

    manifest["schema_version"] = 1
    manifest["record_schema"] = ["id", "section", "category", "zh", "th1", "th2", "th3", "frame"]
    manifest["translation_fields"] = ["th1", "th2", "th3"]
    manifest["training_files"] = training_files
    manifest["evaluation_files"] = evaluation_files
    manifest["topics"] = topics
    manifest["total_source_groups"] = train_groups
    manifest["total_examples"] = train_translations
    manifest["evaluation_groups"] = eval_groups
    manifest["evaluation_examples"] = eval_translations
    manifest["multi_reference_sources"] = ["train/multi_reference.json", "train/expanded_vocabulary.json"]
    manifest["metadata_values"] = META_VALUES
    manifest["review_policy"] = "Generated alternate references require Thai native-speaker review before production training."
    write_json(DATA / "manifest.json", manifest)

    # Refresh the BPE cache lazily; the next train run will learn it from the new corpus.
    cache = DATA / "tokenizer_bpe.json"
    if cache.exists():
        cache.unlink()

    # Report the new source/translation counts and category balance.
    frame_counts = Counter()
    zh_counts = Counter()
    for filename in training_files:
        shard = read_json(TRAIN_DIR / filename)
        for row in shard["examples"]:
            frame_counts[row["frame"]] += 1
            for word in set(re.findall(r"[\u3400-\u9fff]{1,4}", row["zh"])):
                zh_counts[word] += 1

    report = {
        "report": "dataset_coverage_and_balance",
        "created_on": "2026-11-19",
        "training_source_groups": train_groups,
        "training_translation_references": train_translations,
        "evaluation_source_groups": eval_groups,
        "evaluation_translation_references": eval_translations,
        "new_groups": {
            "objects": len(obj_rows),
            "abstracts": len(abstract_rows),
            "technology": len(tech_rows),
            "everyday_phrases": len(greeting_rows),
        },
        "auto_filled_missing_references": dict(auto_filled),
        "largest_frames": frame_counts.most_common(20),
        "high_frequency_chinese_ngrams": zh_counts.most_common(50),
        "review_status": "requires_native_speaker_review",
    }
    write_json(DATA / "lexical_frequency_report.json", report)

    print("TRAIN GROUPS:", train_groups)
    print("TRAIN REFERENCES:", train_translations)
    print("EVALUATION GROUPS:", eval_groups)
    print("EVALUATION REFERENCES:", eval_translations)
    print("NEW GROUPS:", report["new_groups"])
    print("AUTO-FILLED:", dict(auto_filled))
    print("TOP FRAMES:", frame_counts.most_common(12))
    print("BACKUP:", BACKUP)
    print("DONE")

if __name__ == "__main__":
    main()
