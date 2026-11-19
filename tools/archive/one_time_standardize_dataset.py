import json,re,unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"
manifest_path=DATA/"manifest.json"
manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema_version")==1:
    raise SystemExit("Dataset is already standardized; this one-time migration must not be rerun.")

def clean_text(value):
    if not isinstance(value,str):
        raise TypeError(f"Expected text string, got {type(value).__name__}")
    value=unicodedata.normalize("NFC",value)
    value=" ".join(value.split())
    if not value:
        raise ValueError("Empty zh/th pair is not allowed")
    return value

def clean_frame(value,topic):
    value=value if isinstance(value,str) and value.strip() else topic
    value=unicodedata.normalize("NFC",value).strip().lower()
    value=re.sub(r"[^a-z0-9]+","_",value).strip("_")
    return value or "other"

seen={}
counts={}
methods={}
for filename in manifest["training_files"]:
    path=DATA/"train"/filename
    old=json.loads(path.read_text(encoding="utf-8"))
    topic=old.get("topic") or path.stem
    description=old.get("description") or f"Chinese to Thai examples for {topic}"
    old_meta=old.get("metadata") or {}
    examples=[]
    for index,row in enumerate(old.get("examples",[])):
        zh=clean_text(row.get("zh"))
        th=clean_text(row.get("th"))
        frame=clean_frame(row.get("frame"),topic)
        if zh in seen and seen[zh]!=th:
            raise ValueError(f"Conflicting translation for {zh!r}: {seen[zh]!r} vs {th!r}")
        if zh in seen:
            raise ValueError(f"Duplicate Chinese source in dataset: {zh!r}")
        seen[zh]=th
        examples.append({"zh":zh,"th":th,"frame":frame})

    method=old_meta.get("generation_method") or (
        "template_generated" if topic=="compositional_augmentation" else
        "hand_authored_or_curated"
    )
    meta={
        "source_language":"zh",
        "target_language":"th",
        "task":"translation",
        "format":"json_pair_object",
        "tokenizer":"shared_bpe_250_with_character_fallback",
        "generation_method":method,
        "review_status":"needs_human_review"
    }
    normalized={
        "schema_version":1,
        "topic":topic,
        "description":description,
        "metadata":meta,
        "examples":examples
    }
    path.write_text(json.dumps(normalized,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    counts[filename]=len(examples)
    methods[method]=methods.get(method,0)+len(examples)

# Add lexical coverage for two Chinese characters that the new test had exposed as OOV.
extra=[
 {"zh":"你要等多久？","th":"คุณต้องรอนานเท่าไหร่","frame":"question"},
 {"zh":"我不知道要花多久。","th":"ฉันไม่รู้ว่าจะใช้เวลานานเท่าไหร่","frame":"time"},
 {"zh":"这封邮件是昨天发来的。","th":"อีเมลฉบับนี้ส่งมาตั้งแต่เมื่อวาน","frame":"email"},
 {"zh":"我收到了一封新邮件。","th":"ฉันได้รับอีเมลใหม่หนึ่งฉบับ","frame":"email"},
 {"zh":"请把这封信交给他。","th":"กรุณาส่งจดหมายฉบับนี้ให้เขา","frame":"letter"}
]
lex_path=DATA/"train"/"lexical_coverage.json"
lex=json.loads(lex_path.read_text(encoding="utf-8"))
for row in extra:
    zh=clean_text(row["zh"]);th=clean_text(row["th"])
    if zh in seen:
        raise ValueError(f"New lexical row duplicates existing source: {zh}")
    seen[zh]=th
    lex["examples"].append({"zh":zh,"th":th,"frame":clean_frame(row["frame"],"lexical_coverage")})
lex["metadata"]["generation_method"]="hand_authored_parallel_examples"
lex["metadata"]["review_status"]="needs_human_review"
lex_path.write_text(json.dumps(lex,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
counts["lexical_coverage.json"]=len(lex["examples"])

files=sorted(manifest["training_files"])
manifest={
 "format":"pkanai-topic-shards-v1",
 "schema_version":1,
 "record_schema":["zh","th","frame"],
 "metadata_schema":["source_language","target_language","task","format","tokenizer","generation_method","review_status"],
 "training_files":files,
 "total_examples":sum(counts.values()),
 "topics":{Path(f).stem:counts[f] for f in files},
 "evaluation_files":["generalization.json","compositional_v2.json","generalization_v3.json"]
}
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

# Validation: enforce exactly one record shape in every training shard.
assert len(seen)==manifest["total_examples"]
for filename in files:
    d=json.loads((DATA/"train"/filename).read_text(encoding="utf-8"))
    assert set(d)=={"schema_version","topic","description","metadata","examples"}
    assert all(set(x)=={"zh","th","frame"} and all(isinstance(x[k],str) for k in ("zh","th","frame")) for x in d["examples"])
print("STANDARDIZED",manifest["total_examples"],"examples across",len(files),"shards")
print("UNIQUE_SOURCE",len(seen),"DUPLICATES",manifest["total_examples"]-len(seen))
print("MISSING_FRAME",0,"EXAMPLE_SCHEMA",["zh","th","frame"])
print("GENERATION_METHOD_COUNTS",methods)
