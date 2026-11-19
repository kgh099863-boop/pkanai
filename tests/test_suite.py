import json,re,unicodedata,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from translate import load_model,translate_text,neural_translate,HAN
model,vocab,merges=load_model()
phrase_map={}
train_sources=set()
for f in (ROOT/"data"/"train").glob("*.json"):
    obj=json.loads(f.read_text(encoding="utf-8"))
    for x in obj["examples"]:
        train_sources.add(x["zh"])
        key="".join(HAN.findall(x["zh"]))
        primary=x.get("th1")
        if key and isinstance(primary,str) and primary.strip():
            phrase_map[key]=re.sub(r"[。！？!?，,；;：:]+$","",primary).rstrip()

def expected(x):
    suffix=re.search(r"[。！？!?]+$",x["zh"])
    return x["th1"]+(suffix.group(0) if suffix else "")

def char_f1(pred,ref):
    clean=lambda s:[c for c in s if not c.isspace() and not unicodedata.category(c).startswith("P")]
    a,b=Counter(clean(pred)),Counter(clean(ref))
    hit=sum((a&b).values())
    p=hit/max(1,sum(a.values()));r=hit/max(1,sum(b.values()))
    return 2*p*r/max(1e-9,p+r)

def check(name,examples):
    overlap=[x["zh"] for x in examples if x["zh"] in train_sources]
    assert not overlap,(name,"train leakage",overlap)
    exact=0;f1s=[]
    for x in examples:
        out=translate_text(x["zh"],model,vocab,phrase_map,merges)
        ref=expected(x);ok=out==ref;exact+=ok;f1s.append(char_f1(out,ref))
        print(("PASS" if ok else "FAIL"),x["zh"],"=>",out,"| REF:",ref)
    print(f"{name}: {exact}/{len(examples)} exact | char-F1 {sum(f1s)/max(1,len(f1s)):.3f}")
    return exact,len(examples)

special=[
("你好！","สวัสดี！"),
("你好 #AI @user $5.00 100%","สวัสดี #AI @user $5.00 100%"),
("你好 - (test); [OK] / + = _","สวัสดี - (test); [OK] / + = _"),
("你好 😊❤️","สวัสดี 😊❤️"),
("你好\n谢谢","สวัสดี\nขอบคุณ"),
("Hello 你好 123!","Hello สวัสดี 123!"),
("https://example.com 你好","https://example.com สวัสดี"),
("www.example.com 你好","www.example.com สวัสดี"),
("mail@test.com 你好","mail@test.com สวัสดี"),
("你好 {name} %s <b>OK</b>","สวัสดี {name} %s <b>OK</b>"),
("你好 {用户名}","สวัสดี {用户名}"),
("你好，世界！","สวัสดี，世界！"),
]
print("SPECIAL-CHARACTER / TEXT-PRESERVATION")
sp=0
for src,ref in special:
    out=translate_text(src,model,vocab,phrase_map,merges)
    ok=out==ref;sp+=ok
    print(("PASS" if ok else "FAIL"),repr(src),"->",repr(out))
print(f"SPECIAL: {sp}/{len(special)}")

for fn in ["generalization.json","compositional.json","generalization_extended.json"]:
    obj=json.loads((ROOT/"data"/"evaluation"/fn).read_text(encoding="utf-8"))
    check(fn,obj["examples"])

print("NEURAL-ONLY (same Han-span segmentation as production; no phrase lookup or rules)")
def neural_only_text(source):
    return HAN.sub(lambda m: neural_translate(model,vocab,m.group(0),merges=merges),source)
total=0;exact_total=0;all_f1=[]
for fn in ["generalization.json","compositional.json","generalization_extended.json"]:
    obj=json.loads((ROOT/"data"/"evaluation"/fn).read_text(encoding="utf-8"))
    exact=0;f1s=[]
    for x in obj["examples"]:
        out=neural_only_text(x["zh"])
        exact+=out==expected(x);f1s.append(char_f1(out,expected(x)))
    total+=len(obj["examples"]);exact_total+=exact;all_f1.extend(f1s)
    print(f"{fn}: {exact}/{len(obj['examples'])} exact | char-F1 {sum(f1s)/max(1,len(f1s)):.3f}")
print(f"NEURAL_ONLY_TOTAL: {exact_total}/{total} exact | char-F1 {sum(all_f1)/max(1,len(all_f1)):.3f}")
