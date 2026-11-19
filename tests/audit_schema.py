import json, unicodedata, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"
m=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
outer={"schema_version","topic","description","metadata","examples"}
record={"id","section","category","zh","th1","th2","th3","frame"}
meta=set(m["metadata_schema"]); bad=[]; train=set()
tot={"train_groups":0,"train_translations":0,"eval_groups":0,"eval_translations":0}
for kind,folder,names in [("train",DATA/"train",m["training_files"]),("eval",DATA/"evaluation",m["evaluation_files"])]:
 for fn in names:
  p=folder/fn
  if not p.exists(): bad.append((kind,fn,"missing")); continue
  d=json.loads(p.read_text(encoding="utf-8"))
  if set(d)!=outer: bad.append((kind,fn,"outer schema"))
  if d.get("schema_version")!=1: bad.append((kind,fn,"schema version"))
  if set(d.get("metadata",{}))!=meta: bad.append((kind,fn,"metadata keys"))
  for k,v in m["metadata_values"].items():
   if d.get("metadata",{}).get(k)!=v: bad.append((kind,fn,"metadata",k))
  ids=set()
  for n,x in enumerate(d.get("examples",[]),1):
   at=(kind,fn,n)
   if set(x)!=record: bad.append((*at,"record schema")); continue
   if not isinstance(x["id"],int) or x["id"] in ids: bad.append((*at,"id"))
   ids.add(x["id"])
   for k in ("section","category","zh","frame","th1"):
    if not isinstance(x[k],str) or not x[k].strip(): bad.append((*at,k,"empty"))
   refs=[]
   for k in ("th1","th2","th3"):
    v=x[k]
    if v is None and k!="th1": continue
    if not isinstance(v,str) or not v.strip(): bad.append((*at,k,"type/empty")); continue
    if v!=unicodedata.normalize("NFC",v) or v!=v.strip(): bad.append((*at,k,"normalization"))
    if re.search(r"[\u3400-\u9fff]",v): bad.append((*at,k,"Han in Thai"))
    if v in refs: bad.append((*at,k,"duplicate ref"))
    refs.append(v)
   if x["zh"]!=unicodedata.normalize("NFC",x["zh"]) or x["zh"]!=x["zh"].strip(): bad.append((*at,"zh","normalization"))
   if kind=="train":
    if x["zh"] in train: bad.append((*at,"duplicate train source"))
    train.add(x["zh"])
   elif x["zh"] in train: bad.append((*at,"train/eval leakage"))
   tot[kind+"_groups"]+=1
   tot[kind+"_translations"]+=len(refs)
checks={"train_groups":"total_source_groups","train_translations":"total_examples","eval_groups":"evaluation_groups","eval_translations":"evaluation_examples"}
for key,mkey in checks.items():
 if tot[key]!=m.get(mkey): bad.append(("manifest",mkey,tot[key],m.get(mkey)))
print("SCHEMA",m.get("record_schema"))
print("COUNTS",tot)
print("ERRORS",len(bad))
if bad:
 print("SAMPLES",bad[:30]); raise SystemExit(1)
print("AUDIT PASSED")
