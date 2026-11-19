"""Normalize grouped training/evaluation text and refresh manifest counters."""
import json, unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"
MP=DATA/"manifest.json"
m=json.loads(MP.read_text(encoding="utf-8"))
def clean_zh(s):
 return " ".join(unicodedata.normalize("NFC",s).split()).strip()
def clean_th(s):
 s=unicodedata.normalize("NFC",s).replace("，"," ")
 return " ".join(s.split()).strip()
def save(p,o):
 t=p.with_name(p.name+".tmp")
 t.write_text(json.dumps(o,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 t.replace(p)
def refs(d):
 return sum(1 for x in d["examples"] for k in ("th1","th2","th3") if isinstance(x[k],str) and x[k].strip())
for folder,names in [(DATA/"train",m["training_files"]),(DATA/"evaluation",m["evaluation_files"])]:
 for fn in names:
  p=folder/fn; d=json.loads(p.read_text(encoding="utf-8"))
  d["metadata"]["format"]="json_multi_reference_object"
  for x in d["examples"]:
   for k in ("section","category","frame"):
    x[k]=clean_zh(x[k])
   x["zh"]=clean_zh(x["zh"])
   for k in ("th1","th2","th3"):
    if isinstance(x[k],str): x[k]=clean_th(x[k])
  save(p,d)
m["format"]="pkanai-grouped-multi-reference"
m["record_schema"]=["id","section","category","zh","th1","th2","th3","frame"]
m["translation_fields"]=["th1","th2","th3"]
m["metadata_values"]["format"]="json_multi_reference_object"
train=[json.loads((DATA/"train"/f).read_text(encoding="utf-8")) for f in m["training_files"]]
ev=[json.loads((DATA/"evaluation"/f).read_text(encoding="utf-8")) for f in m["evaluation_files"]]
m["topics"]={d["topic"]:refs(d) for d in train}
m["total_source_groups"]=sum(len(d["examples"]) for d in train)
m["total_examples"]=sum(refs(d) for d in train)
m["evaluation_groups"]=sum(len(d["examples"]) for d in ev)
m["evaluation_examples"]=sum(refs(d) for d in ev)
save(MP,m)
print("Normalized grouped data and refreshed manifest.")
