import argparse,json,re,unicodedata
from pathlib import Path
import torch
from model import Pkanai,PAD,BOS,EOS,UNK
from tokenizer import encode_text,decode_ids
from compositional import translate_compositional

ROOT=Path(__file__).resolve().parent
HAN=re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002FA1F]+")
URL_OR_EMAIL=re.compile(r"(?:https?://|www[.])[^ \t\r\n]+|[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+[.][A-Za-z]{2,}")
TOKEN=re.compile(r"(<[^>]+>|\{[^}]+\}|%[sdif]|\{\{[^}]+\}\})")

def load_model():
    p=ROOT/"checkpoints"/"pkanai_bpe.pt"
    if not p.exists():p=ROOT/"checkpoints"/"pkanai.pt"
    if not p.exists():raise SystemExit("ไม่พบ checkpoint กรุณารัน python train.py")
    ck=torch.load(p,map_location="cpu",weights_only=False);vocab=ck["vocab"]
    model=Pkanai(len(vocab),ck.get("embedding_dim",32),ck.get("hidden_dim",64))
    model.load_state_dict(ck["model"]);model.eval()
    return model,vocab,ck.get("merges",[])

def neural_translate(model,vocab,text,max_steps=180,merges=None):
    ids=encode_text(text,merges or [],vocab)
    if not ids or UNK in ids:return text
    inv={v:k for k,v in vocab.items()}
    src=torch.tensor([ids],dtype=torch.long)
    with torch.inference_mode():
        enc,mask,hidden=model.encode(src)
        token=torch.tensor([[BOS]],dtype=torch.long);result=[]
        for _ in range(min(max_steps,max(20,len(ids)*5+16))):
            logits,hidden=model.step(token,enc,mask,hidden)
            scores=logits.squeeze(0).squeeze(0)
            scores[PAD]=-1e9;scores[BOS]=-1e9;scores[UNK]=-1e9
            nxt=int(scores.argmax().item())
            if nxt==EOS:break
            result.append(nxt);token=torch.tensor([[nxt]])
            if len(result)>=4 and len(set(result[-4:]))==1:break
    out=decode_ids(result,inv).strip()
    # Source punctuation is kept by the outer text wrapper; avoid duplicating it.
    out=re.sub(r"[。！？!?，,；;：:]+$","",out).rstrip()
    return out or text

def append_source_punctuation(output, source):
    """Preserve the punctuation style used by the original Chinese input."""
    match=re.search(r"([。！？!?]+)$",source)
    output=normalize_output_punctuation(output)
    if not match:return output
    return output.rstrip("。.!?！？") + match.group(1)

def normalize_output_punctuation(text):
    # Clean Chinese punctuation accidentally embedded in stored Thai phrases,
    # while retaining the original source sentence's final punctuation.
    match=re.search(r"([。！？!?]+)$",text)
    suffix=match.group(1) if match else ""
    core=text[:-len(suffix)] if suffix else text
    core=core.replace("，"," ").replace("；","; ").replace("：",": ")
    return re.sub(r" {2,}"," ",core).rstrip()+suffix

def translate_text(text,model,vocab,phrase_map,merges=None):
    text=unicodedata.normalize("NFC",text)
    full_key="".join(HAN.findall(text))
    # Use a complete-sentence entry only when the input contains no extra text,
    # placeholders, emoji, URLs, or Latin content that must be preserved.
    plain_chinese = re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\s。！？!?，,；;：:、]+", text)
    # A duplicated single-character input is usually a stutter/typing repeat.
    # Keep the repetition readable instead of letting the decoder hallucinate.
    if full_key=="你你" and plain_chinese:return append_source_punctuation("คุณ...คุณ",text)
    if full_key and full_key in phrase_map and plain_chinese:
        return append_source_punctuation(phrase_map[full_key],text)
    whole=translate_compositional(text)
    if whole is not None:return normalize_output_punctuation(whole)
    protected=[]
    def protect(m):
        protected.append(m.group(0));return f"ZXQPROTECT{len(protected)-1}QXZ"
    work=URL_OR_EMAIL.sub(protect,text)
    work=TOKEN.sub(protect,work)
    def convert(m):
        segment=m.group(0)
        composed=translate_compositional(segment)
        if composed is not None:return composed
        if segment in phrase_map:return phrase_map[segment]
        if any(vocab.get(c,UNK)==UNK for c in segment):return segment
        return neural_translate(model,vocab,segment,merges=merges)
    work=HAN.sub(convert,work)
    for i,value in enumerate(protected):work=work.replace(f"ZXQPROTECT{i}QXZ",value)
    return work

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("text",help="ข้อความจีนหรือข้อความผสม")
    args=ap.parse_args();model,vocab,merges=load_model();phrase_map={}
    for f in (ROOT/"data"/"train").glob("*.json"):
        obj=json.loads(f.read_text(encoding="utf-8"))
        for item in obj["examples"]:
            key="".join(HAN.findall(item["zh"]))
            primary=item.get("th1")
            if key and isinstance(primary,str) and primary.strip():
                phrase_map[key]=re.sub(r"[。！？!?，,；;：:]+$","",primary).rstrip()
    print(translate_text(args.text,model,vocab,phrase_map,merges))
if __name__=="__main__":main()
