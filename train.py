import argparse,json,random,time,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader,Sampler
from model import Pkanai,PAD,BOS,EOS,UNK,parameter_count
from tokenizer import learn_bpe,encode_text

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"

class TranslationData(Dataset):
    def __init__(self):
        manifest=json.loads((DATA/"manifest.json").read_text(encoding="utf-8"))
        expected_record={"id","section","category","zh","th1","th2","th3","frame"}
        if manifest.get("schema_version")!=1 or set(manifest.get("record_schema",[]))!=expected_record:
            raise ValueError("Unsupported grouped multi-reference data schema")
        translation_fields=manifest.get("translation_fields",["th1","th2","th3"])
        if translation_fields!=["th1","th2","th3"]:
            raise ValueError("Unexpected translation_fields in manifest")
        self.rows=[]
        for filename in manifest["training_files"]:
            obj=json.loads((DATA/"train"/filename).read_text(encoding="utf-8"))
            if obj.get("schema_version")!=1:
                raise ValueError(f"{filename}: schema_version must be 1")
            expected_meta={"source_language","target_language","task","format","tokenizer","generation_method","review_status"}
            if set(obj.get("metadata",{}))!=expected_meta:
                raise ValueError(f"{filename}: metadata keys do not match the project schema")
            for key,value in manifest.get("metadata_values",{}).items():
                if obj["metadata"].get(key)!=value:
                    raise ValueError(f"{filename}: metadata value mismatch for {key}")
            if set(obj)!={"schema_version","topic","description","metadata","examples"}:
                raise ValueError(f"{filename}: inconsistent shard structure")
            seen_ids=set()
            for line,item in enumerate(obj["examples"],1):
                if set(item)!=expected_record:
                    raise ValueError(f"{filename}:{line}: record keys do not match grouped schema")
                if not isinstance(item["id"],int) or item["id"] in seen_ids:
                    raise ValueError(f"{filename}:{line}: id must be a unique integer inside its shard")
                seen_ids.add(item["id"])
                for key in ("section","category","zh","frame"):
                    if not isinstance(item[key],str) or not item[key].strip():
                        raise ValueError(f"{filename}:{line}: {key} must be non-empty text")
                refs=[]
                for key in translation_fields:
                    value=item[key]
                    if value is None:
                        continue
                    if not isinstance(value,str) or not value.strip():
                        raise ValueError(f"{filename}:{line}: {key} must be null or non-empty text")
                    if value.strip() not in refs:
                        refs.append(value.strip())
                if not refs:
                    raise ValueError(f"{filename}:{line}: at least th1 must contain a translation")
                for translation in refs:
                    self.rows.append((item["zh"].strip(),translation,item["frame"]))
        signature=hashlib.sha256(json.dumps(self.rows,ensure_ascii=False).encode("utf-8")).hexdigest()
        cache_path=DATA/"tokenizer_bpe.json"
        cache=None
        if cache_path.exists():
            try:cache=json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:cache=None
        if cache and cache.get("corpus_sha256")==signature and cache.get("merges_count")==250:
            self.merges=[tuple(x) for x in cache["merges"]]
            self.vocab=cache["vocab"]
            print("Loaded cached BPE tokenizer",flush=True)
        else:
            print("Learning BPE tokenizer (first run or changed corpus)...",flush=True)
            self.merges,self.vocab=learn_bpe([x for z,t,_ in self.rows for x in (z,t)],250,2)
            cache={"corpus_sha256":signature,"merges_count":250,"merges":self.merges,"vocab":self.vocab}
            cache_path.write_text(json.dumps(cache,ensure_ascii=False),encoding="utf-8")
        self.samples=[(encode_text(z,self.merges,self.vocab),[BOS]+encode_text(t,self.merges,self.vocab)+[EOS])
                      for z,t,_ in self.rows]
        self.lengths=[max(len(s),len(t)) for s,t in self.samples]
        self.frames=[frame for _,_,frame in self.rows]
        self.sources=[zh for zh,_,_ in self.rows]
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):return self.samples[i]

def collate(batch):
    ss,tt=zip(*batch);sl=max(map(len,ss));tl=max(map(len,tt))
    src=torch.full((len(batch),sl),PAD,dtype=torch.long)
    din=torch.full((len(batch),tl-1),PAD,dtype=torch.long)
    target=torch.full((len(batch),tl-1),PAD,dtype=torch.long)
    for i,(s,t) in enumerate(zip(ss,tt)):
        src[i,:len(s)]=torch.tensor(s)
        din[i,:len(t)-1]=torch.tensor(t[:-1])
        target[i,:len(t)-1]=torch.tensor(t[1:])
    return src,din,target

class BucketBatchSampler(Sampler):
    def __init__(self,indices,lengths,batch_size,shuffle=True):
        self.indices=list(indices);self.lengths=lengths;self.batch_size=batch_size;self.shuffle=shuffle
    def __iter__(self):
        ids=sorted(self.indices,key=lambda i:self.lengths[i])
        buckets=[ids[i:i+self.batch_size*16] for i in range(0,len(ids),self.batch_size*16)]
        batches=[]
        for bucket in buckets:
            if self.shuffle:random.shuffle(bucket)
            batches.extend([bucket[i:i+self.batch_size] for i in range(0,len(bucket),self.batch_size)])
        if self.shuffle:random.shuffle(batches)
        yield from batches
    def __len__(self):return (len(self.indices)+self.batch_size-1)//self.batch_size

def select_device(requested):
    mps_backend=getattr(torch.backends,"mps",None)
    mps_available=bool(mps_backend is not None and mps_backend.is_available())
    if requested=="cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested, but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if requested=="mps":
        if not mps_available:
            raise SystemExit("--device mps requested, but the PyTorch MPS backend is unavailable")
        return torch.device("mps")
    if requested=="cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model,loader,loss_fn,device,use_amp=False):
    model.eval();total_loss=0;total_tokens=0
    with torch.inference_mode():
        for src,din,target in loader:
            non_blocking=device.type=="cuda"
            src=src.to(device,non_blocking=non_blocking)
            din=din.to(device,non_blocking=non_blocking)
            target=target.to(device,non_blocking=non_blocking)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=use_amp):
                logits=model(src,din)
                n=int(target.ne(PAD).sum().item())
                loss=loss_fn(logits.reshape(-1,logits.size(-1)),target.reshape(-1))
            total_loss+=loss.item()*n;total_tokens+=n
    return total_loss/max(1,total_tokens)


def split_by_frame(data,validation_fraction=0.20):
    """Stratify by frame while keeping every identical Chinese source in one split."""
    if not 0.05 <= validation_fraction <= 0.40:
        raise ValueError("validation_fraction must be between 0.05 and 0.40")

    grouped={}
    for idx,(source,frame) in enumerate(zip(data.sources,data.frames)):
        entry=grouped.setdefault(source,{"indices":[],"frames":{}})
        entry["indices"].append(idx)
        entry["frames"][frame]=entry["frames"].get(frame,0)+1

    by_frame={}
    for source,entry in grouped.items():
        primary_frame=sorted(entry["frames"],key=lambda f:(-entry["frames"][f],f))[0]
        by_frame.setdefault(primary_frame,[]).append((source,entry["indices"]))

    rng=random.Random(20261118)
    train_ids=[];val_ids=[]
    for frame in sorted(by_frame):
        groups=list(by_frame[frame])
        rng.shuffle(groups)
        if len(groups)<2:
            val_groups=[]
        else:
            n_val=max(1,round(len(groups)*validation_fraction))
            n_val=min(n_val,len(groups)-1)
            val_groups=groups[:n_val]
        train_groups=groups[len(val_groups):]
        for _,indices in train_groups:
            train_ids.extend(indices)
        for _,indices in val_groups:
            val_ids.extend(indices)

    if not train_ids or not val_ids:
        raise ValueError("Not enough distinct Chinese source groups for grouped validation")
    train_sources={data.sources[i] for i in train_ids}
    val_sources={data.sources[i] for i in val_ids}
    overlap=train_sources.intersection(val_sources)
    if overlap:
        raise AssertionError(f"Source leakage detected between train and validation: {len(overlap)} sources")
    rng.shuffle(train_ids);rng.shuffle(val_ids)
    return train_ids,val_ids,sorted(by_frame)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=100)
    ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--lr",type=float,default=0.0005)
    ap.add_argument("--threads",type=int,default=2,choices=[1,2,3,4])
    ap.add_argument("--count-params",action="store_true")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--save-every",type=int,default=100,help="Save resumable checkpoints every N epochs (default: 100)")
    ap.add_argument("--validation-fraction",type=float,default=0.20,help="Per-frame validation fraction (default: 0.20)")
    ap.add_argument("--early-stop-patience",type=int,default=50,help="Stop after this many epochs without validation improvement (default: 50)")
    ap.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto",help="Training device (default: auto-detect CUDA/MPS, otherwise CPU)")
    ap.add_argument("--no-amp",action="store_true",help="Disable automatic mixed precision on CUDA")
    args=ap.parse_args()
    if args.save_every < 1:
        ap.error("--save-every must be at least 1")
    torch.set_num_threads(args.threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.manual_seed(23);random.seed(23)
    data=TranslationData()
    device=select_device(args.device)
    use_amp=(device.type=="cuda" and not args.no_amp)
    model=Pkanai(len(data.vocab)).to(device)
    try:
        scaler=torch.amp.GradScaler("cuda",enabled=use_amp)
    except (AttributeError,TypeError):
        scaler=torch.cuda.amp.GradScaler(enabled=use_amp)
    ckpt=ROOT/"checkpoints"/"pkanai_bpe.pt"  # best validation checkpoint used by translate.py
    latest_ckpt=ROOT/"checkpoints"/"pkanai_bpe_latest.pt"  # most recent resumable training state
    resumed_checkpoint=None
    if args.resume:
        resume_path=latest_ckpt if latest_ckpt.exists() else ckpt
        if resume_path.exists():
            saved=torch.load(resume_path,map_location="cpu",weights_only=False)
            if saved.get("vocab")!=data.vocab or saved.get("merges")!=data.merges:
                raise SystemExit("Tokenizer mismatch; train from scratch")
            model.load_state_dict(saved["model"]);resumed_checkpoint=saved
            print(f"Resumed BPE checkpoint: {resume_path.name}",flush=True)
    print(f"Examples: {len(data)} | Unique Chinese sources: {len(set(data.sources))} | BPE vocab: {len(data.vocab)} | Merges: {len(data.merges)} | Parameters: {parameter_count(model)} | Device: {device} | AMP: {use_amp} | CPU threads: {args.threads}",flush=True)
    if args.count_params:return
    train_ids,val_ids,val_frames=split_by_frame(data,args.validation_fraction)
    print(f"Source-grouped validation: {len(train_ids)} train / {len(val_ids)} validation ({len(val_ids)/len(data):.1%}) | train sources: {len({data.sources[i] for i in train_ids})} | val sources: {len({data.sources[i] for i in val_ids})} | frames represented: {len(val_frames)}",flush=True)
    pin_memory=(device.type=="cuda")
    train_loader=DataLoader(data,batch_sampler=BucketBatchSampler(train_ids,data.lengths,args.batch_size,True),collate_fn=collate,pin_memory=pin_memory)
    val_loader=DataLoader(data,batch_sampler=BucketBatchSampler(val_ids,data.lengths,args.batch_size,False),collate_fn=collate,pin_memory=pin_memory)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=5e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=0.5,patience=5,min_lr=1e-5)
    start_epoch=int(resumed_checkpoint.get("epoch",0)) if resumed_checkpoint else 0
    stale=int(resumed_checkpoint.get("stale",0)) if resumed_checkpoint else 0
    if resumed_checkpoint and "optimizer_state" in resumed_checkpoint:
        opt.load_state_dict(resumed_checkpoint["optimizer_state"])
        for group in opt.param_groups:group["lr"]=args.lr
    if resumed_checkpoint and "scheduler_state" in resumed_checkpoint:
        scheduler.load_state_dict(resumed_checkpoint["scheduler_state"])
    if resumed_checkpoint and "scaler_state" in resumed_checkpoint:
        scaler.load_state_dict(resumed_checkpoint["scaler_state"])
    # Restore random streams after model/optimizer initialization for reproducible continuation.
    if resumed_checkpoint:
        if "torch_rng_state" in resumed_checkpoint:torch.set_rng_state(resumed_checkpoint["torch_rng_state"])
        if "python_rng_state" in resumed_checkpoint:random.setstate(resumed_checkpoint["python_rng_state"])
    loss_fn=nn.CrossEntropyLoss(ignore_index=PAD,label_smoothing=0.05)
    best=float(resumed_checkpoint.get("best_val",float("inf"))) if resumed_checkpoint else float("inf")
    start=time.time();last_epoch=start_epoch
    if resumed_checkpoint:print(f"Previous best validation loss: {best:.4f} | continuing from epoch {start_epoch+1}",flush=True)
    if start_epoch>=args.epochs:
        print(f"Target already reached: epoch {start_epoch}/{args.epochs}",flush=True)
        return

    def save_training_checkpoint(epoch):
        payload={"model":model.state_dict(),"vocab":data.vocab,"merges":data.merges,
                 "params":parameter_count(model),"metadata":{"model":"pkanai","architecture":"BiGRU-BPE-attention",
                 "source_language":"zh","target_language":"th","validation":"source_grouped_stratified_by_frame"},
                 "embedding_dim":32,"hidden_dim":64,"best_val":best,"validation_frames":val_frames,
                 "epoch":epoch,"stale":stale,"optimizer_state":opt.state_dict(),
                 "torch_rng_state":torch.get_rng_state(),"python_rng_state":random.getstate(),
                 "scheduler_state":scheduler.state_dict(),"scaler_state":scaler.state_dict(),
                 "save_every":args.save_every,"target_epochs":args.epochs}
        archive=ROOT/"checkpoints"/f"pkanai_bpe_epoch_{epoch:04d}.pt"
        for destination in (archive,latest_ckpt):
            temporary=destination.with_name(destination.name+".tmp")
            payload["model"]={k:v.detach().cpu() for k,v in payload["model"].items()}
            torch.save(payload,temporary)
            temporary.replace(destination)
        print(f"Checkpoint saved at epoch {epoch}: {archive.name} + {latest_ckpt.name}",flush=True)

    # A fresh run gets an initial resumable state, so interruption before epoch 100 is recoverable.
    if not resumed_checkpoint:
        save_training_checkpoint(0)

    for epoch in range(start_epoch+1,args.epochs+1):
        model.train();total_loss=0;total_tokens=0
        for src,din,target in train_loader:
            non_blocking=device.type=="cuda"
            src=src.to(device,non_blocking=non_blocking)
            din=din.to(device,non_blocking=non_blocking)
            target=target.to(device,non_blocking=non_blocking)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=use_amp):
                logits=model(src,din)
                n=int(target.ne(PAD).sum().item())
                loss=loss_fn(logits.reshape(-1,logits.size(-1)),target.reshape(-1))
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt);scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
            total_loss+=loss.item()*n;total_tokens+=n
        tr=total_loss/max(1,total_tokens);vl=evaluate(model,val_loader,loss_fn,device,use_amp)
        scheduler.step(vl)
        if epoch==1 or epoch%5==0:
            print(f"Epoch {epoch}/{args.epochs} train={tr:.4f} val={vl:.4f} time={time.time()-start:.1f}s",flush=True)
        if vl<best-0.0005:
            best=vl;stale=0
            best_model_cpu={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save({"model":best_model_cpu,"vocab":data.vocab,"merges":data.merges,
                        "params":parameter_count(model),"metadata":{"model":"pkanai","architecture":"BiGRU-BPE-attention",
                        "source_language":"zh","target_language":"th","validation":"source_grouped_stratified_by_frame"},
                        "embedding_dim":32,"hidden_dim":64,"best_val":best,"validation_frames":val_frames,
                        "epoch":epoch},ckpt)
        else:stale+=1
        last_epoch=epoch
        if epoch%args.save_every==0:
            save_training_checkpoint(epoch)
        if stale>=args.early_stop_patience:
            print(f"Early stopping: validation loss did not improve for {args.early_stop_patience} epochs",flush=True);break
    # Also save the final epoch when training ends between save intervals or by early stopping.
    if last_epoch>start_epoch and last_epoch%args.save_every!=0:
        save_training_checkpoint(last_epoch)
    print(f"Best checkpoint: {ckpt} | best_val={best:.4f} | last_epoch={last_epoch} | total={time.time()-start:.1f}s",flush=True)

if __name__=="__main__":main()
