import sys,time,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.utils.data import DataLoader
from train import TranslationData, collate, ROOT
from model import Pkanai, PAD, parameter_count
threads=int(sys.argv[1]); batch=int(sys.argv[2])
torch.set_num_threads(threads)
try: torch.set_num_interop_threads(1)
except RuntimeError: pass
torch.manual_seed(17); random.seed(17)
data=TranslationData()
model=Pkanai(len(data.vocab))
loader=DataLoader(data,batch_size=batch,shuffle=True,collate_fn=collate,num_workers=0)
opt=torch.optim.AdamW(model.parameters(),lr=0.004,weight_decay=1e-5)
loss_fn=nn.CrossEntropyLoss(ignore_index=PAD)
start=time.perf_counter()
for epoch in range(3):
    model.train(); total=0
    for src,din,target in loader:
        opt.zero_grad(set_to_none=True)
        logits=model(src,din)
        loss=loss_fn(logits.reshape(-1,logits.size(-1)),target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step(); total+=loss.item()
print(f"threads={threads} batch={batch} epoch_sec={(time.perf_counter()-start)/3:.3f} loss={total/max(1,len(loader)):.4f} params={parameter_count(model)}",flush=True)
