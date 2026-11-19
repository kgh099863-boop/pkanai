import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
PAD,BOS,EOS,UNK=0,1,2,3

class Pkanai(nn.Module):
    def __init__(self,vocab_size,embedding_dim=32,hidden_dim=64,dropout=0.30):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=PAD)
        self.encoder=nn.GRU(embedding_dim,hidden_dim,batch_first=True,bidirectional=True)
        self.bridge=nn.Linear(hidden_dim*2,hidden_dim)
        self.attn_query=nn.Linear(hidden_dim,hidden_dim*2,bias=False)
        self.decoder=nn.GRU(embedding_dim,hidden_dim,batch_first=True)
        self.dropout=nn.Dropout(dropout)
        self.output=nn.Linear(hidden_dim*3,vocab_size)

    def encode(self,source):
        mask=source.ne(PAD)
        lengths=mask.sum(dim=1).clamp(min=1).cpu()
        emb=self.dropout(self.embedding(source))
        packed=pack_padded_sequence(emb,lengths,batch_first=True,enforce_sorted=False)
        packed_out,states=self.encoder(packed)
        enc,_=pad_packed_sequence(packed_out,batch_first=True,total_length=source.size(1))
        hidden=torch.tanh(self.bridge(torch.cat([states[-2],states[-1]],dim=-1))).unsqueeze(0)
        return enc,mask,hidden

    def attend(self,dec,enc,mask):
        query=self.attn_query(dec)
        scores=torch.bmm(query,enc.transpose(1,2)).masked_fill(~mask.unsqueeze(1),-1e4)
        weights=torch.softmax(scores,dim=-1)
        context=torch.bmm(weights,enc)
        return self.output(self.dropout(torch.cat([dec,context],dim=-1)))

    def step(self,token,enc,mask,hidden):
        emb=self.dropout(self.embedding(token))
        dec,hidden=self.decoder(emb,hidden)
        return self.attend(dec,enc,mask),hidden

    def forward(self,source,decoder_input):
        enc,mask,hidden=self.encode(source)
        dec,hidden=self.decoder(self.dropout(self.embedding(decoder_input)),hidden)
        return self.attend(dec,enc,mask)

def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
