from collections import Counter
PAD,BOS,EOS,UNK=0,1,2,3

def learn_bpe(texts, num_merges=250, min_frequency=2):
    sequences=[list(text) for text in texts if text]
    merges=[]
    for _ in range(num_merges):
        counts=Counter()
        for seq in sequences:
            counts.update(zip(seq,seq[1:]))
        if not counts:
            break
        pair,freq=counts.most_common(1)[0]
        if freq<min_frequency:
            break
        left,right=pair
        merged=left+right
        updated=[]
        for seq in sequences:
            out=[];i=0
            while i<len(seq):
                if i+1<len(seq) and seq[i]==left and seq[i+1]==right:
                    out.append(merged);i+=2
                else:
                    out.append(seq[i]);i+=1
            updated.append(out)
        sequences=updated
        merges.append((left,right))
    vocab=set()
    for seq in sequences:vocab.update(seq)
    for text in texts:vocab.update(text)
    ordered=sorted(vocab)
    mapping={"<PAD>":PAD,"<BOS>":BOS,"<EOS>":EOS,"<UNK>":UNK}
    for token in ordered:
        if token not in mapping:mapping[token]=len(mapping)
    return merges,mapping

def apply_bpe(text,merges):
    tokens=list(text)
    for left,right in merges:
        if len(tokens)<2:break
        out=[];i=0
        while i<len(tokens):
            if i+1<len(tokens) and tokens[i]==left and tokens[i+1]==right:
                out.append(left+right);i+=2
            else:
                out.append(tokens[i]);i+=1
        tokens=out
    return tokens

def encode_text(text,merges,vocab):
    return [vocab.get(t,UNK) for t in apply_bpe(text,merges)]

def decode_ids(ids,inv_vocab):
    return "".join(inv_vocab.get(int(i),"") for i in ids if int(i)>=4)
