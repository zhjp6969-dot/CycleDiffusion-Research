from pathlib import Path
import re, json
import numpy as np
import pandas as pd
import torch
from transformers import pipeline

MODEL_ID='openai/whisper-base.en'
ROOT=Path('VCTK_2F2M')
SPEAKERS=['p236','p239','p259','p263']
ALPHAS=[0.00,0.25,0.50,0.75,1.00]
RAW=Path('all12_whisper_content_raw.csv')
BATCH=16

def outpath(src,tgt,s,r,a):
    base=f'{src}_to_{tgt}_s{s}_r{r}'
    if a==0: return Path('converted/all12/baseline')/f'{base}.wav'
    if a==1: return Path('converted/all12/centroid')/f'{base}.wav'
    return Path('converted/alpha_screen')/f'{base}_a{int(a*100)}.wav'
def norm(s): return ' '.join(re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?",str(s).lower()))
def edit(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]
def scores(ref,hyp):
    rn,hn=norm(ref),norm(hyp); rw,hw=rn.split(),hn.split(); e=edit(rw,hw); rc=rn.replace(' ','' ); hc=hn.replace(' ',''); ce=edit(list(rc),list(hc))
    return rn,hn,e,e/max(1,len(rw)),ce,ce/max(1,len(rc))

jobs=[]
for src in SPEAKERS:
    wavs=sorted((ROOT/'wavs'/src).glob('*.wav'))[:5]
    for si,wav in enumerate(wavs,1):
        txt=(ROOT/'txt'/src/(wav.stem.replace('_mic1','').replace('_mic2','')+'.txt')).read_text(errors='ignore').strip()
        jobs.append(dict(src=src,tgt='original',source_idx=si,ref_idx=0,alpha=-1.0,path=str(wav),reference=txt))
        for tgt in SPEAKERS:
            if tgt==src: continue
            for a in ALPHAS:
                for ri in range(1,4): jobs.append(dict(src=src,tgt=tgt,source_idx=si,ref_idx=ri,alpha=a,path=str(outpath(src,tgt,si,ri,a)),reference=txt))
for j in jobs:
    if not Path(j['path']).exists(): raise FileNotFoundError(j['path'])
done=set()
if RAW.exists(): done=set(pd.read_csv(RAW).path.astype(str))
pending=[j for j in jobs if j['path'] not in done]
print('jobs',len(jobs),'done',len(done),'pending',len(pending),flush=True)
dtype=torch.float16 if torch.cuda.is_available() else torch.float32
asr=pipeline('automatic-speech-recognition',model=MODEL_ID,device=0 if torch.cuda.is_available() else -1,torch_dtype=dtype)
for start in range(0,len(pending),BATCH):
    batch=pending[start:start+BATCH]
    outputs=asr([j['path'] for j in batch],batch_size=BATCH)
    rows=[]
    for j,o in zip(batch,outputs):
        hyp=o['text']; rn,hn,e,wer,ce,cer=scores(j['reference'],hyp)
        rows.append({**j,'hypothesis':hyp,'ref_norm':rn,'hyp_norm':hn,'word_edits':e,'wer':wer,'char_edits':ce,'cer':cer})
    pd.DataFrame(rows).to_csv(RAW,mode='a',header=not RAW.exists(),index=False)
    print('completed',min(start+BATCH,len(pending)),'/',len(pending),flush=True)
print('WHISPER_DONE',len(pd.read_csv(RAW)),flush=True)
