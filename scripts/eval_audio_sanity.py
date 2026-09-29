from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import ttest_rel,wilcoxon
raw=pd.read_csv('all12_whisper_content_raw.csv'); meta=raw[['src','tgt','source_idx','ref_idx','alpha','path']].drop_duplicates('path').copy()
def metrics(row):
 x,sr=sf.read(row.path,dtype='float32'); x=x.mean(1) if x.ndim>1 else x; n=len(x); rms=float(np.sqrt(np.mean(x*x)+1e-12)); peak=float(np.max(np.abs(x))) if n else 0.; fl=max(1,int(.025*sr)); hop=max(1,int(.010*sr)); nf=max(1,1+(max(n,fl)-fl)//hop); xx=np.pad(x,(0,max(0,fl-n))); frames=np.array([xx[i*hop:i*hop+fl] for i in range(nf)]); frms=np.sqrt(np.mean(frames*frames,axis=1)+1e-12)
 return dict(path=row.path,duration_s=n/sr,rms_dbfs=20*np.log10(rms+1e-12),peak_dbfs=20*np.log10(peak+1e-12),clip_fraction=float(np.mean(np.abs(x)>=.999)) if n else 0.,silence_fraction=float(np.mean(frms<10**(-50/20))),dc_offset=float(np.mean(x)) if n else 0.)
with ThreadPoolExecutor(max_workers=12) as ex: vals=list(ex.map(metrics,[r for _,r in meta.iterrows()]))
m=meta.merge(pd.DataFrame(vals),on='path'); srcdur=m[m.alpha<0][['src','source_idx','duration_s']].rename(columns={'duration_s':'source_duration_s'}); m=m.merge(srcdur,on=['src','source_idx']); m['duration_ratio']=m.duration_s/m.source_duration_s; m.to_csv('all12_audio_sanity_raw.csv',index=False)
conv=m[m.alpha>=0].copy(); cl=conv.groupby(['src','source_idx','alpha'],as_index=False)[['duration_ratio','rms_dbfs','peak_dbfs','clip_fraction','silence_fraction']].mean(); base=cl[cl.alpha==0].sort_values(['src','source_idx'])
def ci(d,n=50000,seed=20260907):
 d=np.asarray(d); rng=np.random.default_rng(seed); b=d[rng.integers(0,len(d),size=(n,len(d)))].mean(1); return [float(x) for x in np.quantile(b,[.025,.975])]
rows=[]
for a,g in conv.groupby('alpha'):
 x=cl[cl.alpha==a].sort_values(['src','source_idx']); dd=x.duration_ratio.to_numpy()-base.duration_ratio.to_numpy(); rd=x.rms_dbfs.to_numpy()-base.rms_dbfs.to_numpy(); dlo,dhi=ci(dd,seed=20260907+int(a*100)); rlo,rhi=ci(rd,seed=20270000+int(a*100))
 rows.append(dict(alpha=a,n=len(g),duration_ratio=g.duration_ratio.mean(),duration_change=dd.mean(),duration_ci_low=dlo,duration_ci_high=dhi,rms_dbfs=g.rms_dbfs.mean(),rms_change_db=rd.mean(),rms_ci_low=rlo,rms_ci_high=rhi,peak_dbfs=g.peak_dbfs.mean(),clip_fraction=g.clip_fraction.mean(),files_with_clipping=int((g.clip_fraction>0).sum()),silence_fraction=g.silence_fraction.mean(),very_short=int((g.duration_ratio<.6).sum()),very_long=int((g.duration_ratio>1.6).sum())))
s=pd.DataFrame(rows); s.to_csv('all12_audio_sanity_summary.csv',index=False); outliers=conv[(conv.duration_ratio<.6)|(conv.duration_ratio>1.6)|(conv.clip_fraction>.001)|(conv.silence_fraction>.8)].sort_values(['alpha','src','tgt']); outliers.to_csv('all12_audio_sanity_outliers.csv',index=False); Path('all12_audio_sanity_summary.json').write_text(json.dumps({'global':s.to_dict('records'),'outliers':len(outliers)},indent=2)); print(s.round(5).to_string(index=False)); print('OUTLIERS',len(outliers)); print(outliers[['src','tgt','source_idx','ref_idx','alpha','duration_ratio','rms_dbfs','clip_fraction','silence_fraction','path']].head(30).to_string(index=False)); print('FILES_READY')
