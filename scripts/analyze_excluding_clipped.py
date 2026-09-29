from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp,wilcoxon
SPEAKERS=['p236','p239','p259','p263']; ALPHAS=[0.,.25,.5,.75,1.]
z=np.load('all12_disjoint_wavlm_raw.npz'); sanity=pd.read_csv('all12_audio_sanity_raw.csv'); raw=pd.read_csv('all12_whisper_content_raw.csv'); convs=sanity[sanity.alpha>=0]
bad=convs[convs.clip_fraction>.01][['src','tgt','source_idx','ref_idx']].drop_duplicates(); badkeys=set(map(tuple,bad.to_records(index=False))); print('BAD_PAIRED_KEYS',sorted(badkeys))
def ida(s,t,a):
 x=z[f'{s}_to_{t}_a{int(a*100):03d}'].copy();
 for bs,bt,si,ri in badkeys:
  if bs==s and bt==t: x[int(si)-1,int(ri)-1]=np.nan
 return np.nanmean(x,axis=1)
c=raw[raw.alpha>=0].copy(); c=c[~c.apply(lambda r:(r.src,r.tgt,r.source_idx,r.ref_idx) in badkeys,axis=1)].copy(); c['wer_cap']=c.wer.clip(upper=1); c['cer_cap']=c.cer.clip(upper=1); c['robust_error']=c[['wer_cap','cer_cap']].max(axis=1); pc=c.groupby(['src','tgt','source_idx','alpha'],as_index=False).robust_error.mean()
def ca(s,t,a): return pc[(pc.src==s)&(pc.tgt==t)&(pc.alpha==a)].sort_values('source_idx').robust_error.to_numpy()
def ci(x,n=50000,seed=20260907):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); b=x[rng.integers(0,len(x),size=(n,len(x)))].mean(1); return [float(v) for v in np.quantile(b,[.025,.975])]
# fixed alpha identity, clustered by 20 source utterances
igr=[]
for s in SPEAKERS:
 for si in range(5):
  for a in ALPHAS: igr.append(dict(src=s,source_idx=si+1,alpha=a,value=np.mean([ida(s,t,a)[si] for t in SPEAKERS if t!=s])))
ig=pd.DataFrame(igr); ib=ig[ig.alpha==0].sort_values(['src','source_idx']).value.to_numpy(); fix=[]
for a in ALPHAS:
 x=ig[ig.alpha==a].sort_values(['src','source_idx']).value.to_numpy(); d=x-ib; lo,hi=ci(d,seed=20260907+int(a*100)); fix.append(dict(alpha=a,mean=x.mean(),gain=d.mean(),ci_low=lo,ci_high=hi))
fix=pd.DataFrame(fix)
# direction best and identity-only LOSO with held-out content
best={}; cv=[]
for s in SPEAKERS:
 for t in SPEAKERS:
  if s==t: continue
  ids={a:ida(s,t,a) for a in ALPHAS}; cs={a:ca(s,t,a) for a in ALPHAS}; best[(s,t)]=max(ALPHAS,key=lambda a:ids[a].mean())
  for h in range(5):
   tr=[i for i in range(5) if i!=h]; sel=max(ALPHAS,key=lambda a:ids[a][tr].mean()); cv.append(dict(src=s,tgt=t,source_idx=h+1,selected_alpha=sel,identity_gain=ids[sel][h]-ids[0][h],content_change=cs[sel][h]-cs[0][h]))
cv=pd.DataFrame(cv); cv.to_csv('all12_sensitivity_no_clipped_loso.csv',index=False); cl=cv.groupby(['src','source_idx'],as_index=False)[['identity_gain','content_change']].mean(); ilo,ihi=ci(cl.identity_gain); clo,chi=ci(cl.content_change,seed=20261007)
# content fixed alpha after paired exclusion
cc=c.groupby(['src','source_idx','alpha'],as_index=False).robust_error.mean(); cb=cc[cc.alpha==0].sort_values(['src','source_idx']).robust_error.to_numpy(); cr=[]
for a in ALPHAS:
 x=cc[cc.alpha==a].sort_values(['src','source_idx']).robust_error.to_numpy(); d=x-cb; lo,hi=ci(d,seed=20270000+int(a*100)); cr.append(dict(alpha=a,mean=x.mean(),change=d.mean(),ci_low=lo,ci_high=hi))
cont=pd.DataFrame(cr); summary={'excluded_paired_keys':[list(x) for x in sorted(badkeys)],'excluded_output_files':len(badkeys)*len(ALPHAS),'fixed_identity':fix.to_dict('records'),'best_alpha_by_direction':{f'{s}->{t}':a for (s,t),a in best.items()},'identity_only_loso':{'identity_gain':cl.identity_gain.mean(),'identity_ci':[ilo,ihi],'identity_p_t':float(ttest_1samp(cl.identity_gain,0).pvalue),'content_change':cl.content_change.mean(),'content_ci':[clo,chi],'content_p_t':float(ttest_1samp(cl.content_change,0).pvalue)},'fixed_content':cont.to_dict('records')}
Path('all12_sensitivity_no_clipped.json').write_text(json.dumps(summary,indent=2,default=lambda o:o.item() if hasattr(o,'item') else str(o))); fix.to_csv('all12_sensitivity_no_clipped_identity.csv',index=False); cont.to_csv('all12_sensitivity_no_clipped_content.csv',index=False); print('FIXED IDENTITY\n',fix.round(5).to_string(index=False)); print('\nBEST',summary['best_alpha_by_direction']); print('\nLOSO',summary['identity_only_loso']); print('\nCONTENT\n',cont.round(5).to_string(index=False)); print('FILES_READY')
