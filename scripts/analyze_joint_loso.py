from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp,wilcoxon
SPEAKERS=['p236','p239','p259','p263']; ALPHAS=[0.,.25,.5,.75,1.]; MARGINS=[0.,.02,.05,999.]
z=np.load('all12_disjoint_wavlm_raw.npz'); raw=pd.read_csv('all12_whisper_content_raw.csv'); c=raw[raw.alpha>=0].copy(); c['wer_cap']=c.wer.clip(upper=1); c['cer_cap']=c.cer.clip(upper=1); c['robust_error']=c[['wer_cap','cer_cap']].max(axis=1); pc=c.groupby(['src','tgt','source_idx','alpha'],as_index=False).robust_error.mean()
def ida(s,t,a): return z[f'{s}_to_{t}_a{int(a*100):03d}'].mean(axis=1)
def ca(s,t,a): return pc[(pc.src==s)&(pc.tgt==t)&(pc.alpha==a)].sort_values('source_idx').robust_error.to_numpy()
def ci(x,n=50000,seed=20260907):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); b=x[rng.integers(0,len(x),size=(n,len(x)))].mean(1); return [float(v) for v in np.quantile(b,[.025,.975])]
rows=[]
for margin in MARGINS:
 for s in SPEAKERS:
  for t in SPEAKERS:
   if s==t: continue
   ids={a:ida(s,t,a) for a in ALPHAS}; cs={a:ca(s,t,a) for a in ALPHAS}
   for held in range(5):
    tr=[i for i in range(5) if i!=held]; base_c=float(cs[0][tr].mean()); feasible=[a for a in ALPHAS if float(cs[a][tr].mean())<=base_c+margin+1e-12]; selected=max(feasible,key=lambda a:float(ids[a][tr].mean()))
    rows.append(dict(margin=margin,src=s,tgt=t,source_idx=held+1,selected_alpha=selected,identity_base=float(ids[0][held]),identity_adapt=float(ids[selected][held]),identity_gain=float(ids[selected][held]-ids[0][held]),content_base=float(cs[0][held]),content_adapt=float(cs[selected][held]),content_change=float(cs[selected][held]-cs[0][held]),n_feasible=len(feasible)))
df=pd.DataFrame(rows); df.to_csv('all12_joint_loso_raw.csv',index=False)
sums=[]
for margin,g in df.groupby('margin'):
 cl=g.groupby(['src','source_idx'],as_index=False)[['identity_gain','content_change']].mean(); ilo,ihi=ci(cl.identity_gain,seed=20260907+int(min(margin,1)*100)); clo,chi=ci(cl.content_change,seed=20261007+int(min(margin,1)*100)); ip=float(ttest_1samp(cl.identity_gain,0).pvalue); cp=float(ttest_1samp(cl.content_change,0).pvalue)
 try: iwp=float(wilcoxon(cl.identity_gain).pvalue)
 except: iwp=np.nan
 try: cwp=float(wilcoxon(cl.content_change).pvalue)
 except: cwp=np.nan
 counts=g.selected_alpha.value_counts().reindex(ALPHAS,fill_value=0)
 sums.append(dict(margin=margin,n_clusters=20,identity_gain=cl.identity_gain.mean(),identity_ci_low=ilo,identity_ci_high=ihi,identity_p_t=ip,identity_p_wilcoxon=iwp,content_change=cl.content_change.mean(),content_ci_low=clo,content_ci_high=chi,content_p_t=cp,content_p_wilcoxon=cwp,alpha0=int(counts[0]),alpha025=int(counts[.25]),alpha05=int(counts[.5]),alpha075=int(counts[.75]),alpha1=int(counts[1.])))
sdf=pd.DataFrame(sums); sdf.to_csv('all12_joint_loso_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(7,5))
labels=[]
for _,r in sdf.iterrows():
 lab='identity-only' if r.margin>1 else f'margin={r.margin:.2f}'; labels.append(lab); ax.errorbar(r.content_change,r.identity_gain,xerr=[[r.content_change-r.content_ci_low],[r.content_ci_high-r.content_change]],yerr=[[r.identity_gain-r.identity_ci_low],[r.identity_ci_high-r.identity_gain]],fmt='o',capsize=4,label=lab)
ax.axhline(0,color='gray',lw=1); ax.axvline(0,color='gray',lw=1); ax.set(xlabel='Held-out robust content-error change (left is better)',ylabel='Held-out WavLM identity gain',title='Joint LOSO alpha selection (20 source clusters)'); ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig('all12_joint_loso.png',dpi=180); plt.close(fig)
Path('all12_joint_loso_summary.json').write_text(json.dumps(sdf.to_dict('records'),indent=2)); print(sdf.round(5).to_string(index=False)); print('FILES_READY')
