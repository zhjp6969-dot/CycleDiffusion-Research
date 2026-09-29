from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

ALPHAS=[0.00,0.25,0.50,0.75,1.00]
SPEAKERS=['p236','p239','p259','p263']
z=np.load('all12_disjoint_wavlm_raw.npz')
def arr(src,tgt,a): return z[f'{src}_to_{tgt}_a{int(a*100):03d}']
def tests(x,y):
    d=np.asarray(y)-np.asarray(x)
    t,p=ttest_rel(y,x)
    if np.allclose(d,0): w,wp=0.0,1.0
    else:
        try: w,wp=wilcoxon(d)
        except ValueError: w,wp=np.nan,np.nan
    return float(t),float(p),float(w),float(wp)
def boot_ci(d,n=50000,seed=20260907):
    d=np.asarray(d,float); rng=np.random.default_rng(seed)
    b=d[rng.integers(0,len(d),size=(n,len(d)))].mean(axis=1)
    return [float(x) for x in np.quantile(b,[0.025,0.975])]

samples=[]
for src in SPEAKERS:
  for tgt in SPEAKERS:
    if src==tgt: continue
    for a in ALPHAS:
      x=arr(src,tgt,a)
      for si in range(5):
        for ri in range(3): samples.append(dict(src=src,tgt=tgt,source_idx=si+1,ref_idx=ri+1,alpha=a,delta=float(x[si,ri])))
pd.DataFrame(samples).to_csv('all12_disjoint_samples.csv',index=False)

pair_rows=[]; best_map={}; gain_map={}
for src in SPEAKERS:
  for tgt in SPEAKERS:
    if src==tgt: continue
    base=arr(src,tgt,0).mean(axis=1)
    means={a:float(arr(src,tgt,a).mean()) for a in ALPHAS}
    best=max(means,key=means.get); best_map[(src,tgt)]=best; gain_map[(src,tgt)]=means[best]-means[0]
    for a in ALPHAS:
      vals=arr(src,tgt,a).mean(axis=1); d=vals-base; t,p,w,wp=tests(base,vals)
      lo,hi=boot_ci(d,seed=20260907+int(a*100)+SPEAKERS.index(src)*1000+SPEAKERS.index(tgt)*100)
      pair_rows.append(dict(src=src,tgt=tgt,alpha=a,mean_delta=float(vals.mean()),sd_source=float(vals.std(ddof=1)),gain_vs_0=float(d.mean()),gain_ci_low=lo,gain_ci_high=hi,t=t,p_t=p,wilcoxon_w=w,p_wilcoxon=wp,best_for_pair=a==best))
pair_df=pd.DataFrame(pair_rows); pair_df.to_csv('all12_disjoint_pair_alpha_stats.csv',index=False)

unit_rows=[]
for src in SPEAKERS:
  tgts=[x for x in SPEAKERS if x!=src]
  for si in range(5):
    for a in ALPHAS:
      vals=np.concatenate([arr(src,t,a)[si] for t in tgts])
      unit_rows.append(dict(src=src,source_idx=si+1,alpha=a,delta=float(vals.mean())))
units=pd.DataFrame(unit_rows); base=units[units.alpha==0].sort_values(['src','source_idx']).delta.to_numpy()
global_rows=[]
for a in ALPHAS:
  vals=units[units.alpha==a].sort_values(['src','source_idx']).delta.to_numpy(); d=vals-base; t,p,w,wp=tests(base,vals); lo,hi=boot_ci(d,seed=20270000+int(a*100))
  global_rows.append(dict(alpha=a,n_clusters=len(vals),mean_delta=float(vals.mean()),gain_vs_0=float(d.mean()),gain_ci_low=lo,gain_ci_high=hi,t=t,p_t=p,wilcoxon_w=w,p_wilcoxon=wp))
global_df=pd.DataFrame(global_rows); global_df.to_csv('all12_disjoint_global_cluster_stats.csv',index=False)

cv=[]
for src in SPEAKERS:
  for tgt in SPEAKERS:
    if src==tgt: continue
    srcmeans={a:arr(src,tgt,a).mean(axis=1) for a in ALPHAS}
    for held in range(5):
      train=[i for i in range(5) if i!=held]
      selected=max(ALPHAS,key=lambda a:float(srcmeans[a][train].mean()))
      b=float(srcmeans[0][held]); ad=float(srcmeans[selected][held])
      cv.append(dict(src=src,tgt=tgt,source_idx=held+1,selected_alpha=selected,baseline=b,adaptive=ad,gain=ad-b))
cvdf=pd.DataFrame(cv); cvdf.to_csv('all12_disjoint_loso_cv.csv',index=False)
cluster=cvdf.groupby(['src','source_idx'],as_index=False)[['baseline','adaptive','gain']].mean()
t,p,w,wp=tests(cluster.baseline,cluster.adaptive); lo,hi=boot_ci(cluster.gain)
cv_summary=dict(n_clusters=len(cluster),baseline=float(cluster.baseline.mean()),adaptive=float(cluster.adaptive.mean()),gain=float(cluster.gain.mean()),gain_ci_low=lo,gain_ci_high=hi,t=t,p_t=p,wilcoxon_w=w,p_wilcoxon=wp,positive_clusters=int((cluster.gain>0).sum()))

M=np.full((4,4),np.nan); G=np.full((4,4),np.nan)
for i,s in enumerate(SPEAKERS):
  for j,tg in enumerate(SPEAKERS):
    if s!=tg: M[i,j]=best_map[(s,tg)]; G[i,j]=gain_map[(s,tg)]
fig,axs=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
for ax,data,title,cmap,vmin,vmax,fmt in [(axs[0],M,'Best alpha by direction','viridis',0,1,'.2f'),(axs[1],G,'Oracle gain vs alpha=0','RdYlGn',-np.nanmax(abs(G)),np.nanmax(abs(G)),'.3f')]:
  im=ax.imshow(data,cmap=cmap,vmin=vmin,vmax=vmax); ax.set_xticks(range(4),SPEAKERS); ax.set_yticks(range(4),SPEAKERS); ax.set_xlabel('Target'); ax.set_ylabel('Source'); ax.set_title(title)
  for i in range(4):
    for j in range(4):
      if np.isfinite(data[i,j]): ax.text(j,i,format(data[i,j],fmt),ha='center',va='center',color='white' if abs(data[i,j])>0.45 else 'black',fontsize=9)
  fig.colorbar(im,ax=ax,shrink=.8)
means=global_df.mean_delta.to_numpy(); sem=units.groupby('alpha').delta.std(ddof=1).to_numpy()/np.sqrt(20)
axs[2].errorbar(ALPHAS,means,yerr=1.96*sem,marker='o',capsize=4); axs[2].axhline(means[0],ls='--',color='gray'); axs[2].set(xlabel='Fixed alpha',ylabel='Mean target-source cosine delta',title='Global fixed-alpha curve (20 clusters)'); axs[2].grid(alpha=.25)
fig.savefig('all12_disjoint_summary.png',dpi=180); plt.close(fig)
summary={'best_global_alpha':float(global_df.loc[global_df.mean_delta.idxmax(),'alpha']),'direction_oracle_gain':float(np.mean(list(gain_map.values()))),'loso_clustered':cv_summary,'best_alpha_by_direction':{f'{s}->{t}':float(a) for (s,t),a in best_map.items()}}
Path('all12_disjoint_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('FILES_READY')
