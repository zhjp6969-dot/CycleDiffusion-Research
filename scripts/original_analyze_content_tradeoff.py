from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
ALPHAS=[0.00,0.25,0.50,0.75,1.00]
raw=pd.read_csv('all12_whisper_content_raw.csv')
conv=raw[raw.alpha>=0].copy(); orig=raw[raw.alpha<0].copy()
identity=pd.read_csv('all12_disjoint_pair_alpha_stats.csv')
global_identity=pd.read_csv('all12_disjoint_global_cluster_stats.csv')
def tests(x,y):
 d=np.asarray(y)-np.asarray(x); t,p=ttest_rel(y,x)
 try: w,wp=wilcoxon(d) if not np.allclose(d,0) else (0.0,1.0)
 except ValueError: w,wp=np.nan,np.nan
 return float(t),float(p),float(w),float(wp)
def ci(d,n=50000,seed=20260907):
 d=np.asarray(d,float); rng=np.random.default_rng(seed); b=d[rng.integers(0,len(d),size=(n,len(d)))].mean(1); return [float(x) for x in np.quantile(b,[.025,.975])]
cluster=conv.groupby(['src','source_idx','alpha'],as_index=False)[['wer','cer']].mean()
base=cluster[cluster.alpha==0].sort_values(['src','source_idx'])
grows=[]
for a in ALPHAS:
 x=cluster[cluster.alpha==a].sort_values(['src','source_idx']); dw=x.wer.to_numpy()-base.wer.to_numpy(); dc=x.cer.to_numpy()-base.cer.to_numpy(); t,p,w,wp=tests(base.wer,x.wer); lo,hi=ci(dw,seed=20260907+int(a*100))
 grows.append(dict(alpha=a,n_clusters=20,mean_wer=float(x.wer.mean()),mean_cer=float(x.cer.mean()),wer_change_vs_0=float(dw.mean()),wer_ci_low=lo,wer_ci_high=hi,p_t=p,p_wilcoxon=wp,cer_change_vs_0=float(dc.mean())))
gdf=pd.DataFrame(grows); gdf.to_csv('all12_content_global_cluster_stats.csv',index=False)
pairc=conv.groupby(['src','tgt','source_idx','alpha'],as_index=False)[['wer','cer']].mean()
trade=[]
for (src,tgt),g in pairc.groupby(['src','tgt']):
 b=g[g.alpha==0].sort_values('source_idx')
 for a in ALPHAS:
  x=g[g.alpha==a].sort_values('source_idx'); dw=x.wer.to_numpy()-b.wer.to_numpy(); lo,hi=ci(dw,seed=20270000+int(a*100)); ident=identity[(identity.src==src)&(identity.tgt==tgt)&(identity.alpha==a)].iloc[0]; t,p,w,wp=tests(b.wer,x.wer)
  trade.append(dict(src=src,tgt=tgt,alpha=a,identity_delta=ident.mean_delta,identity_gain_vs_0=ident.gain_vs_0,mean_wer=float(x.wer.mean()),wer_change_vs_0=float(dw.mean()),wer_ci_low=lo,wer_ci_high=hi,p_t=p,p_wilcoxon=wp,best_identity_alpha=bool(ident.best_for_pair),win_win=bool(ident.gain_vs_0>0 and dw.mean()<=0)))
tdf=pd.DataFrame(trade); tdf.to_csv('all12_identity_content_tradeoff.csv',index=False)
best=tdf[tdf.best_identity_alpha].copy(); best.to_csv('all12_best_identity_alpha_content_effect.csv',index=False)
orig_wer=float(orig.wer.mean()); orig_cer=float(orig.cer.mean())
fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
means=gdf.mean_wer.to_numpy(); sem=cluster.groupby('alpha').wer.std(ddof=1).to_numpy()/np.sqrt(20)
axs[0].errorbar(ALPHAS,means,yerr=1.96*sem,marker='o',capsize=4,label='converted'); axs[0].axhline(orig_wer,ls='--',color='gray',label=f'original={orig_wer:.3f}'); axs[0].set(xlabel='alpha',ylabel='WER (lower is better)',title='Content preservation (20 source clusters)'); axs[0].legend(); axs[0].grid(alpha=.25)
mi=global_identity.set_index('alpha').loc[ALPHAS].mean_delta.to_numpy(); axs[1].plot(means,mi,'o-')
for a,x,y in zip(ALPHAS,means,mi): axs[1].annotate(str(a),(x,y),xytext=(4,4),textcoords='offset points')
axs[1].set(xlabel='WER',ylabel='WavLM target-source delta',title='Global identity-content tradeoff'); axs[1].grid(alpha=.25)
axs[2].axhline(0,color='gray',lw=1); axs[2].axvline(0,color='gray',lw=1); axs[2].scatter(best.wer_change_vs_0,best.identity_gain_vs_0,c=best.alpha,cmap='viridis',vmin=0,vmax=1,s=55)
for _,r in best.iterrows(): axs[2].annotate(f"{r.src[1:]}->{r.tgt[1:]}",(r.wer_change_vs_0,r.identity_gain_vs_0),fontsize=7,xytext=(3,3),textcoords='offset points')
axs[2].set(xlabel='WER change vs alpha=0 (left is better)',ylabel='Identity gain vs alpha=0',title='Best-identity alpha by direction'); axs[2].grid(alpha=.2)
fig.savefig('all12_identity_content_tradeoff.png',dpi=180); plt.close(fig)
summary={'n_transcriptions':int(len(raw)),'original_wer':orig_wer,'original_cer':orig_cer,'global_by_alpha':gdf.to_dict('records'),'best_identity_alpha_mean_wer_change':float(best.wer_change_vs_0.mean()),'best_identity_alpha_win_win_directions':int(best.win_win.sum()),'best_identity_alpha_directions':int(len(best))}
Path('all12_content_summary.json').write_text(json.dumps(summary,indent=2))
print('ORIGINAL WER/CER',orig_wer,orig_cer); print(gdf.to_string(index=False)); print('\nBEST IDENTITY ALPHA CONTENT EFFECT'); print(best[['src','tgt','alpha','identity_gain_vs_0','wer_change_vs_0','wer_ci_low','wer_ci_high','p_wilcoxon','win_win']].to_string(index=False)); print('\nWIN_WIN',best.win_win.sum(),'/',len(best)); print('FILES_READY')
