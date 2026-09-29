from pathlib import Path
import json,re
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel,wilcoxon
raw=pd.read_csv('all12_whisper_content_raw.csv'); c=raw[raw.alpha>=0].copy(); orig=raw[raw.alpha<0].copy(); identity=pd.read_csv('all12_disjoint_pair_alpha_stats.csv')
for d in [c,orig]:
 d['ref_words']=d.ref_norm.fillna('').map(lambda x:max(1,len(str(x).split())))
 d['ref_chars']=d.ref_norm.fillna('').str.replace(' ','',regex=False).str.len().clip(lower=1)
 d['hyp_chars']=d.hyp_norm.fillna('').str.replace(' ','',regex=False).str.len()
 d['wer_cap']=d.wer.clip(upper=1); d['cer_cap']=d.cer.clip(upper=1); d['robust_error']=d[['wer_cap','cer_cap']].max(axis=1)
 d['degenerate']=(d.hyp_chars>3*d.ref_chars)|d.hyp_norm.fillna('').str.replace(' ','',regex=False).str.contains(r'(.)\1{19,}',regex=True)
def ci(x,n=50000,seed=20260907):
 x=np.asarray(x,float); rng=np.random.default_rng(seed); b=x[rng.integers(0,len(x),size=(n,len(x)))].mean(1); return [float(v) for v in np.quantile(b,[.025,.975])]
def pv(x,y):
 d=np.asarray(y)-np.asarray(x); tp=float(ttest_rel(y,x).pvalue)
 try: wp=float(wilcoxon(d).pvalue) if not np.allclose(d,0) else 1.0
 except ValueError: wp=np.nan
 return tp,wp
rows=[]; cl=c.groupby(['src','source_idx','alpha'],as_index=False).agg(robust_error=('robust_error','mean'),wer_cap=('wer_cap','mean'),cer_cap=('cer_cap','mean')); bcl=cl[cl.alpha==0].sort_values(['src','source_idx'])
for a,g in c.groupby('alpha'):
 xcl=cl[cl.alpha==a].sort_values(['src','source_idx']); diff=xcl.robust_error.to_numpy()-bcl.robust_error.to_numpy(); lo,hi=ci(diff,seed=20260907+int(a*100)); tp,wp=pv(bcl.robust_error,xcl.robust_error)
 rows.append(dict(alpha=a,n=len(g),macro_wer=g.wer.mean(),median_wer=g.wer.median(),micro_wer=g.word_edits.sum()/g.ref_words.sum(),macro_cer=g.cer.mean(),micro_cer=g.char_edits.sum()/g.ref_chars.sum(),mean_wer_capped=g.wer_cap.mean(),mean_cer_capped=g.cer_cap.mean(),mean_robust_error=g.robust_error.mean(),degenerate_n=int(g.degenerate.sum()),degenerate_rate=g.degenerate.mean(),cluster_robust_change_vs_0=diff.mean(),cluster_ci_low=lo,cluster_ci_high=hi,p_t=tp,p_wilcoxon=wp))
gdf=pd.DataFrame(rows); gdf.to_csv('all12_content_robust_global.csv',index=False)
pc=c.groupby(['src','tgt','source_idx','alpha'],as_index=False)[['robust_error','wer_cap','cer_cap']].mean(); trade=[]
for (src,tgt),g in pc.groupby(['src','tgt']):
 b=g[g.alpha==0].sort_values('source_idx'); best=float(identity[(identity.src==src)&(identity.tgt==tgt)&(identity.best_for_pair==True)].alpha.iloc[0])
 for a in sorted(g.alpha.unique()):
  x=g[g.alpha==a].sort_values('source_idx'); diff=x.robust_error.to_numpy()-b.robust_error.to_numpy(); lo,hi=ci(diff,seed=20270000+int(a*100)); ident=identity[(identity.src==src)&(identity.tgt==tgt)&(identity.alpha==a)].iloc[0]
  trade.append(dict(src=src,tgt=tgt,alpha=a,identity_gain_vs_0=ident.gain_vs_0,robust_error=float(x.robust_error.mean()),robust_change_vs_0=float(diff.mean()),robust_ci_low=lo,robust_ci_high=hi,best_identity_alpha=a==best,content_noninferior=hi<=0.02,win_win=ident.gain_vs_0>0 and diff.mean()<=0))
tdf=pd.DataFrame(trade); tdf.to_csv('all12_identity_content_robust_tradeoff.csv',index=False); best=tdf[tdf.best_identity_alpha]; best.to_csv('all12_best_alpha_robust_content.csv',index=False)
summary={'original_macro_wer':float(orig.wer.mean()),'original_robust_error':float(orig.robust_error.mean()),'converted_by_alpha':gdf.to_dict('records'),'best_identity_win_win':int(best.win_win.sum()),'best_identity_content_noninferior_margin_0.02':int(best.content_noninferior.sum())}
Path('all12_content_robust_summary.json').write_text(json.dumps(summary,indent=2)); print(gdf.round(4).to_string(index=False)); print('\nBEST IDENTITY ALPHA'); print(best[['src','tgt','alpha','identity_gain_vs_0','robust_change_vs_0','robust_ci_low','robust_ci_high','content_noninferior','win_win']].round(4).to_string(index=False)); print('\nORIGINAL robust',orig.robust_error.mean(),'degenerate',orig.degenerate.sum()); print('FILES_READY')
