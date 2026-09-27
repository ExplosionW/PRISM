"""Reproduce PRISM selection with the existing Pareto-Max implementation."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'predictor/vendor'))
import selection as sel

def read_prediction(folder,split,seed,sequences,targets):
    z=np.load(folder/f'{split}_seed{seed}.npz')
    if not np.array_equal(z['sequences'],sequences):raise ValueError(f'{split}: sequence order mismatch')
    if z['targets'].tolist()!=targets:raise ValueError(f'{split}: target order mismatch')
    p=z['predictions']
    if p.shape!=(len(sequences),len(targets)) or not np.isfinite(p).all():raise ValueError('Invalid prediction matrix')
    return p.T

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);ap.add_argument('--check-reference',action='store_true');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data={s:dict(np.load(ROOT/f'data/benchmark/{s}.npz')) for s in ['train','val','test']}
    targets=json.loads((ROOT/'data/benchmark/target_order.json').read_text());ood=dict(np.load(ROOT/'data/ood/ood.npz'))
    yt=data['test']['labels'].T;yv=data['val']['labels'].T;ye=ood['labels'].T
    ta,td=sel.thresholds(data['train']['labels'].T);ea,ed=sel.thresholds(ye)
    rows=[];members=[];regression=[];threshold_rows=[]
    for cohort,ts,aa,dd in [('benchmark',targets,ta,td),('ood',ood['targets'].tolist(),ea,ed)]:
        threshold_rows.extend(dict(dataset=cohort,target=t,activity_threshold=aa[j],margin_threshold=dd[j]) for j,t in enumerate(ts))
    for seed in a.seeds:
        pv=read_prediction(a.predictions,'val',seed,data['val']['sequences'],targets)
        pt=read_prediction(a.predictions,'test',seed,data['test']['sequences'],targets)
        pe=read_prediction(a.predictions,'ood',seed,ood['sequences'],targets)
        for cohort,protocol,truth,ts,K,aa,dd,pred,seqs in [('benchmark','count',yt,targets,100,ta,td,pt,data['test']['sequences']),('ood','ood_k5',ye,ood['targets'].tolist(),5,ea,ed,pe,ood['sequences'])]:
            margin=sel.margins(truth);hit=(truth>=aa[:,None])&(margin>=dd[:,None])
            for t,target in enumerate(ts):
                pi=targets.index(target);order=np.argsort(-pred[pi],kind='stable')
                if cohort=='benchmark':
                    vo=np.argsort(-pv[pi],kind='stable');vr=np.delete(pv,pi,0).max(0);ref=yv[pi,vo[:K]].mean();feasible=[]
                    if ref<=0:raise ValueError('Validation activity reference must be positive')
                    for frac in sel.FRACS:
                        n=K if frac==0 else max(K,int(np.ceil(frac*len(data['val']['sequences']))))
                        vi=sel._pareto(pv[pi],vr,vo[:n],K)
                        if yv[pi,vi].mean()/ref>=.8:feasible.append(n)
                    n=max(feasible)
                else:n=len(seqs)
                # Predictions compare all 18 enzymes; OOD labels cover the measured 12.
                chosen=sel._pareto(pred[pi],np.delete(pred,pi,0).max(0),order[:n],K)
                rows.append(dict(seed=seed,dataset=cohort,protocol=protocol,target=target,precision=float(hit[t,chosen].mean()),SR=float((margin[t,chosen]<=0).mean()),candidates=n,selected=K,hits=int(hit[t,chosen].sum())))
                members.extend(dict(seed=seed,dataset=cohort,target=target,rank=rank+1,sequence=str(seqs[i]),observed_activity=float(truth[t,i]),positive=bool(hit[t,i]),shortcut=bool(margin[t,i]<=0)) for rank,i in enumerate(chosen))
                regression.append(dict(seed=seed,dataset=cohort,target=target,Pearson=float(pearsonr(truth[t],pred[pi]).statistic),Spearman=float(spearmanr(truth[t],pred[pi]).statistic),MAE=float(np.abs(truth[t]-pred[pi]).mean()) if cohort=='benchmark' else np.nan))
    df=pd.DataFrame(rows);by_seed=df.groupby(['seed','dataset','protocol'])[['precision','SR']].mean().reset_index()
    summary=by_seed.groupby('dataset')[['precision','SR']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    df.to_csv(a.output/'by_target.csv',index=False);by_seed.to_csv(a.output/'by_seed.csv',index=False);summary.to_csv(a.output/'summary.csv')
    pd.DataFrame(members).to_csv(a.output/'selected_peptides.csv',index=False);pd.DataFrame(regression).to_csv(a.output/'regression_by_target.csv',index=False);pd.DataFrame(threshold_rows).to_csv(a.output/'thresholds.csv',index=False)
    if a.check_reference:
        expected=pd.read_csv(ROOT/'reference_results/PRISM_by_seed.csv');joint=by_seed.merge(expected,on=['seed','protocol'],suffixes=('_new','_expected'),validate='one_to_one');assert len(joint)==2*len(a.seeds)
        for metric in ['precision','SR']:np.testing.assert_allclose(joint[metric+'_new'],joint[metric+'_expected'],rtol=0,atol=1e-10)
        print('Published selection values reproduced for all requested seeds.')
    print(summary.to_string());print('Metrics are fractions; multiply by 100 for percentages.')
if __name__=='__main__':main()
