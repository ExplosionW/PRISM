"""Score generated sequences with the released three-seed PRISM predictor ensemble."""
from pathlib import Path
import sys,argparse,subprocess,json,re,importlib.util
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'predictor'));from predict import load,predict
_spec=importlib.util.spec_from_file_location('generator_selection',R/'generator/selection_rules.py');_selection=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_selection)

def main():
    p=argparse.ArgumentParser();p.add_argument('--pools',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--precision',choices=['bf16','fp32'],default='bf16');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    known=set((R/'generator/data/known_sequences.txt').read_text().splitlines())
    rows=[];files=sorted(a.pools.glob('PRISM*.csv'))
    if not files:raise ValueError('No PRISM pools found; run generator or pass a pool directory')
    for file in files:
        raw=pd.read_csv(file,keep_default_na=False);attempts=len(raw);d=raw;d=d[d.stopped_normally.astype(str).str.lower().eq('true')&d.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{10}')&~d.sequence.isin(known)].drop_duplicates('sequence');seq=d.sequence.tolist()
        if not seq:raise ValueError(f'No novel canonical peptides in {file.name}')
        work=a.output/file.stem;work.mkdir(exist_ok=True);inp=work/'peptides.csv';pd.DataFrame({'sequence':seq}).to_csv(inp,index=False);feat=work/'features.npz'
        if not feat.exists():subprocess.run([sys.executable,str(R/'predictor/extract_features.py'),'--input',str(inp),'--output',str(feat),'--device',a.device,'--precision',a.precision],check=True)
        z=np.load(feat);assert z['sequences'].tolist()==seq
        all_pred=[]
        for seed in range(3):
            net,targets=load(R/f'checkpoints/predictor/seed{seed}.pt',a.device);all_pred.append(predict(net,seq,z['esm33'],a.device));del net
        mean=np.stack(all_pred).mean(0);t=targets.index('MMP13');activity=mean[:,t];off=np.delete(mean,t,axis=1);selectivity=activity-off.mean(1);margin=activity-off.max(1);joint=(activity>1)&(margin>0);top=_selection.select(seq,mean,100,t);top24=top[:24]
        np.savez_compressed(work/'predictions.npz',sequences=seq,targets=targets,predictions=np.stack(all_pred),mean=mean)
        match=re.search(r'_seed(\d+)_sample(\d+)_T([\d.]+)_(\w+)$',file.stem)
        if not match:raise ValueError(f'Unexpected pool name: {file.stem}')
        selected=pd.DataFrame({'rank':np.arange(1,len(top)+1),'sequence':np.asarray(seq)[top],'MMP13':activity[top],'max_competitor':off.max(1)[top],'bottleneck':np.minimum(activity-1,margin)[top],'predicted_joint':joint[top]})
        selected.to_csv(work/'selected_top100.csv',index=False)
        rows.append(dict(pool=file.stem,seed=int(match[1]),replicate=int(match[2]),temperature=float(match[3]),mode=match[4],scorer='PRISM ensemble (seeds 0,1,2)',attempts=attempts,unique_novel=len(seq),MMP13_predicted_activity_mean=float(activity.mean()),mean_contrast_selectivity_mean=float(selectivity.mean()),predicted_target_leads_fraction=float((margin>0).mean()),predicted_joint_unique=int(joint.sum()),predicted_joint_per_attempt=float(joint.sum()/attempts),top24_n=len(top24),top100_n=len(top),top24_joint_fraction=float(joint[top24].mean()),top100_joint_fraction=float(joint[top].mean()),top100_selectivity_mean=float(selectivity[top].mean())))
        print(json.dumps(rows[-1]),flush=True)
    table=pd.DataFrame(rows);table.to_csv(a.output/'predicted_function.csv',index=False)
    metrics=[c for c in table.select_dtypes(include='number').columns if c not in ['seed','replicate','temperature']]
    byseed=table.groupby(['mode','temperature','seed'])[metrics].mean();byseed.to_csv(a.output/'predicted_function_by_seed.csv')
    byseed.groupby(['mode','temperature'])[metrics].agg(['mean','std']).to_csv(a.output/'predicted_function_summary.csv')
if __name__=='__main__':main()
