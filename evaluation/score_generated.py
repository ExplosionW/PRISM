"""Score generated sequences with the released three-seed PRISM predictor ensemble."""
from pathlib import Path
import sys,argparse,subprocess,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'predictor'));from predict import load,predict

def main():
    p=argparse.ArgumentParser();p.add_argument('--pools',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--precision',choices=['bf16','fp32'],default='bf16');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    known=set()
    for split in ['train','val','test']:known.update(np.load(R/f'data/benchmark/{split}.npz')['sequences'].tolist())
    rows=[];files=sorted(a.pools.glob('PRISM*.csv'))
    if not files:raise ValueError('No PRISM pools found; run generator or pass a pool directory')
    for file in files:
        d=pd.read_csv(file,keep_default_na=False);d=d[d.stopped_normally.astype(str).str.lower().eq('true')&d.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{10}')&~d.sequence.isin(known)].drop_duplicates('sequence');seq=d.sequence.tolist()
        if not seq:raise ValueError(f'No novel canonical peptides in {file.name}')
        work=a.output/file.stem;work.mkdir(exist_ok=True);inp=work/'peptides.csv';pd.DataFrame({'sequence':seq}).to_csv(inp,index=False);feat=work/'features.npz'
        if not feat.exists():subprocess.run([sys.executable,str(R/'predictor/extract_features.py'),'--input',str(inp),'--output',str(feat),'--device',a.device,'--precision',a.precision],check=True)
        z=np.load(feat);assert z['sequences'].tolist()==seq
        all_pred=[]
        for seed in range(3):
            net,targets=load(R/f'checkpoints/predictor/seed{seed}.pt',a.device);all_pred.append(predict(net,seq,z['esm33'],a.device));del net
        mean=np.stack(all_pred).mean(0);t=targets.index('MMP13');activity=mean[:,t];off=np.delete(mean,t,axis=1);selectivity=activity-off.mean(1);top=np.argsort(-selectivity,kind='stable')[:100]
        np.savez_compressed(work/'predictions.npz',sequences=seq,targets=targets,predictions=np.stack(all_pred),mean=mean)
        rows.append(dict(pool=file.stem,scorer='PRISM ensemble (seeds 0,1,2)',unique_novel=len(seq),MMP13_predicted_activity_mean=float(activity.mean()),mean_contrast_selectivity_mean=float(selectivity.mean()),predicted_target_leads_fraction=float((activity>off.max(1)).mean()),top100_selectivity_mean=float(selectivity[top].mean())))
        print(json.dumps(rows[-1]),flush=True)
    pd.DataFrame(rows).to_csv(a.output/'predicted_function.csv',index=False)
if __name__=='__main__':main()
