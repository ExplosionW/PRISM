"""Sequence validity, novelty and motif diversity for the current released generator."""
from pathlib import Path
import argparse,collections,re
import numpy as np,pandas as pd
from scipy.stats import entropy
ROOT=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pools',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    known=set((ROOT/'generator/data/known_sequences.txt').read_text().splitlines())
    rows=[]
    for p in sorted(a.pools.glob('PRISM*.csv')):
        d=pd.read_csv(p,keep_default_na=False);valid=d[d.stopped_normally.astype(str).str.lower().eq('true')&d.sequence.str.fullmatch('[ACDEFGHIKLMNPQRSTVWY]{10}')]
        if d.empty or valid.empty:raise ValueError(f'No valid peptide attempts in {p.name}')
        novel=valid[~valid.sequence.isin(known)].sequence.tolist();m=re.search(r'_seed(\d+)_sample(\d+)_T([\d.]+)_(\w+)$',p.stem)
        if not m:raise ValueError(f'Unexpected pool filename: {p.name}')
        row=dict(pool=p.stem,seed=int(m[1]),replicate=int(m[2]),temperature=float(m[3]),mode=m[4],attempts=len(d),valid_fraction=len(valid)/len(d),unique_fraction=valid.sequence.nunique()/len(valid),known_fraction=valid.sequence.isin(known).mean(),unique_novel=len(set(novel)),novel_unique_per_attempt=len(set(novel))/len(d))
        for k in [3,4,5,6]:
            counts=collections.Counter(s[i:i+k] for s in novel for i in range(11-k));n=sum(counts.values());row.update({f'k{k}_distinct':len(counts),f'k{k}_entropy':float(entropy(list(counts.values()),base=2)) if n else np.nan,f'k{k}_top10_mass':sum(v for _,v in counts.most_common(10))/n if n else np.nan})
        rows.append(row)
    if not rows:raise ValueError('No PRISM generation pools found')
    df=pd.DataFrame(rows);df.to_csv(a.output/'quality_by_repeat.csv',index=False)
    metrics=['valid_fraction','unique_fraction','known_fraction','unique_novel','novel_unique_per_attempt','k3_entropy','k6_entropy'];byseed=df.groupby(['temperature','mode','seed'])[metrics].mean();byseed.to_csv(a.output/'quality_by_seed.csv');g=byseed.groupby(['temperature','mode'])[metrics].agg(['mean','std']);g.columns=['_'.join(x) for x in g.columns];g.to_csv(a.output/'quality_summary.csv')
    print(g.to_string())
if __name__=='__main__':main()
