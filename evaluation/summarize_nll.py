from pathlib import Path
import argparse,json,re
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);a=p.parse_args()
rows=[dict(seed=int(re.search(r'seed(\d+)',f.name)[1]),**json.loads(f.read_text())) for f in sorted(a.input.glob('nll_seed*.json'))]
if not rows:raise SystemExit('No NLL results found')
df=pd.DataFrame(rows);df.to_csv(a.input/'nll_by_seed.csv',index=False);df.drop(columns='seed').agg(['mean','std']).to_csv(a.input/'nll_summary.csv');print(df.to_string(index=False))
