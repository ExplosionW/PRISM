"""Verify split identities, target order, label arrays and release checksums."""
from pathlib import Path
import hashlib,json,itertools
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1]
ids=json.loads((R/'data/split_ids.json').read_text());targets=json.loads((R/'data/benchmark/target_order.json').read_text());sets={}
for split,n in [('train',13666),('val',1200),('test',2901)]:
    z=np.load(R/f'data/benchmark/{split}.npz');d=pd.read_csv(R/f'data/benchmark/{split}.csv');seq=z['sequences'].tolist()
    assert len(seq)==n and len(set(seq))==n and seq==ids[split]==d.sequence.tolist()
    assert all(len(s)==10 and set(s)<=set('ACDEFGHIKLMNPQRSTVWY') for s in seq)
    assert z['labels'].shape==(n,18) and z['conditions'].shape==(n,18) and z['tokens'].shape==(n,10)
    assert np.isfinite(z['labels']).all();np.testing.assert_allclose(d[targets].to_numpy(),z['labels'],rtol=1e-6,atol=1e-6)
    expected_tokens=np.array([['ACDEFGHIKLMNPQRSTVWY'.index(a) for a in s] for s in seq]);np.testing.assert_array_equal(expected_tokens,z['tokens']);sets[split]=set(seq)
z=np.load(R/'data/ood/ood.npz');d=pd.read_csv(R/'data/ood/ood.csv');assert len(z['sequences'])==80 and z['labels'].shape==(80,12);assert z['sequences'].tolist()==d.sequence.tolist();assert z['targets'].tolist()==json.loads((R/'data/ood/target_order.json').read_text());assert set(z['targets'])<=set(targets)
np.testing.assert_allclose(d[z['targets'].tolist()].to_numpy(),z['labels'],rtol=0,atol=1e-12);assert np.isfinite(z['labels']).all();sets['ood']=set(z['sequences']);assert len(sets['ood'])==80
for a,b in itertools.combinations(sets,2):assert not sets[a]&sets[b],f'Overlap between {a} and {b}'
manifest=R/'SHA256SUMS'
if manifest.exists():
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1);p=R/name;assert p.is_file(),f'Missing {name}'
        with p.open('rb') as f:
            h=hashlib.sha256()
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        assert h.hexdigest()==digest,f'Checksum mismatch: {name}'
print(json.dumps({'split_sizes':{k:len(v) for k,v in sets.items()},'pairwise_sequence_overlap':0,'benchmark_targets':18,'measured_ood_targets':12,'checksums':'passed' if manifest.exists() else 'not present'},indent=2))
