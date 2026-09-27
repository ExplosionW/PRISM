"""Frozen ESM2 L33 residue features, historical mixed precision by default.
Input CSV column 'sequence'; exact 10-mer strings only. Requires GPU for bf16.
Encoder revision and numerical precision are recorded in the feature archive.
"""
from pathlib import Path
import argparse
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer,AutoModel
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--encoder',default='facebook/esm2_t33_650M_UR50D');p.add_argument('--revision',default='08e4846e537177426273712802403f7ba8261b6c');p.add_argument('--device',default='cuda');p.add_argument('--precision',choices=['bf16','fp32'],default='bf16');a=p.parse_args()
seq=pd.read_csv(a.input).sequence.astype(str).tolist();assert len(seq)>0 and all(len(s)==10 and set(s)<=set('ACDEFGHIKLMNPQRSTVWY') for s in seq)
if a.precision=='bf16':assert a.device.startswith('cuda'),'Historical bf16 extraction requires CUDA; use fp32 explicitly on CPU'
torch.backends.cuda.matmul.allow_tf32=False;tok=AutoTokenizer.from_pretrained(a.encoder,revision=a.revision);enc=AutoModel.from_pretrained(a.encoder,revision=a.revision,torch_dtype=torch.float32,add_pooling_layer=False).to(a.device).eval();enc.requires_grad_(False);out=[]
with torch.inference_mode():
 for i in range(0,len(seq),256):
  chunk=seq[i:i+256];padded=chunk+[chunk[-1]]*(256-len(chunk));b=tok(padded,return_tensors='pt',padding=True)
  assert tok.convert_ids_to_tokens(b['input_ids'][0][1:11].tolist())==list(chunk[0])
  with torch.autocast('cuda',dtype=torch.bfloat16,enabled=a.precision=='bf16'):
   z=enc(**{k:v.to(a.device) for k,v in b.items()},output_hidden_states=True).hidden_states[33][:len(chunk),1:11]
  out.append(z.float().cpu().numpy().astype(np.float16 if a.precision=='bf16' else np.float32))
np.savez_compressed(a.output,sequences=seq,esm33=np.concatenate(out),feature_precision=a.precision,encoder=a.encoder,revision=a.revision)
