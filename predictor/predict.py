"""Predict activity from an NPZ containing sequences and ESM-2 residue features."""
from pathlib import Path
import sys,argparse
import numpy as np
import torch
import model as m
from model import DeCleaveLM

def load(path,device='cpu'):
 ck=torch.load(path,map_location='cpu',weights_only=False);s=ck['state'];c=ck['model_cfg']
 assert ck['recipe']['model']=='new:adapter:shared_query'
 lm=DeCleaveLM(L=10,d=256,layers=8,heads=4,ff=512)
 b=m.AblatedBase(18,L=10,d=c['dim'],m=c['pair_dim'],hid=c['hid'],pdrop=c['dropout'],de=c['de'],Emb=s['base.Ep_raw'].numpy(),esm_dim=1280,zdim=c['zdim'],kappa=c['kappa'],use_metric=True,lm=lm,lm_freeze=False,lm_residual=True,arm='full')
 net=m.AblatedAdapter(b,[s[k] for k in ['gf','gd','gm','gh']],arm='shared_query');net.load_state_dict(s,strict=True);net.to(device).eval()
 return net,ck['ids']['targets']

def predict(net,sequences,features,device='cpu',batch=128):
 assert features.shape==(len(sequences),10,1280)
 assert all(len(s)==10 and set(s)<=set('ACDEFGHIKLMNPQRSTVWY') for s in sequences)
 b=net.base;tokens=torch.tensor(m.encode(sequences));out=[]
 with torch.inference_mode():
  for i in range(0,len(tokens),batch):
   e=b.residues(tokens[i:i+batch].to(device),torch.as_tensor(np.array(features[i:i+batch]),device=device));hp=b.body(b.condition(e).flatten(0,1)).reshape(len(e),18,-1);raw=b.raw(hp);y=b.a[None]+b.g(b.body(e))+raw-raw.mean(1,keepdim=True);out.append(net(e,y,raw).cpu().numpy())
 return np.concatenate(out)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--checkpoint',required=True);a.add_argument('--input',required=True);a.add_argument('--output',required=True);a.add_argument('--device',default='cpu');p=a.parse_args();z=np.load(p.input);net,targets=load(p.checkpoint,p.device);y=predict(net,z['sequences'].tolist(),z['esm33'],p.device);np.savez_compressed(p.output,predictions=y,sequences=z['sequences'],targets=targets)
