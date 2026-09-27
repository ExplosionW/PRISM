"""Portable G2 sampling. Raw attempts retained; invalid outputs are not refilled."""
from pathlib import Path
import sys,argparse,json,csv
import numpy as np,tensorflow as tf
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'src'))
from generator_model import build,AA,START,STOP
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--temperature',type=float,default=1.2);p.add_argument('--mode',choices=['selective','efficient','unconditional'],default='selective');p.add_argument('--per-template',type=int,default=400);p.add_argument('--batch',type=int,default=512);a=p.parse_args()
m=build('G2');m.load_weights(a.checkpoint);tr=np.load(R.parent/'data/benchmark/train.npz');y=tr['labels'];score=y[:,4] if a.mode=='efficient' else y[:,4]-np.delete(y,4,axis=1).mean(1);templates=np.argsort(-score,kind='stable')[:50];ix=np.repeat(templates,a.per_template);conditions=tr['conditions'][ix] if a.mode!='unconditional' else np.zeros((len(ix),18),np.float32);known=set()
for split in ['train','val','test']:known.update(np.load(R.parent/f'data/benchmark/{split}.npz')['sequences'].tolist())
@tf.function(reduce_retracing=True)
def sample(y,temp,seed):
 n=tf.shape(y)[0];x=tf.zeros([n,0],tf.int32);st=tf.zeros([n],tf.bool);length=tf.fill([n],15)
 for k in range(15):
  z=m((x,y,tf.fill([n],4),tf.fill([n],a.mode!='unconditional')),training=False)[:,-1,:];draw=tf.cast(tf.random.stateless_categorical(z/temp,1,[seed,k*2])[:,0],tf.int32)
  if k>0:
   pen=z*(1.-tf.one_hot(draw,22)*(1.-1./1.2));again=tf.cast(tf.random.stateless_categorical(pen,1,[seed,k*2+1])[:,0],tf.int32);draw=tf.where(draw==x[:,-1],again,draw)
  first=(~st)&(draw==STOP);length=tf.where(first,tf.fill([n],k),length);st=st|(draw==STOP);x=tf.concat([x,draw[:,None]],1)
 return x,length,st
with open(a.output,'w') as f:
 w=csv.DictWriter(f,fieldnames=['sample_id','sequence','condition_train_index','stopped_normally','raw_length','filter_reason']);w.writeheader()
 for lo in range(0,len(ix),a.batch):
  x,L,stop=sample(tf.constant(conditions[lo:lo+a.batch]),tf.constant(a.temperature),tf.constant(20260922+a.seed*10000+lo+['unconditional','efficient','selective'].index(a.mode)*100000))
  for j,(xx,ll,ss) in enumerate(zip(x.numpy(),L.numpy(),stop.numpy())):
   tok=xx[:ll];s=''.join(AA[t] if t<20 else ('$' if t==START else '*') for t in tok);reason='no_stop' if not ss else 'special_token' if any(tok>=20) else 'length' if ll!=10 else 'known_sequence' if s in known else ''
   w.writerow(dict(sample_id=lo+j,sequence=s,condition_train_index=int(ix[lo+j]) if a.mode!='unconditional' else -1,stopped_normally=bool(ss),raw_length=int(ll),filter_reason=reason))
