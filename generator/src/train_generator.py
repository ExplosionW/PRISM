import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2');os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH','true')
from pathlib import Path
import argparse,json,time,hashlib,pickle,random,csv,sys,shutil
import numpy as np
import tensorflow as tf
from generator_model import build,Schedule,STOP
R=Path(__file__).resolve().parents[1]
def save_json(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
@tf.function(reduce_retracing=True)
def batch_nll(model,x,y,target,flag):
 logits=model((x,y,tf.fill([tf.shape(x)[0]],target),tf.fill([tf.shape(x)[0]],flag)),training=False);labels=tf.concat([x,tf.fill([tf.shape(x)[0],1],STOP)],1)
 return tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True))
def evaluate(model,X,Y,batch=256,all_targets=True):
 cond=[];uncond=[]
 for flag in [False,True]:
  targets=range(18) if flag and model.variant!='G0' and all_targets else [4]
  total=0.;n=0
  for target in targets:
   for i in range(0,len(X),batch):
    x=X[i:i+batch];y=Y[i:i+batch];total+=float(batch_nll(model,x,y,tf.constant(target),tf.constant(flag)));n+=len(x)*11
  (cond if flag else uncond).append(total/n)
 return dict(conditional_nll=cond[0],unconditional_nll=uncond[0],joint_nll=(cond[0]+uncond[0])/2,conditional_perplexity=float(np.exp(cond[0])),unconditional_perplexity=float(np.exp(uncond[0])))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--variant',required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();cfg=json.loads((R/'configs/generator_joint.json').read_text());out=R/'checkpoints'/a.variant/f'seed{a.seed}';out.mkdir(parents=True,exist_ok=True)
 tf.keras.utils.set_random_seed(a.seed);tf.config.experimental.enable_op_determinism();tf.config.threading.set_inter_op_parallelism_threads(2);tf.config.threading.set_intra_op_parallelism_threads(4)
 assert tf.config.list_physical_devices('GPU'),'Full training requires requested GPU'
 model=build(a.variant);opt=tf.keras.optimizers.Adam(Schedule(),beta_1=.9,beta_2=.999,epsilon=1e-7);opt.build(model.trainable_variables);rng=tf.random.Generator.from_seed(a.seed+931)
 tr=np.load(R.parent/'data/benchmark/train.npz');va=np.load(R.parent/'data/benchmark/val.npz');X=tf.constant(tr['tokens']);Y=tf.constant(tr['conditions']);VX=tf.constant(va['tokens']);VY=tf.constant(va['conditions']);epoch=tf.Variable(0,dtype=tf.int64,trainable=False);best=tf.Variable(float('inf'),dtype=tf.float32,trainable=False)
 # Include Keras layer seed states explicitly for exact dropout resume.
 states={f's{i}':v.value for i,v in enumerate(model.non_trainable_variables)};ck=tf.train.Checkpoint(model=model,optimizer=opt,rng=rng,epoch=epoch,best=best,dropout_states=tf.train.Checkpoint(**states));manager=tf.train.CheckpointManager(ck,str(out/'last_checkpoint'),max_to_keep=1)
 history=[]
 if manager.latest_checkpoint:
  ck.restore(manager.latest_checkpoint).assert_existing_objects_matched();history=json.loads((out/'history.json').read_text())[:int(epoch.numpy())]
 save_json(out/'config.json',dict(**cfg,variant=a.variant,seed=a.seed,parameters=model.count_params(),tensorflow=tf.__version__))
 save_json(out/'manifest.json',dict(data_manifest_sha256=hashlib.sha256((R.parent/'data/benchmark/manifest.json').read_bytes()).hexdigest(),source_hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((R/'src').glob('*.py'))+list((R/'configs').glob('*.json'))},gpu=[str(q) for q in tf.config.list_physical_devices('GPU')],test_labels_used=False))
 @tf.function(reduce_retracing=True)
 def step(x,y):
  n=tf.shape(x)[0];target=tf.cast(rng.uniform([n],minval=0,maxval=18,dtype=tf.int32),tf.int32);flag=rng.uniform([n])<.5;labels=tf.concat([x,tf.fill([n,1],STOP)],1)
  with tf.GradientTape() as tape:
   logits=model((x,y,target,flag),training=True);loss=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True))
  grads=tape.gradient(loss,model.trainable_variables);opt.apply_gradients(zip(grads,model.trainable_variables));return loss
 for ep in range(int(epoch.numpy()),cfg['epochs']):
  start=time.time();order=np.random.default_rng(a.seed*100000+ep).permutation(len(X));total=0
  for i in range(0,len(order),cfg['batch_size']):
   ix=order[i:i+cfg['batch_size']];loss=step(tf.gather(X,ix),tf.gather(Y,ix));total+=float(loss)*len(ix)
  metrics=evaluate(model,VX,VY)
  if shutil.disk_usage(out).free<80*1024**2:raise RuntimeError('Checkpoint save aborted: data disk reserve <80 MiB')
  if metrics['joint_nll']<float(best.numpy()):
   best.assign(metrics['joint_nll']);model.save_weights(out/'best.weights.h5');save_json(out/'validation_metrics.json',dict(epoch=ep+1,**metrics))
  epoch.assign(ep+1);history.append(dict(epoch=ep+1,train_nll=total/len(X),**metrics,seconds=time.time()-start,steps=int(opt.iterations.numpy()),learning_rate=float(Schedule()(opt.iterations)),best_joint_nll=float(best.numpy())))
  save_json(out/'history.json',history)
  with (out/'history.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
  manager.save(checkpoint_number=ep+1);print(json.dumps(dict(variant=a.variant,seed=a.seed,**history[-1])),flush=True)
 save_json(out/'DONE.json',dict(variant=a.variant,seed=a.seed,epochs=int(epoch.numpy()),parameters=model.count_params(),best=json.loads((out/'validation_metrics.json').read_text()),best_sha256=hashlib.sha256((out/'best.weights.h5').read_bytes()).hexdigest(),seconds=sum(q['seconds'] for q in history),test_labels_used=False))
if __name__=='__main__':main()
