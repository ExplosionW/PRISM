from pathlib import Path
import os,sys,json,argparse,time,hashlib,shutil
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH','true');os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
import numpy as np,tensorflow as tf
from generator_model import build,Schedule,STOP
from train_generator import evaluate,save_json
P=json.loads((R/'optimization_v2/protocol.json').read_text())
a=argparse.ArgumentParser();a.add_argument('--seed',type=int,required=True);a.add_argument('--prob',type=float,required=True);a.add_argument('--stage',choices=['long','contrast'],required=True);a.add_argument('--weight',type=float,default=0);a.add_argument('--source');a.add_argument('--smoke',action='store_true');args=a.parse_args()
tag=f'p{int(args.prob*100)}' if args.stage=='long' else f'contrast{args.weight:g}'
O=R/'optimization_v2'/('smoke' if args.smoke else 'runs')/args.stage/tag/f'seed{args.seed}';O.mkdir(parents=True,exist_ok=True)
if (O/'DONE.json').exists():sys.exit(0)
tf.keras.utils.set_random_seed(args.seed);tf.config.experimental.enable_op_determinism();tf.config.threading.set_inter_op_parallelism_threads(2);tf.config.threading.set_intra_op_parallelism_threads(4)
assert tf.config.list_physical_devices('GPU')
m=build('G2');opt=tf.keras.optimizers.Adam(Schedule() if args.stage=='long' else P['contrast_learning_rate'],epsilon=1e-7);opt.build(m.trainable_variables);rng=tf.random.Generator.from_seed(args.seed+931);epoch=tf.Variable(0,dtype=tf.int64,trainable=False);best=tf.Variable(float('inf'),dtype=tf.float32,trainable=False)
ck=tf.train.Checkpoint(model=m,optimizer=opt,rng=rng,epoch=epoch,best=best,dropout_states=tf.train.Checkpoint(**{f's{i}':v.value for i,v in enumerate(m.non_trainable_variables)}));manager=tf.train.CheckpointManager(ck,str(O/'last_checkpoint'),max_to_keep=1);hist=[]
tr=np.load(R.parent/'data/benchmark/train.npz');va=np.load(R.parent/'data/benchmark/val.npz');X=tf.constant(tr['tokens']);Y=tf.constant(tr['conditions']);VX=tf.constant(va['tokens']);VY=tf.constant(va['conditions'])
if args.smoke:X=X[:128];Y=Y[:128];VX=VX[:32];VY=VY[:32]
if manager.latest_checkpoint:
 ck.restore(manager.latest_checkpoint).assert_existing_objects_matched();hist=json.loads((O/'history.json').read_text())[:int(epoch.numpy())]
elif args.stage=='long' and args.prob==.5 and not args.smoke:
 src=R/'checkpoints/G2'/f'seed{args.seed}';last=tf.train.latest_checkpoint(str(src/'last_checkpoint'));assert last;ck.restore(last).assert_existing_objects_matched();assert int(epoch.numpy())==50
 hist=json.loads((src/'history.json').read_text());shutil.copy2(src/'best.weights.h5',O/'best.weights.h5');shutil.copy2(src/'validation_metrics.json',O/'validation_metrics.json')
elif args.stage=='contrast':
 assert args.source;m.load_weights(Path(args.source)/'best.weights.h5');initial=evaluate(m,VX,VY);best.assign(initial['joint_nll']);m.save_weights(O/'best.weights.h5');save_json(O/'validation_metrics.json',dict(epoch=0,**initial))
save_json(O/'config.json',dict(**vars(args),protocol=P,parameters=m.count_params(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256((R/'optimization_v2/protocol.json').read_bytes()).hexdigest(),data_sha256=hashlib.sha256((R.parent/'data/benchmark/manifest.json').read_bytes()).hexdigest(),test_labels_used=False))
def token_loss(labels,logits):return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels,logits,from_logits=True),axis=1)
def wrong_profile(y,target):
 # Same target identity throughout each minibatch; donor close in target activity.
 activity=y[:,target];order=tf.argsort(activity,stable=True);n=tf.shape(y)[0];pos=tf.range(n);start=(pos//16)*16;length=tf.minimum(16,n-start);donor_sorted=start+tf.math.floormod(pos-start+1,length);donors=tf.scatter_nd(order[:,None],tf.gather(order,donor_sorted),[n]);wrong=tf.gather(y,donors);mask=tf.one_hot(target,18);return wrong*(1-mask)+y*mask
@tf.function(reduce_retracing=True)
def step(x,y):
 n=tf.shape(x)[0];t=rng.uniform([],minval=0,maxval=18,dtype=tf.int32) if args.stage=='contrast' else None
 target=tf.fill([n],t) if args.stage=='contrast' else rng.uniform([n],minval=0,maxval=18,dtype=tf.int32);flag=rng.uniform([n])<args.prob;labels=tf.concat([x,tf.fill([n,1],STOP)],1)
 with tf.GradientTape() as tape:
  ce=tf.reduce_mean(token_loss(labels,m((x,y,target,flag),training=True)));con=tf.constant(0.,tf.float32)
  if args.weight>0:
   wrong=wrong_profile(y,t);right_loss=token_loss(labels,m((x,y,target,tf.ones([n],tf.bool)),training=True));wrong_loss=token_loss(labels,m((x,wrong,target,tf.ones([n],tf.bool)),training=True));valid=tf.cast(tf.reduce_any(tf.abs(wrong-y)>1e-6,axis=1),tf.float32);con=tf.math.divide_no_nan(tf.reduce_sum(valid*tf.nn.relu(P['contrast_margin_nats_per_token']+right_loss-wrong_loss)),tf.reduce_sum(valid))
  loss=ce+args.weight*con
 grads=tape.gradient(loss,m.trainable_variables);tf.debugging.assert_all_finite(loss,'nonfinite loss');opt.apply_gradients(zip(grads,m.trainable_variables));return ce,con
# Extra stop state is a deterministic function of complete logged validation history.
limit=2 if args.smoke else P['max_epochs'] if args.stage=='long' else P['contrast_epochs']
anchor=float('inf');stale=0
for h in hist:
 if h['joint_nll']<anchor-P['min_delta']:anchor=h['joint_nll'];stale=0
 else:stale+=1
for ep in range(int(epoch.numpy()),limit):
 if stale>=P['patience']:break
 start=time.time();order=np.random.default_rng(args.seed*100000+ep).permutation(len(X));ce_sum=con_sum=0.
 for i in range(0,len(X),P['batch_size']):
  ix=order[i:i+P['batch_size']];ce,con=step(tf.gather(X,ix),tf.gather(Y,ix));ce_sum+=float(ce)*len(ix);con_sum+=float(con)*len(ix)
 metrics=evaluate(m,VX,VY)
 if shutil.disk_usage(O).free<80*1024**2:raise RuntimeError('Data disk free below checkpoint safety reserve')
 if metrics['joint_nll']<float(best.numpy()):best.assign(metrics['joint_nll']);m.save_weights(O/'best.weights.h5');save_json(O/'validation_metrics.json',dict(epoch=ep+1,**metrics))
 if metrics['joint_nll']<anchor-P['min_delta']:anchor=metrics['joint_nll'];stale=0
 else:stale+=1
 epoch.assign(ep+1);hist.append(dict(epoch=ep+1,train_nll=ce_sum/len(X),contrast_loss=con_sum/len(X),**metrics,seconds=time.time()-start,stale_epochs=stale,steps=int(opt.iterations.numpy())));save_json(O/'history.json',hist);manager.save(checkpoint_number=ep+1);print(json.dumps(dict(stage=args.stage,tag=tag,seed=args.seed,**hist[-1])),flush=True)
save_json(O/'DONE.json',dict(stage=args.stage,tag=tag,seed=args.seed,epochs=int(epoch.numpy()),best=json.loads((O/'validation_metrics.json').read_text()),best_sha256=hashlib.sha256((O/'best.weights.h5').read_bytes()).hexdigest(),early_stopped=stale>=P['patience'],test_labels_used=False))
