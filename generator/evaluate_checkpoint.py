from pathlib import Path
import sys,argparse,json,numpy as np,tensorflow as tf
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'src'))
from generator_model import build
from train_generator import evaluate
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data',default=str(R.parent/'data/benchmark/test.npz'));p.add_argument('--output',required=True);a=p.parse_args();m=build('G2');m.load_weights(a.checkpoint);z=np.load(a.data);r=evaluate(m,tf.constant(z['tokens']),tf.constant(z['conditions']));Path(a.output).write_text(json.dumps(r,indent=2));print(r)
