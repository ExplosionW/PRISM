import numpy as np
import tensorflow as tf
AA='ACDEFGHIKLMNPQRSTVWY';START=20;STOP=21
class Schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
 def __call__(self,step):
  s=tf.cast(step,tf.float32);return 64.**-.5*tf.minimum(tf.math.rsqrt(tf.maximum(s,1.)),s*4000.**-1.5)
 def get_config(self):return {}
class Block(tf.keras.layers.Layer):
 def __init__(self,cross=False):
  super().__init__();self.att=tf.keras.layers.MultiHeadAttention(num_heads=6,key_dim=64,dropout=.25);self.ln=tf.keras.layers.LayerNormalization(epsilon=.001)
  self.ff=tf.keras.Sequential([tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(64),tf.keras.layers.Dropout(.1)]);self.ln2=tf.keras.layers.LayerNormalization(epsilon=.001);self.cross=cross
  if cross:self.ca=tf.keras.layers.MultiHeadAttention(num_heads=6,key_dim=64,dropout=.25);self.cln=tf.keras.layers.LayerNormalization(epsilon=.001)
 def call(self,x,memory=None,training=False):
  x=self.ln(x+self.att(x,x,use_causal_mask=True,training=training))
  if self.cross:x=self.cln(x+self.ca(x,memory,training=training))
  return self.ln2(x+self.ff(x,training=training))
class Generator(tf.keras.Model):
 def __init__(self,variant):
  super().__init__();assert variant in ['G0','G1','G2'];self.variant=variant
  self.emb=tf.keras.layers.Embedding(22,64,mask_zero=False);self.cond=tf.keras.layers.Dense(64);self.drop=tf.keras.layers.Dropout(.25);self.blocks=[Block(variant=='G2') for _ in range(3)];self.out=tf.keras.layers.Dense(22)
  angles=np.arange(32)[:,None]/(10000**(np.arange(32)[None,:]/32));self.pos=tf.constant(np.concatenate([np.sin(angles),np.cos(angles)],axis=-1),tf.float32)
  if variant=='G2':
   self.enzyme=tf.keras.layers.Embedding(18,64);self.profile=tf.keras.layers.Dense(64);self.null_memory=self.add_weight(name='null_memory',shape=(1,1,64),initializer='zeros')
 def call(self,inputs,training=False):
  tokens,y,target,conditional=inputs;b=tf.shape(tokens)[0];conditional=tf.cast(conditional,tf.bool);yt=tf.gather(y,target,axis=1,batch_dims=1)
  if self.variant=='G1':
   mask=tf.logical_not(tf.one_hot(target,18,on_value=True,off_value=False));diff=tf.reshape(tf.boolean_mask(yt[:,None]-y,mask),(-1,17));features=tf.concat([tf.one_hot(target,18),yt[:,None],diff],1)
  else:features=y
  prefix=tf.where(conditional[:,None],self.cond(features),self.emb(tf.fill([b],START)))
  x=tf.concat([prefix[:,None,:],self.emb(tokens)],1);x=self.drop(x*8.+self.pos[None,:tf.shape(x)[1],:],training=training);memory=None
  if self.variant=='G2':
   features=tf.stack([y,yt[:,None]-y,tf.one_hot(target,18)],axis=-1);memory=self.enzyme(tf.range(18))[None]+self.profile(features);memory=tf.where(conditional[:,None,None],memory,tf.broadcast_to(self.null_memory,tf.shape(memory)))
  for block in self.blocks:x=block(x,memory=memory,training=training)
  return self.out(x)
def build(variant):
 m=Generator(variant);m((tf.zeros((2,10),tf.int32),tf.zeros((2,18)),tf.zeros((2,),tf.int32),tf.ones((2,),tf.bool)));return m
