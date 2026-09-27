#!/usr/bin/env python3
"""Three-seed, train-time component ablations of the released own model.

Run from gpu_pkg with the release source snapshot adjacent to this file.
No test labels enter training, early stopping, or candidate-pool calibration.
Historical winning checkpoints and matched retraining controls stay separate.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import pearsonr, spearmanr
import gpu_pdb_pair as pp

r8 = pp.r8
HERE = Path(__file__).resolve().parent
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
CODE_FILES = ('own_model_ablations.py','gpu_pdb_pair.py','gpu_r8_metric.py',
              'gpu_r7_pairwise.py','gpu_mlm_pretrain.py','runmeta.py','selection.py')


def sha(p):
    return pp.digest(p)


def js(p, v):
    pp.save_json(p, v)


def save_torch(p, v):
    p = Path(p); tmp = p.with_suffix('.tmp.pt')
    torch.save(v, tmp); tmp.replace(p)


def cpu_state(net):
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


class AblatedBase(r8.R8Net):
    """Same parameter names/shapes; one train-time intervention per arm."""
    def __init__(self, *args, arm='full', **kwargs):
        super().__init__(*args, **kwargs)
        self.arm = arm

    def residues(self, z, f):
        e = torch.stack([self.E[i][z[:, i]] for i in range(self.L)], 1)
        if self.arm == 'no_lookup': e = e * 0.
        if self.esm:
            fe = self.eproj(f.float())
            if self.arm == 'no_esm': fe = fe * 0.
            e = self.fuse(torch.cat([e, fe], -1))
        if self.lm is not None:
            zl = self.lm.encode(z)
            coeff = self.alpha_lm * (0. if self.arm == 'no_task_lm' else 1.)
            e = e + coeff * self.lmproj(zl)
        return e

    def condition(self, e):
        gam, bet = self.film(self.ep_vec()).chunk(2, -1)
        if self.arm == 'no_film': gam, bet = gam * 0., bet * 0.
        return e[:, None] * (1 + gam[None, :, None]) + bet[None, :, None]

    def pairs_of(self, e):
        ii, jj = zip(*self.pairs)
        left, right = e[..., list(ii), :], e[..., list(jj), :]
        if self.arm == 'additive_pairs':
            d = self.d_e
            u = (torch.relu(torch.einsum('...pk,pkm->...pm', left, self.Wp[:, :d]) + self.bp/2)
                 + torch.relu(torch.einsum('...pk,pkm->...pm', right, self.Wp[:, d:]) + self.bp/2))
        else:
            u = torch.relu(torch.einsum('...pk,pkm->...pm', torch.cat([left, right], -1), self.Wp) + self.bp)
        if self.arm == 'no_pairs': u = u * 0.
        return u

    def body(self, e):
        u = self.pairs_of(e) * torch.sigmoid(self.z)[..., None]
        return self.trunk(torch.cat([e.flatten(-2), u.flatten(-2)], -1))

    def raw(self, hp):
        raw = self.rout(hp).squeeze(-1)
        if self.use_metric:
            z = nn.functional.normalize(self.Wz(hp), dim=-1)
            v = nn.functional.normalize(self.Wv(self.ep_vec()), dim=-1)
            scale = 0. if self.arm == 'no_metric' else 1.
            raw = raw + scale * self.alpha * self.kappa * (z * v[None]).sum(-1)
        return raw

    def forward(self, z, f):
        e = self.residues(z, f)
        hp = self.body(self.condition(e).flatten(0, 1)).reshape(len(e), self.P, -1)
        raw = self.raw(hp)
        return self.a[None] + self.g(self.body(e)) + raw - raw.mean(1, keepdim=True)


class AblatedAdapter(pp.PocketPair):
    def __init__(self, base, geo, arm='full'):
        super().__init__(base, geo, 'sequence', width=32)
        self.arm = arm

    def forward(self, e, baseline_y, baseline_raw):
        b = self.base; em = b.condition(e)
        pair = torch.cat([em[:, :, self.ii], em[:, :, self.jj]], -1)
        u = b.pairs_of(em)
        if self.arm == 'shared_query':
            uncond = torch.cat([e[:, self.ii], e[:, self.jj]], -1)
            q = self.q(uncond)[:, None].expand(-1, b.P, -1, -1)
        else:
            q = self.q(pair)
        sites = self.sites()
        att = torch.einsum('btpd,tsd->btps', q, self.key(sites)) / self.width**.5
        att = att.masked_fill(self.gm[None, :, None] < .5, -1e4).softmax(-1)
        ctx = torch.einsum('btps,tsd->btpd', att, self.value(sites))
        prod = q * ctx * (0. if self.arm == 'no_qc_product' else 1.)
        logits = self.combine(torch.cat([q, ctx, prod], -1))
        delta = .25 * (logits if self.arm == 'linear_update' else torch.tanh(logits))
        u = u * (1 + delta * self.gh[None, :, None, None])
        u = u * torch.sigmoid(b.z)[None, None, :, None]
        hp = b.trunk(torch.cat([em.flatten(2), u.flatten(2)], -1))
        return baseline_y + (b.raw(hp) - baseline_raw) * self.gh[None]


def construct(ctx, seed, arm, with_lm, state=None):
    torch.manual_seed(seed); np.random.seed(seed)
    c = ctx.cfg
    lm = r8.load_lm(str(ctx.root / c['lm_init']), ctx.dev) if with_lm else None
    net = AblatedBase(18, L=10, d=c['dim'], m=c['pair_dim'], hid=c['hid'],
        pdrop=c['dropout'], de=c['de'], Emb=ctx.emb, esm_dim=1280,
        zdim=c['zdim'], kappa=c['kappa'], use_metric=True,
        lm=lm, lm_freeze=False, lm_residual=True, arm=arm).to(ctx.dev)
    matched = []
    if state is not None:
        own = net.state_dict()
        matched = [k for k in state if k in own and own[k].shape == state[k].shape]
        net.load_state_dict({**own, **{k: state[k] for k in matched}}, strict=True)
    return net, matched


class Context:
    def __init__(self, root, out, device):
        self.root, self.out, self.dev = Path(root).resolve(), Path(out).resolve(), torch.device(device)
        self.out.mkdir(parents=True, exist_ok=True)
        ck = torch.load(self.root / 'ck/t2_res_stage_seed0.pt', map_location='cpu', weights_only=False)
        self.cfg, self.ids = ck['cfg'], ck['ids']
        self.emb = ck['state']['Ep_raw'].numpy()
        for s in (1, 2):
            sc = torch.load(self.root / f'ck/t2_res_stage_seed{s}.pt', map_location='cpu', weights_only=False)
            assert sc['ids'] == self.ids
            assert np.array_equal(sc['state']['Ep_raw'].numpy(), self.emb)
        os.environ['PROTEASE_DATA'] = str(self.root/'data')
        r8.r7.B = str(self.root/'data')
        pool_s, pool_y, test_s, test_y = r8.r7.load_raw()
        assert list(test_s) == self.ids['test']
        assert [len(self.ids[k]) for k in ('train', 'val', 'test', 'targets')] == [13666,1200,2901,18]
        assert len(set(pool_s)) == len(pool_s)
        assert set(self.ids['train']) | set(self.ids['val']) == set(pool_s)
        assert not (set(self.ids['train']) & set(self.ids['val']))
        assert not (set(pool_s) & set(test_s))
        assert self.ids['targets'] == sorted(self.ids['targets'])
        ix = {str(s): i for i,s in enumerate(pool_s)}
        self.y = {k: torch.as_tensor(pool_y[[ix[s] for s in self.ids[k]]].astype(np.float32), device=self.dev)
                  for k in ('train', 'val')}
        self.test_y = test_y.astype(np.float32)
        z = np.load(self.root/self.cfg['esm_feats'], allow_pickle=True)
        assert len(z['seqs']) == len(set(map(str,z['seqs'])))
        fmap = {str(s): i for i,s in enumerate(z['seqs'])}
        self.feats = torch.as_tensor(z['L33'], device=self.dev)
        self.frows = {k: torch.as_tensor([fmap[s] for s in self.ids[k]],device=self.dev) for k in ('train','val','test')}
        self.z = {k: torch.as_tensor(r8.encode(self.ids[k]),device=self.dev) for k in ('train','val','test')}
        self.geo_path = self.root/'pdb_pair_code_v2/mmp_geom_v2.npz'
        geo = np.load(self.geo_path,allow_pickle=True)
        assert list(geo['genes']) == self.ids['targets']
        self.geo = [torch.as_tensor(geo[k],dtype=torch.float32,device=self.dev) for k in ('feat','dist','mask','has_struct')]
        trip = r8.build_pairs(self.y['train'].cpu().numpy(), mode='clean', seed=0)
        self.trip = torch.as_tensor(trip,device=self.dev)
        assert len(trip) > 0 and trip[:,1:].max() < 13666
        code = {n:sha(HERE/n) for n in CODE_FILES}
        self.signature = hashlib.sha256(json.dumps({
            'protocol':PROTOCOL, 'code':code,
            'ids': self.ids, 'lm':sha(self.root/self.cfg['lm_init']),
            'features':sha(self.root/self.cfg['esm_feats']), 'geometry':sha(self.geo_path),
            'base':{str(s):sha(self.root/f'ck/t2_res_stage_seed{s}.pt') for s in range(3)}
        }, sort_keys=True).encode()).hexdigest()
        m = self.out/'manifest.json'
        if m.exists():
            assert json.loads(m.read_text())['signature'] == self.signature, 'Run signature changed; use a new output directory'
        else:
            js(m,dict(signature=self.signature,protocol=PROTOCOL,ids=self.ids,started_unix=time.time(),
                      environment=dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),
                      code=code))
        self.active = {}

    def status(self, **v):
        self.active.update(v)
        js(self.out/'status.json',{**self.active,'updated_unix':time.time()})

    def data(self,k,ix):
        return self.z[k][ix], self.feats[self.frows[k][ix]]


@torch.no_grad()
def pred_base(net,ctx,k,batch=512):
    net.eval()
    return torch.cat([net(*ctx.data(k,slice(i,i+batch))) for i in range(0,len(ctx.z[k]),batch)])


@torch.no_grad()
def cache(net,ctx,k):
    net.eval(); a,b,c = [],[],[]
    for i in range(0,len(ctx.z[k]),128):
        z,f = ctx.data(k,slice(i,i+128)); e = net.residues(z,f)
        hp = net.body(net.condition(e).flatten(0,1)).reshape(len(e),18,-1)
        raw = net.raw(hp)
        y = net.a[None] + net.g(net.body(e)) + raw - raw.mean(1,keepdim=True)
        a.append(e); b.append(y); c.append(raw)
    return tuple(torch.cat(v) for v in (a,b,c))


def attempts(path):
    path.mkdir(parents=True,exist_ok=True)
    done = path/'complete.json'
    if done.exists(): return None,json.loads(done.read_text())
    nums = [int(x.name.split('_')[-1]) for x in path.glob('attempt_*') if x.is_dir()]
    p = path/f'attempt_{max(nums,default=0)+1}'
    p.mkdir(); return p,None


def train_loss(y,truth,activity_only=False):
    return (y-truth).square().mean() if activity_only else pp.profile_loss(y,truth)


def fit_base(net,ctx,seed,arm,stage,path,matched,smoke=False):
    att,done = attempts(path)
    if done:
        assert done.get('smoke',False)==smoke, 'Never reuse smoke training as a full run'
        net.load_state_dict(torch.load(done['checkpoint'],map_location=ctx.dev,weights_only=False)['state'])
        return net,done
    cfg = PROTOCOL['training']; total=cfg['max_epochs_per_stage']; patience=cfg['patience']
    torch.manual_seed(seed); np.random.seed(seed)
    sampler=torch.Generator(device=ctx.dev).manual_seed(seed)
    frozen = stage == 'lm_stage'
    for n,p in net.named_parameters(): p.requires_grad_(not frozen or n not in matched)
    opt=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=.001,weight_decay=.001)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total)
    best=float('inf'); bad=0; bestep=0; hist=[]; started=time.time()
    bestfile=att/'best.pt'
    js(att/'initialization.json',dict(signature=ctx.signature,state_hash=pp.state_digest(net.state_dict()),
       seed=seed,arm=arm,stage=stage,matched_tensors=len(matched),
       nominal_parameters=sum(p.numel() for p in net.parameters()),max_epochs=total))
    for ep in range(1,total+1):
        tic=time.time(); net.train(); order=torch.randperm(len(ctx.y['train']),device=ctx.dev,generator=sampler)
        losses=[]
        for j in range(0,len(order),64):
            ix=order[j:j+64]
            t=ctx.trip[torch.randint(len(ctx.trip),(256,),device=ctx.dev,generator=sampler)]
            pix=torch.cat([t[:,1],t[:,2]])
            opt.zero_grad(set_to_none=True)
            yp=net(*ctx.data('train',ix)); pairpred=net(*ctx.data('train',pix))
            loss=train_loss(yp,ctx.y['train'][ix],arm=='activity_only')
            loss=loss + (0. if arm=='activity_only' else .5)*pp.pair_loss(pairpred,t,256)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite base loss')
            loss.backward()
            if ep==1 and j==0:
                alive={n:p.numel() for n,p in net.named_parameters() if p.grad is not None and bool(p.grad.abs().max()>0)}
                js(att/'gradient_probe.json',dict(active_parameters=sum(alive.values()),active_names=list(alive),
                   phase='initial freeze phase' if frozen else 'unfrozen task training',
                   zero_grad_allowed=arm=='no_task_lm' and frozen))
                if not alive and not (arm=='no_task_lm' and frozen): raise RuntimeError('No active gradients')
            if stage=='lm_stage' and ep==11 and j==0:
                inherited = [p for n,p in net.named_parameters() if n in matched]
                assert all(p.requires_grad for p in inherited)
                assert any(p.grad is not None and bool(p.grad.abs().max()>0) for p in inherited)
                js(att/'unfreeze_check.json',dict(epoch=11,passed=True))
            opt.step(); losses.append(float(loss.detach()))
            if j//64 % 50 == 0:
                ctx.status(status='training',family='backbone',seed=seed,arm=arm,stage=stage,epoch=ep,batch=j//64+1)
            if smoke and j>=64: break
        sch.step()
        if frozen and ep==10:
            for p in net.parameters():p.requires_grad_(True)
            opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
            sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,total-10)
            frozen=False
        net.eval()
        with torch.no_grad(): val=float(pp.profile_loss(pred_base(net,ctx,'val'),ctx.y['val']))
        if val < best-1e-7:
            best,bestep,bad=val,ep,0
            save_torch(bestfile,dict(state=cpu_state(net),epoch=ep,arm=arm,stage=stage,
                                    ids=ctx.ids,signature=ctx.signature))
        else:bad+=1
        rec=dict(epoch=ep,val_loss=val,train_loss=float(np.mean(losses)),seconds=time.time()-tic,lr=sch.get_last_lr()[0])
        hist.append(rec);js(att/'history.json',hist)
        ctx.status(epoch=ep,best_epoch=bestep,best_val=best,epoch_seconds=rec['seconds'])
        print('EPOCH',seed,arm,stage,ep,round(val,6),round(rec['seconds'],2),flush=True)
        if bad>=patience or (smoke and (stage!='lm_stage' or ep>=11)):break
    net.load_state_dict(torch.load(bestfile,map_location=ctx.dev,weights_only=False)['state']);net.eval()
    result=dict(checkpoint=str(bestfile),seed=seed,arm=arm,stage=stage,best_epoch=bestep,
                epochs=ep,best_val=best,seconds=time.time()-started,signature=ctx.signature,smoke=smoke)
    js(path/'complete.json',result);return net,result


def fit_adapter(base,ctx,seed,arm,path,activity_only=False,smoke=False):
    base.eval().requires_grad_(False)
    base.zero_grad(set_to_none=True)
    basehash=pp.state_digest(base.state_dict())
    torch.manual_seed(seed);np.random.seed(seed)
    net=AblatedAdapter(base,ctx.geo,arm).to(ctx.dev)
    att,done=attempts(path)
    if done:
        assert done.get('smoke',False)==smoke, 'Never reuse smoke training as a full run'
        net.load_state_dict({**net.state_dict(),**torch.load(done['checkpoint'],map_location=ctx.dev,weights_only=False)['adapter']})
        return net,done
    caches={k:cache(base,ctx,k) for k in ('train','val')}
    pars=[p for n,p in net.named_parameters() if not n.startswith('base.')]
    opt=torch.optim.AdamW(pars,lr=.001,weight_decay=.001)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,300)
    sampler=torch.Generator(device=ctx.dev).manual_seed(seed)
    with torch.no_grad():
        pv=pp.predict(net,caches['val']);initerr=float((pv-caches['val'][1]).abs().max())
        assert initerr<2e-5,('nonidentity adapter init',initerr)
        best=float(pp.profile_loss(pv,ctx.y['val']))
    bestep=0;bad=0;hist=[dict(epoch=0,val_loss=best,initial_error=initerr)];started=time.time()
    bestfile=att/'best.pt'
    save_torch(bestfile,dict(adapter=net.adapter_state(),signature=ctx.signature,epoch=0))
    if base.arm=='no_pairs':
        # There is no modulated branch, so these parameters have no causal path to output.
        result=dict(checkpoint=str(bestfile),seed=seed,arm=arm,stage='adapter',epochs=0,best_epoch=0,
                    seconds=0,initial_error=initerr,skipped='no pair branch: exactly inactive',signature=ctx.signature)
        result['smoke']=smoke
        js(path/'complete.json',result);return net,result
    for ep in range(1,301):
        tic=time.time();net.train();order=torch.randperm(len(ctx.y['train']),device=ctx.dev,generator=sampler)
        losses=[]
        for j in range(0,len(order),64):
            ix=order[j:j+64];t=ctx.trip[torch.randint(len(ctx.trip),(256,),device=ctx.dev,generator=sampler)]
            pix=torch.cat([t[:,1],t[:,2]])
            opt.zero_grad(set_to_none=True)
            yp=net(*(v[ix] for v in caches['train']))
            pairpred=net(*(v[pix] for v in caches['train']))
            loss=train_loss(yp,ctx.y['train'][ix],activity_only)
            weight=0. if activity_only or arm=='no_pair_rank' else .5
            loss=loss+weight*pp.pair_loss(pairpred,t,256)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite adapter loss')
            loss.backward()
            if ep==1 and j==0:
                assert all(p.grad is None for p in base.parameters())
                assert any(p.grad is not None and bool(p.grad.abs().max()>0) for p in pars)
            nn.utils.clip_grad_norm_(pars,10.);opt.step();losses.append(float(loss.detach()))
            if j//64 % 50 == 0:ctx.status(status='training',seed=seed,arm=arm,stage='adapter',epoch=ep,batch=j//64+1)
            if smoke and j>=64:break
        sch.step()
        with torch.no_grad():val=float(pp.profile_loss(pp.predict(net,caches['val']),ctx.y['val']))
        if val<best-1e-7:
            best,bestep,bad=val,ep,0
            save_torch(bestfile,dict(adapter=net.adapter_state(),signature=ctx.signature,epoch=ep))
        else:bad+=1
        rec=dict(epoch=ep,val_loss=val,train_loss=float(np.mean(losses)),seconds=time.time()-tic,lr=sch.get_last_lr()[0])
        hist.append(rec);js(att/'history.json',hist)
        ctx.status(epoch=ep,best_epoch=bestep,best_val=best,epoch_seconds=rec['seconds'])
        print('EPOCH',seed,arm,'adapter',ep,round(val,6),round(rec['seconds'],2),flush=True)
        if bad>=25 or smoke:break
    assert pp.state_digest(base.state_dict())==basehash,'Frozen base changed'
    net.load_state_dict({**net.state_dict(),**torch.load(bestfile,map_location=ctx.dev,weights_only=False)['adapter']})
    result=dict(checkpoint=str(bestfile),seed=seed,arm=arm,stage='adapter',epochs=ep,best_epoch=bestep,
                best_val=best,seconds=time.time()-started,base_unchanged=True,initial_error=initerr,
                signature=ctx.signature,smoke=smoke)
    js(path/'complete.json',result);return net,result


@torch.no_grad()
def export_predictions(net,ctx,path,family,arm,seed):
    net.eval()
    pv=pp.predict(net,cache(net.base,ctx,'val')).cpu().numpy().T.astype(np.float32)
    pt=pp.predict(net,cache(net.base,ctx,'test')).cpu().numpy().T.astype(np.float32)
    assert np.isfinite(pv).all() and np.isfinite(pt).all()
    key=f'{family}:{arm}|{seed}'
    pp.save_npz(path/'predictions.npz',{key:pt,f'VAL@{key}':pv,'_Yf':ctx.test_y})
    rr,_=r8.selection.evaluate(pv,ctx.y['val'].cpu().numpy().T,pt,ctx.test_y,ctx.y['train'].cpu().numpy().T)
    for q in rr:q.update(family=family,arm=arm,seed=seed,gene=ctx.ids['targets'][q['target']])
    pd.DataFrame(rr).to_csv(path/'selection.csv',index=False)
    assert len(rr)==36,'Do not silently drop a nonpositive-reference target'
    mt,mp=r8.selection.margins(ctx.test_y),r8.selection.margins(pt)
    ct,cp=ctx.test_y-ctx.test_y.mean(0),pt-pt.mean(0)
    result=dict(family=family,arm=arm,seed=seed,selection=r8.selection.summarize(rr),
        margin_spearman=float(np.mean([spearmanr(a,b).statistic for a,b in zip(mt,mp)])),
        centred_pearson=float(np.mean([pearsonr(a,b).statistic for a,b in zip(ct,cp)])),
        prediction_sha256=sha(path/'predictions.npz'),signature=ctx.signature)
    js(path/'result.json',result)
    print('ARM_DONE',family,arm,seed,flush=True)


def existing_base(ctx,seed):
    ck=torch.load(ctx.root/f'ck/t2_res_stage_seed{seed}.pt',map_location='cpu',weights_only=False)
    net,_=construct(ctx,seed,'full',True,state=ck['state'])
    assert set(net.state_dict())==set(ck['state'])
    net.eval().requires_grad_(False);return net


def preflight(ctx):
    records=[]
    for seed in (0,1,2):
        base=existing_base(ctx,seed)
        original,ck=pp.load_base(str(ctx.root/f'ck/t2_res_stage_seed{seed}.pt'),ctx.dev)
        z,f=ctx.data('val',slice(0,12))
        with torch.no_grad():
            y=base(z,f);ref=original(z,torch.arange(len(z),device=ctx.dev),f)
            assert float((y-ref).abs().max())<2e-5
        bc=tuple(v[:12] for v in cache(base,ctx,'val'))
        # Compare the trained adapter as well as identity initialization; a zero
        # initialization alone would fail to detect dead or miswired branches.
        chk=torch.load(ctx.root/f'pdb_pair_train_v2_20260910/seed{seed}/sequence_best.pt',map_location=ctx.dev,weights_only=False)['adapter']
        our=AblatedAdapter(base,ctx.geo).to(ctx.dev).eval()
        old=pp.PocketPair(original,ctx.geo,'sequence').to(ctx.dev).eval()
        our.load_state_dict({**our.state_dict(),**chk});old.load_state_dict({**old.state_dict(),**chk})
        with torch.no_grad():
            err=float((our(*bc)-old(*bc)).abs().max());assert err<2e-5
        records.append(dict(seed=seed,kind='trained_release_parity',max_error=err))
        for arm in PROTOCOL['adapter_family']['arms']:
            torch.manual_seed(seed);net=AblatedAdapter(base,ctx.geo,arm).to(ctx.dev)
            loss=pp.profile_loss(net(*bc),ctx.y['val'][:12]);loss.backward()
            assert any(v.grad is not None and bool(v.grad.abs().max()>0) for n,v in net.named_parameters() if not n.startswith('base.'))
            assert all(v.grad is None for v in base.parameters())
            records.append(dict(seed=seed,kind='adapter_gradient',arm=arm,passed=True))
        del base,original,our,old,bc
    # Validate every training-time intervention and identical initial tensors.
    for with_lm in (False,True):
        anchor=None
        for arm in PROTOCOL['backbone_family']['arms']:
            b,_=construct(ctx,0,arm,with_lm)
            digest=pp.state_digest(b.state_dict())
            if anchor is None:anchor=digest
            assert digest==anchor,('different initial weights',arm)
            b.train();z,f=ctx.data('train',slice(0,12));y=b(z,f)
            loss=train_loss(y,ctx.y['train'][:12],arm=='activity_only');loss.backward()
            assert torch.isfinite(loss) and any(v.grad is not None and bool(v.grad.abs().max()>0) for v in b.parameters())
            b.eval()
            with torch.no_grad():
                if arm=='no_esm':assert torch.equal(b(z,f),b(z,torch.randn_like(f)))
                if arm=='no_lookup':assert b.E.grad is not None and not bool(b.E.grad.abs().max()>0)
                if arm=='no_film':assert torch.equal(b.condition(b.residues(z,f))[:,0],b.condition(b.residues(z,f))[:,1])
                if arm=='no_pairs':assert not bool(b.pairs_of(b.residues(z,f)).abs().max()>0)
            records.append(dict(seed=0,kind='base_gradient',with_lm=with_lm,arm=arm,initial_hash=digest,passed=True))
            del b
    js(ctx.out/'preflight.json',dict(passed=True,checks=records,signature=ctx.signature))
    ctx.status(status='preflight_passed',checks=len(records))
    print('PREFLIGHT_PASSED',len(records),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='/root/autodl-tmp/gpu_pkg')
    ap.add_argument('--out',required=True);ap.add_argument('--device',default='cuda')
    ap.add_argument('--preflight',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--family',choices=['adapter','backbone','all'],default='all')
    a=ap.parse_args();assert a.device=='cuda','Training must run on the remote GPU'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    os.chdir(a.root)
    ctx=Context(a.root,a.out,a.device)
    if a.preflight:preflight(ctx);return
    assert (ctx.out/'preflight.json').exists(),'Preflight required in this immutable run directory'
    assert json.loads((ctx.out/'preflight.json').read_text())['signature']==ctx.signature
    errors=[]
    for family in ('adapter','backbone'):
        if a.family not in (family,'all'):continue
        arms=PROTOCOL[f'{family}_family']['arms']
        for seed in (0,1,2):
            for arm in arms:
                p=ctx.out/family/f'seed{seed}'/arm;p.mkdir(parents=True,exist_ok=True)
                if (p/'result.json').exists():continue
                ctx.status(status='starting_arm',family=family,seed=seed,arm=arm,stage=None,epoch=0)
                try:
                    if family=='adapter':
                        b=existing_base(ctx,seed)
                        net,_=fit_adapter(b,ctx,seed,arm,p/'adapter',smoke=a.smoke)
                    else:
                        b,_=construct(ctx,seed,arm,False)
                        b,_=fit_base(b,ctx,seed,arm,'task_base',p/'task_base',[],a.smoke)
                        init=cpu_state(b);del b
                        b,matched=construct(ctx,seed,arm,True,init)
                        b,_=fit_base(b,ctx,seed,arm,'lm_stage',p/'lm_stage',matched,a.smoke)
                        net,_=fit_adapter(b,ctx,seed,'full',p/'adapter',arm=='activity_only',a.smoke)
                    if a.smoke:
                        ctx.status(status='smoke_passed',family=family,seed=seed,arm=arm)
                        print('SMOKE_PASSED',flush=True);return
                    export_predictions(net,ctx,p,family,arm,seed)
                    del net,b
                    torch.cuda.empty_cache()
                except Exception as exc:
                    e=dict(family=family,seed=seed,arm=arm,error=repr(exc),traceback=traceback.format_exc(),unix=time.time())
                    js(p/'failure.json',e);errors.append(e);print('ARM_FAILED',json.dumps(e),flush=True)
                    if a.smoke:
                        ctx.status(status='smoke_failed',errors=errors)
                        raise
                    torch.cuda.empty_cache()
        ctx.status(status='family_finished',family=family,failures=len(errors))
    ctx.status(status='finished_with_errors' if errors else 'completed',errors=errors)
    js(ctx.out/'queue_result.json',dict(status='finished_with_errors' if errors else 'completed',errors=errors,signature=ctx.signature))
    print('QUEUE_DONE',len(errors),flush=True)


if __name__=='__main__':main()
