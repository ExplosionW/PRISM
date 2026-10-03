"""PRISM predictor architecture."""
import numpy as np
import torch
from torch import nn

AA = "ACDEFGHIKLMNPQRSTVWY"

def encode(seqs, L=10):
    ix = {c: i for i, c in enumerate(AA)}
    return np.array([[ix.get(c, 20) for c in str(s).strip()[:L]] for s in seqs], np.int64)

class DeCleaveLM(nn.Module):
    """Peptide language model with residue embeddings and masked-token prediction."""

    def __init__(s, L=10, d=128, layers=4, heads=4, ff=256, pdrop=0.1):
        super().__init__()
        s.L, s.d = L, d
        s.emb = nn.Embedding(22, d)
        s.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        s.tr = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, ff, dropout=pdrop,
                                       batch_first=True, norm_first=True),
            num_layers=layers)
        s.norm = nn.LayerNorm(d)
        s.head = nn.Linear(d, 20)                                  

    def encode(s, Z):
        return s.norm(s.tr(s.emb(Z) + s.pos))                   

    def forward(s, Z):
        return s.head(s.encode(Z))

class R8Net(nn.Module):
    """Enzyme-conditioned peptide predictor with a metric branch."""

    def __init__(s, P, L=10, d=96, m=32, hid=384, pdrop=0.15, de=32,
                 Emb=None, esm_dim=0, zdim=64, kappa=10.0, use_metric=True,
                 lm=None, lm_freeze=False, film_pos_rank=0, cross_attn=False,
                 lm_residual=False, ladder="none", n_layer=0,
                 geom_mode="none", geom_dim=28,
                 cdr_mode="off", cdr_dim=32, cdr_eps=0.1):
        super().__init__()
        import itertools
        s.P, s.L, s.use_metric, s.kappa, s.d_e = P, L, use_metric, kappa, d
        s.pairs = list(itertools.combinations(range(L), 2))
        s.E = nn.Parameter(torch.randn(L, 21, d) * 0.05)
                                
                                               
                                              
        s.gmode = geom_mode
        if geom_mode != "none":
            s.gnode = nn.Sequential(nn.Linear(geom_dim, d), nn.LayerNorm(d), nn.ReLU())
            s.gmsg = nn.ModuleList([nn.Sequential(nn.Linear(2 * d + 16, d), nn.ReLU(),
                                                  nn.Linear(d, d)) for _ in range(2)])
            s.gupd = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
                                    for _ in range(2)])
                                                    
            s.gfall = nn.Parameter(torch.randn(d) * 0.02)
            if geom_mode == "interact":
                                            
                                                      
                                               
                s.gq = nn.Linear(d, d); s.gk = nn.Linear(d, d); s.gv = nn.Linear(d, d)
                s.gout = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
                s.beta_g = nn.Parameter(torch.zeros(1))
            else:
                s.gemb = nn.Sequential(nn.Linear(d, de), nn.LayerNorm(de))
                                                                                 
                                                                           
                                                      
         
                                               
                                                
                                                     
                                                   
                                              
                                                               
                                   
        s.cdr, s.cdr_eps = cdr_mode, cdr_eps
        if cdr_mode != "off":
            H_ = hid // 2
            s.cdr_phi = nn.Sequential(nn.Linear(H_ + de, cdr_dim), nn.ReLU())
            s.cdr_w = nn.Linear(H_ + de + 1, 1)                              
            s.cdr_psi = nn.Linear(cdr_dim, 1)
            s.eta_cdr = nn.Parameter(torch.zeros(1))
            s.register_buffer("cdr_eye", torch.eye(P))
        s.ladder = ladder
        if ladder != "none":
            nL = n_layer
                                           
            s.lproj = nn.ModuleList([nn.Sequential(nn.Linear(esm_dim, d), nn.LayerNorm(d))
                                     for _ in range(nL)])
            if ladder == "linear":
                                                   
                s.lmix = nn.Linear(nL * d, d)
            else:
                                                  
                                                 
                               
                s.lupd = nn.ModuleList([nn.GRUCell(d, d) for _ in range(nL)])
                if ladder == "cond":
                    s.lfilm = nn.ModuleList([nn.Sequential(
                        nn.Linear(de, 64), nn.ReLU(), nn.Linear(64, 2 * d))
                        for _ in range(nL)])
        s.esm = esm_dim > 0 and ladder == "none"
        if s.esm:
            s.eproj = nn.Sequential(nn.Linear(esm_dim, d), nn.LayerNorm(d))
            s.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
                                                          
        s.lm, s.lm_freeze, s.lm_residual = lm, lm_freeze, lm_residual
        if lm is not None:
            s.lmproj = nn.Sequential(nn.Linear(lm.d, d), nn.LayerNorm(d))
            s.lmfuse = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d))
            s.alpha_lm = nn.Parameter(torch.zeros(1))
            if lm_freeze:
                for q in lm.parameters():
                    q.requires_grad_(False)
        s.Wp = nn.Parameter(torch.randn(len(s.pairs), 2 * d, m) * (1.0 / (2 * d) ** .5))
        s.bp = nn.Parameter(torch.zeros(len(s.pairs), m))
        s.z = nn.Parameter(torch.zeros(len(s.pairs)))
        din = L * d + len(s.pairs) * m
        s.trunk = nn.Sequential(nn.Linear(din, hid), nn.ReLU(), nn.Dropout(pdrop),
                                nn.Linear(hid, hid // 2), nn.ReLU())
        s.a = nn.Parameter(torch.zeros(P))
        s.g = nn.Linear(hid // 2, 1)
        s.rout = nn.Linear(hid // 2, 1)
        if Emb is None:
            s.ep = nn.Parameter(torch.randn(P, de) * 0.05); s.Eproj = None
        else:
            s.register_buffer("Ep_raw", torch.tensor(Emb))
            s.Eproj = nn.Sequential(nn.Linear(Emb.shape[1], de), nn.LayerNorm(de))
        s.film = nn.Sequential(nn.Linear(de, 64), nn.ReLU(), nn.Linear(64, 2 * d))
                                                                 
                                                  
                                        
                                                                       
                                                       
                                             
                                                  
        s.fp_r = film_pos_rank
                                                  
                                                                           
                                      
                                                      
                                                     
                                    
        s.xattn = cross_attn
        if cross_attn:
            s.xln = nn.LayerNorm(hid // 2)
            s.xmha = nn.MultiheadAttention(hid // 2, 4, batch_first=True)
            s.beta_x = nn.Parameter(torch.zeros(1))
        if s.fp_r > 0:
            s.filmP = nn.Linear(de, L * 2 * s.fp_r)
            nn.init.zeros_(s.filmP.weight); nn.init.zeros_(s.filmP.bias)
            s.basisG = nn.Parameter(torch.randn(s.fp_r, d) * 0.02)
            s.basisB = nn.Parameter(torch.randn(s.fp_r, d) * 0.02)
        if use_metric:
            s.Wz = nn.Linear(hid // 2, zdim)
            s.Wv = nn.Linear(de, zdim)
            s.alpha = nn.Parameter(torch.zeros(1))                   

    def geom_sites(s):
        """(P,S,d) 亚位点表示。RBF 边、掩码求和；无结构的酶整体换成回退向量。"""
        GF, GD, GM, GH = s.geo
        h = s.gnode(GF)
        c = torch.linspace(2., 20., 16, device=GF.device)
        rbf = torch.exp(-((GD[..., None] - c) ** 2) / 4.)                      
        m = GM[:, None, :, None]
        for msg, upd in zip(s.gmsg, s.gupd):
            n_ = h.shape[1]
            pair = torch.cat([h[:, :, None].expand(-1, -1, n_, -1),
                              h[:, None].expand(-1, n_, -1, -1), rbf], -1)
            agg = (msg(pair) * m).sum(2) / m.sum(2).clamp_min(1.)
            h = h + upd(torch.cat([h, agg], -1))
        return torch.where(GH[:, None, None] > .5, h, s.gfall[None, None])

    def geom_ep(s):
        S_ = s.geom_sites(); m = s.geo[2][..., None]
        return s.gemb((S_ * m).sum(1) / m.sum(1).clamp_min(1.))

    def ep_vec(s):
        base = s.ep if s.Eproj is None else s.Eproj(s.Ep_raw)
        if s.gmode == "embed":
                                                 
                                              
                                            
            gp = s.geom_ep()
            h = (s.geo[3] > .5).float()[:, None]
            return h * gp + (1 - h) * base
        return base

    def body(s, e):
        i_, j_ = zip(*s.pairs)
        cat = torch.cat([e[:, list(i_)], e[:, list(j_)]], -1)
        u = torch.relu(torch.einsum("bpk,pkm->bpm", cat, s.Wp) + s.bp)
        gg = torch.sigmoid(s.z)[None, :, None]
        return s.trunk(torch.cat([e.flatten(1), (u * gg).flatten(1)], 1))

    def forward(s, Z, Zr=None, F=None, ret_r=False):
        Bn = Z.shape[0]
        e = torch.stack([s.E[i][Z[:, i]] for i in range(s.L)], 1)
        if s.esm:
            e = s.fuse(torch.cat([e, s.eproj(F[Zr].float())], -1))
        if s.lm is not None:
                                                      
            if s.lm_freeze:
                was = s.lm.training; s.lm.eval()
                with torch.no_grad():
                    zl = s.lm.encode(Z)
                if was:
                    s.lm.train()
            else:
                zl = s.lm.encode(Z)
            if s.lm_residual:
                                                      
                                                         
                                                
                e = e + s.alpha_lm * s.lmproj(zl)
            else:
                e = s.lmfuse(torch.cat([e, s.lmproj(zl)], -1))
        if s.ladder == "linear":
            xs = [s.lproj[k](F[Zr][:, k].float()) for k in range(len(s.lproj))]
            e = e + s.lmix(torch.cat(xs, -1))
        h0 = s.body(e); gg = s.g(h0)
        ev = s.ep_vec()
        if s.ladder in ("side", "cond"):
                                                 
                                           
                                               
            st = e[:, None].expand(-1, s.P, -1, -1).reshape(-1, s.d_e)
            for k in range(len(s.lproj)):
                x = s.lproj[k](F[Zr][:, k].float())                       
                if s.ladder == "cond":
                    gl, bl = s.lfilm[k](s.ep_vec()).chunk(2, -1)         
                    x = x[:, None] * (1 + gl[None, :, None]) + bl[None, :, None]
                else:
                    x = x[:, None].expand(-1, s.P, -1, -1)
                st = s.lupd[k](x.reshape(-1, s.d_e), st)
            hp = s.body(st.reshape(Bn * s.P, s.L, -1)).reshape(Bn, s.P, -1)
            R = s.rout(hp).squeeze(-1)
            r = None
            if s.use_metric:
                z = nn.functional.normalize(s.Wz(hp), dim=-1)
                v = nn.functional.normalize(s.Wv(ev), dim=-1)
                r = s.kappa * (z * v[None]).sum(-1)
                R = R + s.alpha * (r - r.mean(1, keepdim=True))
            Y = s.a[None, :] + gg + (R - R.mean(1, keepdim=True))
            return (Y, r) if ret_r else Y
        gam, bet = s.film(ev).chunk(2, -1)
        if s.fp_r > 0:
            c = s.filmP(ev).view(s.P, s.L, 2, s.fp_r)
            gm = gam[:, None] + c[:, :, 0] @ s.basisG                     
            bt = bet[:, None] + c[:, :, 1] @ s.basisB
            em = e[:, None] * (1 + gm[None]) + bt[None]
        else:
            em = e[:, None] * (1 + gam[None, :, None]) + bet[None, :, None]
        hp = s.body(em.reshape(Bn * s.P, s.L, -1)).reshape(Bn, s.P, -1)                
        if s.xattn:
            q = s.xln(hp)
            hp = hp + s.beta_x * s.xmha(q, q, q, need_weights=False)[0]
        R = s.rout(hp).squeeze(-1)
        if s.cdr != "off":
            Bn_, P_ = R.shape
            ev_ = s.ep_vec()                                                  
            if s.cdr == "plain":
                dd = hp[:, None].expand(-1, P_, -1, -1)                             
                de_ = ev_[None].expand(P_, -1, -1)                                  
            else:
                dd = hp[:, :, None] - hp[:, None]                                         
                de_ = ev_[:, None] - ev_[None]                                            
            x = torch.cat([dd, de_[None].expand(Bn_, -1, -1, -1)], -1)
            phi = s.cdr_phi(x)                                                   
            off_ = (1 - s.cdr_eye)[None, :, :, None]                          
            if s.cdr == "uniform":
                w = off_ / (P_ - 1)
            else:
                                              
                risk = (R[:, None, :] - R[:, :, None]).unsqueeze(-1)
                lg = s.cdr_w(torch.cat([x, risk], -1)).squeeze(-1)             
                lg = lg.masked_fill(s.cdr_eye[None] > .5, -1e9)
                                               
                w = ((1 - s.cdr_eps) * lg.softmax(-1)
                     + s.cdr_eps / (P_ - 1)).unsqueeze(-1) * off_
            zc = (w * phi).sum(2)                                              
            rc = s.cdr_psi(zc).squeeze(-1)                                   
            R = R + s.eta_cdr * (rc - rc.mean(1, keepdim=True))
        if s.gmode == "interact":
            S_ = s.geom_sites()                                                    
            q = s.gq(e); k = s.gk(S_); v = s.gv(S_)
            att = torch.einsum("bld,psd->bpls", q, k) / (q.shape[-1] ** .5)
            att = att.masked_fill(s.geo[2][None, :, None] < .5, -1e9).softmax(-1)
            ctx = torch.einsum("bpls,psd->bpld", att, v)
            R = R + s.beta_g * s.gout(ctx).squeeze(-1).mean(-1)
        r = None
        if s.use_metric:
            z = nn.functional.normalize(s.Wz(hp), dim=-1)                             
            v = nn.functional.normalize(s.Wv(ev), dim=-1)                           
            r = s.kappa * (z * v[None]).sum(-1)                                  
            R = R + s.alpha * (r - r.mean(1, keepdim=True))                      
        Y = s.a[None, :] + gg + (R - R.mean(1, keepdim=True))
        return (Y, r) if ret_r else Y

def condition(base, e):
    ev = base.ep_vec(); gam, bet = base.film(ev).chunk(2, -1)
    return e[:, None] * (1 + gam[None, :, None]) + bet[None, :, None]

def raw_scores(base, hp):
    raw = base.rout(hp).squeeze(-1)
    if base.use_metric:
        z = nn.functional.normalize(base.Wz(hp), dim=-1)
        v = nn.functional.normalize(base.Wv(base.ep_vec()), dim=-1)
        raw = raw + base.alpha * base.kappa * (z * v[None]).sum(-1)
    return raw

class PocketPair(nn.Module):
    def __init__(self, base, geometry, mode, width=32):
        super().__init__()
        self.base = base
        gf, gd, gm, gh = [x.clone() for x in geometry]
        if mode == 'sequence': gf.zero_(); gd.zero_()
        if mode == 'identity': gf[:, :, 21:] = 0; gd.zero_()
                                                                           
                                                                             
        if mode == 'geometry':
            gf[:, :, 21:23] /= 20.; gf[:, :, 24:27] /= 20.
        for name, arr in zip(('gf', 'gd', 'gm', 'gh'), (gf, gd, gm, gh)):
            self.register_buffer(name, arr)
        self.node = nn.Sequential(nn.Linear(27, width), nn.LayerNorm(width), nn.GELU())
        self.msg = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * width, width),
                                               nn.LayerNorm(width), nn.GELU()) for _ in range(2)])
        self.q = nn.Sequential(nn.Linear(2 * base.d_e, width), nn.LayerNorm(width), nn.GELU())
        self.key = nn.Linear(width, width); self.value = nn.Linear(width, width)
        self.combine = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(),
                                     nn.Linear(width, base.Wp.shape[-1]))
        nn.init.zeros_(self.combine[-1].weight); nn.init.zeros_(self.combine[-1].bias)
        ii, jj = zip(*base.pairs)
        self.register_buffer('ii', torch.tensor(ii)); self.register_buffer('jj', torch.tensor(jj))
        self.width = width

    def train(self, mode=True):
        super().train(mode); self.base.eval(); return self

    def adapter_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if not k.startswith('base.')}

    def sites(self):
        h = self.node(self.gf) * self.gm[..., None]
                                                                                     
        w = torch.exp(-self.gd.square() / 64.) * self.gm[:, None]
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-6)
        for msg, upd in zip(self.msg, self.upd):
            agg = w @ msg(h)
            h = (h + upd(torch.cat([h, agg], -1))) * self.gm[..., None]
        return h

    def forward(self, e, baseline_y, baseline_raw):
        b = self.base; em = condition(b, e)
        pair = torch.cat([em[:, :, self.ii], em[:, :, self.jj]], -1)
        u = torch.relu(torch.einsum('btpk,pkm->btpm', pair, b.Wp) + b.bp)
        q = self.q(pair); sites = self.sites()
        att = torch.einsum('btpd,tsd->btps', q, self.key(sites)) / self.width ** .5
        att = att.masked_fill(self.gm[None, :, None] < .5, -1e4).softmax(-1)
        ctx = torch.einsum('btps,tsd->btpd', att, self.value(sites))
        delta = .25 * torch.tanh(self.combine(torch.cat([q, ctx, q * ctx], -1)))
        u = u * (1 + delta * self.gh[None, :, None, None])
        u = u * torch.sigmoid(b.z)[None, None, :, None]
        h = b.trunk(torch.cat([em.flatten(2), u.flatten(2)], -1))
        raw = raw_scores(b, h)
                                                                                 
                                                                                 
        return baseline_y + (raw - baseline_raw) * self.gh[None]

class AblatedBase(R8Net):
    """Peptide backbone; released models use the full configuration."""
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

class AblatedAdapter(PocketPair):
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
