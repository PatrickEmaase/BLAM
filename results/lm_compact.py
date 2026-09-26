import argparse, os, json, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn

def corpus(path, n=400000, seed=0):
    if path and os.path.exists(path):
        return open(path, encoding='utf-8', errors='ignore').read()
    rng = np.random.default_rng(seed); V = 96
    T = rng.dirichlet(np.ones(V) * 0.3, size=(V,)); out, prev = [], 0
    for _ in range(n):
        prev = int(rng.choice(V, p=T[prev])); out.append(prev)
    return ''.join(chr(32 + c) for c in out)

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.att = nn.MultiheadAttention(d, h, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x, m):
        h = s.ln1(x); a, _ = s.att(h, h, h, attn_mask=m, need_weights=False)
        x = x + a; return x + s.mlp(s.ln2(x))

class TiedLM(nn.Module):
    def __init__(s, V, d, nl, nh, ctx):
        super().__init__()
        s.emb = nn.Embedding(V, d); s.pos = nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)]); s.ln = nn.LayerNorm(d)
    def feats(s, idx):
        T = idx.shape[1]
        x = s.emb(idx) + s.pos(torch.arange(T, device=idx.device))
        m = torch.triu(torch.full((T, T), float('-inf'), device=idx.device), 1)
        for b in s.blocks: x = b(x, m)
        return s.ln(x)

p = argparse.ArgumentParser()
for a, t, d in [('--dim',int,64),('--layers',int,4),('--heads',int,4),('--ctx',int,128),
                ('--seed',int,0),('--steps',int,8000),('--bs',int,32),('--lr',float,3e-4),
                ('--n_eval',int,4000)]: p.add_argument(a, type=t, default=d)
p.add_argument('--text', default=None); p.add_argument('--out', default='runs/lm')
p.add_argument('--smoke', action='store_true'); a = p.parse_args()
if a.smoke: a.steps = 50
if a.dim % a.heads: a.heads = 1          # heads must divide dim
os.makedirs(a.out, exist_ok=True)

text = corpus(a.text); chars = sorted(set(text)); stoi = {c:i for i,c in enumerate(chars)}
data = np.array([stoi[c] for c in text], dtype=np.int64); V = len(chars)
ntr = int(.9*len(data)); tr, te = data[:ntr], data[ntr:]
print(f"vocab {V}  diversity needs V>M={a.dim}: {'OK' if V>a.dim else 'FAILS'}", flush=True)
if V <= a.dim: raise SystemExit("diversity fails: use a larger corpus/vocab or smaller --dim")

torch.manual_seed(a.seed); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = TiedLM(V, a.dim, a.layers, a.heads, a.ctx).to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr); g = torch.Generator().manual_seed(a.seed+7)
def batch(src, bs):
    i = torch.randint(0, len(src)-a.ctx-1, (bs,), generator=g)
    x = torch.stack([torch.from_numpy(src[j:j+a.ctx]) for j in i])
    y = torch.stack([torch.from_numpy(src[j+1:j+a.ctx+1]) for j in i])
    return x.to(dev), y.to(dev)
m.train()
for t in range(1, a.steps+1):
    x, y = batch(tr, a.bs)
    loss = Fn.cross_entropy((m.feats(x) @ m.emb.weight.T).reshape(-1,V), y.reshape(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if t % 2000 == 0: print(f"  step {t}  loss {loss.item():.4f}  ppl {math.exp(loss.item()):.2f}", flush=True)

m.eval(); F_, LP, Y = [], [], []
with torch.no_grad():
    W = m.emb.weight; need = a.n_eval
    while need > 0:
        b = min(64, need)
        i = torch.randint(0, len(te)-a.ctx-1, (b,), generator=g)
        x = torch.stack([torch.from_numpy(te[j:j+a.ctx]) for j in i]).to(dev)
        yt = torch.stack([torch.from_numpy(te[j+a.ctx:j+a.ctx+1]) for j in i]).squeeze(1)
        f = m.feats(x)[:, -1, :]; lg = f @ W.T
        LP.append((lg - torch.logsumexp(lg,1,keepdim=True)).cpu().numpy())
        F_.append(f.cpu().numpy()); Y.append(yt.numpy()); need -= b
f = np.concatenate(F_); lp = np.concatenate(LP); y = np.concatenate(Y)
acc = float((lp.argmax(1)==y).mean()); nll = float(-lp[np.arange(len(y)),y].mean())
tag = f"lm_d{a.dim}_s{a.seed}"
np.savez_compressed(os.path.join(a.out, tag+'.npz'), f=f.astype('f4'),
    g=m.emb.weight.detach().cpu().numpy().astype('f4'), logp=lp.astype('f4'),
    y=y.astype('i8'), acc=acc, loss=nll, dim=a.dim, seed=a.seed, smooth=0.0,
    width=0, nclass=V)
print(json.dumps(dict(tag=tag, acc=acc, nll=nll, ppl=math.exp(nll))), flush=True)
