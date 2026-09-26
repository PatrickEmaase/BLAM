import argparse, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
def make_data(c,n,seed,sigma=3.0):
    rng=np.random.default_rng(seed); X=rng.normal(0,sigma,size=(n,2))
    th=np.arctan2(X[:,1],X[:,0])%(2*np.pi)
    return X.astype(np.float32),(np.floor(th/(np.pi/c)).astype(np.int64))%c
class Embed(nn.Module):
    def __init__(s,w,M):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(2,w),nn.LeakyReLU(0.01),nn.Linear(w,w),
            nn.LeakyReLU(0.01),nn.Linear(w,w),nn.LeakyReLU(0.01),nn.Linear(w,M))
    def forward(s,x): return s.net(x)
p=argparse.ArgumentParser()
for a,t,d in [('--classes',int,6),('--width',int,64),('--dim',int,2),('--seed',int,0),
  ('--smooth',float,0.0),('--steps',int,15000),('--bs',int,128),('--lr',float,1e-3),
  ('--n',int,20000)]: p.add_argument(a,type=t,default=d)
p.add_argument('--out',default='runs/synth'); a=p.parse_args()
os.makedirs(a.out,exist_ok=True); torch.manual_seed(a.seed)
dev='cuda' if torch.cuda.is_available() else 'cpu'
X,y=make_data(a.classes,a.n,0)
Xt=torch.tensor(X,device=dev); yt=torch.tensor(y,device=dev)
emb=Embed(a.width,a.dim).to(dev); G=nn.Parameter(torch.randn(a.classes,a.dim,device=dev))
opt=torch.optim.Adam(list(emb.parameters())+[G],lr=a.lr)
g=torch.Generator().manual_seed(a.seed+999)
for t in range(a.steps):
    i=torch.randint(0,a.n,(a.bs,),generator=g).to(dev)
    loss=Fn.cross_entropy(emb(Xt[i])@G.T,yt[i],label_smoothing=a.smooth)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    f=emb(Xt); lg=f@G.T; lp=lg-torch.logsumexp(lg,1,keepdim=True)
    f,Gn,lp=f.cpu().numpy(),G.detach().cpu().numpy(),lp.cpu().numpy()
acc=float((lp.argmax(1)==y).mean()); ls=float(-lp[np.arange(len(y)),y].mean())
tag=f"c{a.classes}_w{a.width}_d{a.dim}_s{a.seed}_sm{a.smooth}"
np.savez_compressed(os.path.join(a.out,tag+'.npz'),f=f.astype('f4'),g=Gn.astype('f4'),
    logp=lp.astype('f4'),y=y.astype('i8'),acc=acc,loss=ls,dim=a.dim,seed=a.seed,
    smooth=a.smooth,width=a.width,nclass=a.classes)
print(json.dumps(dict(tag=tag,acc=acc,loss=ls)),flush=True)
