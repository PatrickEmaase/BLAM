"""Why did option 2 produce no valid configurations? Count rejection reasons."""
import numpy as np, traceback
from a3 import make_pair, gaps_and_margin, alpha_star_m, dfg

for irreg, nu in [(False,False),(False,True),(True,False),(True,True)]:
    cnt = dict(exception=0, argmax=0, few_inputs=0, nonfinite=0,
               low_dfg=0, accepted=0)
    exmsg=None
    okvals=[]
    for trial in range(120):
        rng=np.random.default_rng(2000+10*int(irreg)+int(nu)+7*trial)
        k=int(rng.choice([8,12,18,26])); sd=int(rng.choice([1,2,3]))
        sp=float(rng.choice([0.12,0.2,0.3]))
        try:
            F1,G,F2,Gp,lab=make_pair(2,k,24,rng,spread=sp,irregular=irreg,
                                     nonuniform=nu,swap_dist=sd)
            a,b,off,margin,ok=gaps_and_margin(F1,G,F2,Gp)
            okvals.append(ok.mean())
            if ok.mean()<0.9: cnt['argmax']+=1; continue
            A,ns=alpha_star_m(a,b,off,margin,ok,0.10)
            if ns<20: cnt['few_inputs']+=1; continue
            if not np.isfinite(A): cnt['nonfinite']+=1; continue
            d=dfg(F1,G,F2,Gp,lab,2)
            if d<0.02: cnt['low_dfg']+=1; continue
            cnt['accepted']+=1
        except Exception as e:
            cnt['exception']+=1
            if exmsg is None: exmsg=f"{type(e).__name__}: {e}"
    print(f"irregular={irreg!s:<5} nonuniform={nu!s:<5}  {cnt}")
    print(f"   mean shared-argmax fraction = {np.mean(okvals):.3f}"
          f"  (min {np.min(okvals):.3f})")
    if exmsg: print(f"   first exception: {exmsg}")
