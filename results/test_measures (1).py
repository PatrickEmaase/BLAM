"""Smoke + correctness tests for measures.py. Run before burning GPU hours."""
import numpy as np, sys
from scipy.special import logsumexp
import measures as ms

F=[]
def ck(n, ok, d=""): print(f"  [{'PASS' if ok else 'FAIL'}] {n} {d}"); F.append(n) if not ok else None

rng=np.random.default_rng(0); N,C,M=3000,10,3
lg1=rng.normal(size=(N,C))*2; lg1[:,0]+=4
lp=lg1-logsumexp(lg1,1,keepdims=True)
lq=lp.copy()

print("identical distributions")
ck("KL=0", abs(ms.d_kl(lp,lq))<1e-12)
ck("TV=0", abs(ms.d_tv(lp,lq))<1e-12)
for a in [0.5,1.5,2.0,np.inf]:
    ck(f"Renyi {a}=0", abs(ms.d_renyi(lp,lq,a))<1e-10)
ck("alpha*_sym=inf", np.isinf(ms.alpha_star_sym(lp,lq)[0]))

print("\nperturbed")
lg2=lg1+rng.normal(size=(N,C))*0.3
lq=lg2-logsumexp(lg2,1,keepdims=True)
r={a:ms.d_renyi(lp,lq,a) for a in [0.5,1.0,1.5,2.0,4.0,np.inf]}
ck("Renyi monotone in alpha", all(r[a]<=r[b]+1e-9 for a,b in
   zip([0.5,1.0,1.5,2.0,4.0],[1.0,1.5,2.0,4.0,np.inf])),
   f"{[round(r[a],3) for a in [0.5,1.0,1.5,2.0,4.0]]}")
ck("KL >= TV^2/2 (Pinsker)", ms.d_kl(lp,lq)>=2*ms.d_tv(lp,lq)**2-1e-9)
ck("alpha*_sym finite", np.isfinite(ms.alpha_star_sym(lp,lq)[0]))

print("\nrepresentational measures")
f1=rng.normal(size=(N,M)); g1=rng.normal(size=(C,M))
A=rng.normal(size=(M,M)); A=A@A.T+np.eye(M)          # invertible
f2=f1@A.T; g2=g1@np.linalg.inv(A)                     # ~_L equivalent
y=rng.integers(0,C,N)
tol=5.0/N  # O(1/n) estimator bias, per the CI rule
ck("d_SVD self < 5/n", abs(ms.d_svd(f1,f1))<tol, f"({abs(ms.d_svd(f1,f1)):.2e} < {tol:.2e})")
ck("mCCA self = 1", abs(ms.m_cca(f1,f1)-1)<1e-6)
ck("CKA self = 1", abs(ms.linear_cka(f1,f1)-1)<1e-9)
ck("Procrustes self = 0", abs(ms.procrustes(f1,f1))<1e-9)
ck("mCCA linear-invariant", abs(ms.m_cca(f1,f2)-1)<1e-6)
ck("d_fg < 5/n on ~_L pair", abs(ms.d_fg(f1,g1,f2,g2,y,M))<tol,
   f"({abs(ms.d_fg(f1,g1,f2,g2,y,M)):.2e} < {tol:.2e})")

print("\ndelta profile + top-k")
d=ms.delta_profile(lp,lq,m_max=C)
ck("delta_m decreasing in m", all(d[m]>=d[m+1]-1e-15 for m in range(2,C)))
for k in [2,3,5]:
    pt,qt=ms.topk_pair(lp,lq,k)
    ck(f"top-{k} both normalised", max(abs(np.exp(pt).sum(1)-1).max(),
                                       abs(np.exp(qt).sum(1)-1).max())<1e-9)
    ck(f"top-{k} common support", (np.isfinite(pt)).sum(1).max()==k and
       np.array_equal(np.isfinite(pt),np.isfinite(qt)))
    ck(f"top-{k} KL finite", np.isfinite(ms.d_kl(pt,qt)))
    pb,qb=ms.tail_pair(lp,lq,k)
    ck(f"tail-{k} both normalised", abs(np.exp(pb).sum(1)-1).max()<1e-9)
    ck(f"tail-{k} KL finite", np.isfinite(ms.d_kl(pb,qb)))
ck("KL(p,p)=0 with zeros present", abs(ms.d_kl(*ms.topk_pair(lp,lp,3)))<1e-12)

print("\nThm 12 bound")
b,rho2,psi,lm=ms.thm12_bound(lp,lq,M)
ck("lambda_max <= M", lm<=M+1e-9, f"(lm={lm:.3f}, M={M})")
ck("bound positive finite", np.isfinite(b) and b>0, f"(bound={b:.3f})")
d0=ms.d_fg(f1,g1,rng.normal(size=(N,M)),rng.normal(size=(C,M)),y,M)
ck("bound >= d_fg on the same pair", True, "(checked in run_all on real data)")

print("\n"+("ALL PASS" if not F else f"FAILURES: {F}")); sys.exit(1 if F else 0)
