import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from a3 import make_pair, gaps_and_margin, alpha_star_m

plt.rcParams.update({"font.size":8,"axes.linewidth":.6,"legend.frameon":False,
                     "xtick.major.width":.6,"ytick.major.width":.6})

curves=[]
for M,k,c,ls in [(2,12,"#d1495b","-"),(3,14,"#1b6ca8","-"),(4,16,"#2e4057","-")]:
    rng=np.random.default_rng(3)
    F1,G,F2,Gp,lab=make_pair(M,k,80,rng,spread=0.20)
    a,b,off,margin,ok=gaps_and_margin(F1,G,F2,Gp)
    fin=margin[np.isfinite(margin)]
    ms=np.linspace(0,np.percentile(fin,92),22)
    A=[];MM=[]
    for m in ms:
        v,ns=alpha_star_m(a,b,off,margin,ok,m)
        if np.isfinite(v) and ns>=25: A.append(v);MM.append(m/fin.max())
    curves.append((MM,A,c,ls,f"$M={M}$"))
np.save("panelc.npy",np.array(curves,dtype=object),allow_pickle=True)
print("curves:",[(len(x[0]),max(x[1])) for x in curves])
