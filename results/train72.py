"""
Section 7.2 -- synthetic angular classification, run for real.

Data      2D Gaussian (sigma=3), c classes, each a circle slice plus its
          opposite (App. F.2 of the prior work).
Model     embedding net: 3 fully connected LeakyReLU layers, 2 -> w -> w -> w -> 2;
          unembedding: a free c x 2 table (the model class only constrains g
          through its values, so a table is equivalent).
Training  Adam, batch 128, cross-entropy, optional label smoothing.

Arms
  A  width sweep   w in {16,32,64,128,256}, c in {4,6}, 6 seeds
  B  smoothing     s in {0,0.05,0.1,0.2},   c = 6, w = 64, 6 seeds
"""

import numpy as np

# ------------------------------------------------------------------ data ----
def make_data(c, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 3, size=(n, 2))
    th = np.arctan2(X[:, 1], X[:, 0]) % (2 * np.pi)
    y = (np.floor(th / (np.pi / c)).astype(int)) % c
    return X, y


# ----------------------------------------------------------------- model ----
def init(w, c, M, seed):
    rng = np.random.default_rng(seed)
    def he(a, b):
        return rng.normal(0, np.sqrt(2.0 / a), size=(a, b))
    return dict(
        W1=he(2, w), b1=np.zeros(w),
        W2=he(w, w), b2=np.zeros(w),
        W3=he(w, M), b3=np.zeros(M),
        G=rng.normal(0, 1.0, size=(c, M)))


def lrelu(z, a=0.01):
    return np.where(z > 0, z, a * z)


def dlrelu(z, a=0.01):
    return np.where(z > 0, 1.0, a)


def embed(P, X):
    z1 = X @ P['W1'] + P['b1']; h1 = lrelu(z1)
    z2 = h1 @ P['W2'] + P['b2']; h2 = lrelu(z2)
    f = h2 @ P['W3'] + P['b3']
    return f, (z1, h1, z2, h2)


def logprob(P, X):
    f, _ = embed(P, X)
    lg = f @ P['G'].T
    return lg - np.log(np.exp(lg - lg.max(1, keepdims=True)).sum(1, keepdims=True)) \
           - lg.max(1, keepdims=True)


def train(X, y, c, w, seed, smooth=0.0, steps=3000, bs=128, lr=3e-3, M=2):
    P = init(w, c, M, seed)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(v) for k, v in P.items()}
    rng = np.random.default_rng(seed + 999)
    n = len(X)
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, bs)
        xb, yb = X[idx], y[idx]
        f, (z1, h1, z2, h2) = embed(P, xb)
        lg = f @ P['G'].T
        lg -= lg.max(1, keepdims=True)
        p = np.exp(lg); p /= p.sum(1, keepdims=True)
        T = np.full((bs, c), smooth / c)
        T[np.arange(bs), yb] += 1 - smooth
        dlg = (p - T) / bs
        gG = dlg.T @ f
        gf = dlg @ P['G']
        gW3 = h2.T @ gf; gb3 = gf.sum(0)
        gh2 = gf @ P['W3'].T; gz2 = gh2 * dlrelu(z2)
        gW2 = h1.T @ gz2; gb2 = gz2.sum(0)
        gh1 = gz2 @ P['W2'].T; gz1 = gh1 * dlrelu(z1)
        gW1 = xb.T @ gz1; gb1 = gz1.sum(0)
        g = dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2, W3=gW3, b3=gb3, G=gG)
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            mh = m[k] / (1 - 0.9 ** t); vh = v[k] / (1 - 0.999 ** t)
            P[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return P


def evaluate(P, X, y):
    lp = logprob(P, X)
    acc = (lp.argmax(1) == y).mean()
    loss = -lp[np.arange(len(y)), y].mean()
    return acc, loss


# ------------------------------------------------------------------ run -----
if __name__ == "__main__":
    import pickle
    out = {}

    print("ARM A: width sweep")
    for c in [4, 6]:
        X, y = make_data(c, seed=0)
        for w in [16, 32, 64, 128, 256]:
            for s in range(6):
                P = train(X, y, c, w, seed=s)
                acc, loss = evaluate(P, X, y)
                out[('A', c, w, 0.0, s)] = (P, acc, loss)
            accs = [out[('A', c, w, 0.0, s)][1] for s in range(6)]
            lss = [out[('A', c, w, 0.0, s)][2] for s in range(6)]
            print(f"  c={c} w={w:>3}  acc {np.mean(accs):.3f}"
                  f"  loss {np.mean(lss):.4f}")

    print("\nARM B: label smoothing (c=6, w=64)")
    X, y = make_data(6, seed=0)
    for sm in [0.0, 0.05, 0.1, 0.2]:
        for s in range(6):
            P = train(X, y, 6, 64, seed=100 + s, smooth=sm)
            acc, loss = evaluate(P, X, y)
            out[('B', 6, 64, sm, s)] = (P, acc, loss)
        accs = [out[('B', 6, 64, sm, s)][1] for s in range(6)]
        print(f"  smooth={sm:<5} acc {np.mean(accs):.3f}")

    with open("models.pkl", "wb") as fh:
        pickle.dump(out, fh)
    print(f"\nsaved {len(out)} models")
