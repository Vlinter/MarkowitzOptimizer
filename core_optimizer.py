# core_optimizer.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Optionnel: Ledoit–Wolf (scikit-learn)
try:
    from sklearn.covariance import LedoitWolf
    HAS_LW = True
except Exception:
    HAS_LW = False

# ---- Fonctions de base du portefeuille ----
def portfolio_perf(w, mu, cov):
    ret = float(w @ mu)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    return ret, vol

def risk_contrib(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    tot_var = float(w @ cov @ w)
    if tot_var <= 1e-12: return np.zeros_like(w) # Éviter la division par zéro
    mrc = cov @ w
    return (w * mrc) / tot_var

# ---- Solver et Helpers ----
def feasible_start(n: int, min_w: float, max_w: float) -> np.ndarray:
    if min_w < 0 or max_w <= 0 or min_w > max_w: raise ValueError("Bornes incohérentes.")
    w = np.full(n, min_w, dtype=float); R = 1.0 - n * min_w
    if R < -1e-12: raise ValueError("Bornes infaisables : min_w * n > 1.")
    if R <= 1e-12: return w
    cap = np.maximum(0.0, max_w - w)
    while R > 1e-12:
        idx = np.where(cap > 1e-12)[0]
        if idx.size == 0: break
        inc = R / idx.size
        add = np.minimum(cap[idx], inc)
        w[idx] += add; cap[idx] -= add; R -= float(add.sum())
    np.clip(w, min_w, max_w, out=w)
    s = w.sum()
    if abs(s - 1.0) > 1e-10:
        w /= s; np.clip(w, min_w, max_w, out=w)
    return w

def _solve_slsqp(objective, n, bounds, cons, w0=None, maxiter=2000, ftol=1e-10):
    if w0 is None: w0 = feasible_start(n, bounds[0][0], bounds[0][1])
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': maxiter, 'ftol': ftol, 'disp': False})
    if res.success: return res.x
    w0b = np.ones(n)/n
    res2 = minimize(objective, w0b, method='SLSQP', bounds=bounds, constraints=cons,
                    options={'maxiter': maxiter*2, 'ftol': ftol*10, 'disp': False})
    if res2.success: return res2.x
    try:
        w_fallback = feasible_start(n, bounds[0][0], bounds[0][1])
        return w_fallback
    except ValueError:
        return np.ones(n) / n

# ---- Stratégies d'optimisation ----
# (mu, cov, min_w, max_w, rf=0.0, tickers=None)
# Toutes les stratégies reçoivent les mêmes arguments pour avoir une API cohérente
def max_sharpe_weights(mu, cov, min_w, max_w, rf=0.0, tickers=None): 
    n = len(mu); bounds = [(min_w, max_w)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0 = feasible_start(n, min_w, max_w)
    def obj(w):
        ret, vol = portfolio_perf(w, mu, cov)
        return 1e6 if vol <= 1e-12 else - (ret - rf) / vol
    return _solve_slsqp(obj, n, bounds, cons, w0=w0)

def min_var_weights(mu, cov, min_w, max_w, rf=0.0, tickers=None):
    n = len(mu); bounds = [(min_w, max_w)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0 = feasible_start(n, min_w, max_w)
    return _solve_slsqp(lambda w: float(w @ cov @ w), n, bounds, cons, w0=w0)

def max_return_weights(mu, cov, min_w, max_w, rf=0.0, tickers=None):
    n = len(mu); bounds = [(min_w, max_w)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0 = feasible_start(n, min_w, max_w)
    return _solve_slsqp(lambda w: - float(w @ mu), n, bounds, cons, w0=w0)

def risk_parity_weights(mu, cov, min_w, max_w, rf=0.0, tickers=None):
    n = cov.shape[0]
    bounds = [(min_w, max_w)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    w0 = feasible_start(n, min_w, max_w)
    def obj_rp(w, cov):
        port_var = float(w @ cov @ w)
        if port_var <= 1e-12:
            return 0.0
        rc_pct = (w * (cov @ w)) / port_var
        return np.std(rc_pct)
    return _solve_slsqp(lambda w: obj_rp(w, cov), n, bounds, cons, w0=w0)

# ---- [SUPPRIMÉ] HRP (Hierarchical Risk Parity) a été supprimé. ----


# ---- Frontière Efficiente ----
def _min_var_for_target_return(mu, cov, r_target, min_w, max_w, w_start=None):
    n = len(mu); bounds = [(min_w, max_w)] * n
    cons = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w, rt=r_target: float(w @ mu) - rt}
    ]
    if w_start is None:
        w_start = feasible_start(n, min_w, max_w)
        w_start = np.clip(w_start + 0.01 * (mu / (np.linalg.norm(mu)+1e-12)), min_w, max_w)
        w_start = w_start / w_start.sum()
    def obj(w): return float(w @ cov @ w)
    return _solve_slsqp(obj, n, bounds, cons, w0=w_start, maxiter=3000, ftol=1e-12)

def efficient_frontier(mu, cov, min_w, max_w, npts=160, eps_keep=1e-10):
    try:
        w_mv = min_var_weights(mu, cov, min_w, max_w); r_mv, _ = portfolio_perf(w_mv, mu, cov)
        w_mr = max_return_weights(mu, cov, min_w, max_w); r_mr, _ = portfolio_perf(w_mr, mu, cov)
    except Exception:
        return np.empty((0, 2))
    r_grid = np.linspace(r_mv, r_mr, npts)
    pts = []; w_prev = None
    for rt in r_grid:
        try:
            w_t = _min_var_for_target_return(mu, cov, rt, min_w, max_w, w_start=w_prev)
            r_t, v_t = portfolio_perf(w_t, mu, cov)
            pts.append((r_t, v_t)); w_prev = w_t
        except Exception:
            continue
    if not pts: return np.empty((0, 2))
    pts = np.array(pts); order = np.argsort(pts[:, 1]); pts = pts[order]
    cleaned = [pts[0]]
    for r, v in pts[1:]:
        if r >= cleaned[-1][0] - eps_keep:
            cleaned.append([r, v])
    return np.array(cleaned)

# ---- Monte Carlo ----
def sample_bounded_simplex(n: int, N: int, min_w: float, max_w: float, rng: np.random.Generator) -> np.ndarray:
    base = np.full(n, min_w, dtype=float)
    cap = np.full(n, max_w - min_w, dtype=float)
    Rtot = 1.0 - n * min_w
    if Rtot < -1e-12: raise ValueError("Bornes infaisables : min_w * n > 1.")
    if Rtot <= 1e-12: return np.tile(base, (N, 1))
    alpha = np.where(cap > 0, cap, 1e-12)
    W = np.empty((N, n), dtype=float)
    for k in range(N):
        p = rng.dirichlet(alpha)
        add = np.minimum(p * Rtot, cap)
        R = Rtot - float(add.sum())
        free = cap - add
        while R > 1e-12 and (free > 1e-12).any():
            q = rng.dirichlet(np.where(free > 1e-12, free, 1e-12))
            inc = np.minimum(free, q * R)
            add += inc; R -= float(inc.sum()); free = cap - add
        W[k, :] = base + add
    W = np.clip(W, min_w, max_w); s = W.sum(axis=1, keepdims=True); W = W / s
    return W