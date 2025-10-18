import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import deque
import itertools

# --------- Utils ---------
def robust_mad(x, eps=1e-12):
    med = np.nanmedian(x)
    return med, np.nanmedian(np.abs(x - med)) + eps

def z_mad(x, ref_med, ref_mad, eps=1e-12):
    return (x - ref_med) / max(ref_mad, eps)

# ========= PERMUTATION ENTROPY =========
def permutation_entropy(series: np.ndarray, m=5, tau=1, window=300):
    """Bandt–Pompe PE normalized to [0,1] over a rolling window."""
    x = np.asarray(series[-(window + (m - 1) * tau):], float)
    if x.size < m or np.nanstd(x) < 1e-14:
        return np.nan

    patterns = []
    jitter_base = 1e-12
    for start in range(0, len(x) - (m - 1) * tau):
        v = x[start:start + m * tau:tau]
        if np.all(np.isfinite(v)):
            jitter = jitter_base * np.arange(len(v))
            ranks = np.argsort(np.argsort(v + jitter))
            patterns.append(tuple(ranks))
    if not patterns:
        return np.nan

    total = int(np.math.factorial(m))
    perm_index = {perm: i for i, perm in enumerate(itertools.permutations(range(m)))}
    freqs = np.zeros(total, float)
    for p in patterns:
        freqs[perm_index[p]] += 1.0

    p = freqs / max(freqs.sum(), 1.0)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = -(p * np.log(p + 1e-12)).sum()
    return float(H / np.log(total))

class PEStateMachine:
    """Compressed / Transition / Chaotic with hysteresis from PE."""
    def __init__(self, p20, p80, med, mad, on_z=1.5, off_z=0.5, on_min=3, off_min=5):
        self.p20, self.p80 = p20, p80
        self.med, self.mad = med, mad
        self.on_z, self.off_z = on_z, off_z
        self.on_min, self.off_min = on_min, off_min
        self.state = "Compressed"
        self._counter = 0

    def step(self, pe_value):
        if not np.isfinite(pe_value):
            return self.state

        z = z_mad(pe_value, self.med, self.mad)
        prev = self.state

        if self.state == "Compressed":
            if z >= self.on_z or pe_value >= self.p80:
                self._counter += 1
                if self._counter >= self.on_min:
                    self.state = "Chaotic"
                    self._counter = 0
            else:
                self._counter = 0

        elif self.state == "Chaotic":
            if z <= self.off_z and pe_value <= self.p80:
                self._counter += 1
                if self._counter >= self.off_min:
                    self.state = "Compressed"
                    self._counter = 0
            else:
                self._counter = 0

        else:
            self.state = "Compressed"  # default fallback

        return self.state

# ========= LAPLACE KERNEL + KF =========
def laplace_kernel(L: int, sigma: float):
    n = np.arange(-L, L + 1, dtype=float)
    k = np.exp(-np.abs(n) / max(sigma, 1e-9))
    k /= k.sum()
    return k

@dataclass
class SSConfig:
    model: str = "ar1"  # "local" or "ar1"
    phi: float = 0.6
    q: float = 1.0      # process noise var
    r: float = 2.0      # measurement noise var

def kalman_filter(y: np.ndarray, cfg: SSConfig):
    n = len(y)
    m = np.zeros(n); P = np.zeros(n)
    m_prev = float(y[0]) if np.isfinite(y[0]) else 0.0
    P_prev = 10.0
    for t in range(n):
        a = m_prev if cfg.model == "local" else cfg.phi * m_prev
        Rv = (P_prev + cfg.q) if cfg.model == "local" else (cfg.phi**2 * P_prev + cfg.q)
        K = Rv / (Rv + cfg.r)
        yt = float(y[t]) if np.isfinite(y[t]) else a
        m_t = a + K * (yt - a)
        P_t = (1 - K) * Rv
        m[t], P[t] = m_t, P_t
        m_prev, P_prev = m_t, P_t
    return m, P

def kf_predict_means_vars(m_t, P_t, cfg: SSConfig, H: int):
    means = np.zeros(H); vars_ = np.zeros(H)
    mt, Pt = m_t, P_t
    for h in range(1, H + 1):
        if cfg.model == "local":
            mt = mt
            Pt = Pt + cfg.q
        else:
            mt = cfg.phi * mt
            Pt = cfg.phi**2 * Pt + cfg.q
        means[h - 1] = mt
        vars_[h - 1] = Pt + cfg.r
    return means, vars_

# ========= KF-GUIDED INDEPENDENCE MH (optional) =========
def logpdf_student_t(x, loc, scale, nu):
    z = (x - loc) / scale
    # constant term per element:
    c = np.log(np.exp(np.log(np.math.gamma((nu+1)/2)) - np.log(np.math.gamma(nu/2))) /
               (np.sqrt(nu*np.pi) * scale))
    return c - 0.5*(nu+1)*np.log1p((z**2)/nu)

def mh_future_paths_kf(means, variances, N=150, burn=60, nu=5.0, rng=None):
    if rng is None: rng = np.random.default_rng(123)
    L = len(means)
    std = np.sqrt(np.maximum(variances, 1e-12))

    # proposal q: N(means, vars)
    def log_q(path):
        z = (path - means) / std
        return -0.5*np.sum(z**2 + np.log(2*np.pi*variances))

    # target p: Student-t centered at means, match variance roughly
    scale = std * np.sqrt((nu - 2) / nu) if nu > 2 else std * 0.7
    def log_p(path):
        return np.sum(logpdf_student_t(path, means, scale, nu))

    cur = means.copy()
    cur_lp, cur_lq = log_p(cur), log_q(cur)
    samples = []
    for i in range(burn + N):
        prop = rng.normal(means, std)
        prop_lp, prop_lq = log_p(prop), log_q(prop)
        log_alpha = (prop_lp + cur_lq) - (cur_lp + prop_lq)
        if np.log(rng.random()) < log_alpha:
            cur, cur_lp, cur_lq = prop, prop_lp, prop_lq
        if i >= burn:
            samples.append(cur.copy())
    return np.vstack(samples)

# ========= FULL LAPLACE MEAN (KF, with optional MH) =========
def full_laplace_mean_series(x: pd.Series, L=8, sigma_kernel=5.0,
                             cfg=SSConfig(), use_mh=False, mh_n=150, mh_burn=60, mh_nu=5.0):
    xv = x.astype(float).to_numpy()
    n = len(xv)
    mu = np.full(n, np.nan); resid = np.full(n, np.nan)
    m, P = kalman_filter(xv, cfg)
    k = laplace_kernel(L, sigma_kernel)

    rng = np.random.default_rng(2024)
    for t in range(L, n):
        past = xv[t - L:t]
        x_t = xv[t]
        means, vars_ = kf_predict_means_vars(m[t], P[t], cfg, H=L)
        if use_mh:
            fut_paths = mh_future_paths_kf(means, vars_, N=mh_n, burn=mh_burn, nu=mh_nu, rng=rng)
            fut_mean = fut_paths.mean(axis=0)
        else:
            fut_mean = means  # use KF expectation only
        window = np.concatenate([past, [x_t], fut_mean])
        mu[t] = float(np.dot(k, window))
        resid[t] = x_t - mu[t]
    return pd.DataFrame({"mu": mu, "resid": resid, "kf_mean": m}, index=x.index)

# ========= TAIL METRICS =========
def threshold_by_quantile(x_abs: np.ndarray, q=0.975):
    thr = np.nanquantile(x_abs, q)
    mask = x_abs > thr
    y = x_abs[mask] - thr
    return thr, y

def fit_gpd_mle(y):
    # minimalist, with guard rails; returns (xi, beta)
    y = np.asarray(y, float)
    if len(y) < 30 or np.nanmax(y) <= 0:
        return np.nan, np.nan
    from scipy.optimize import minimize

    def nll(params):
        xi, beta = params
        if beta <= 0: return np.inf
        z = 1 + xi * y / beta
        if np.any(z <= 0): return np.inf
        return len(y) * np.log(beta) + (1/xi + 1) * np.sum(np.log(z))

    res = minimize(nll, x0=[0.2, np.nanstd(y) if np.nanstd(y) > 0 else 1.0],
                   bounds=[(-0.49, 2.0), (1e-9, None)])
    return (res.x[0], res.x[1]) if res.success else (np.nan, np.nan)

def hill_index(x_abs, k=50):
    x = np.sort(x_abs[~np.isnan(x_abs)])
    if len(x) < max(10, k+1) or x[0] <= 0:
        return np.nan
    tail = x[-k:]
    return 1.0 / np.mean(np.log(tail / tail[0]))

def l_moments_quick(x):
    x = x[np.isfinite(x)]
    if len(x) < 80: return np.nan, np.nan
    q = np.quantile(x, [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    L1 = np.median(x)
    L2 = (q[7]-q[1])/2
    L3 = (q[8]-2*q[6]+2*q[2]-q[0])/6
    L4 = (q[8]-3*q[6]+3*q[4]-q[2]+q[0])/5
    tau3 = L3/L2 if L2!=0 else np.nan
    tau4 = L4/L2 if L2!=0 else np.nan
    return tau3, tau4

def tail_asymmetry(x):
    q1, med, q99 = np.quantile(x, [0.01, 0.5, 0.99])
    den = (med - q1)
    return np.nan if den == 0 else (q99 - med) / den

def compute_tail_metrics(resid: pd.Series, pot_q=0.975, hill_k=0.02):
    r = resid.to_numpy()
    x_abs = np.abs(r)
    thr, y = threshold_by_quantile(x_abs, q=pot_q)
    xi, beta = fit_gpd_mle(y)
    k = max(10, int(hill_k * len(x_abs)))
    hill = hill_index(x_abs, k=k)
    L3, L4 = l_moments_quick(r)
    asym = tail_asymmetry(r)
    return {"pot_thr": thr, "gpd_xi": xi, "gpd_beta": beta,
            "hill": hill, "L_skew": L3, "L_kurt": L4, "tail_asym": asym}

# ========= OPTIONAL: WAVELET DIAGNOSTIC =========
def wavelet_energy(resid: pd.Series, levels=3, wave='db4'):
    try:
        import pywt
    except ImportError:
        return None  # wavelet optional
    x = resid.fillna(0.0).to_numpy()
    coeffs = pywt.wavedec(x, wave, level=levels)
    energies = [np.sum(c**2) for c in coeffs]
    tot = sum(energies) + 1e-12
    shares = [e / tot for e in energies]  # [A, D3, D2, D1] style
    return {"levels": levels, "shares": shares}

# ========= MAIN PIPELINE (per stream) =========
@dataclass
class BaselineParams:
    L: int = 8
    sigma_kernel: float = 5.0
    ss_cfg: SSConfig = SSConfig()
    use_mh: bool = False
    mh_n: int = 150
    mh_burn: int = 60
    mh_nu: float = 5.0
    pe_m: int = 5
    pe_tau: int = 1
    pe_win: int = 300
    pe_smooth_n: int = 3
    pot_q: float = 0.975
    hill_k_frac: float = 0.02
    do_wavelet: bool = True

def run_pipeline(series: pd.Series,
                 bands: dict | None = None,
                 params: BaselineParams = BaselineParams()):
    """
    series: pd.Series of the order-flow signal (delta, trade_size, imbalance)
    bands: optional dict with {'p20':..,'p80':..,'med':..,'mad':..} from your historical PE study
    returns: dict of DataFrames/metrics
    """
    s = series.astype(float).copy()
    # 1) Laplace–KF (with optional MH) -> residuals
    base = full_laplace_mean_series(
        s, L=params.L, sigma_kernel=params.sigma_kernel,
        cfg=params.ss_cfg, use_mh=params.use_mh,
        mh_n=params.mh_n, mh_burn=params.mh_burn, mh_nu=params.mh_nu
    )
    resid = base["resid"]

    # 2) Permutation Entropy stream
    # build a simple rolling minute-return proxy on this stream to define order pattern
    x = s.diff().fillna(0.0).to_numpy()
    pe_buf = deque(maxlen=params.pe_smooth_n)
    pe_values = np.full(len(s), np.nan)
    for i in range(len(s)):
        if i < params.pe_win + (params.pe_m - 1) * params.pe_tau:
            continue
        window = x[:i+1]
        pe = permutation_entropy(window, m=params.pe_m, tau=params.pe_tau, window=params.pe_win)
        if np.isfinite(pe):
            pe_buf.append(pe)
            pe_values[i] = np.median(pe_buf) if len(pe_buf) == params.pe_smooth_n else pe
    pe_series = pd.Series(pe_values, index=s.index, name="PE")

    # 3) Bands (if not provided, infer rough ones from current run)
    if bands is None:
        vals = pe_series[np.isfinite(pe_series)].values
        if len(vals) < 100:
            med, mad = 0.95, 0.005
            p20, p80 = 0.945, 0.962
        else:
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) + 1e-12
            p20 = float(np.percentile(vals, 20))
            p80 = float(np.percentile(vals, 80))
        bands = {"med": med, "mad": mad, "p20": p20, "p80": p80}

    sm = PEStateMachine(bands["p20"], bands["p80"], bands["med"], bands["mad"])
    states = []
    for val in pe_series.fillna(bands["med"]).values:
        states.append(sm.step(val))
    state_series = pd.Series(states, index=s.index, name="state")

    # 4) Tail metrics on residuals
    tails = compute_tail_metrics(resid, pot_q=params.pot_q, hill_k=params.hill_k_frac)

    # 5) Optional wavelet diagnostic if chaotic detected
    wave_diag = None
    if params.do_wavelet and (state_series == "Chaotic").any():
        wave_diag = wavelet_energy(resid)

    return {
        "baseline": base,           # columns: mu, resid, kf_mean
        "pe": pe_series,            # permutation entropy over time
        "state": state_series,      # Compressed / Chaotic
        "bands": bands,             # p20/p80/median/mad used
        "tails": tails,             # dict of tail metrics
        "wavelet": wave_diag        # None or dict with energy shares
    }



import matplotlib.pyplot as plt
# visualization functions
# --- helpers ---
def _shade_state(ax, state: pd.Series, label_map=None, alpha=0.09):
    lab = label_map or {"Compressed":"#cccccc", "Chaotic":"#999999"}
    # don't specify colors explicitly? then just use default facecolor with alpha via ax.axvspan
    # we'll just shade by drawing transparent spans without explicit color
    current = None
    start = None
    for t, s in state.items():
        if current is None:
            current, start = s, t
            continue
        if s != current:
            ax.axvspan(start, t, alpha=alpha)
            current, start = s, t
    if start is not None:
        ax.axvspan(start, state.index[-1], alpha=alpha)

def _roll_exceed_count(x: pd.Series, thr: float, win: int = 50):
    m = (np.abs(x) > thr).astype(int)
    return m.rolling(win, min_periods=1).sum()

# 1) Stream + baseline mean + KF band
def plot_stream_and_mu(stream: pd.Series, mu: pd.Series, kf_mean: pd.Series,
                       kf_var: float | pd.Series | None = None,
                       state: pd.Series | None = None, title="Stream vs Laplace–KF mean"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(stream.index, stream.values, label="stream")
    ax.plot(mu.index, mu.values, label="Laplace–KF mean")
    # KF predictive band (±1σ)
    if kf_var is not None:
        if isinstance(kf_var, pd.Series):
            std = np.sqrt(kf_var.values)
        else:
            std = np.sqrt(float(kf_var)) * np.ones_like(kf_mean.values)
        upper = kf_mean.values + std
        lower = kf_mean.values - std
        ax.fill_between(kf_mean.index, lower, upper, alpha=0.15, label="KF ±1σ")
    if state is not None:
        _shade_state(ax, state)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax

# 2) Residuals + POT thresholds + rolling exceed count
def plot_residuals_pot(resid: pd.Series, pot_thr: float, pot_thr_hi: float | None = None,
                       win_count: int = 50, state: pd.Series | None = None, title="Residuals & POT"):
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(resid.index, resid.values, label="residual")
    ax.axhline(pot_thr, linestyle="--", linewidth=1, label=f"POT thr +{pot_thr:.4g}")
    ax.axhline(-pot_thr, linestyle="--", linewidth=1, label=f"POT thr -{pot_thr:.4g}")
    if pot_thr_hi:
        ax.axhline(pot_thr_hi, linestyle=":", linewidth=1, label=f"Hi thr +{pot_thr_hi:.4g}")
        ax.axhline(-pot_thr_hi, linestyle=":", linewidth=1, label=f"Hi thr -{pot_thr_hi:.4g}")

    exc = _roll_exceed_count(resid, pot_thr, win=win_count)
    ax2 = ax.twinx()
    ax2.plot(exc.index, exc.values, linestyle="-.", label=f"exceed count({win_count})")
    ax2.set_ylabel("exceed count")
    if state is not None:
        _shade_state(ax, state)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("residual")
    # combine legends
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    return fig, ax

# 3) Log–log CCDF of |resid| with GPD tail fit
def plot_ccdf_gpd(resid: pd.Series, pot_thr: float, xi: float, beta: float, title="CCDF |resid| with GPD tail"):
    x = np.abs(resid.dropna().values)
    x = np.sort(x)
    n = len(x)
    p = 1.0 - (np.arange(1, n+1) / (n+1))  # empirical CCDF
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(x, p, marker='.', linestyle='none', label="empirical CCDF")
    # theoretical GPD tail beyond threshold
    tail = x[x >= pot_thr]
    if np.isfinite(xi) and np.isfinite(beta) and len(tail) > 0:
        # CCDF for GPD: S(y) = (1 + xi*(y-thr)/beta)^(-1/xi), y>=thr
        y = tail
        S = np.power(1 + xi * (y - pot_thr) / beta, -1/xi)
        # scale S by fraction of observations beyond threshold
        frac = (x >= pot_thr).mean()
        ax.loglog(y, S * frac, label=f"GPD fit (xi={xi:.2f}, beta={beta:.2g})")
    ax.set_title(title)
    ax.set_xlabel("|residual|")
    ax.set_ylabel("CCDF")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax

# 4) Hill plot (pick k via stability)
def plot_hill(resid: pd.Series, k_grid: list[int] | None = None, title="Hill plot"):
    x = np.sort(np.abs(resid.dropna().values))
    n = len(x)
    if k_grid is None:
        # 15 points between 1% and 10% of tail size
        ks = np.unique((np.linspace(max(10, int(0.01*n)), max(20, int(0.1*n)), 15)).astype(int))
    else:
        ks = np.array(k_grid, dtype=int)
    vals = []
    for k in ks:
        if n < k + 1 or x[-k] <= 0:
            vals.append(np.nan)
            continue
        tail = x[-k:]
        vals.append(1.0 / np.mean(np.log(tail / tail[0])))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vals, marker='o')
    ax.set_title(title)
    ax.set_xlabel("top-k order statistics")
    ax.set_ylabel("Hill estimate")
    fig.tight_layout()
    return fig, ax

# 5) QQ plot vs GPD (tail only)
def plot_qq_gpd(resid: pd.Series, pot_thr: float, xi: float, beta: float, title="GPD QQ plot (tail)"):
    r = np.abs(resid.dropna().values)
    tail = r[r > pot_thr] - pot_thr
    tail = np.sort(tail)
    if len(tail) == 0 or not np.isfinite(xi) or not np.isfinite(beta):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title(title + " (insufficient tail)")
        return fig, ax
    # empirical probs
    probs = (np.arange(1, len(tail)+1) - 0.5) / len(tail)
    # GPD quantiles
    q = genpareto.ppf(probs, c=xi, scale=beta, loc=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(q, tail, marker='.', linestyle='none')
    lim = [min(q.min(), tail.min()), max(q.max(), tail.max())]
    ax.plot(lim, lim, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("GPD quantile")
    ax.set_ylabel("Empirical tail (|r|-thr)")
    fig.tight_layout()
    return fig, ax

# 6) PE monitor + bands + controller parameters
def plot_pe_and_controls(pe: pd.Series, bands: dict, state: pd.Series,
                         sigma_series: pd.Series | None = None,
                         q_series: pd.Series | None = None,
                         title="Permutation Entropy (PE) & control tracks"):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(pe.index, pe.values, label="PE")
    ax.axhline(bands["p20"], linestyle="--", linewidth=1, label=f"P20={bands['p20']:.3f}")
    ax.axhline(bands["p80"], linestyle="--", linewidth=1, label=f"P80={bands['p80']:.3f}")
    _shade_state(ax, state)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("PE (0–1)")
    ax.legend(loc="best")
    fig.tight_layout()

    if sigma_series is not None or q_series is not None:
        fig2, ax2 = plt.subplots(figsize=(11, 2.6))
        if sigma_series is not None:
            ax2.step(sigma_series.index, sigma_series.values, where="post", label="σ_kernel")
        if q_series is not None:
            ax2.step(q_series.index, q_series.values, where="post", label="KF q")
        _shade_state(ax2, state)
        ax2.set_title("Controller tracks")
        ax2.set_xlabel("time")
        ax2.set_ylabel("value")
        ax2.legend(loc="best")
        fig2.tight_layout()
        return (fig, ax), (fig2, ax2)

    return (fig, ax)

# 7) Wavelet diagnostic: simple energy bars per band (when wavelet mode is on)
def plot_wavelet_energy_bars(wavelet_diag: dict, title="Wavelet energy shares"):
    if wavelet_diag is None:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_title(title + " (no wavelet data)")
        return fig, ax
    shares = wavelet_diag["shares"]
    labels = [f"Band {i}" for i in range(len(shares))]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(labels, shares)
    ax.set_title(title)
    ax.set_ylabel("energy share")
    fig.tight_layout()
    return fig, ax
