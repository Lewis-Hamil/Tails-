import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- IO helpers ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_baseline_csv(outdir: str, name: str, baseline_df: pd.DataFrame):
    ensure_dir(outdir)
    baseline_df.to_csv(os.path.join(outdir, f"{name}_baseline.csv"))


def save_tail_json(outdir: str, name: str, tails: dict):
    ensure_dir(outdir)
    def _to_py(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    clean = {k: _to_py(v) for k, v in tails.items()}
    with open(os.path.join(outdir, f"{name}_tails.json"), "w") as f:
        json.dump(clean, f, indent=2)


# ---------- CSV loader (futures-tolerant) ----------
def _pick(norm, *cands):
    for c in cands:
        if c in norm:
            return norm[c]
    return None


def _parse_time_column(df: pd.DataFrame, csv_tz: str | None, dayfirst: bool):
    """Return tz-aware UTC Timestamp index from a wide range of CSV time formats."""
    norm = {c.lower().strip(): c for c in df.columns}

    # Single datetime column variants
    cand = _pick(norm, "bar end time", "datetime", "date time", "timestamp", "time")
    if cand is not None:
        col = df[cand]
        # Handle numeric epoch (sec/ms)
        if pd.api.types.is_numeric_dtype(col):
            unit = "ms" if col.max() > 1e12 else "s"
            dt = pd.to_datetime(col, unit=unit, utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(col, utc=False, errors="coerce", dayfirst=dayfirst)
    else:
        # Date + Time split
        c_date = _pick(norm, "date")
        c_time = _pick(norm, "time")
        if c_date is None or c_time is None:
            raise ValueError(
                "No time-like column found. Expected one of: "
                "'Bar End Time', 'Datetime', 'Timestamp', 'Time' "
                "or 'Date' + 'Time'."
            )
        dt = pd.to_datetime(
            df[c_date].astype(str) + " " + df[c_time].astype(str),
            utc=False, errors="coerce", dayfirst=dayfirst
        )

    # Localize if naïve; otherwise just convert to UTC
    if dt.dt.tz is None:
        if not csv_tz:
            dt = dt.dt.tz_localize("UTC")
        else:
            dt = dt.dt.tz_localize(csv_tz, nonexistent="shift_forward", ambiguous="NaT")
    dt = dt.dt.tz_convert("UTC")
    return dt


def load_bars(csv_path: str, csv_tz: str | None, dayfirst: bool, drop_zero_vol: bool = True) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    norm = {c.lower().strip(): c for c in df.columns}

    # Build UTC time index
    time = _parse_time_column(df, csv_tz=csv_tz, dayfirst=dayfirst)

    # Map OHLCV with Sierra/IBKR-friendly aliases
    c_open = _pick(norm, "open", "o")
    c_high = _pick(norm, "high", "h")
    c_low = _pick(norm, "low", "l")
    c_close = _pick(norm, "close", "last", "final", "settlement", "c", "price")
    c_vol = _pick(norm, "volume", "total volume", "vol", "nbvolume", "v")

    missing = [nm for nm, col in {
        "open": c_open, "high": c_high, "low": c_low, "close": c_close, "volume": c_vol
    }.items() if col is None]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame({
        "time": time,
        "open": pd.to_numeric(df[c_open], errors="coerce"),
        "high": pd.to_numeric(df[c_high], errors="coerce"),
        "low": pd.to_numeric(df[c_low], errors="coerce"),
        "close": pd.to_numeric(df[c_close], errors="coerce"),
        "volume": pd.to_numeric(df[c_vol], errors="coerce"),
    })

    # Optional L1 sizes if present (various names)
    c_bid_sz = _pick(norm, "bid_size", "bid size", "bidvolume", "bid volume", "bidsize")
    c_ask_sz = _pick(norm, "ask_size", "ask size", "askvolume", "ask volume", "asksize")
    if c_bid_sz:
        out["bid_size"] = pd.to_numeric(df[c_bid_sz], errors="coerce")
    if c_ask_sz:
        out["ask_size"] = pd.to_numeric(df[c_ask_sz], errors="coerce")

    out = out.dropna(subset=["time"]).set_index("time").sort_index()
    if drop_zero_vol:
        out = out[out["volume"] > 0]
    return out


# ---------- feature builders ----------
def build_delta_tick_test(bars: pd.DataFrame) -> pd.Series:
    """Tick test from close changes: sign * volume (works for futures too)."""
    close = bars["close"].astype(float)
    vol = bars["volume"].fillna(0.0).astype(float)
    sign = np.sign(close.diff().fillna(0.0).to_numpy())
    # carry forward sign on flats
    for i in range(1, len(sign)):
        if sign[i] == 0.0:
            sign[i] = sign[i - 1]
    delta = sign * vol.to_numpy()
    return pd.Series(delta, index=close.index, name="delta")


def build_trade_size(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(bars["volume"].fillna(0.0).values, index=bars.index, name="trade_size")


def build_imbalance(bars: pd.DataFrame) -> pd.Series:
    if "bid_size" in bars.columns and "ask_size" in bars.columns:
        qb = bars["bid_size"].astype(float)
        qa = bars["ask_size"].astype(float)
        denom = (qb + qa).replace(0, np.nan)
        imb = (qb - qa) / denom
        return pd.Series(imb.clip(-1.0, 1.0).values, index=bars.index, name="imbalance")
    return pd.Series(np.nan, index=bars.index, name="imbalance")


# ---------- particle filter + tails (inlined) ----------
def particle_filter_1d(
    y: np.ndarray,
    n_particles: int = 500,
    proc_var: float = 0.5,
    obs_var: float = 1.0,
    init_mean: float | None = None,
    init_var: float = 1.0,
):
    T = len(y)
    means = np.zeros(T, dtype=float)

    if init_mean is None:
        init_mean = float(y[0]) if np.isfinite(y[0]) else 0.0

    parts = np.random.normal(loc=init_mean, scale=np.sqrt(init_var), size=n_particles)
    weights = np.ones(n_particles, dtype=float) / n_particles

    for t in range(T):
        yt = y[t]

        # propagate
        parts = parts + np.random.normal(0.0, np.sqrt(proc_var), size=n_particles)

        # weight
        if np.isfinite(yt):
            ll = np.exp(-0.5 * (yt - parts) ** 2 / obs_var) / np.sqrt(2.0 * np.pi * obs_var)
            weights *= ll
        wsum = weights.sum()
        if wsum <= 0 or not np.isfinite(wsum):
            weights[:] = 1.0 / n_particles
        else:
            weights /= wsum

        means[t] = np.sum(weights * parts)

        # resample if degenerate
        neff = 1.0 / np.sum(weights ** 2)
        if neff < n_particles / 2.0:
            idx = np.random.choice(n_particles, size=n_particles, p=weights)
            parts = parts[idx]
            weights[:] = 1.0 / n_particles

    return means


def peaks_over_threshold(resid: np.ndarray, q: float = 0.98):
    thr = np.quantile(resid, q)
    excess = resid[resid > thr] - thr
    return thr, excess


def fit_gpd_mom(excess: np.ndarray):
    if len(excess) < 5:
        return np.nan, np.nan
    m1 = excess.mean()
    m2 = ((excess - m1) ** 2).mean()
    if m2 <= 0:
        return np.nan, np.nan
    xi = 0.5 * (1 - (m1 ** 2) / m2)
    beta = 0.5 * m1 * (1 + (m1 ** 2) / m2)
    return float(xi), float(beta)


def hill_estimator(resid: np.ndarray, k: int = 50):
    x = np.sort(resid)
    x = x[np.isfinite(x)]
    if len(x) < k + 1:
        return np.nan
    xk = x[-k:]
    xk_min = xk[0]
    if xk_min <= 0:
        return np.nan
    hill = (1.0 / k) * np.sum(np.log(xk / xk_min))
    return float(hill)


def run_pipeline(series: pd.Series, bands: dict | None = None, pf_cfg: dict | None = None):
    if pf_cfg is None:
        pf_cfg = {}

    y = series.astype(float).to_numpy()
    data_std = np.nanstd(y)
    if not np.isfinite(data_std) or data_std == 0:
        data_std = 1.0

    pf_means = particle_filter_1d(
        y,
        n_particles=pf_cfg.get("n_particles", 500),
        proc_var=pf_cfg.get("proc_var", 0.2 * data_std),
        obs_var=pf_cfg.get("obs_var", 0.5 * data_std ** 2),
        init_mean=pf_cfg.get("init_mean", None),
        init_var=pf_cfg.get("init_var", data_std ** 2),
    )

    resid = y - pf_means
    pos_resid = resid[resid > 0]
    pot_thr, excess = peaks_over_threshold(pos_resid, q=0.98)
    gpd_xi, gpd_beta = fit_gpd_mom(excess)
    hill = hill_estimator(pos_resid, k=min(50, max(5, len(pos_resid) // 10)))

    baseline = pd.DataFrame(
        {
            "obs": series.astype(float).values,
            "pf_mean": pf_means,
            "resid": resid,
        },
        index=series.index,
    )

    out = {
        "baseline": baseline,
        "tails": {
            "pot_thr": float(pot_thr),
            "gpd_xi": gpd_xi,
            "gpd_beta": gpd_beta,
            "hill": hill,
            "n_exceed": int(len(excess)),
        },
        "state": {
            "pf": {
                "n_particles": pf_cfg.get("n_particles", 500),
            }
        },
        "bands": bands if bands is not None else {},
    }
    return out


# ---------- plotting ----------
def maybe_plot_and_save(imgdir: str, prefix: str, series: pd.Series, out: dict):
    try:
        os.makedirs(imgdir, exist_ok=True)
        baseline = out["baseline"]
        tails = out["tails"]

        # 1) stream vs pf_mean
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(series.index, series.values, label="stream", linewidth=1)
        ax.plot(series.index, baseline["pf_mean"].values, label="pf_mean", linewidth=1)
        ax.set_title(f"{prefix} stream and PF mean")
        ax.legend()
        fig.savefig(os.path.join(imgdir, f"{prefix}_stream_pf.png"), dpi=130)
        plt.close(fig)

        # 2) residuals with POT threshold
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(series.index, baseline["resid"].values, label="resid", linewidth=1)
        ax.axhline(tails["pot_thr"], color="red", linestyle="--", label="POT thr")
        ax.set_title(f"{prefix} residuals + POT") 
        ax.legend()
        fig.savefig(os.path.join(imgdir, f"{prefix}_resid_pot.png"), dpi=130)
        plt.close(fig)

    except Exception as e:
        print(f"[warn] plotting failed: {e}")


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Run PF-based orderflow tails on OHLCV CSV")
    p.add_argument("--csv", required=True, help="Path to OHLCV CSV (Sierra/IBKR/Schwab export)")
    p.add_argument("--symbol", required=True, help="Symbol label for outputs (e.g., NQZ25)")
    p.add_argument("--out", default="out", help="Output root directory (default: out)")
    p.add_argument("--plots", action="store_true", help="Save matplotlib PNGs")
    p.add_argument("--session-tz", default=None, help="Optional IANA TZ for session clip (e.g., America/Chicago)")
    p.add_argument("--session-window", default=None, help="Optional session window HH:MM-HH:MM in session-tz")
    p.add_argument("--csv-tz", default=None, help="Timezone for naïve timestamps in CSV")
    p.add_argument("--dayfirst", action="store_true", help="Parse dates as day-first")
    p.add_argument("--bands", default=None, help="PE bands JSON to apply to all streams")
    p.add_argument("--bands-delta", default=None, help="PE bands JSON for delta only")
    p.add_argument("--bands-size", default=None, help="PE bands JSON for size only")
    p.add_argument("--bands-imb", default=None, help="PE bands JSON for imbalance only")
    p.add_argument("--pf-particles", type=int, default=500)
    return p.parse_args()


def load_bands(path: str | None):
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)


def session_clip(df: pd.DataFrame, session_tz: str | None, session_window: str | None) -> pd.DataFrame:
    if not session_tz or not session_window:
        return df
    try:
        t1_str, t2_str = [s.strip() for s in session_window.split("-", 1)]
    except Exception:
        raise ValueError(f"Bad --session-window '{session_window}'. Use HH:MM-HH:MM")

    df_loc = df.tz_convert(session_tz)
    start_local = df_loc.index.min().normalize()
    end_local = df_loc.index.max().normalize()
    days = pd.date_range(start_local, end_local, freq="D")

    kept = []
    for day in days:
        d1 = pd.Timestamp(f"{day.date()} {t1_str}").tz_localize(session_tz)
        d2 = pd.Timestamp(f"{day.date()} {t2_str}").tz_localize(session_tz)
        if d2 <= d1:
            kept.append(df_loc[(df_loc.index >= d1) & (df_loc.index < d1.normalize() + pd.Timedelta(days=1))])
            kept.append(df_loc[(df_loc.index >= d2.normalize()) & (df_loc.index < d2)])
        else:
            kept.append(df_loc[(df_loc.index >= d1) & (df_loc.index < d2)])
    out = pd.concat(kept) if kept else df_loc.iloc[0:0]
    return out.tz_convert("UTC").sort_index()


def main():
    args = parse_args()

    bars = load_bars(args.csv, csv_tz=args.csv_tz, dayfirst=args.dayfirst)
    if bars.index.tz is None:
        bars = bars.tz_localize("UTC")

    if args.session_tz and args.session_window:
        bars = session_clip(bars, args.session_tz, args.session_window)

    delta = build_delta_tick_test(bars)
    size = build_trade_size(bars)
    imb = build_imbalance(bars)

    # bands
    def _bands(path, fallback):
        return load_bands(path) if path else fallback
    bands_all = load_bands(args.bands)
    bands_delta = _bands(args.bands_delta, bands_all)
    bands_size = _bands(args.bands_size, bands_all)
    bands_imb = _bands(args.bands_imb, bands_all)

    pf_cfg = {"n_particles": args.pf_particles}

    out_delta = run_pipeline(delta, bands=bands_delta, pf_cfg=pf_cfg)
    out_size = run_pipeline(size, bands=bands_size, pf_cfg=pf_cfg)
    out_imb = run_pipeline(imb, bands=bands_imb, pf_cfg=pf_cfg)

    root = os.path.join(args.out, args.symbol)

    for label, out, series in [
        ("delta", out_delta, delta),
        ("size", out_size, size),
        ("imb", out_imb, imb),
    ]:
        d = os.path.join(root, label)
        save_baseline_csv(d, label, out["baseline"])
        save_tail_json(d, label, out["tails"])
        if args.plots:
            imgdir = os.path.join(d, "plots")
            maybe_plot_and_save(imgdir, label, series, out)

        t = out["tails"]
        bands = out["bands"]
        def _fmt(v):
            return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.3g}" if isinstance(v, float) else str(v))
        print(
            f"[{label.upper()}] POT thr={_fmt(t.get('pot_thr'))} | xi={_fmt(t.get('gpd_xi'))} "
            f"| Hill={_fmt(t.get('hill'))} | PE bands: P20={_fmt((bands or {}).get('p20'))}, "
            f"P80={_fmt((bands or {}).get('p80'))}, med={_fmt((bands or {}).get('med'))}"
        )


if __name__ == "__main__":
    main()
