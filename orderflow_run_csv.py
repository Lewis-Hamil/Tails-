
import os
import json
import argparse
import numpy as np
import pandas as pd

# Use your existing model exactly as-is
import orderflow_tails as oft

# ---------- IO helpers ----------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_baseline_csv(outdir: str, name: str, baseline_df: pd.DataFrame):
    ensure_dir(outdir)
    baseline_df.to_csv(os.path.join(outdir, f"{name}_baseline.csv"))

def save_tail_json(outdir: str, name: str, tails: dict):
    ensure_dir(outdir)
    # cast np types to python for json
    def _to_py(v):
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    clean = {k: _to_py(v) for k, v in tails.items()}
    with open(os.path.join(outdir, f"{name}_tails.json"), "w") as f:
        json.dump(clean, f, indent=2)

def detect_time_col(df: pd.DataFrame):
    for cand in ("time", "datetime", "timestamp", "Date", "date"):
        if cand in df.columns:
            return cand
    raise ValueError("No time-like column found. Expected one of: time, datetime, timestamp, Date, date.")

def load_bars(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    tcol = detect_time_col(df)
    bars = df.rename(columns={tcol: "time"}).copy()
    bars["time"] = pd.to_datetime(bars["time"], utc=True, errors="coerce")
    bars = bars.set_index("time").sort_index()

    # Coerce numeric columns if present
    for c in ["open","high","low","close","volume","bid_size","ask_size"]:
        if c in bars.columns:
            bars[c] = pd.to_numeric(bars[c], errors="coerce")
    # minimal required
    for req in ["close","volume"]:
        if req not in bars.columns:
            raise ValueError(f"CSV missing required column: {req}")
    return bars

# ---------- Feature builders ----------
def build_delta_tick_test(bars: pd.DataFrame) -> pd.Series:
    """Option A: Tick test from close changes: sign * volume."""
    close = bars["close"].astype(float)
    vol = bars["volume"].fillna(0.0).astype(float)
    sign = np.sign(close.diff().fillna(0.0).to_numpy())
    # Replace zeros (flat close) with previous sign to avoid dropping flow
    for i in range(1, len(sign)):
        if sign[i] == 0.0:
            sign[i] = sign[i-1]
    delta = sign * vol.to_numpy()
    return pd.Series(delta, index=close.index, name="delta")

def build_trade_size(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(bars["volume"].fillna(0.0).values, index=bars.index, name="trade_size")

def build_imbalance(bars: pd.DataFrame) -> pd.Series:
    if "bid_size" in bars.columns and "ask_size" in bars.columns:
        qb = bars["bid_size"]; qa = bars["ask_size"]
        denom = (qb + qa).replace(0, np.nan)
        imb = (qb - qa) / denom
        return pd.Series(imb.clip(-1.0, 1.0).values, index=bars.index, name="imbalance")
    # If L1 not available, return NaNs so downstream ignores it
    return pd.Series(np.nan, index=bars.index, name="imbalance")

# ---------- Session clip (optional) ----------
def session_clip(df: pd.DataFrame, session_tz: str | None, session_window: str | None) -> pd.DataFrame:
    if not session_tz or not session_window:
        return df
    try:
        t1_str, t2_str = [s.strip() for s in session_window.split("-", 1)]
    except Exception:
        raise ValueError(f"Bad --session-window '{session_window}'. Use HH:MM-HH:MM, e.g. 08:00-21:00")

    df_loc = df.tz_convert(session_tz)
    # build per-day windows across the date span
    start_local = df_loc.index.min().tz_convert(session_tz).normalize()
    end_local   = df_loc.index.max().tz_convert(session_tz).normalize()
    days = pd.date_range(start_local, end_local, freq="D", inclusive="both")

    kept = []
    for day in days:
        d1 = pd.Timestamp(f"{day.date()} {t1_str}").tz_localize(session_tz)
        d2 = pd.Timestamp(f"{day.date()} {t2_str}").tz_localize(session_tz)
        if d2 <= d1:
            # wrap midnight
            kept.append(df_loc[(df_loc.index >= d1) & (df_loc.index < d1.normalize() + pd.Timedelta(days=1))])
            kept.append(df_loc[(df_loc.index >= d2.normalize()) & (df_loc.index < d2)])
        else:
            kept.append(df_loc[(df_loc.index >= d1) & (df_loc.index < d2)])
    out = pd.concat(kept) if kept else df_loc.iloc[0:0]
    return out.tz_convert("UTC").sort_index()

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Run original orderflow_tails model from a CSV dump (no API calls)")
    p.add_argument("--csv", required=True, help="Path to OHLCV CSV (from schwab_dump_csv.py or similar)")
    p.add_argument("--symbol", required=True, help="Symbol label for outputs (e.g., SPY)")
    p.add_argument("--out", default="out", help="Output root directory (default: out)")
    p.add_argument("--plots", action="store_true", help="Save matplotlib PNGs using orderflow_tails helpers")
    p.add_argument("--session-tz", default=None, help="Optional IANA TZ for session clip, e.g. Europe/London")
    p.add_argument("--session-window", default=None, help="Optional session window HH:MM-HH:MM in session-tz")
    p.add_argument("--bands", default=None, help="PE bands JSON to apply to all streams (p20,p80,med,mad)" )
    p.add_argument("--bands-delta", default=None, help="PE bands JSON for delta only")
    p.add_argument("--bands-size",  default=None, help="PE bands JSON for size only")
    p.add_argument("--bands-imb",   default=None, help="PE bands JSON for imbalance only")
    return p.parse_args()

def load_bands(path: str | None):
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)

def maybe_plot_and_save(imgdir: str, prefix: str, series: pd.Series, out: dict):
    try:
        b = out["baseline"]; pe = out["pe"]; state = out["state"]; tails = out["tails"]; bands = out["bands"]; wave = out["wavelet"]
        figs = []
        figs.append(("stream_mu",        oft.plot_stream_and_mu(series, b["mu"], b["kf_mean"], kf_var=None, state=state)[0]))
        figs.append(("resid_pot",        oft.plot_residuals_pot(b["resid"], tails["pot_thr"], state=state)[0]))
        figs.append(("ccdf_gpd",         oft.plot_ccdf_gpd(b["resid"], tails["pot_thr"], tails["gpd_xi"], tails["gpd_beta"])[0]))
        figs.append(("hill",             oft.plot_hill(b["resid"])[0]))
        figs.append(("qq_gpd",           oft.plot_qq_gpd(b["resid"], tails["pot_thr"], tails["gpd_xi"], tails["gpd_beta"])[0]))
        pe_figs = oft.plot_pe_and_controls(pe, bands, state)
        if isinstance(pe_figs, tuple) and isinstance(pe_figs[0], tuple):
            figs.append(("pe", pe_figs[0][0])); figs.append(("controls", pe_figs[1][0]))
        else:
            figs.append(("pe", pe_figs[0]))
        figs.append(("wavelet_energy",   oft.plot_wavelet_energy_bars(wave)[0]))

        os.makedirs(imgdir, exist_ok=True)
        for name, fig in figs:
            fig.savefig(os.path.join(imgdir, f"{prefix}_{name}.png"), dpi=130)
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
    except Exception as e:
        print(f"[warn] plotting failed: {e}")

def main():
    args = parse_args()

    # Load & basic sanity
    bars = load_bars(args.csv)
    # give the index a tz if missing
    if bars.index.tz is None:
        bars = bars.tz_localize("UTC")
    if args.session_tz and args.session_window:
        bars = session_clip(bars, args.session_tz, args.session_window)

    # Feature streams
    delta = build_delta_tick_test(bars)
    size  = build_trade_size(bars)
    imb   = build_imbalance(bars)

    # Bands
    bands_all   = load_bands(args.bands)
    bands_delta = load_bands(args.bands_delta) if args.bands_delta else bands_all
    bands_size  = load_bands(args.bands_size)  if args.bands_size  else bands_all
    bands_imb   = load_bands(args.bands_imb)   if args.bands_imb   else bands_all

    # Run original model unmodified
    out_delta = oft.run_pipeline(delta,      bands=bands_delta)
    out_size  = oft.run_pipeline(size,       bands=bands_size)
    out_imb   = oft.run_pipeline(imb,        bands=bands_imb)

    # Save outputs
    root = os.path.join(args.out, args.symbol)
    for label, out, series in [("delta", out_delta, delta), ("size", out_size, size), ("imb", out_imb, imb)]:
        d = os.path.join(root, label)
        save_baseline_csv(d, label, out["baseline"])
        save_tail_json(d, label, out["tails"])
        if args.plots:
            imgdir = os.path.join(d, "plots")
            maybe_plot_and_save(imgdir, label, series, out)

        # Print short summary
        t = out["tails"]; bands = out["bands"]
        def _fmt(v):
            return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.3g}" if isinstance(v, float) else str(v))
        print(f"[{label.upper()}] POT thr={_fmt(t.get('pot_thr'))} | xi={_fmt(t.get('gpd_xi'))} | Hill={_fmt(t.get('hill'))} | PE bands: P20={_fmt(bands.get('p20'))}, P80={_fmt(bands.get('p80'))}, med={_fmt(bands.get('med'))}")

if __name__ == "__main__":
    main()
