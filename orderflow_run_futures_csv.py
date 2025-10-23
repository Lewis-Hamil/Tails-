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
            # Heuristic: >1e12 likely ms, >1e9 sec
            unit = "ms" if col.max() > 1e12 else "s"
            dt = pd.to_datetime(col, unit=unit, utc=True, errors="coerce")
        else:
            dt = pd.to_datetime(col, utc=False, errors="coerce", dayfirst=dayfirst)

    else:
        # Date + Time split
        c_date = _pick(norm, "date")
        c_time = _pick(norm, "time")
        if c_date is None or c_time is None:
            raise ValueError("No time-like column found. Expected one of: "
                             "'Bar End Time', 'Datetime', 'Timestamp', 'Time' "
                             "or 'Date' + 'Time'.")
        dt = pd.to_datetime(
            df[c_date].astype(str) + " " + df[c_time].astype(str),
            utc=False, errors="coerce", dayfirst=dayfirst
        )

    # Localize if naïve; otherwise just convert to UTC
    if dt.dt.tz is None:
        if not csv_tz:
            # Default to UTC if user didn't tell us; avoids silent local-time assumptions
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
    c_open  = _pick(norm, "open", "o")
    c_high  = _pick(norm, "high", "h")
    c_low   = _pick(norm, "low", "l")
    c_close = _pick(norm, "close", "last", "final", "settlement", "c", "price")
    c_vol   = _pick(norm, "volume", "total volume", "vol", "nbvolume", "v")

    missing = [nm for nm, col in {
        "open": c_open, "high": c_high, "low": c_low, "close": c_close, "volume": c_vol
    }.items() if col is None]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame({
        "time":   time,
        "open":   pd.to_numeric(df[c_open], errors="coerce"),
        "high":   pd.to_numeric(df[c_high], errors="coerce"),
        "low":    pd.to_numeric(df[c_low],  errors="coerce"),
        "close":  pd.to_numeric(df[c_close],errors="coerce"),
        "volume": pd.to_numeric(df[c_vol],  errors="coerce"),
    })

    # Optional L1 sizes if present (various names)
    c_bid_sz = _pick(norm, "bid_size", "bid size", "bidvolume", "bid volume", "bidsize")
    c_ask_sz = _pick(norm, "ask_size", "ask size", "askvolume", "ask volume", "asksize")
    if c_bid_sz: out["bid_size"] = pd.to_numeric(df[c_bid_sz], errors="coerce")
    if c_ask_sz: out["ask_size"] = pd.to_numeric(df[c_ask_sz], errors="coerce")

    out = out.dropna(subset=["time"]).set_index("time").sort_index()
    if drop_zero_vol:
        out = out[out["volume"] > 0]
    return out

# ---------- Feature builders ----------
def build_delta_tick_test(bars: pd.DataFrame) -> pd.Series:
    """Tick test from close changes: sign * volume (works for futures too)."""
    close = bars["close"].astype(float)
    vol = bars["volume"].fillna(0.0).astype(float)
    sign = np.sign(close.diff().fillna(0.0).to_numpy())
    # carry forward sign on flats to avoid losing flow
    for i in range(1, len(sign)):
        if sign[i] == 0.0:
            sign[i] = sign[i-1]
    delta = sign * vol.to_numpy()
    return pd.Series(delta, index=close.index, name="delta")

def build_trade_size(bars: pd.DataFrame) -> pd.Series:
    return pd.Series(bars["volume"].fillna(0.0).values, index=bars.index, name="trade_size")

def build_imbalance(bars: pd.DataFrame) -> pd.Series:
    if "bid_size" in bars.columns and "ask_size" in bars.columns:
        qb = bars["bid_size"].astype(float); qa = bars["ask_size"].astype(float)
        denom = (qb + qa).replace(0, np.nan)
        imb = (qb - qa) / denom
        return pd.Series(imb.clip(-1.0, 1.0).values, index=bars.index, name="imbalance")
    return pd.Series(np.nan, index=bars.index, name="imbalance")

# ---------- Session clip (optional) ----------
def session_clip(df: pd.DataFrame, session_tz: str | None, session_window: str | None) -> pd.DataFrame:
    if not session_tz or not session_window:
        return df
    try:
        t1_str, t2_str = [s.strip() for s in session_window.split("-", 1)]
    except Exception:
        raise ValueError(f"Bad --session-window '{session_window}'. Use HH:MM-HH:MM, e.g. 08:30-15:00")

    df_loc = df.tz_convert(session_tz)
    start_local = df_loc.index.min().normalize()
    end_local   = df_loc.index.max().normalize()
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

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="Run orderflow_tails on OHLCV CSV (futures-friendly loader)")
    p.add_argument("--csv", required=True, help="Path to OHLCV CSV (Sierra/IBKR/Schwab export)")
    p.add_argument("--symbol", required=True, help="Symbol label for outputs (e.g., NQZ25)")
    p.add_argument("--out", default="out", help="Output root directory (default: out)")
    p.add_argument("--plots", action="store_true", help="Save matplotlib PNGs using orderflow_tails helpers")
    p.add_argument("--session-tz", default=None, help="Optional IANA TZ for session clip (e.g., America/Chicago)")
    p.add_argument("--session-window", default=None, help="Optional session window HH:MM-HH:MM in session-tz")
    # Futures-specific parsing knobs:
    p.add_argument("--csv-tz", default=None, help="Timezone for naïve timestamps in CSV (e.g., America/Chicago). If omitted, assumes timestamps are already UTC.")
    p.add_argument("--dayfirst", action="store_true", help="Parse dates as day-first (useful for Sierra dd/mm/yyyy)")
    # Optional PE band files
    p.add_argument("--bands", default=None, help="PE bands JSON to apply to all streams (p20,p80,med,mad)")
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
    bars = load_bars(args.csv, csv_tz=args.csv_tz, dayfirst=args.dayfirst)
    if bars.index.tz is None:
        bars = bars.tz_localize("UTC")  # belt-and-suspenders

    if args.session_tz and args.session_window:
        bars = session_clip(bars, args.session_tz, args.session_window)

    # Feature streams
    delta = build_delta_tick_test(bars)
    size  = build_trade_size(bars)
    imb   = build_imbalance(bars)

    # Bands
    def _bands(path, fallback):
        return load_bands(path) if path else fallback
    bands_all   = load_bands(args.bands)
    bands_delta = _bands(args.bands_delta, bands_all)
    bands_size  = _bands(args.bands_size,  bands_all)
    bands_imb   = _bands(args.bands_imb,   bands_all)

    # Run original model unmodified
    out_delta = oft.run_pipeline(delta, bands=bands_delta)
    out_size  = oft.run_pipeline(size,  bands=bands_size)
    out_imb   = oft.run_pipeline(imb,   bands=bands_imb)

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
