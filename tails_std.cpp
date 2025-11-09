// Orderflow Tails Live (Delta / Size / Imbalance) with EVT + Z-Score
// Wavelets dropped, everything else kept.
// This mirrors your Python pipeline structure.

#include "sierrachart.h"

SCDLLName("Orderflow Tails EVT")

// ---------------------- helpers ----------------------

// safe clip
float ClipFloat(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// compute rolling mean over last N bars (including idx)
double RollingMean(const SCFloatArray& arr, int idx, int len)
{
    double s = 0.0;
    int count = 0;
    int start = idx - len + 1;
    if (start < 0) start = 0;
    for (int i = start; i <= idx; ++i)
    {
        s += arr[i];
        ++count;
    }
    if (count == 0) return 0.0;
    return s / (double)count;
}

// compute rolling std over last N bars
double RollingStd(const SCFloatArray& arr, int idx, int len, double mean)
{
    double ss = 0.0;
    int count = 0;
    int start = idx - len + 1;
    if (start < 0) start = 0;
    for (int i = start; i <= idx; ++i)
    {
        double d = arr[i] - mean;
        ss += d * d;
        ++count;
    }
    if (count <= 1) return 0.0;
    return sqrt(ss / (double)(count - 1));
}

// pick POT threshold = empirical quantile in last N residuals
double RollingQuantile(const SCFloatArray& arr, int idx, int len, double q)
{
    // collect into temp buffer
    int start = idx - len + 1;
    if (start < 0) start = 0;
    int n = idx - start + 1;
    if (n <= 0) return 0.0;

    // Sierra doesn't have std::vector guaranteed here, so use static max
    const int MAXWIN = 2000;
    double buf[MAXWIN];
    if (n > MAXWIN)
        n = MAXWIN;

    int j = 0;
    for (int i = idx - n + 1; i <= idx; ++i)
        buf[j++] = arr[i];

    // simple selection sort to get ascending (n is small)
    for (int a = 0; a < n - 1; ++a)
    {
        int m = a;
        for (int b = a + 1; b < n; ++b)
            if (buf[b] < buf[m])
                m = b;
        if (m != a)
        {
            double tmp = buf[a];
            buf[a] = buf[m];
            buf[m] = tmp;
        }
    }

    double pos = q * (n - 1);
    int idx_low = (int)pos;
    int idx_high = idx_low + 1;
    if (idx_high >= n) idx_high = n - 1;
    double frac = pos - (double)idx_low;
    return buf[idx_low] * (1.0 - frac) + buf[idx_high] * frac;
}

// fit GPD to exceedances using simple method-of-moments
// inputs: exceedances e[0..k-1]
// outputs: xi (shape), beta (scale)
// if fail, return xi=0, beta=0
void FitGPD_MoM(const double* e, int k, double& xi, double& beta)
{
    if (k < 2)
    {
        xi = 0.0;
        beta = 0.0;
        return;
    }

    double m1 = 0.0;
    double m2 = 0.0;
    for (int i = 0; i < k; ++i)
    {
        m1 += e[i];
        m2 += e[i] * e[i];
    }
    m1 /= (double)k;
    m2 /= (double)k;

    double s2 = m2 - m1 * m1;
    if (s2 <= 0.0 || m1 <= 0.0)
    {
        xi = 0.0;
        beta = m1;
        return;
    }

    // MoM formulas for GPD
    double ratio = (m1 * m1) / s2;
    xi = 0.5 * (ratio - 1.0);
    if (xi < -0.49) // avoid weird negatives
        xi = -0.49;
    beta = 0.5 * m1 * (ratio + 1.0);
    if (beta <= 0.0)
        beta = m1;
}

// compute tail probability for a single exceedance y over GPD(u, xi, beta)
// using GPD CDF: F(y) = 1 - (1 + xi*y/beta)^(-1/xi)
// tail prob = 1 - F(y)
double GPDTailProb(double y, double xi, double beta)
{
    if (y <= 0.0) return 0.0;
    if (beta <= 0.0) return 0.0;

    if (fabs(xi) < 1e-6)
    {
        // exponential limit
        double F = 1.0 - exp(-y / beta);
        return 1.0 - F;
    }
    double inner = 1.0 + xi * y / beta;
    if (inner <= 0.0)
        inner = 1e-6;
    double F = 1.0 - pow(inner, -1.0 / xi);
    double tail = 1.0 - F;
    if (tail < 0.0) tail = 0.0;
    if (tail > 1.0) tail = 1.0;
    return tail;
}

// Hill estimator for top-k tail
double HillEstimator(const SCFloatArray& arr, int idx, int len, int k)
{
    // collect positives only
    int start = idx - len + 1;
    if (start < 0) start = 0;
    int n = idx - start + 1;
    if (n <= 0) return 0.0;

    const int MAXWIN = 2000;
    double buf[MAXWIN];
    if (n > MAXWIN)
        n = MAXWIN;

    int j = 0;
    for (int i = idx - n + 1; i <= idx; ++i)
    {
        double v = arr[i];
        if (v > 0.0)
            buf[j++] = v;
    }
    int m = j;
    if (m < k + 1)
        return 0.0;

    // sort descending
    for (int a = 0; a < m - 1; ++a)
    {
        int mx = a;
        for (int b = a + 1; b < m; ++b)
            if (buf[b] > buf[mx])
                mx = b;
        if (mx != a)
        {
            double tmp = buf[a];
            buf[a] = buf[mx];
            buf[mx] = tmp;
        }
    }

    double xk1 = buf[k]; // (k+1)-th largest (0-based)
    if (xk1 <= 0.0)
        return 0.0;

    double sumlog = 0.0;
    for (int i = 0; i < k; ++i)
        sumlog += log(buf[i] / xk1);

    return sumlog / (double)k;
}


// ---------------------- main study ----------------------

SCSFExport scsf_OrderflowTailsEVT(SCStudyInterfaceRef sc)
{
    // ---- defaults ----
    if (sc.SetDefaults)
    {
        sc.GraphName = "Orderflow Tails EVT (Delta/Size/Imb)";

        sc.AutoLoop = 1;
        sc.GraphRegion = 1;

        // subgraphs
        sc.Subgraph[0].Name = "Delta";
        sc.Subgraph[0].DrawStyle = DRAWSTYLE_LINE;
        sc.Subgraph[0].PrimaryColor = RGB(0, 128, 255);

        sc.Subgraph[1].Name = "Size";
        sc.Subgraph[1].DrawStyle = DRAWSTYLE_LINE;
        sc.Subgraph[1].PrimaryColor = RGB(0, 200, 0);

        sc.Subgraph[2].Name = "Imbalance";
        sc.Subgraph[2].DrawStyle = DRAWSTYLE_LINE;
        sc.Subgraph[2].PrimaryColor = RGB(255, 128, 0);

        sc.Subgraph[3].Name = "Delta Tail Score (p)";
        sc.Subgraph[3].DrawStyle = DRAWSTYLE_LINE;
        sc.Subgraph[3].PrimaryColor = RGB(255, 0, 0);

        sc.Subgraph[4].Name = "Size Tail Score (p)";
        sc.Subgraph[4].DrawStyle = DRAWSTYLE_LINE;
        sc.Subgraph[4].PrimaryColor = RGB(200, 0, 200);

        sc.Subgraph[5].Name = "Imb Tail Score (p)";
        sc.Subgraph[5].DrawStyle = DRAWSTYLE_LINE;
        sc.Subgraph[5].PrimaryColor = RGB(180, 180, 0);

        sc.Subgraph[6].Name = "Delta Tail Z";
        sc.Subgraph[6].DrawStyle = DRAWSTYLE_LINE;
        sc.Subgraph[6].PrimaryColor = RGB(255, 0, 0);

        sc.Subgraph[7].Name = "Size Tail Z";
        sc.Subgraph[7].DrawStyle = DRAWSTYLE_LINE;
        sc.Subgraph[7].PrimaryColor = RGB(200, 0, 200);

        sc.Subgraph[8].Name = "Imb Tail Z";
        sc.Subgraph[8].DrawStyle = DRAWSTYLE_LINE;
        sc.Subgraph[8].PrimaryColor = RGB(180, 180, 0);

        sc.Subgraph[9].Name = "Alert Marker";
        sc.Subgraph[9].DrawStyle = DRAWSTYLE_COLOR_BAR;
        sc.Subgraph[9].PrimaryColor = RGB(255, 0, 0);

        // inputs
        sc.Input[0].Name = "Lookback Length (residuals/tails)";
        sc.Input[0].SetInt(300);
        sc.Input[0].SetIntLimits(50, 2000);

        sc.Input[1].Name = "POT Quantile (0.95 - 0.995)";
        sc.Input[1].SetFloat(0.98f);

        sc.Input[2].Name = "Tail Z Alert Level";
        sc.Input[2].SetFloat(2.5f);

        sc.Input[3].Name = "Use Bid/Ask for Imbalance";
        sc.Input[3].SetYesNo(1);

        sc.Input[4].Name = "Hill k (top tail count)";
        sc.Input[4].SetInt(20);

        sc.Input[5].Name = "Z Lookback";
        sc.Input[5].SetInt(200);

        return;
    }

    int idx = sc.Index;
    int lookback = sc.Input[0].GetInt();
    double pot_q = sc.Input[1].GetFloat();
    double z_alert = sc.Input[2].GetFloat();
    int use_ba = sc.Input[3].GetYesNo();
    int hill_k = sc.Input[4].GetInt();
    int z_lookback = sc.Input[5].GetInt();

    // ------------- 1) build streams: delta, size, imbalance -------------
    // price move sign
    float close_curr = sc.Close[idx];
    float close_prev = (idx > 0 ? sc.Close[idx - 1] : close_curr);
    float vol_curr = sc.Volume[idx];

    // delta with carry sign
    static float last_sign = 0.0f;
    float sign = 0.0f;
    if (close_curr > close_prev) sign = 1.0f;
    else if (close_curr < close_prev) sign = -1.0f;
    else sign = last_sign;

    float delta_val = sign * vol_curr;
    last_sign = sign;
    sc.Subgraph[0][idx] = delta_val;

    // size
    sc.Subgraph[1][idx] = vol_curr;

    // imbalance
    float imb_val = 0.0f;
    if (use_ba)
    {
        // depends on what data is available in your chart; if none, stays 0
        float bidv = sc.BidVolume[idx];
        float askv = sc.AskVolume[idx];
        float denom = bidv + askv;
        if (denom > 0.0f)
            imb_val = ClipFloat((bidv - askv) / denom, -1.0f, 1.0f);
        else
            imb_val = 0.0f;
    }
    sc.Subgraph[2][idx] = imb_val;

    // ------------- 2) residuals (stream - rolling mean) -------------
    // We’ll store residuals in extra arrays (not displayed)
    SCFloatArrayRef delta_resid = sc.Subgraph[0].Arrays[0];
    SCFloatArrayRef size_resid  = sc.Subgraph[1].Arrays[0];
    SCFloatArrayRef imb_resid   = sc.Subgraph[2].Arrays[0];

    double mu_delta = RollingMean(sc.Subgraph[0], idx, lookback);
    double mu_size  = RollingMean(sc.Subgraph[1], idx, lookback);
    double mu_imb   = RollingMean(sc.Subgraph[2], idx, lookback);

    delta_resid[idx] = sc.Subgraph[0][idx] - (float)mu_delta;
    size_resid[idx]  = sc.Subgraph[1][idx] - (float)mu_size;
    imb_resid[idx]   = sc.Subgraph[2][idx] - (float)mu_imb;

    // ------------- 3) POT threshold for each stream -------------
    double thr_delta = RollingQuantile(delta_resid, idx, lookback, pot_q);
    double thr_size  = RollingQuantile(size_resid,  idx, lookback, pot_q);
    double thr_imb   = RollingQuantile(imb_resid,   idx, lookback, pot_q);

    // ------------- 4) collect exceedances and fit GPD -------------
    // We'll do it per stream
    const int MAXEX = 2000;
    double ex_delta[MAXEX];
    double ex_size[MAXEX];
    double ex_imb[MAXEX];
    int cnt_delta = 0, cnt_size = 0, cnt_imb = 0;

    int start = idx - lookback + 1;
    if (start < 0) start = 0;
    for (int i = start; i <= idx; ++i)
    {
        double rd = delta_resid[i];
        if (rd > thr_delta && cnt_delta < MAXEX)
            ex_delta[cnt_delta++] = rd - thr_delta;

        double rs = size_resid[i];
        if (rs > thr_size && cnt_size < MAXEX)
            ex_size[cnt_size++] = rs - thr_size;

        double ri = imb_resid[i];
        if (ri > thr_imb && cnt_imb < MAXEX)
            ex_imb[cnt_imb++] = ri - thr_imb;
    }

    double xi_d = 0.0, beta_d = 0.0;
    double xi_s = 0.0, beta_s = 0.0;
    double xi_i = 0.0, beta_i = 0.0;

    FitGPD_MoM(ex_delta, cnt_delta, xi_d, beta_d);
    FitGPD_MoM(ex_size,  cnt_size,  xi_s, beta_s);
    FitGPD_MoM(ex_imb,   cnt_imb,   xi_i, beta_i);

    // ------------- 5) Hill estimator (optional, just compute; not plotted) -------------
    double hill_delta = HillEstimator(delta_resid, idx, lookback, hill_k);
    double hill_size  = HillEstimator(size_resid,  idx, lookback, hill_k);
    double hill_imb   = HillEstimator(imb_resid,   idx, lookback, hill_k);
    // (you can print to log if you want, but we’ll keep it silent)

    // ------------- 6) compute tail severity per bar (as tail probability) -------------
    double tail_d = 0.0;
    double tail_s = 0.0;
    double tail_i = 0.0;

    double rd_curr = delta_resid[idx];
    double rs_curr = size_resid[idx];
    double ri_curr = imb_resid[idx];

    if (rd_curr > thr_delta && cnt_delta > 1)
        tail_d = GPDTailProb(rd_curr - thr_delta, xi_d, beta_d);
    else
        tail_d = 0.0;

    if (rs_curr > thr_size && cnt_size > 1)
        tail_s = GPDTailProb(rs_curr - thr_size, xi_s, beta_s);
    else
        tail_s = 0.0;

    if (ri_curr > thr_imb && cnt_imb > 1)
        tail_i = GPDTailProb(ri_curr - thr_imb, xi_i, beta_i);
    else
        tail_i = 0.0;

    // store tail probs
    sc.Subgraph[3][idx] = (float)tail_d;
    sc.Subgraph[4][idx] = (float)tail_s;
    sc.Subgraph[5][idx] = (float)tail_i;

    // ------------- 7) standardize the outcomes (z over tail-prob) -------------
    double mu_td = RollingMean(sc.Subgraph[3], idx, z_lookback);
    double sd_td = RollingStd(sc.Subgraph[3], idx, z_lookback, mu_td);
    double z_td = 0.0;
    if (sd_td > 0.0)
        z_td = (tail_d - mu_td) / sd_td;

    double mu_ts = RollingMean(sc.Subgraph[4], idx, z_lookback);
    double sd_ts = RollingStd(sc.Subgraph[4], idx, z_lookback, mu_ts);
    double z_ts = 0.0;
    if (sd_ts > 0.0)
        z_ts = (tail_s - mu_ts) / sd_ts;

    double mu_ti = RollingMean(sc.Subgraph[5], idx, z_lookback);
    double sd_ti = RollingStd(sc.Subgraph[5], idx, z_lookback, mu_ti);
    double z_ti = 0.0;
    if (sd_ti > 0.0)
        z_ti = (tail_i - mu_ti) / sd_ti;

    sc.Subgraph[6][idx] = (float)z_td;
    sc.Subgraph[7][idx] = (float)z_ts;
    sc.Subgraph[8][idx] = (float)z_ti;

    // ------------- 8) alerting / marking -------------
    bool alert_bar = false;
    if (fabs(z_td) >= z_alert) alert_bar = true;
    if (fabs(z_ts) >= z_alert) alert_bar = true;
    if (fabs(z_ti) >= z_alert) alert_bar = true;

    if (alert_bar)
    {
        sc.Subgraph[9][idx] = 1.0f;
        // optional: sc.AddMessageToLog("Orderflow tail anomaly detected", 0);
    }
    else
    {
        sc.Subgraph[9][idx] = 0.0f;
    }
}
