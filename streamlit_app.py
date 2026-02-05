# streamlit_app.py
# EL image → Dark I–V (module) with robust per-cell diode solve
# Fixes included:
# - Adds Ns (series cell count) and uses per-cell voltage in the diode exponent
# - Guards coarse search so 'best' is always valid; clips exp() argument
# - Handles NaNs/overflow robustly

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="EL → Dark I–V (Module)", layout="wide")

st.title("EL image → Dark I–V (Module)")

st.markdown("""
Upload **EL images** at **multiple forward-bias setpoints** and a **metadata CSV**:

- CSV columns (header required): `filename, V_applied_V, I_meas_A, exposure_s, gain_iso, temp_C`
- Optional: dark frame image and flat-field image for camera corrections
- The app:
  1) Computes a trimmed log-intensity per image,
  2) Fits EL slope → ideality `n`,
  3) Fits single-diode dark parameters (`I0, n, Rs, Rsh`),
  4) Generates and exports a dark I–V sweep.
""")

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Preprocessing")
    border_crop_px = st.number_input("Crop border (pixels)", min_value=0, max_value=500, value=10, step=2)
    trim_low = st.slider("Trim lowest % pixels", 0, 40, 5, step=1)
    trim_high = st.slider("Trim highest % pixels", 0, 40, 5, step=1)
    eps = st.number_input("Min intensity clamp (epsilon)", 1e-12, 1e-3, 1e-6, format="%.1e")
    roi_box = st.text_input("ROI (x0,y0,w,h) optional", "")

    st.header("Fitting")
    # Physical constants
    q = 1.602176634e-19
    kB = 1.380649e-23
    # Ns: number of cells in series (IMPORTANT for per-cell voltage)
    Ns = st.number_input("Series cell count (Ns)", min_value=1, max_value=200, value=60, step=1)
    ref_T = st.number_input("Assume module T for EL slope (°C)", -20.0, 120.0, 25.0)
    n_reg = st.number_input("n regularization weight (0=off)", 0.0, 10.0, 1.0, step=0.5)
    max_iter = st.number_input("Max iterations (implicit solve)", 10, 500, 60, step=10)

    st.header("Sweep")
    v_min = st.number_input("Vmin for dark sweep (V)", 0.0, 2000.0, 0.0)
    v_max = st.number_input("Vmax for dark sweep (V)", 0.1, 2000.0, 40.0)
    v_pts = st.number_input("Points", 10, 5000, 400)

# -----------------------
# File inputs
# -----------------------
csv_file = st.file_uploader("Metadata CSV", type=["csv"])
imgs = st.file_uploader("EL images (multi-select)", type=["png","tif","tiff","jpg","jpeg"], accept_multiple_files=True)
dark_frame_file = st.file_uploader("Optional: dark frame", type=["png","tif","tiff","jpg","jpeg"])
flat_field_file = st.file_uploader("Optional: flat-field", type=["png","tif","tiff","jpg","jpeg"])

if not csv_file or not imgs:
    st.info("Upload metadata CSV and at least 3 EL images to begin.")
    st.stop()

# -----------------------
# Load metadata
# -----------------------
meta = pd.read_csv(csv_file)
required_cols = ["filename","V_applied_V","I_meas_A","exposure_s","gain_iso","temp_C"]
missing = [c for c in required_cols if c not in meta.columns]
if missing:
    st.error(f"CSV missing columns: {missing}")
    st.stop()

# Index uploaded images by name
img_by_name = {f.name: f for f in imgs}

# -----------------------
# Load helper images
# -----------------------
def load_pil(file):
    try:
        if file is None:
            return None
        return Image.open(file)
    except Exception:
        return None

dark_frame = load_pil(dark_frame_file)
flat_field = load_pil(flat_field_file)

# -----------------------
# Image helpers
# -----------------------
def pil_to_float(im: Image.Image) -> np.ndarray:
    """Convert PIL image to float64, normalized to ~[0, 1]."""
    if im is None:
        return None
    if im.mode in ("I;16", "I"):
        arr = np.array(im, dtype=np.float64)
        # If 16-bit like I;16: use 65535 for scaling; else just max() to be safe
        scale = 65535.0 if arr.max() > 255 else float(arr.max() or 1.0)
        arr = arr / (scale if scale > 0 else 1.0)
    else:
        arr = np.array(im.convert("L"), dtype=np.float64)
        scale = float(arr.max() or 1.0)
        arr = arr / (scale if scale > 0 else 1.0)
    return arr

def apply_corrections(arr: np.ndarray, dark: Image.Image = None, flat: Image.Image = None) -> np.ndarray:
    a = arr.copy()
    if dark is not None:
        d = pil_to_float(dark)
        if d is not None:
            if d.shape != a.shape:
                d = np.resize(d, a.shape)
            a = np.clip(a - d, 0, None)
    if flat is not None:
        f = pil_to_float(flat)
        if f is not None:
            if f.shape != a.shape:
                f = np.resize(f, a.shape)
            f = np.where(f <= 0, 1.0, f)
            a = a / f
    return a

def crop_to_roi(arr: np.ndarray, border_px=0, roi_spec=""):
    h, w = arr.shape
    x0, y0, ww, hh = 0, 0, w, h
    if roi_spec.strip():
        try:
            x0, y0, ww, hh = [int(v) for v in roi_spec.split(",")]
            x0 = np.clip(x0, 0, w-1); y0 = np.clip(y0, 0, h-1)
            ww = np.clip(ww, 1, w-x0); hh = np.clip(hh, 1, h-y0)
        except Exception:
            pass
    arr2 = arr[y0:y0+hh, x0:x0+ww]
    if border_px > 0 and arr2.shape[0] > 2*border_px and arr2.shape[1] > 2*border_px:
        arr2 = arr2[border_px:-border_px, border_px:-border_px]
    return arr2

# -----------------------
# Compute robust metric per image
# -----------------------
rows = []
for _, r in meta.iterrows():
    name = str(r["filename"])
    if name not in img_by_name:
        st.error(f"Image '{name}' not uploaded.")
        st.stop()
    im = Image.open(img_by_name[name])
    arr = pil_to_float(im)
    arr = apply_corrections(arr, dark_frame, flat_field)
    arr = crop_to_roi(arr, border_px=border_crop_px, roi_spec=roi_box)
    arr = np.clip(arr, eps, None)

    # robust trimmed log-mean
    logA = np.log(arr)
    flat = np.sort(logA.flatten())
    n = len(flat)
    lo = int(n * (trim_low/100.0))
    hi = int(n * (1.0 - trim_high/100.0))
    lo = np.clip(lo, 0, n-1); hi = np.clip(hi, lo+1, n)
    trimmed = flat[lo:hi]
    log_mean = float(np.mean(trimmed))
    rows.append({
        "filename": name,
        "log_mean": log_mean,
        "V": float(r["V_applied_V"]),
        "I": float(r["I_meas_A"]),
        "exposure_s": float(r["exposure_s"]),
        "gain_iso": float(r["gain_iso"]),
        "temp_C": float(r["temp_C"]),
    })

df = pd.DataFrame(rows).sort_values("V").reset_index(drop=True)

# -----------------------
# EL slope fit → ideality n (using per-cell slope)
# -----------------------
T_K = (df["temp_C"].mean() if df["temp_C"].notna().all() else ref_T) + 273.15

A = np.vstack([np.ones(len(df)), df["V"].values]).T
y = df["log_mean"].values
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
b = coef[1]  # slope vs module voltage
# n ≈ q * Ns / (b * kT)
n_EL = float(q * Ns / (b * kB * T_K)) if b > 0 else 2.0

col1, col2 = st.columns(2)
with col1:
    st.write("**EL fit (log-intensity vs Vmodule):**")
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(df["V"], df["log_mean"], color="tab:blue", label="data")
    ax.plot(df["V"], coef[0]+coef[1]*df["V"], color="tab:orange", label=f"fit: n ≈ {n_EL:.2f}")
    ax.set_xlabel("Applied V (module, V)"); ax.set_ylabel("log mean intensity")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
with col2:
    st.metric("Estimated ideality n (EL-derived)", f"{n_EL:.2f}")

# ======================================================
# Single-diode (dark) model fit  — PASTE-IN BLOCK START
# I = I0*(exp(q*Vd_cell/(n*kT)) - 1) + Vd/Rsh
# with Vd = Vmodule - I*Rs,  Vd_cell = Vd/Ns
# ======================================================
V_meas = df["V"].values.astype(float)
I_meas = df["I"].values.astype(float)

def solve_current(V_module, I0, n, Rs, Rsh, T=T_K, Ns=Ns, iters=60):
    """Fixed-point solve for I at module voltage, using per-cell junction voltage."""
    V_module = np.asarray(V_module, dtype=float)
    I = np.zeros_like(V_module)
    Ns_eff = max(int(Ns), 1)
    for _ in range(int(iters)):
        Vd = V_module - I*Rs         # module-level diode voltage
        Vd_cell = Vd / Ns_eff        # per-cell diode voltage
        # Clip the exponent to avoid overflow/underflow
        expo_arg = np.clip(q * Vd_cell / (n * kB * T), -100.0, 100.0)
        Id = I0 * (np.exp(expo_arg) - 1.0)
        Ish = Vd / Rsh if Rsh > 0 else 0.0
        I_new = Id + Ish
        # Numerical safety
        if not np.all(np.isfinite(I_new)):
            I_new = np.nan_to_num(I_new, nan=0.0, posinf=1e30, neginf=-1e30)
        # Simple damping for convergence
        if np.max(np.abs(I_new - I)) < 1e-9:
            I = I_new
            break
        I = 0.5*I + 0.5*I_new
    return I

def loss(params):
    I0, n, Rs, Rsh = params
    # Guard invalid regions
    if I0 <= 0 or n <= 0 or Rs < 0 or Rsh <= 0:
        return 1e99
    I_pred = solve_current(V_meas, I0, n, Rs, Rsh, T=T_K, Ns=Ns, iters=max_iter)
    if not np.all(np.isfinite(I_pred)):
        return 1e99
    # Gentle regularization of n around EL-derived value
    reg = n_reg * (n - n_EL)**2
    return float(np.mean((I_pred - I_meas)**2) + reg)

# -----------------------
# Coarse search with safe initialization
# -----------------------
rng = np.random.default_rng(42)

def safe_seed():
    # A conservative seed that evaluates everywhere
    return [1e-9, max(1.0, min(2.0, float(n_EL))), 0.1, 1e4]  # [I0, n, Rs, Rsh]

best = safe_seed()
best_loss = loss(best)
if not np.isfinite(best_loss):
    best_loss = 1e99

# Modest grids (UI-friendly; widen if your devices differ significantly)
I0_grid  = np.logspace(-12, -7, 6)                             # A
n_grid   = np.linspace(max(1.0, n_EL-0.4), min(2.5, n_EL+0.4), 6)
Rs_grid  = np.linspace(0.0, 1.0, 6)                             # Ω (module)
Rsh_grid = np.logspace(2, 5, 6)                                 # Ω

for I0 in I0_grid:
    for n_try in n_grid:
        for Rs in Rs_grid:
            for Rsh in Rsh_grid:
                L = loss((I0, n_try, Rs, Rsh))
                if np.isfinite(L) and L < best_loss:
                    best_loss = L
                    best = [float(I0), float(n_try), float(Rs), float(Rsh)]

# If nothing better was found, ensure best is still a valid seed
if (not isinstance(best, (list, tuple))) or len(best) != 4 or (not np.isfinite(best_loss)):
    best = safe_seed()
    best_loss = loss(best)

# -----------------------
# Random refinement around current best
# -----------------------
for _ in range(200):
    trial = [
        max(1e-14, best[0] * 10**rng.normal(0, 0.2)),
        float(np.clip(best[1] + rng.normal(0, 0.05), 0.9, 3.0)),
        float(np.clip(best[2] + rng.normal(0, 0.05), 0.0, 5.0)),
        float(np.clip(best[3] * 10**rng.normal(0, 0.2), 1.0, 1e7)),
    ]
    L = loss(trial)
    if np.isfinite(L) and L < best_loss:
        best_loss = L
        best = trial

I0_fit, n_fit, Rs_fit, Rsh_fit = best
st.write(
    f"**Fitted dark parameters @ {T_K:.1f} K (Ns={Ns}):**  "
    f"I0 = {I0_fit:.3e} A,  n = {n_fit:.2f},  Rs = {Rs_fit:.3f} Ω,  Rsh = {Rsh_fit:.1f} Ω"
)

# -----------------------
# Generate dark I–V sweep & plot
# -----------------------
V_sweep = np.linspace(v_min, v_max, int(v_pts))
I_sweep = solve_current(V_sweep, I0_fit, n_fit, Rs_fit, Rsh_fit, T=T_K, Ns=Ns, iters=max_iter)

fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(V_sweep, I_sweep, label="Dark I–V (fit)")
ax2.scatter(V_meas, I_meas, color="tab:red", zorder=5, label="Measured points")
ax2.set_xlabel("Voltage (V, module)"); ax2.set_ylabel("Current (A)")
ax2.grid(True, alpha=0.3); ax2.legend()
st.pyplot(fig2)

# -----------------------
# Export CSV
# -----------------------
out = pd.DataFrame({"V_V": V_sweep, "I_A": I_sweep})
st.download_button("Download dark IV (CSV)", out.to_csv(index=False).encode("utf-8"),
                   file_name="dark_IV_from_EL.csv", mime="text/csv")
# =========================
# Single-diode fit — PASTE-IN BLOCK END
# =========================
