# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from io import BytesIO

st.set_page_config(page_title="EL → Dark I–V (Module)", layout="wide")

st.title("EL image → Dark I–V (module)")

st.markdown("""
**Upload EL images at multiple forward-bias setpoints** + a small **metadata CSV**:

- CSV columns (header required): `filename, V_applied_V, I_meas_A, exposure_s, gain_iso, temp_C`
- Optional: dark frame image (same exposure), flat-field (for vignetting)
- The app estimates ideality `n` from EL slope, then fits a single-diode dark model (`J0, Rs, Rsh, n`) to your electrical data.
""")

# -----------------------
# Sidebar controls
# -----------------------
with st.sidebar:
    st.header("Preprocessing")
    border_crop_px = st.number_input("Crop border (pixels)", 0, 200, 10, step=2)
    trim_low = st.slider("Trim lowest % pixels", 0, 40, 5, step=1)
    trim_high = st.slider("Trim highest % pixels", 0, 40, 5, step=1)
    eps = st.number_input("Min intensity clamp (epsilon)", 1e-9, 1e-3, 1e-6, format="%.1e")
    roi_box = st.text_input("ROI (x0,y0,w,h) optional", "")

    st.header("Fitting")
    k_B = 1.380649e-23
    q = 1.602176634e-19
    ref_T = st.number_input("Assume module T for EL slope (°C)", -20.0, 120.0, 25.0)
    n_reg = st.number_input("n regularization weight (0=off)", 0.0, 10.0, 1.0, step=0.5)
    max_iter = st.number_input("Max iterations (Rs/Rsh implicit solve)", 10, 200, 50, step=10)

    st.header("Sweep")
    v_min = st.number_input("Vmin for dark sweep (V)", 0.0, 1000.0, 0.0)
    v_max = st.number_input("Vmax for dark sweep (V)", 0.1, 2000.0, 40.0)
    v_pts = st.number_input("Points", 10, 2000, 400)

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

# Load metadata
meta = pd.read_csv(csv_file)
required_cols = ["filename","V_applied_V","I_meas_A","exposure_s","gain_iso","temp_C"]
missing = [c for c in required_cols if c not in meta.columns]
if missing:
    st.error(f"CSV missing columns: {missing}")
    st.stop()

# Index images by name
img_by_name = {f.name: f for f in imgs}

# Load optional frames
def load_image(file):
    im = Image.open(file).convert("I") if file else None  # 'I' for 32-bit integer pixels if possible; else convert to L and np.float
    return im

dark_frame = load_image(dark_frame_file)
flat_field = load_image(flat_field_file)

def pil_to_float(im):
    if im.mode in ("I;16", "I"):
        arr = np.array(im, dtype=np.float64)
    else:
        arr = np.array(im.convert("L"), dtype=np.float64)  # grayscale
    # Normalize by max to keep numbers in a reasonable range (0..1 or so)
    m = arr.max()
    if m > 0:
        arr = arr / m
    return arr

def apply_corrections(arr, dark=None, flat=None):
    a = arr.copy()
    if dark is not None:
        d = pil_to_float(dark)
        d = d if d.shape == a.shape else np.resize(d, a.shape)
        a = np.clip(a - d, 0, None)
    if flat is not None:
        f = pil_to_float(flat)
        f = f if f.shape == a.shape else np.resize(f, a.shape)
        f = np.where(f <= 0, 1.0, f)
        a = a / f
    return a

def crop_to_roi(arr, border_px=0, roi_spec=""):
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
    if border_px > 0:
        arr2 = arr2[border_px:-border_px, border_px:-border_px]
    return arr2

# Compute robust log-intensity metric per image
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

# EL slope → ideality n
T_K = (df["temp_C"].mean() if df["temp_C"].notna().all() else ref_T) + 273.15
q = 1.602176634e-19
kB = 1.380649e-23

# linear fit: log_mean = a + b*V  =>  b ≈ q/(n kT)
A = np.vstack([np.ones(len(df)), df["V"].values]).T
y = df["log_mean"].values
coef, *_ = np.linalg.lstsq(A, y, rcond=None)
b = coef[1]
n_EL = float(q / (b * kB * T_K)) if b > 0 else 2.0

col1, col2 = st.columns(2)
with col1:
    st.write("**EL fit (log-intensity vs V):**")
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(df["V"], df["log_mean"], color="tab:blue", label="data")
    ax.plot(df["V"], coef[0]+coef[1]*df["V"], color="tab:orange", label=f"fit: n ≈ {n_EL:.2f}")
    ax.set_xlabel("Applied V (V)"); ax.set_ylabel("log mean intensity")
    ax.legend(); ax.grid(True, alpha=0.3)
    st.pyplot(fig)
with col2:
    st.metric("Estimated ideality n (EL-derived)", f"{n_EL:.2f}")

# Fit single-diode (dark) model to (V,I) with gentle n-regularization
# Model: I = I0*(exp(q*(V - I*Rs)/(n*kT)) - 1) + (V - I*Rs)/Rsh
# We'll fit on measured (V,I) to get (I0, Rs, Rsh, n), starting at n_EL and penalizing |n-n_EL|.

V_meas = df["V"].values
I_meas = df["I"].values

def solve_current(V, I0, n, Rs, Rsh, T=T_K, iters=50):
    """Fixed-point solve for I at given V (dark diode with Rs, Rsh)."""
    I = np.copy(V)*0  # initial guess 0 A
    for _ in range(iters):
        Vd = V - I*Rs
        Id = I0*(np.exp(q*Vd/(n*kB*T)) - 1.0)
        Ish = Vd/Rsh if Rsh>0 else 0.0
        I_new = Id + Ish
        if np.max(np.abs(I_new - I)) < 1e-9:
            break
        I = 0.5*I + 0.5*I_new
    return I

def loss(params):
    I0, n, Rs, Rsh = params
    I_pred = solve_current(V_meas, I0, n, Rs, Rsh, T_K, iters=int(max_iter))
    w = 1.0  # could weight by |I| if desired
    # regularize n near n_EL
    reg = n_reg * (n - n_EL)**2
    return np.mean((I_pred - I_meas)**2) + reg

# Simple grid / random search + small local tweak (keep robust & dependency-free)
rng = np.random.default_rng(42)
best = None
best_loss = 1e99
# search ranges (tune if needed)
I0_grid = np.logspace(-12, -6, 7)      # A
n_grid  = np.linspace(max(1.0, n_EL-0.5), min(2.5, n_EL+0.5), 6)
Rs_grid = np.linspace(0.0, 1.0, 6)      # Ω (module-level)
Rsh_grid= np.linspace(20.0, 1e5, 6)     # Ω

for I0 in I0_grid:
    for n_try in n_grid:
        for Rs in Rs_grid:
            for Rsh in Rsh_grid:
                L = loss((I0, n_try, Rs, Rsh))
                if L < best_loss:
                    best_loss = L
                    best = [I0, n_try, Rs, Rsh]

# small local random refinement
for _ in range(200):
    trial = [
        best[0]*10**rng.normal(0, 0.2),
        np.clip(best[1]+rng.normal(0,0.05), 0.9, 3.0),
        np.clip(best[2]+rng.normal(0,0.05), 0.0, 5.0),
        np.clip(best[3]*10**rng.normal(0,0.2), 1.0, 1e6),
    ]
    L = loss(trial)
    if L < best_loss:
        best_loss = L
        best = trial

I0_fit, n_fit, Rs_fit, Rsh_fit = best
st.write(f"**Fitted parameters (dark model at {T_K:.1f} K):**  "
         f"I0={I0_fit:.3e} A,  n={n_fit:.2f},  Rs={Rs_fit:.3f} Ω,  Rsh={Rsh_fit:.1f} Ω")

# Generate dark IV
V_sweep = np.linspace(v_min, v_max, int(v_pts))
I_sweep = solve_current(V_sweep, I0_fit, n_fit, Rs_fit, Rsh_fit, T_K, iters=int(max_iter))

fig2, ax2 = plt.subplots(figsize=(6,4))
ax2.plot(V_sweep, I_sweep, label="Dark I–V (fit)")
ax2.scatter(V_meas, I_meas, color="tab:red", zorder=5, label="Measured points")
ax2.set_xlabel("Voltage (V)"); ax2.set_ylabel("Current (A)")
ax2.grid(True, alpha=0.3); ax2.legend()
st.pyplot(fig2)

# Export
out = pd.DataFrame({"V_V": V_sweep, "I_A": I_sweep})
st.download_button("Download dark IV (CSV)", out.to_csv(index=False).encode("utf-8"),
                   file_name="dark_IV_from_EL.csv", mime="text/csv")
