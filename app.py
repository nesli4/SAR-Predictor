import streamlit as st
import numpy as np

# Güvenli Matematik
def safe_calc(val): return np.nan_to_num(val, nan=0.0)

st.set_page_config(page_title="AI-SAR Predictor", layout="wide")
st.title("🧬 AI-Powered SAR Prediction Framework")
st.write("Symbolic Regression Model ($R^2=0.801$)")

# Parametre Paneli
st.sidebar.header("🔬 Girişler")
H = st.sidebar.slider("Field (H)", 10, 1000, 250)
f = st.sidebar.slider("Freq (f)", 0.01, 1.0, 0.25)
Ms = st.sidebar.slider("Ms", 10, 150, 44)
mu = st.sidebar.slider("mu", 0.1, 5.0, 2.1)
C = st.sidebar.slider("Conc (C)", 0.1, 10.0, 1.0)
rejim = st.sidebar.selectbox("Rejim", ["SPM (Hc=0)", "SM (Hc>0)"])

def calculate_sar(H, f, Ms, mu, C, rejim):
    if "SPM" in rejim:
        # PDF'deki SPM Denklemi
        res = np.log(mu) * (np.sqrt(H*f + C) + (np.log(mu)*(f**2)*H)/(C/(Ms + 1e-9)))
    else:
        # PDF'deki SM Denklemi
        res = (1/Ms - np.sqrt(Ms))/0.08 + (20 - Ms)/(f + 1e-9) + H*(f+1)
    return safe_calc(res)

if st.button("HESAPLA", use_container_width=True):
    res = calculate_sar(H, f, Ms, mu, C, rejim)
    st.header(f"Sonuç: {res:.2f} W/g")
    h_vals = np.linspace(10, 1000, 20)
    y_vals = [calculate_sar(h, f, Ms, mu, C, rejim) for h in h_vals]
    st.line_chart(y_vals)
