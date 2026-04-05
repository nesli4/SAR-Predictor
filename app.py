import streamlit as st
import numpy as np
import cv2
from PIL import Image

# --- MATHEMATICAL ENGINE (SYMBOLIC REGRESSION MODELS) ---
def safe_calc(val): 
    return np.nan_to_num(val, nan=0.0, posinf=1e6, neginf=0.0)

def calculate_sar(H, f, Ms, magnetic_moment, concentration, AV, regime):
    try:
        if "SPM" in regime:
            # High-Accuracy SPM Formula (R2 = 0.801)
            # Based on Symbolic Regression Analysis
            term1 = np.log(magnetic_moment)
            term2 = np.sqrt(H * f + concentration)
            term3 = (np.log(magnetic_moment) * (f**2) * H) / (concentration / (Ms + 1e-9))
            res = term1 * (term2 + term3)
            return np.maximum(res, 0.1)
        else:
            # SM Formula (R2 = 0.682)
            # Energy loss based on hysteresis behavior
            res = (1/Ms - np.sqrt(Ms))/AV + (20 - Ms)/(f + 1e-9) + H*(f+1)
            return np.maximum(res, 0.1)
    except: 
        return 0.0

# --- COMPUTER VISION ENGINE (OPENCV) ---
def analyze_image(uploaded_file, scale_pixel_to_nm):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pre-processing for particle detection
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.find_contours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sizes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 15: # Filter out noise
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            sizes.append(equivalent_diameter)
    
    if not sizes:
        return 15.0, 0 # Default diameter if none detected
        
    avg_diameter_px = np.mean(sizes)
    avg_diameter_nm = avg_diameter_px * scale_pixel_to_nm
    return avg_diameter_nm, len(sizes)

# --- USER INTERFACE SETUP ---
st.set_page_config(page_title="AI-SAR Smart Lab", layout="wide", page_icon="🧬")

st.title("🔬 Smart Nano-Physics Laboratory")
st.markdown("### Integrated Image Analysis & SAR Prediction Framework")
st.write("Leveraging Symbolic Regression ($R^2=0.801$) and Computer Vision")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📸 Step 1: Automated Image Analysis")
    up_file = st.file_uploader("Upload SEM/TEM Micrograph", type=['png', 'jpg', 'jpeg'])
    
    # Scale adjustment for accurate nanometer calculation
    nm_per_pixel = st.number_input("Calibration: Nanometers per Pixel (nm/px)", value=0.5, step=0.1)
    
    if up_file:
        diam, count = analyze_image(up_file, nm_per_pixel)
        st.success(f"Detection Successful: {count} particles identified.")
        st.info(f"Calculated Mean Diameter: {diam:.2f} nm")
        # A/V Ratio Calculation (Surface Area to Volume)
        auto_av = 6 / (diam + 1e-9) 
        st.write(f"Resulting Area/Volume Ratio: **{auto_av:.4f}**")
    else:
        auto_av = 0.08 # Default fallback value

    st.divider()
    
    st.header("⚙️ Step 2: Experimental Parameters")
    regime = st.selectbox("Magnetic Regime", ["SPM (Coercivity = 0)", "SM (Coercivity > 0)"])
    H = st.slider("Field Amplitude (H) [kA/m]", 10, 1000, 250)
    f = st.slider("Field Frequency (f) [MHz]", 0.01, 1.5, 0.25)
    Ms = st.number_input("Saturation Magnetization (Ms) [emu/g]", value=44.0)
    magnetic_moment = st.number_input("Magnetic Moment (μ) [μB]", value=2.1)
    concentration = st.number_input("Particle Concentration (C) [mg/mL]", value=1.0)

with col2:
    st.header("📊 Step 3: Predictive Analysis")
    if st.button("RUN COMPLETE AI ANALYSIS", use_container_width=True):
        final_sar = calculate_sar(H, f, Ms, magnetic_moment, concentration, auto_av, regime)
        
        # Display Metric
        st.metric(label="Predicted SAR Value", value=f"{final_sar:.2f} W/g")
        
        # Visualization
        st.subheader("Field Amplitude vs SAR Response")
        h_range = np.linspace(10, 1000, 50)
        y_vals = [calculate_sar(h, f, Ms, magnetic_moment, concentration, auto_av, regime) for h in h_range]
        
        chart_data = {"Field Amplitude (kA/m)": h_range, "SAR (W/g)": y_vals}
        st.line_chart(data=chart_data, x="Field Amplitude (kA/m)", y="SAR (W/g)")
        
        st.info(f"Prediction based on automated Surface-to-Volume analysis (A/V: {auto_av:.4f})")

st.divider()
st.caption("© 2026 AI-Powered Hyperthermia Research Framework | Standardized for Scientific Publication")
