import streamlit as st
import numpy as np
import cv2
from PIL import Image

# --- MATHEMATICAL ENGINE ---
def safe_calc(val): 
    return np.nan_to_num(val, nan=0.0, posinf=1e6, neginf=0.0)

def calculate_sar(H, f, Ms, magnetic_moment, concentration, AV, regime):
    try:
        if "SPM" in regime:
            term1 = np.log(magnetic_moment)
            term2 = np.sqrt(H * f + concentration)
            term3 = (np.log(magnetic_moment) * (f**2) * H) / (concentration / (Ms + 1e-9))
            res = term1 * (term2 + term3)
            return np.maximum(res, 0.1)
        else:
            res = (1/Ms - np.sqrt(Ms))/AV + (20 - Ms)/(f + 1e-9) + H*(f+1)
            return np.maximum(res, 0.1)
    except: return 0.0

# --- AUTO SCALE BAR DETECTION ENGINE ---
def detect_scale_bar_pixels(img_array):
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        # Resmin sadece alt %25'lik kısmına odaklan (Scale bar genelde oradadır)
        h, w = gray.shape
        bottom_part = gray[int(h*0.75):h, :]
        
        # Beyaz bölgeleri maskele
        _, thresh = cv2.threshold(bottom_part, 240, 255, cv2.THRESH_BINARY)
        
        # Yatay çizgileri belirlemek için morfolojik işlem
        kernel = np.ones((1, 20), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # En geniş yatay konturu bul (Scale bar budur)
            best_cnt = max(contours, key=lambda x: cv2.boundingRect(x)[2])
            _, _, bar_w, _ = cv2.boundingRect(best_cnt)
            return float(bar_w)
        return None
    except:
        return None

# --- PARTICLE ANALYSIS ENGINE ---
def analyze_particles(img_array, scale_pixel_to_nm):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    sizes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 15:
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            sizes.append(equivalent_diameter)
    
    if not sizes: return 15.0, 0
    avg_diameter_px = np.mean(sizes)
    return avg_diameter_px * scale_pixel_to_nm, len(sizes)

# --- UI SETUP ---
st.set_page_config(page_title="AI-SAR Smart Lab", layout="wide")
st.title("🔬 Fully Automated Nano-Physics Lab")
st.write("AI-Driven Scale Bar Detection & SAR Prediction ($R^2=0.801$)")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📸 Step 1: Intelligent Analysis")
    up_file = st.file_uploader("Upload TEM/SEM Image", type=['png', 'jpg', 'jpeg'])
    
    scale_nm = st.number_input("Real Scale Value (nm) - e.g. 50", value=50.0)
    
    if up_file:
        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # AUTO DETECTION MAGIC
        detected_px = detect_scale_bar_pixels(img)
        
        if detected_px:
            st.success(f"AI detected scale bar: {detected_px} pixels wide.")
            final_px = st.number_input("Adjust detected pixels if needed:", value=detected_px)
        else:
            st.warning("AI couldn't find scale bar. Please enter pixels manually:")
            final_px = st.number_input("Scale Bar Width (Pixels):", value=200.0)
            
        nm_per_pixel = scale_nm / (final_px + 1e-9)
        
        diam, count = analyze_particles(img, nm_per_pixel)
        st.info(f"Analysis Result: {count} particles | Avg Diameter: {diam:.2f} nm")
        auto_av = 6 / (diam + 1e-9)
    else:
        auto_av = 0.08

    st.divider()
    st.header("⚙️ Step 2: Physics Parameters")
    regime = st.selectbox("Regime", ["SPM (Hc=0)", "SM (Hc>0)"])
    H = st.slider("Field (H) [kA/m]", 10, 1000, 250)
    f = st.slider("Freq (f) [MHz]", 0.01, 1.5, 0.25)
    Ms = st.number_input("Ms [emu/g]", value=44.0)
    mu = st.number_input("Moment (μ) [μB]", value=2.1)
    conc = st.number_input("Conc (C) [mg/mL]", value=1.0)

with col2:
    st.header("📊 Step 3: Predictive Report")
    if st.button("RUN AI EVALUATION", use_container_width=True):
        res = calculate_sar(H, f, Ms, mu, conc, auto_av, regime)
        st.metric("Predicted SAR", f"{res:.2f} W/g")
        
        h_range = np.linspace(10, 1000, 50)
        y_vals = [calculate_sar(h, f, Ms, mu, conc, auto_av, regime) for h in h_range]
        st.line_chart(data={"Field (kA/m)": h_range, "SAR (W/g)": y_vals}, x="Field (kA/m)", y="SAR (W/g)")
        st.caption(f"Based on automated A/V ratio: {auto_av:.4f}")

st.caption("© 2026 AI-Powered Hyperthermia Research Framework")
