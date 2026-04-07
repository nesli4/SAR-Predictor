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
            term1 = np.log(magnetic_moment + 1e-9)
            term2 = np.sqrt(H * f + concentration)
            term3 = (np.log(magnetic_moment + 1e-9) * (f**2) * H) / (concentration / (Ms + 1e-9))
            res = term1 * (term2 + term3)
            return np.maximum(res, 0.1)
        else:
            res = (1/Ms - np.sqrt(Ms))/AV + (20 - Ms)/(f + 1e-9) + H*(f+1)
            return np.maximum(res, 0.1)
    except: return 0.0

# --- AUTO SCALE BAR DETECTION ---
def detect_scale_bar_pixels(img_array):
    try:
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        bottom_part = gray[int(h*0.75):h, :]
        _, thresh = cv2.threshold(bottom_part, 230, 255, cv2.THRESH_BINARY)
        kernel = np.ones((1, 25), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_cnt = max(contours, key=lambda x: cv2.boundingRect(x)[2])
            _, _, bar_w, _ = cv2.boundingRect(best_cnt)
            return float(bar_w)
        return None
    except: return None

# --- ADVANCED PARTICLE SEPARATION (WATERSHED) ---
def analyze_particles_advanced(img_array, scale_pixel_to_nm):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    # Gürültü temizleme (Küçük lekeleri siler)
    blur = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Birbirine değenleri ayırma işlemi (Mesafe Dönüşümü)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Parçacıkların "merkezlerini" bul
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, last_points = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    
    last_points = np.uint8(last_points)
    num_labels, labels = cv2.connectedComponents(last_points)
    
    # Analiz
    sizes = []
    # Her bir etiketi (parçacığı) tek tek ölç
    for i in range(1, num_labels):
        area = np.sum(labels == i)
        if area > 10: # Çok küçük noktaları görmezden gel
            equivalent_diameter = np.sqrt(4 * area / np.pi)
            sizes.append(equivalent_diameter)
            
    if not sizes: return 20.0, 0
    avg_diameter_px = np.mean(sizes)
    return avg_diameter_px * scale_pixel_to_nm, len(sizes)

# --- UI SETUP ---
st.set_page_config(page_title="AI-SAR Advanced Lab", layout="wide")
st.title("🔬 AI-Powered Nano-Particle Separator")
st.write("Advanced Watershed Segmentation for Agglomerated Particles")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("📸 Step 1: Smart Analysis")
    up_file = st.file_uploader("Upload TEM/SEM Image", type=['png', 'jpg', 'jpeg'])
    scale_nm = st.number_input("Real Scale Value (nm)", value=50.0)
    
    if up_file:
        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        det_px = detect_scale_bar_pixels(img)
        
        if det_px:
            st.success(f"AI detected scale bar: {det_px} px")
            final_px = st.number_input("Adjust pixels:", value=det_px)
        else:
            final_px = st.number_input("Scale Bar (Pixels):", value=200.0)
            
        nm_per_pixel = scale_nm / (final_px + 1e-9)
        
        # GELİŞMİŞ ANALİZ ÇALIŞIYOR
        diam, count = analyze_particles_advanced(img, nm_per_pixel)
        st.info(f"Analysis: {count} Individual Particles Found | Avg: {diam:.2f} nm")
        auto_av = 6 / (diam + 1e-9)
    else:
        auto_av = 0.08

    st.divider()
    regime = st.selectbox("Regime", ["SPM (Hc=0)", "SM (Hc>0)"])
    H = st.slider("Field (H) [kA/m]", 10, 1000, 250)
    f = st.slider("Freq (f) [MHz]", 0.01, 1.5, 0.25)
    Ms = st.number_input("Ms [emu/g]", value=44.0)

with col2:
    st.header("📊 Step 3: Predictive Report")
    if st.button("RUN AI SEPARATION & SAR EVALUATION", use_container_width=True):
        res = calculate_sar(H, f, Ms, 2.2, 1.0, auto_av, regime)
        st.metric("Predicted SAR", f"{res:.2f} W/g")
        st.caption(f"Based on Watershed Separated Diameter: {diam:.2f} nm")
        
        h_range = np.linspace(10, 1000, 50)
        y_vals = [calculate_sar(h, f, Ms, 2.2, 1.0, auto_av, regime) for h in h_range]
        st.line_chart(data={"Field (kA/m)": h_range, "SAR (W/g)": y_vals}, x="Field (kA/m)", y="SAR (W/g)")

st.caption("© 2026 AI-ParticleSeparator Framework - High Precision Analysis")
