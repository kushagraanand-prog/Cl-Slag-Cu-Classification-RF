import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Cl Slag Cu loss Prediction", layout="wide")

# ===============================
# Load Model & Scaler
# ===============================
model = joblib.load("cu_rf_model.pkl")
scaler = joblib.load("cu_scaler.pkl")

st.title("Cl Slag Cu Range considered(0.70–0.75  &  0.80–0.85)")
st.markdown("### Enter process parameters:")

# ===============================
# 1. BLEND GROUP
# ===============================
st.markdown("## Blend Composition")
col1, col2, col3 = st.columns(3)

with col1:
    Fe = st.number_input("Fe (%)", value=0.0)
    SiO2 = st.number_input("SiO₂ (%)", value=0.0)

with col2:
    CaO = st.number_input("CaO (%)", value=0.0)
    MgO = st.number_input("MgO (%)", value=0.0)

with col3:
    Al2O3 = st.number_input("Al₂O₃ (%)", value=0.0)
    S_Cu = st.number_input("S/Cu Ratio", value=0.0)


# ===============================
# 2. S-FURNACE PARAMETERS
# ===============================
st.markdown("## S-Furnace Parameters")
col4, col5 = st.columns(2)

with col4:
    conc_feed = st.number_input("Concentrate Feed Rate", value=0.0)
    silica_feed = st.number_input("Silica Feed Rate", value=0.0)
    cslag_feed = st.number_input("C-Slag S-Furnace Feed Rate", value=0.0)

with col5:
    s_air = st.number_input("S-Furnace Air", value=0.0)
    s_oxygen = st.number_input("S-Furnace Oxygen", value=0.0)


# ===============================
# 3. Fe/SiO2 + Fe3O4 (CLS)
# ===============================
st.markdown("## Fe/SiO₂ & Fe₃O₄(CL Slag) Composition")
col6, col7 = st.columns(2)

with col6:
    fe_sio2_ratio = st.number_input("Fe/SiO₂ Ratio", value=0.0)
with col7:
    fe3o4_cls = st.number_input("Fe₃O₄ (CLS)", value=0.0)


# ===============================
# 4. MATTE GRADE
# ===============================
st.markdown("## Matte Grade")
matte_grade = st.number_input("Matte Grade (%)", value=0.0)


# ===============================
# 5. C-SLAG ANALYSIS
# ===============================
st.markdown("##  C-Slag Analysis")
col8, col9, col10, col11 = st.columns(4)

with col8:
    cu_cslag = st.number_input("Cu(%)", value=0.0)
with col9:
    fe_cslag = st.number_input("Fe(%)", value=0.0)
with col10:
    cao_cslag = st.number_input("CaO(%)", value=0.0)
with col11:
    fe3o4_cslag = st.number_input("Fe₃O₄(%)", value=0.0)


# ===============================
# Collect all inputs into 1 feature array
# (Ensure order matches training data!)
# ===============================
features = np.array([[
    Fe, SiO2, Al2O3, CaO, MgO, S_Cu,
    conc_feed, silica_feed, cslag_feed, s_air, s_oxygen,
    fe_sio2_ratio, fe3o4_cls,
    matte_grade,
    cu_cslag, fe_cslag, cao_cslag, fe3o4_cslag
]])



# ===============================
# Prediction Section
# ===============================
st.markdown("---")

if st.button("Predict Cl Slag Cu Class"):
    X_scaled = scaler.transform(features)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    if pred == 1:
        st.success(f"(0.70–0.75 Cu%) — Probability: {proba[1]:.2f}")
    else:
        st.error(f"(0.80–0.85 Cu%) — Probability: {proba[0]:.2f}")
