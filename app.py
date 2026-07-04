from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "style.css"
MODEL_PATH = BASE_DIR / "model" / "car_price_model.pkl"

st.set_page_config(
    page_title="AutoVal",
    page_icon="🚗",
    layout="wide"
)

# Load CSS
with CSS_PATH.open(encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Model
model = joblib.load(MODEL_PATH)

# ----------------------------
# Dictionaries
# ----------------------------

car_names={
"Alto":0,
"City":1,
"Civic":2,
"Corolla":3,
"Cultus":4,
"Fortuner":5,
"Hilux":6,
"Mehran":7,
"Prius":8,
"Sportage":9,
"Swift":10,
"Tucson":11,
"Vitz":12,
"WagonR":13,
"Yaris":14
}

fuel={
"Petrol":2,
"Diesel":0,
"Hybrid":1,
"CNG":3
}

seller={
"Dealer":0,
"Individual":1
}

transmission={
"Automatic":0,
"Manual":1
}

st.markdown("""
<div class='top'>
<h1>🚗 AutoVal</h1>
<p>Price Intelligence Engine</p>
</div>
""",unsafe_allow_html=True)

left,right=st.columns([1,1])

with left:

    st.markdown("<div class='card'>",unsafe_allow_html=True)

    st.subheader("Vehicle Parameters")

    car=st.selectbox("Car Name",list(car_names.keys()))

    year=st.slider("Manufacturing Year",2010,2024,2020)

    present_price=st.number_input(
        "Present Price (Lakh)",
        min_value=1.0
    )

    kms=st.slider(
        "Kilometers Driven",
        0,
        200000,
        80000
    )

    fuel_type=st.radio(
        "Fuel Type",
        list(fuel.keys()),
        horizontal=True
    )

    gear=st.radio(
        "Transmission",
        list(transmission.keys()),
        horizontal=True
    )

    seller_type=st.selectbox(
        "Seller Type",
        list(seller.keys())
    )

    owner=st.radio(
        "Previous Owners",
        [0,1,2,3],
        horizontal=True
    )

    predict=st.button(
        "Predict Fair Value",
        use_container_width=True
    )

    st.markdown("</div>",unsafe_allow_html=True)

with right:

    st.markdown("<div class='card'>",unsafe_allow_html=True)

    st.subheader("Estimated Selling Price")

    if predict:

        sample=pd.DataFrame({

            "Car_Name":[car_names[car]],
            "Year":[year],
            "Present_Price":[present_price],
            "Kms_Driven":[kms],
            "Fuel_Type":[fuel[fuel_type]],
            "Seller_Type":[seller[seller_type]],
            "Transmission":[transmission[gear]],
            "Owner":[owner]

        })

        prediction=model.predict(sample)[0]

        st.markdown(f"""
        <div class='price'>
        ₹ {prediction:.2f} Lakh
        </div>
        """,unsafe_allow_html=True)

        st.progress(84)

        st.markdown("### Decision Flow")

        st.info(f"✔ Year : {year}")

        st.info(f"✔ {kms:,} km Driven")

        st.info(f"✔ {fuel_type} • {gear}")

        st.info(f"✔ Owner : {owner}")

    else:

        st.markdown("<br><br>",unsafe_allow_html=True)

        st.write("Prediction will appear here.")

    st.markdown("</div>",unsafe_allow_html=True)