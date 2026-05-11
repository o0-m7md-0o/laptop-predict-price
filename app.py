import streamlit as st
import pandas as pd
import pickle
import datetime

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="AI Laptop Price Predictor",
    page_icon="💻",
    layout="centered"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
h1, h2, h3 {
    text-align: center;
}
.stButton>button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    with open('catboost_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    files = {
        "Company": "Company_label_encoder.pkl",
        "TypeName": "TypeName_label_encoder.pkl",
        "ScreenResolution": "ScreenResolution_label_encoder.pkl",
        "Cpu": "Cpu_label_encoder.pkl",
        "Memory": "Memory_label_encoder.pkl",
        "Gpu": "Gpu_label_encoder.pkl",
        "OpSys": "OpSys_label_encoder.pkl",
    }
    encoders = {}
    for key, file in files.items():
        with open(file, 'rb') as f:
            encoders[key] = pickle.load(f)
    return encoders

# ------------------ Load Dataset (optional now) ------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("laptopData.xlsx")
    except:
        try:
            df = pd.read_csv("laptop_data.csv")
        except:
            return pd.DataFrame()

    if "Price_euros" in df.columns:
        df.rename(columns={"Price_euros": "Price"}, inplace=True)

    return df

model = load_model()
encoders = load_encoders()
df = load_data()  # مش مستخدمة دلوقتي بس ممكن تحتاجها بعدين

# ------------------ Encode ------------------
def encode_input(data, encoders):
    encoded = {}
    for col, value in data.items():
        if col in encoders:
            try:
                encoded[col] = encoders[col].transform([value])[0]
            except:
                encoded[col] = -1
        else:
            encoded[col] = value
    return pd.DataFrame([encoded])

# ------------------ Header ------------------
st.markdown("""
<h1>💻 AI Laptop Price Predictor</h1>
<p style='text-align:center; color:gray;'>
Fast & Accurate Laptop Price Estimation using AI
</p>
""", unsafe_allow_html=True)

# ------------------ Form ------------------
with st.form("form"):

    col1, col2 = st.columns(2)

    with col1:
        Company = st.selectbox("🏢 Company", encoders["Company"].classes_)
        TypeName = st.selectbox("💼 Type", encoders["TypeName"].classes_)
        Inches = st.slider("📏 Screen Size", 10.0, 20.0, 15.6)
        Ram = st.selectbox("🧠 RAM (GB)", [4, 8, 16, 32, 64])

    with col2:
        ScreenResolution = st.selectbox("🖥 Resolution", encoders["ScreenResolution"].classes_)
        Cpu = st.selectbox("⚙ CPU", encoders["Cpu"].classes_)
        Memory = st.selectbox("💾 Storage", encoders["Memory"].classes_)
        Gpu = st.selectbox("🎮 GPU", encoders["Gpu"].classes_)
        OpSys = st.selectbox("💿 Operating System", encoders["OpSys"].classes_)
        Weight = st.slider("⚖ Weight (kg)", 0.5, 5.0, 2.0)

    submit = st.form_submit_button("🔮 Predict Price")

# ------------------ Prediction ------------------
if submit:

    if Inches <= 0 or Weight <= 0:
        st.error("❌ Invalid input values")
        st.stop()

    input_dict = {
        "Company": Company,
        "TypeName": TypeName,
        "Inches": Inches,
        "ScreenResolution": ScreenResolution,
        "Cpu": Cpu,
        "Ram": Ram,
        "Memory": Memory,
        "Gpu": Gpu,
        "OpSys": OpSys,
        "Weight": Weight
    }

    input_df = encode_input(input_dict, encoders)

    with st.spinner("🤖 AI is predicting..."):
        try:
            prediction = model.predict(input_df)[0]

            st.success("✅ Prediction Ready")

            st.metric("💰 Estimated Price", f"${round(prediction,2):,}")

            # Price Category
            if prediction < 500:
                st.success("💚 Budget Laptop")
            elif prediction < 1500:
                st.info("💙 Mid-range Laptop")
            else:
                st.warning("💎 Premium Laptop")

            # Save logs
            log = input_df.copy()
            log["prediction"] = prediction
            log["time"] = datetime.datetime.now()
            log.to_csv("logs.csv", mode="a", header=False, index=False)

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# ------------------ Footer ------------------
st.markdown("---")
st.caption("🚀 Built with Streamlit | AI Powered App")