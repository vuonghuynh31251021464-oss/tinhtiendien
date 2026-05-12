import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

st.set_page_config(page_title="Ezetric - AI Tiền Điện", layout="wide")

st.markdown("""
<style>
.stSlider > div[data-baseweb="slider"] > div > div > div {
    background-color: #FFD700 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;800&display=swap');

.big-title {
    font-family: 'Poppins', sans-serif;
    font-size: clamp(60px, 8vw, 110px);
    font-weight: 800;
    text-align: center;

    background: linear-gradient(90deg, #00c6ff, #0072ff, #00eaff, #0072ff);
    background-size: 300%;

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    animation: shine 6s linear infinite;
}

@keyframes shine {
    0% { background-position: 0% }
    100% { background-position: 300% }
}

.sub-title {
    text-align: center;
    font-family: 'Poppins', sans-serif;
    font-size: 20px;
    color: #ccefff;
    margin-top: -15px;
    margin-bottom: 25px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.block-container {
    padding: 2rem 3rem;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px;
}

.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    font-size: 16px;
    height: 3em;
}

section[data-testid="stSidebar"] {
    background: #0f2027;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">⚡ EZETRIC</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI dự đoán tiền điện thông minh</p>', unsafe_allow_html=True)

st.sidebar.markdown("## 📄 Thông tin dự án")
st.sidebar.write("""
- 📊 Dữ liệu: 200 hộ gia đình (giả lập)
- ⚠️ Bias: dữ liệu nhỏ
- 🧹 Data cleaning: encode + one-hot
- 🤖 Model: Random Forest
""")

menu = st.sidebar.radio(
    "📌 Chức năng",
    ["📊 Khảo sát & Dự đoán", "📈 Phân tích dữ liệu"]
)

df = pd.read_csv("200_ho_gia_dinh_tien_dien.csv")

df["Co_tu_lanh"] = df["Co_tu_lanh"].map({"Co": 1, "Khong": 0})
df = pd.get_dummies(df, columns=["Loai_nha"])

df["So_kWh"] = (
    df["So_may_lanh"] * df["So_gio_bat_may_lanh_ngay"] * 1.2 +
    df["So_quat"] * 0.08 * 24 +
    df["So_nguoi_o"] * 1.5 +
    df["Co_tu_lanh"] * 30
)

X = df.drop(["Tien_dien_thang_VND", "STT", "So_kWh"], axis=1)
y = df["So_kWh"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

def tinh_tien_dien(kwh):
    bac = [
        (50, 1806),
        (50, 1866),
        (100, 2167),
        (100, 2729),
        (100, 3050),
        (float("inf"), 3151)
    ]
    tong = 0
    for muc, gia in bac:
        if kwh > muc:
            tong += muc * gia
            kwh -= muc
        else:
            tong += kwh * gia
            break
    return tong

if menu == "📊 Khảo sát & Dự đoán":

    st.header("🧠 Nhập thông tin")

    col1, col2 = st.columns(2)

    with col1:
        so_nguoi = st.slider("Số người ở", 1, 10, 2, key="so_nguoi")
        so_may_lanh = st.slider("Số máy lạnh", 0, 5, 1, key="so_may_lanh")
        so_quat = st.slider("Số quạt", 0, 10, 2, key="so_quat")
        co_tu_lanh = st.selectbox("Có tủ lạnh?", ["Co", "Khong"], key="co_tu_lanh")

    with col2:
        gio_may_lanh = st.slider("Giờ máy lạnh", 0.0, 24.0, 5.0, key="gio_may_lanh")
        dien_tich = st.slider("Diện tích", 10, 200, 30, key="dien_tich")
        tang = st.slider("Tầng", 1, 30, 1, key="tang")
        loai_nha = st.selectbox("Loại nhà",
            ["Phong tro", "Can ho", "Nha cap 4", "Nha pho"],
            key="loai_nha"
        )

    input_dict = {
        "So_nguoi_o": so_nguoi,
        "So_may_lanh": so_may_lanh,
        "So_quat": so_quat,
        "Co_tu_lanh": 1 if co_tu_lanh == "Co" else 0,
        "So_gio_bat_may_lanh_ngay": gio_may_lanh,
        "Dien_tich_phong_m2": dien_tich,
        "Tang_lau": tang,
    }

    for col in X.columns:
        if "Loai_nha_" in col:
            input_dict[col] = 1 if col == f"Loai_nha_{loai_nha}" else 0

    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    if st.button("🚀 Dự đoán ngay"):

        with st.spinner("🤖 AI đang phân tích dữ liệu..."):
            import time
            time.sleep(1.5)
            kwh_pred = model.predict(input_df)[0]
            tien = tinh_tien_dien(kwh_pred)

        placeholder = st.empty()
        final_money = int(tien)
        current = 0
        step = max(final_money // 50, 1)

        while current < final_money:
            current += step
            if current > final_money:
                current = final_money
            placeholder.success(f"💰 {current:,} VND / tháng")
            time.sleep(0.02)

        st.session_state.last_prediction = final_money

        history_file = "history.csv"

        record = input_dict.copy()
        record["Tien_du_doan"] = final_money
        record["Thoi_gian"] = datetime.now()

        if os.path.exists(history_file):
            old = pd.read_csv(history_file)
            new = pd.concat([old, pd.DataFrame([record])], ignore_index=True)
        else:
            new = pd.DataFrame([record])

        new.to_csv(history_file, index=False)

        for key in [
            "so_nguoi","so_may_lanh","so_quat","co_tu_lanh",
            "gio_may_lanh","dien_tich","tang","loai_nha"
        ]:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()

    if "last_prediction" in st.session_state:
        st.success(f"💰 {st.session_state.last_prediction:,} VND / tháng")

    st.subheader("📜 Lịch sử")

    if os.path.exists("history.csv"):
        history_df = pd.read_csv("history.csv")
        st.dataframe(history_df.tail(10))
    else:
        st.write("Chưa có dữ liệu")

elif menu == "📈 Phân tích dữ liệu":

    st.header("📊 Phân tích AI")

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    col1, col2, col3 = st.columns(3)
    col1.metric("R2", f"{r2:.2f}")
    col2.metric("MAE", f"{int(mae):,}")
    col3.metric("RMSE", f"{int(rmse):,}")

    st.subheader("So sánh thực tế vs dự đoán")

    fig = plt.figure()
    plt.scatter(y_test, y_pred)
    st.pyplot(fig)

    st.subheader("Ảnh hưởng yếu tố")

    importance = model.feature_importances_
    feature_names = X.columns

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig2 = plt.figure()
    plt.barh(imp_df["Feature"], imp_df["Importance"])
    plt.gca().invert_yaxis()
    st.pyplot(fig2)

    st.subheader("Dữ liệu mẫu")
    st.dataframe(df.head(20))
