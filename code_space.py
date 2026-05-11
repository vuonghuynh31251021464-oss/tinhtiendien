import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

# ================= CẤU HÌNH =================
st.set_page_config(page_title="Ezetric - AI Tiền Điện", layout="wide")

# ===== CSS HIỆN ĐẠI MÀU XANH =====
st.markdown("""
<style>
/* nền tổng */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* title */
.big-title {
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:#00c6ff;
}

/* card */
.block-container {
    padding: 2rem 3rem;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px;
}

/* button */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    font-size: 16px;
    height: 3em;
}

/* sidebar */
section[data-testid="stSidebar"] {
    background: #0f2027;
}
</style>
""", unsafe_allow_html=True)

# ===== TITLE =====
st.markdown('<p class="big-title">⚡ EZETRIC</p>', unsafe_allow_html=True)
st.markdown("<center>AI dự đoán tiền điện thông minh</center>", unsafe_allow_html=True)

# ================= SIDEBAR INFO =================
st.sidebar.markdown("## 📄 Thông tin dự án")
st.sidebar.write("""
- 📊 Dữ liệu: 200 hộ gia đình (giả lập)
- ⚠️ Bias: dữ liệu nhỏ
- 🧹 Data cleaning: encode + one-hot
- 🤖 Model: Random Forest
""")

# ================= MENU =================
menu = st.sidebar.radio(
    "📌 Chức năng",
    ["📊 Khảo sát & Dự đoán", "📈 Phân tích dữ liệu"]
)

# ================= LOAD DATA =================
df = pd.read_csv("200_ho_gia_dinh_tien_dien.csv")

# ================= XỬ LÝ =================
df["Co_tu_lanh"] = df["Co_tu_lanh"].map({"Co": 1, "Khong": 0})
df = pd.get_dummies(df, columns=["Loai_nha"])

X = df.drop(["Tien_dien_thang_VND", "STT"], axis=1)
y = df["Tien_dien_thang_VND"]

# ================= TRAIN =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

# ================= MỤC 1 =================
if menu == "📊 Khảo sát & Dự đoán":

    st.header("🧠 Nhập thông tin")

    col1, col2 = st.columns(2)

    with col1:
        so_nguoi = st.slider("Số người ở", 1, 10, 2)
        so_may_lanh = st.slider("Số máy lạnh", 0, 5, 1)
        so_quat = st.slider("Số quạt", 0, 10, 2)
        co_tu_lanh = st.selectbox("Có tủ lạnh?", ["Co", "Khong"])

    with col2:
        gio_may_lanh = st.slider("Giờ máy lạnh", 0.0, 24.0, 5.0)
        dien_tich = st.slider("Diện tích", 10, 200, 30)
        tang = st.slider("Tầng", 1, 30, 1)
        loai_nha = st.selectbox("Loại nhà", ["Phong tro", "Can ho", "Nha cap 4", "Nha pho"])

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

    if st.button("🚀 Dự đoán ngay"):
        prediction = model.predict(input_df)[0]
        st.success(f"💰 {int(prediction):,} VND / tháng")

        history_file = "history.csv"

        record = input_dict.copy()
        record["Tien_du_doan"] = int(prediction)
        record["Thoi_gian"] = datetime.now()

        if os.path.exists(history_file):
            old = pd.read_csv(history_file)
            new = pd.concat([old, pd.DataFrame([record])], ignore_index=True)
        else:
            new = pd.DataFrame([record])

        new.to_csv(history_file, index=False)

    st.subheader("📜 Lịch sử")

    if os.path.exists("history.csv"):
        history_df = pd.read_csv("history.csv")
        st.dataframe(history_df.tail(10))
    else:
        st.write("Chưa có dữ liệu")

# ================= MỤC 2 =================
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
