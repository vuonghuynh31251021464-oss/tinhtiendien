import pandas as pd
import streamlit as st
import os
from datetime import datetime

# Deep Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ================= CONFIG =================
st.set_page_config(page_title="AI Tiền Điện", layout="wide")

st.title("⚡ AI Dự Đoán Tiền Điện (Deep Learning)")

# ================= LOAD DATA =================
df = pd.read_csv("200_ho_gia_dinh_tien_dien.csv")

# Encode
df["Co_tu_lanh"] = df["Co_tu_lanh"].map({"Co": 1, "Khong": 0})
df = pd.get_dummies(df, columns=["Loai_nha"])

X = df.drop(["Tien_dien_thang_VND", "STT"], axis=1)
y = df["Tien_dien_thang_VND"]

# ================= SCALE =================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ================= TRAIN =================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=50, verbose=0)

# ================= EVALUATE =================
y_pred = model.predict(X_test).flatten()

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("🎯 R2", f"{r2:.2f}")
col2.metric("📉 MAE", f"{int(mae):,} VND")

# ================= INPUT =================
st.subheader("Nhập dữ liệu")

col1, col2 = st.columns(2)

with col1:
    so_nguoi = st.slider("Số người", 1, 10, 2)
    so_may_lanh = st.slider("Máy lạnh", 0, 5, 1)
    so_quat = st.slider("Quạt", 0, 10, 2)
    co_tu_lanh = st.selectbox("Tủ lạnh", ["Co", "Khong"])

with col2:
    gio_may_lanh = st.slider("Giờ máy lạnh", 0.0, 24.0, 5.0)
    dien_tich = st.slider("Diện tích", 10, 200, 30)
    tang = st.slider("Tầng", 1, 30, 1)
    loai_nha = st.selectbox("Loại nhà", ["Phong tro", "Can ho", "Nha cap 4", "Nha pho"])

# ================= PREP INPUT =================
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
input_scaled = scaler.transform(input_df)

# ================= PREDICT =================
if st.button("🚀 Dự đoán"):
    pred = model.predict(input_scaled)[0][0]

    st.success(f"💰 {int(pred):,} VND / tháng")

    # ===== LƯU LỊCH SỬ =====
    history_file = "history.csv"

    record = input_dict.copy()
    record["Tien_du_doan"] = int(pred)
    record["Thoi_gian"] = datetime.now()

    if os.path.exists(history_file):
        old = pd.read_csv(history_file)
        new = pd.concat([old, pd.DataFrame([record])], ignore_index=True)
    else:
        new = pd.DataFrame([record])

    new.to_csv(history_file, index=False)

# ================= HIỂN THỊ HISTORY =================
st.subheader("📜 Lịch sử dự đoán")

if os.path.exists("history.csv"):
    history_df = pd.read_csv("history.csv")
    st.dataframe(history_df.tail(10))
else:
    st.write("Chưa có dữ liệu")
