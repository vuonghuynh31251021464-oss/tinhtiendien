import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ================= CẤU HÌNH TRANG =================
st.set_page_config(page_title="Dự đoán tiền điện", layout="wide")

st.markdown("""
<style>
.big-title {font-size:40px; font-weight:bold; color:#00c3ff;}
.card {
    background-color:#1e1e1e;
    padding:20px;
    border-radius:15px;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">⚡ AI Dự Đoán Tiền Điện</p>', unsafe_allow_html=True)

# ================= LOAD DATA =================
df = pd.read_csv("200_ho_gia_dinh_tien_dien.csv")

# ================= XỬ LÝ =================
df["Co_tu_lanh"] = df["Co_tu_lanh"].map({"Co": 1, "Khong": 0})
df = pd.get_dummies(df, columns=["Loai_nha"])

X = df.drop(["Tien_dien_thang_VND", "STT"], axis=1)
y = df["Tien_dien_thang_VND"]

# ================= TRAIN / TEST =================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=120, random_state=42)
model.fit(X_train, y_train)


# ================= ĐÁNH GIÁ =================
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

col1, col2 = st.columns(2)
col1.metric("🎯 R2 Score", f"{r2:.2f}")
col2.metric("📉 MAE (Sai số trung bình)", f"{int(mae):,} VND")
st.info(f"Model dự đoán với sai số trung bình ~ {int(mae):,} VND")

# ================= BIỂU ĐỒ =================
st.subheader("📊 So sánh Giá trị thật vs Dự đoán")

fig = plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Thực tế")
plt.ylabel("Dự đoán")
plt.title("Actual vs Predicted")

st.pyplot(fig)

# ================= FEATURE IMPORTANCE =================
st.subheader("🔥 Mức độ ảnh hưởng của các yếu tố")

importance = model.feature_importances_
feature_names = X.columns

imp_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig2 = plt.figure()
plt.barh(imp_df["Feature"], imp_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance")

st.pyplot(fig2)

# ================= INPUT USER =================
st.subheader("🧠 Nhập thông tin dự đoán")

col1, col2 = st.columns(2)

with col1:
    so_nguoi = st.slider("Số người ở", 1, 10, 2)
    so_may_lanh = st.slider("Số máy lạnh", 0, 5, 1)
    so_quat = st.slider("Số quạt", 0, 10, 2)
    co_tu_lanh = st.selectbox("Có tủ lạnh?", ["Co", "Khong"])

with col2:
    gio_may_lanh = st.slider("Giờ dùng máy lạnh", 0.0, 24.0, 5.0)
    dien_tich = st.slider("Diện tích (m2)", 10, 200, 30)
    tang = st.slider("Tầng", 1, 30, 1)
    loai_nha = st.selectbox("Loại nhà", ["Phong tro", "Can ho", "Nha cap 4", "Nha pho"])

# ================= XỬ LÝ INPUT =================
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

# ================= DỰ ĐOÁN =================
if st.button("🚀 Dự đoán ngay"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Tiền điện dự đoán: {int(prediction):,} VND/tháng")

    # ===== LƯU HISTORY =====
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
st.subheader("📜 Lịch sử dự đoán")

if os.path.exists("history.csv"):
    history_df = pd.read_csv("history.csv")
    st.dataframe(history_df.tail(10))
else:
    st.write("Chưa có dữ liệu")

