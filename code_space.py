import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ================= LOAD DATA =================
df = pd.read_csv("200_ho_gia_dinh_tien_dien.csv")

# ================= XỬ LÝ DỮ LIỆU =================
# Encode Có/Không
df["Co_tu_lanh"] = df["Co_tu_lanh"].map({"Co": 1, "Khong": 0})

# Encode loại nhà
df = pd.get_dummies(df, columns=["Loai_nha"])

# ================= TÁCH INPUT / OUTPUT =================
X = df.drop(["Tien_dien_thang_VND", "STT"], axis=1)
y = df["Tien_dien_thang_VND"]

# ================= TRAIN MODEL =================
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ================= GIAO DIỆN =================
st.title("🔌 App Dự Đoán Tiền Điện Phòng Trọ")

st.write("Nhập thông tin để dự đoán tiền điện:")

# INPUT USER
so_nguoi = st.slider("Số người ở", 1, 10, 2)
so_may_lanh = st.slider("Số máy lạnh", 0, 5, 1)
so_quat = st.slider("Số quạt", 0, 10, 2)
co_tu_lanh = st.selectbox("Có tủ lạnh?", ["Co", "Khong"])
gio_may_lanh = st.slider("Số giờ bật máy lạnh/ngày", 0.0, 24.0, 5.0)
dien_tich = st.slider("Diện tích phòng (m2)", 10, 200, 30)
tang = st.slider("Tầng lầu", 1, 30, 1)
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

# thêm cột loại nhà (one-hot)
for col in X.columns:
    if "Loai_nha_" in col:
        input_dict[col] = 1 if col == f"Loai_nha_{loai_nha}" else 0

input_df = pd.DataFrame([input_dict])

# ================= DỰ ĐOÁN =================
if st.button("Dự đoán"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Tiền điện dự đoán: {int(prediction):,} VND / tháng")
