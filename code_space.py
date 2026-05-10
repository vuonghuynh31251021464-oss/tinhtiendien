import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# =============================
# 🔥 1. LOAD DATA EXCEL
# =============================
FILE_PATH = "sheet_200_ho_gia_dinh_du_doan_tien_dien.xlsx"

df = pd.read_excel(FILE_PATH)

# ⚠️ SỬA TÊN CỘT DATE NẾU KHÁC
DATE_COL = "date"

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce', dayfirst=True)
df = df.dropna(subset=[DATE_COL])

# =============================
# 📅 2. FEATURE TIME
# =============================
df['month'] = df[DATE_COL].dt.month
df['dayofweek'] = df[DATE_COL].dt.dayofweek
df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)

# =============================
# 🧠 3. FUZZY LOGIC
# =============================
def low(x, a, b):
    return max(0, min(1, (b - x) / (b - a + 1e-6)))

def high(x, a, b):
    return max(0, min(1, (x - a) / (b - a + 1e-6)))

def fuzzy_features(people, ac_hours):
    people_low = low(people, 1, 3)
    people_high = high(people, 2, 6)

    ac_low = low(ac_hours, 0, 4)
    ac_high = high(ac_hours, 4, 10)

    consumption_high = min(people_high, ac_high)
    consumption_low = min(people_low, ac_low)

    return [
        people_low,
        people_high,
        ac_low,
        ac_high,
        consumption_low,
        consumption_high
    ]

# =============================
# 🔧 4. BUILD FEATURE
# =============================
def build_features(row):
    base = [
        row['people'],
        row['ac'],
        row['fan'],
        row['fridge'],
        row['ac_hours'],
        row['area'],
        row['floor'],
        row['house_type'],
        row['month'],
        row['dayofweek'],
        row['is_weekend']
    ]

    fuzzy = fuzzy_features(row['people'], row['ac_hours'])
    return base + fuzzy

# =============================
# 🤖 5. TRAIN MODEL
# =============================
X = []
y = df['bill']

for _, row in df.iterrows():
    X.append(build_features(row))

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# =============================
# 🌐 6. STREAMLIT UI
# =============================
st.set_page_config(page_title="Electricity AI", layout="centered")

st.title("⚡ Dự đoán tiền điện (AI + Fuzzy + Time)")

# Input
people = st.slider("Số người", 1, 10)
ac = st.slider("Số máy lạnh", 0, 5)
fan = st.slider("Số quạt", 0, 10)
fridge = st.selectbox("Có tủ lạnh?", [0, 1])
ac_hours = st.slider("Giờ dùng máy lạnh", 0, 24)
area = st.slider("Diện tích", 10, 100)
floor = st.slider("Tầng", 1, 20)
house_type = st.selectbox("Loại nhà", ["Trọ", "Chung cư"])
date_input = st.date_input("📅 Chọn ngày")

house_type = 0 if house_type == "Trọ" else 1

# Extract time
month = date_input.month
dayofweek = date_input.weekday()
is_weekend = 1 if dayofweek >= 5 else 0

# =============================
# 🔮 7. PREDICT
# =============================
if st.button("🚀 Dự đoán"):
    row = {
        "people": people,
        "ac": ac,
        "fan": fan,
        "fridge": fridge,
        "ac_hours": ac_hours,
        "area": area,
        "floor": floor,
        "house_type": house_type,
        "month": month,
        "dayofweek": dayofweek,
        "is_weekend": is_weekend
    }

    features = build_features(row)
    result = model.predict([features])[0]

    st.success(f"💰 Tiền điện dự đoán: {int(result):,} VND")

    # Explain
    st.subheader("🔍 Giải thích")
    st.write(f"- Tháng: {month}")
    st.write(f"- Cuối tuần: {'Có' if is_weekend else 'Không'}")

    if ac_hours > 6:
        st.write("- Dùng máy lạnh nhiều → điện tăng")
    if people > 3:
        st.write("- Nhiều người → tiêu thụ cao")

    st.write("### Fuzzy values:")
    st.write(fuzzy_features(people, ac_hours))

# =============================
# 📊 8. BIỂU ĐỒ (ăn điểm)
# =============================
st.subheader("📈 Trung bình tiền điện theo tháng")

monthly = df.groupby('month')['bill'].mean()

fig, ax = plt.subplots()
ax.plot(monthly.index, monthly.values)
ax.set_xlabel("Month")
ax.set_ylabel("Avg Bill")

st.pyplot(fig)
