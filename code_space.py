import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# =============================
# 🔥 1. LOAD DATA CSV
# =============================
FILE_PATH = "project4 - Trang tính1.csv"

df = pd.read_csv(FILE_PATH, encoding="utf-8")
df.columns = df.columns.str.strip()

print("COLUMNS:", df.columns.tolist())
df = df.rename(columns={
    'nguoi o': 'people',
    'so may lanh': 'ac',
    'so quat': 'fan',
    'tu lanh': 'fridge',
    'so gio bat may lanh': 'ac_hours',
    'dien tich phong m2': 'area',
    'so tang': 'floor',
    'loai nha': 'house_type',
})
# =============================
# 🧠 2. FUZZY LOGIC
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
# 🔧 3. BUILD FEATURE
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
        row['house_type']
    ]

    fuzzy = fuzzy_features(row['people'], row['ac_hours'])
    return base + fuzzy

# =============================
# 🤖 4. TRAIN MODEL
# =============================
required_cols = ['people','ac','fan','fridge','ac_hours','area','floor','house_type','bill']
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"❌ Thiếu cột: {missing}")
    st.stop()

X = []
y = df['bill']

for _, row in df.iterrows():
    X.append(build_features(row))

model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# =============================
# 🌐 5. STREAMLIT UI
# =============================
st.set_page_config(page_title="Electricity AI", layout="centered")

st.title("⚡ Dự đoán tiền điện (AI + Fuzzy)")

people = st.slider("Số người", 1, 10)
ac = st.slider("Số máy lạnh", 0, 5)
fan = st.slider("Số quạt", 0, 10)
fridge = st.selectbox("Có tủ lạnh?", [0, 1])
ac_hours = st.slider("Giờ dùng máy lạnh", 0, 24)
area = st.slider("Diện tích", 10, 100)
floor = st.slider("Tầng", 1, 20)
house_type = st.selectbox("Loại nhà", ["Trọ", "Chung cư"])

house_type = 0 if house_type == "Trọ" else 1

# =============================
# 🔮 6. PREDICT
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
        "house_type": house_type
    }

    features = build_features(row)
    result = model.predict([features])[0]

    st.success(f"💰 Tiền điện dự đoán: {int(result):,} VND")

    # Explain
    st.subheader("🔍 Giải thích")

    if ac_hours > 6:
        st.write("- Dùng máy lạnh nhiều → điện tăng")
    if people > 3:
        st.write("- Nhiều người → tiêu thụ cao")

    st.write("### Giá trị fuzzy:")
    st.write(fuzzy_features(people, ac_hours))

# =============================
# 📊 7. BIỂU ĐỒ
# =============================
st.subheader("📈 Phân bố tiền điện")

fig, ax = plt.subplots()
ax.hist(df['bill'], bins=10)
ax.set_xlabel("Bill")
ax.set_ylabel("Count")

st.pyplot(fig)
