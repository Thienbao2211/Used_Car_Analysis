import re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as pit
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Dự đoán giá xe cũ", page_icon="🚗", layout="wide")
st.title("Phân tích & Dự đoán Giá Xe Cũ")
st.caption("Dựa trên bộ dữ liệu used_cars.csv — làm sạch, trực quan hóa và huấn luyện mô hình dự đoán giá.")

CURRENT_YEAR = 2026

# ============================================================
# 1. NẠP & LÀM SẠCH DỮ LIỆU
# ============================================================
st.sidebar.header("⚙️ Dữ liệu")
uploaded = st.sidebar.file_uploader("Upload used_cars.csv (tùy chọn)", type=["csv"])

@st.cache_data
def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # Giá tiền: "$10,300" -> 10300
    df["price"] = (
        df["price"].astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    # Số km: "51,000 mi." -> 51000
    df["milage"] = (
        df["milage"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("mi.", "", regex=False)
        .str.strip()
        .astype(int)
    )

    # Tuổi xe
    df["age"] = CURRENT_YEAR - df["model_year"]

    # Trích công suất (HP) và dung tích (L) từ chuỗi "engine"
    def extract_hp(text):
        m = re.search(r"([\d.]+)\s*HP", str(text))
        return float(m.group(1)) if m else np.nan

    def extract_liter(text):
        m = re.search(r"([\d.]+)\s*L\b", str(text))
        return float(m.group(1)) if m else np.nan

    df["horsepower"] = df["engine"].apply(extract_hp)
    df["engine_liter"] = df["engine"].apply(extract_liter)

    # Điền thiếu horsepower/engine_liter bằng trung vị (median)
    df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
    df["engine_liter"] = df["engine_liter"].fillna(df["engine_liter"].median())

    # fuel_type, accident: xóa dòng thiếu (là cột phân loại quan trọng, thiếu ít)
    df = df.dropna(subset=["fuel_type", "accident"])

    # clean_title thiếu -> "Unknown"
    df["clean_title"] = df["clean_title"].fillna("Unknown")

    return df.reset_index(drop=True)


if uploaded is not None:
    raw_df = pd.read_csv(uploaded)
else:
    try:
        raw_df = pd.read_csv("used_cars.csv")
        st.sidebar.success("✅ Đã nạp used_cars.csv có sẵn trong thư mục.")
    except FileNotFoundError:
        st.warning("👈 Vui lòng upload file used_cars.csv ở thanh bên trái.")
        st.stop()

df = clean_data(raw_df)

remove_outliers = st.sidebar.checkbox(
    "Loại bỏ outlier giá xe (1%–99% percentile)",
    value=True,
    help="Bộ dữ liệu có nhiều siêu xe giá rất cao gây nhiễu mô hình. "
         "Bật tùy chọn này giúp mô hình dự đoán chính xác hơn đáng kể."
)
if remove_outliers:
    q_low, q_high = df["price"].quantile([0.01, 0.99])
    df_analysis = df[(df["price"] >= q_low) & (df["price"] <= q_high)].reset_index(drop=True)
else:
    df_analysis = df.copy()

st.sidebar.metric("Số dòng sau làm sạch", len(df_analysis))

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Dữ liệu đã làm sạch", "📈 Biểu đồ khám phá (EDA)", "🔗 Tương quan", "🤖 Dự đoán giá xe"]
)

# ------------------------------------------------------------
# TAB 1: DỮ LIỆU ĐÃ LÀM SẠCH
# ------------------------------------------------------------
with tab1:
    st.subheader("Xem trước dữ liệu đã làm sạch")
    st.dataframe(df_analysis.head(20), use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số dòng", len(df_analysis))
    c2.metric("Số hãng xe (brand)", df_analysis["brand"].nunique())
    c3.metric("Giá trung bình", f"${df_analysis['price'].mean():,.0f}")
    c4.metric("Số km trung bình", f"{df_analysis['milage'].mean():,.0f} mi")

    st.subheader("Thống kê mô tả")
    st.dataframe(
        df_analysis[["price", "milage", "model_year", "age", "horsepower", "engine_liter"]].describe().T,
        use_container_width=True
    )

# ------------------------------------------------------------
# TAB 2: EDA
# ------------------------------------------------------------
with tab2:
    st.subheader("Số lượng xe theo năm của top 5 hãng phổ biến nhất")
    top_brand = df_analysis["brand"].value_counts().head(5).index
    df_top = df_analysis[df_analysis["brand"].isin(top_brand)]
    pivot_df = df_top.pivot_table(index="model_year", columns="brand", values="model", aggfunc="count", fill_value=0)

    fig, ax = pit.subplots(figsize=(11, 5))
    for brand in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[brand], label=brand)
    ax.set_xlabel("Năm")
    ax.set_ylabel("Số lượng xe")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 hãng xe có tổng giá trị lớn nhất")
        df_sum_brand = df_analysis.groupby("brand")["price"].sum().sort_values(ascending=False).head(10)
        fig, ax = pit.subplots(figsize=(7, 5))
        ax.bar(df_sum_brand.index, df_sum_brand.values, color="steelblue")
        pit.xticks(rotation=45, ha="right")
        ax.set_ylabel("Tổng giá trị ($)")
        st.pyplot(fig)

    with col2:
        st.subheader("Tỉ lệ loại nhiên liệu")
        fuel_counts = df_analysis["fuel_type"].value_counts()
        fig, ax = pit.subplots(figsize=(6, 5.5))
        ax.pie(fuel_counts, labels=fuel_counts.index, autopct="%1.1f%%", startangle=90)
        st.pyplot(fig)

    st.subheader("Phân bố giá trị & phát hiện ngoại lai (Boxplot)")
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = pit.subplots(figsize=(5, 4))
        ax.boxplot(df_analysis["price"], patch_artist=True)
        ax.set_title("Price")
        st.pyplot(fig)
    with col4:
        fig, ax = pit.subplots(figsize=(5, 4))
        ax.boxplot(df_analysis["milage"], patch_artist=True)
        ax.set_title("Milage")
        st.pyplot(fig)

    st.subheader("Mối quan hệ giữa các đặc trưng và giá xe")
    features = ["milage", "model_year", "age", "horsepower"]
    fig, axes = pit.subplots(1, 4, figsize=(20, 4.5))
    for i, col in enumerate(features):
        x, y = df_analysis[col], df_analysis["price"]
        axes[i].scatter(x, y, alpha=0.3, s=10)
        a, b = np.polyfit(x, y, 1)
        x_sorted = np.sort(x)
        axes[i].plot(x_sorted, a * x_sorted + b, color="red")
        axes[i].set_title(f"{col} vs price")
        axes[i].set_xlabel(col)
    st.pyplot(fig)

    st.subheader("Top 10 hãng xe phổ biến trong 10 năm gần đây")
    df_10y = df_analysis[df_analysis["model_year"] >= CURRENT_YEAR - 10]
    top10_recent = df_10y["brand"].value_counts().head(10)
    fig, ax = pit.subplots(figsize=(7, 6))
    ax.pie(top10_recent, labels=top10_recent.index, autopct="%1.1f%%", startangle=90)
    st.pyplot(fig)

# ------------------------------------------------------------
# TAB 3: TƯƠNG QUAN
# ------------------------------------------------------------
with tab3:
    st.subheader("Hệ số tương quan với giá xe (price)")
    corr_milage = df_analysis["milage"].corr(df_analysis["price"])
    corr_age = df_analysis["age"].corr(df_analysis["price"])
    corr_hp = df_analysis["horsepower"].corr(df_analysis["price"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Số km đã đi", f"{corr_milage:.3f}")
    c2.metric("Tuổi xe", f"{corr_age:.3f}")
    c3.metric("Công suất (HP)", f"{corr_hp:.3f}")

    st.subheader("Ma trận tương quan")
    corr_cols = ["price", "milage", "model_year", "age", "horsepower", "engine_liter"]
    fig, ax = pit.subplots(figsize=(8, 6))
    sns.heatmap(df_analysis[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.info(
        "📌 Nhận xét: Số km đã đi (milage) và tuổi xe (age) có tương quan **âm** với giá — "
        "xe càng cũ, đi càng nhiều thì giá càng giảm. Công suất (horsepower) có tương quan "
        "**dương** — xe càng mạnh thường giá càng cao."
    )

# ------------------------------------------------------------
# TAB 4: DỰ ĐOÁN GIÁ XE (ML)
# ------------------------------------------------------------
with tab4:
    st.subheader("🤖 Huấn luyện mô hình dự đoán giá xe")

    feature_cols = ["brand", "model_year", "milage", "fuel_type", "age", "horsepower", "accident"]

    algo = st.selectbox(
        "Chọn thuật toán",
        ["Random Forest Regressor (khuyến nghị)", "Linear Regression", "Decision Tree Regressor"]
    )
    test_size = st.slider("Tỉ lệ dữ liệu test (%)", 10, 40, 20) / 100

    if st.button("🚀 Huấn luyện mô hình"):
        data = df_analysis.copy()

        le_brand = LabelEncoder()
        le_fuel = LabelEncoder()
        le_accident = LabelEncoder()
        data["brand_enc"] = le_brand.fit_transform(data["brand"])
        data["fuel_enc"] = le_fuel.fit_transform(data["fuel_type"])
        data["accident_enc"] = le_accident.fit_transform(data["accident"])

        X = data[["brand_enc", "model_year", "milage", "fuel_enc", "age", "horsepower", "accident_enc"]]
        y = data["price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        use_scaled = algo == "Linear Regression"
        scaler = StandardScaler()
        if use_scaled:
            X_train_fit = scaler.fit_transform(X_train)
            X_test_fit = scaler.transform(X_test)
        else:
            X_train_fit, X_test_fit = X_train, X_test

        model = {
            "Random Forest Regressor (khuyến nghị)": RandomForestRegressor(n_estimators=300, random_state=42),
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(max_depth=10, random_state=42),
        }[algo]

        model.fit(X_train_fit, y_train)
        y_pred = model.predict(X_test_fit)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("✅ Huấn luyện hoàn tất!")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score", f"{r2:.3f}")
        c2.metric("RMSE", f"${rmse:,.0f}")
        c3.metric("MAE", f"${mae:,.0f}")
        c4.metric("MSE", f"{mse:,.0f}")

        fig, ax = pit.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred, alpha=0.4)
        lims = [y.min(), y.max()]
        ax.plot(lims, lims, "r--", label="Đường lý tưởng (y=x)")
        ax.set_xlabel("Giá thực tế")
        ax.set_ylabel("Giá dự đoán")
        ax.legend()
        st.pyplot(fig)

        if hasattr(model, "feature_importances_"):
            st.subheader("🌟 Mức độ quan trọng của đặc trưng")
            imp_df = pd.DataFrame({
                "Đặc trưng": ["brand", "model_year", "milage", "fuel_type", "age", "horsepower", "accident"],
                "Mức độ quan trọng": model.feature_importances_
            }).sort_values("Mức độ quan trọng", ascending=False)
            fig, ax = pit.subplots(figsize=(7, 4))
            sns.barplot(data=imp_df, x="Mức độ quan trọng", y="Đặc trưng", ax=ax)
            st.pyplot(fig)

        # Lưu vào session_state để dùng cho form dự đoán bên dưới
        st.session_state["cars_model"] = model
        st.session_state["cars_le_brand"] = le_brand
        st.session_state["cars_le_fuel"] = le_fuel
        st.session_state["cars_le_accident"] = le_accident
        st.session_state["cars_scaler"] = scaler
        st.session_state["cars_use_scaled"] = use_scaled

    # ---------------- FORM DỰ ĐOÁN XE MỚI ----------------
    if "cars_model" in st.session_state:
        st.divider()
        st.subheader("🔮 Dự đoán giá cho một chiếc xe cụ thể")

        le_brand = st.session_state["cars_le_brand"]
        le_fuel = st.session_state["cars_le_fuel"]
        le_accident = st.session_state["cars_le_accident"]

        c1, c2, c3 = st.columns(3)
        with c1:
            brand_in = st.selectbox("Hãng xe", sorted(le_brand.classes_))
            model_year_in = st.number_input("Năm sản xuất", min_value=1990, max_value=CURRENT_YEAR, value=2020)
        with c2:
            milage_in = st.number_input("Số km đã đi", min_value=0, value=40000, step=1000)
            fuel_in = st.selectbox("Loại nhiên liệu", sorted(le_fuel.classes_))
        with c3:
            hp_in = st.number_input("Công suất (HP)", min_value=0.0, value=250.0, step=10.0)
            accident_in = st.selectbox("Tình trạng tai nạn", sorted(le_accident.classes_))

        if st.button("Dự đoán giá"):
            input_df = pd.DataFrame({
                "brand_enc": [le_brand.transform([brand_in])[0]],
                "model_year": [model_year_in],
                "milage": [milage_in],
                "fuel_enc": [le_fuel.transform([fuel_in])[0]],
                "age": [CURRENT_YEAR - model_year_in],
                "horsepower": [hp_in],
                "accident_enc": [le_accident.transform([accident_in])[0]],
            })

            if st.session_state["cars_use_scaled"]:
                input_fit = st.session_state["cars_scaler"].transform(input_df)
            else:
                input_fit = input_df

            pred_price = st.session_state["cars_model"].predict(input_fit)[0]
            st.success(f"### 💰 Giá xe dự đoán: **${pred_price:,.0f}**")

st.sidebar.divider()
st.sidebar.caption("Được xây dựng với Streamlit + scikit-learn 🐍")