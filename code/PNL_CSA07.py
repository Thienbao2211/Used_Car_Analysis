# -*- coding: utf-8 -*-
"""
=====================================================================================
 CAR PRICE INTELLIGENCE DASHBOARD
 Ứng dụng Streamlit phân tích dữ liệu & dự đoán giá xe ô tô đã qua sử dụng
 Được xây dựng dựa trên pipeline xử lý dữ liệu và mô hình trong notebook PNL_CSA07.ipynb
=====================================================================================
"""

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# CẤU HÌNH CHUNG
st.set_page_config(
    page_title="Car Price Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

CURRENT_YEAR = 2026     # Năm hiện tại dùng để tính tuổi xe (giống notebook gốc)
MIN_MODEL_COUNT = 10    # Ngưỡng số lượng tối thiểu để giữ nguyên tên model, dưới ngưỡng gom vào "Other"

CUSTOM_CSS = """
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .subtitle { color: #6b7280; font-size: 1.05rem; margin-top: -6px; }
    div[data-testid="stMetric"] {
        background-color: rgba(37, 99, 235, 0.06);
        border: 1px solid rgba(37, 99, 235, 0.15);
        padding: 14px 16px;
        border-radius: 12px;
    }
    section[data-testid="stSidebar"] { border-right: 1px solid rgba(0,0,0,0.08); }
    .result-box {
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        padding: 28px;
        border-radius: 16px;
        color: white;
        text-align: center;
    }
    .result-box h1 { font-size: 2.6rem; margin: 4px 0; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 1. ĐỌC DỮ LIỆU

@st.cache_data(show_spinner="Đang tải dữ liệu...")
def load_data() -> pd.DataFrame:
    # =================== ĐIỀN CÁCH ĐỌC FILE CSV CỦA BẠN VÀO ĐÂY ===================
    df = pd.read_csv("../data/used_cars.csv")
    # ================================================================================
    return df


# 2. CÁC HÀM TRÍCH XUẤT / LÀM SẠCH (giống hệt logic trong notebook)

def extract_hp(text):
    m = re.search(r"([\d.]+)\s*HP", str(text))
    return float(m.group(1)) if m else np.nan


def extract_liter(text):
    m = re.search(r"([\d.]+)\s*L\b", str(text))
    return float(m.group(1)) if m else np.nan


def extract_cylinders(text):
    m = re.search(r"(\d+)\s*Cylinder", str(text))
    return int(m.group(1)) if m else np.nan


def clean_transmission(transmission):
    if pd.isna(transmission):
        return "Unknown"
    T = str(transmission).upper()
    if "M/T" in T or "MT" in T or "MANUAL" in T:
        return "Manual"
    elif "CVT" in T or "VARIABLE" in T or "SINGLE-SPEED" in T:
        return "CVT"
    elif "A/T" in T or "AT" in T or "AUTOMATIC" in T or "AUTO" in T:
        return "Automatic"
    elif "DUAL" in T or "DCT" in T:
        return "Dual-Clutch"
    elif "SEMI" in T:
        return "Semi-Automatic"
    else:
        return "Other"


@st.cache_data(show_spinner="Đang xử lý & làm sạch dữ liệu...")
def preprocess(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # Giá tiền
    df["price"] = (
        df["price"].astype(str).str.replace(r"[\$,]", "", regex=True)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    #  Số dặm
    df["milage"] = (
        df["milage"].astype(str).str.replace(",", "", regex=False).str.replace("mi.", "", regex=False)
    )
    df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

    #  Tuổi xe 
    df["age"] = CURRENT_YEAR - df["model_year"]

    #  Trích xuất mã lực / dung tích / số xy-lanh từ cột engine 
    df["engine_hp"] = df["engine"].apply(extract_hp)
    df["engine_liter"] = df["engine"].apply(extract_liter)
    df["engine_cylinder"] = df["engine"].apply(extract_cylinders)

    for col in ["engine_hp", "engine_liter", "engine_cylinder"]:
        df[col] = df[col].fillna(df[col].median())

    df = df.drop(columns=["engine"])

    #  Xóa các dòng thiếu giá / số dặm
    df = df.dropna(subset=["price", "milage", "model_year"])
    df["price"] = df["price"].astype(int)
    df["milage"] = df["milage"].astype(int)

    #  Chuẩn hoá hộp số 
    df["transmission"] = df["transmission"].apply(clean_transmission)

    #  Chuẩn hoá các cột phân loại còn thiếu dữ liệu 
    df["fuel_type"] = df["fuel_type"].replace(["–", "not supported"], np.nan)
    df["fuel_type"] = df["fuel_type"].fillna("unknown")
    df["accident"] = df["accident"].fillna("unknown").astype(str).str.lower()
    df["clean_title"] = df["clean_title"].fillna("unknown").astype(str).str.lower()
    for col in ["ext_col", "int_col", "brand", "model"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    #  Loại bỏ các hàng còn giá trị null khác 
    df = df.dropna(axis=0).reset_index(drop=True)

    #  Loại outlier giá xe theo percentile 99 
    q99 = df["price"].quantile(0.99)
    df = df[df["price"] <= q99].reset_index(drop=True)

    #  Gom các model hiếm (< MIN_MODEL_COUNT lần) thành "Other" 
    model_counts = df["model"].value_counts()
    rare_models = model_counts[model_counts < MIN_MODEL_COUNT].index.tolist()
    df["model_grouped"] = df["model"].where(~df["model"].isin(rare_models), "Other")

    return df, rare_models


# 3. HUẤN LUYỆN MÔ HÌNH (Random Forest Regressor)
@st.cache_resource(show_spinner="Đang huấn luyện mô hình Random Forest...")
def train_model(df_clean: pd.DataFrame):
    data = df_clean.drop(columns=["model", "model_year"]).copy()

    X = data.drop(columns=["price"])
    y = np.log1p(data["price"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #  Target encoding cho model_grouped (fit trên tập train) 
    tmp = X_train.copy()
    tmp["price_log"] = y_train.values
    model_target_map = tmp.groupby("model_grouped")["price_log"].mean()
    global_mean = float(y_train.mean())

    X_train["model_te"] = X_train["model_grouped"].map(model_target_map)
    X_test["model_te"] = X_test["model_grouped"].map(model_target_map).fillna(global_mean)

    X_train = X_train.drop(columns=["model_grouped"])
    X_test = X_test.drop(columns=["model_grouped"])

    #  One-Hot Encoding cho các cột phân loại còn lại 
    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    #  Chuẩn hoá các cột số 
    numerical_cols = [
        "age", "milage", "engine_hp", "engine_liter", "engine_cylinder", "model_te",
    ]
    numerical_cols = [c for c in numerical_cols if c in X_train.columns]

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    #  Huấn luyện Random Forest 
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_orig = np.expm1(y_test)

    metrics = {
        "MAE": mean_absolute_error(y_test_orig, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_orig, y_pred))),
        "R2": r2_score(y_test_orig, y_pred),
    }

    feature_importance = (
        pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
    )

    return {
        "model": model,
        "scaler": scaler,
        "feature_columns": X_train.columns.tolist(),
        "numerical_cols": numerical_cols,
        "cat_cols": cat_cols,
        "model_target_map": model_target_map,
        "global_mean": global_mean,
        "metrics": metrics,
        "y_test": y_test_orig,
        "y_pred": y_pred,
        "feature_importance": feature_importance,
    }


def predict_price(input_data: dict, rare_models: list, artifacts: dict) -> float:
    row = pd.DataFrame([input_data])
    row["age"] = CURRENT_YEAR - row["model_year"]

    model_name = row.loc[0, "model"]
    row["model_grouped"] = "Other" if model_name in rare_models else model_name
    row = row.drop(columns=["model", "model_year"])

    row["model_te"] = row["model_grouped"].map(artifacts["model_target_map"])
    row["model_te"] = row["model_te"].fillna(artifacts["global_mean"])
    row = row.drop(columns=["model_grouped"])

    cat_cols_present = [c for c in artifacts["cat_cols"] if c in row.columns]
    row = pd.get_dummies(row, columns=cat_cols_present)
    row = row.reindex(columns=artifacts["feature_columns"], fill_value=0)

    row[artifacts["numerical_cols"]] = artifacts["scaler"].transform(row[artifacts["numerical_cols"]])

    pred_log = artifacts["model"].predict(row)[0]
    return float(np.expm1(pred_log))


# 4. CÁC TRANG GIAO DIỆN

    # Trang tổng quan dữ liệu

def page_overview(df_raw, df_clean):
    st.header("Tổng quan bộ dữ liệu")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số dòng gốc", f"{len(df_raw):,}")
    c2.metric("Số dòng sau làm sạch", f"{len(df_clean):,}")
    c3.metric("Giá trung bình", f"${df_clean['price'].mean():,.0f}")
    c4.metric("Số hãng xe", f"{df_clean['brand'].nunique()}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Dữ liệu mẫu", "Thống kê mô tả", "Giá trị thiếu (dữ liệu gốc)"])
    with tab1:
        st.dataframe(df_clean.head(20), use_container_width=True)
    with tab2:
        st.dataframe(df_clean.describe().T, use_container_width=True)
    with tab3:
        missing = df_raw.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if len(missing) == 0:
            st.success("Dữ liệu gốc không có giá trị thiếu.")
        else:
            fig = px.bar(
                x=missing.values, y=missing.index, orientation="h",
                labels={"x": "Số lượng thiếu", "y": "Cột"},
                title="Số lượng giá trị thiếu theo từng cột",
                color=missing.values, color_continuous_scale="Reds",
            )
            st.plotly_chart(fig, use_container_width=True)

    # Trang biểu đồ phân tích

def page_charts(df):
    st.header("Biểu đồ phân tích dữ liệu")

        #  1. Biểu đồ đường: giá trung bình theo năm của top 5 hãng 
    st.subheader("Xu hướng giá trung bình theo năm của Top 5 hãng xe phổ biến nhất")
    top_brand = df["brand"].value_counts().head(5).index
    df_top_brand = df[df["brand"].isin(top_brand)]
    pivot_df = df_top_brand.pivot_table(
        index="model_year", columns="brand", values="price", aggfunc="mean"
    ).reset_index()
    fig1 = px.line(
        pivot_df, x="model_year", y=top_brand.tolist(), markers=True,
        labels={"model_year": "Năm sản xuất", "value": "Giá trung bình ($)", "variable": "Hãng xe"},
    )
    fig1.update_layout(legend_title_text="Hãng xe")
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)

        #  2. Top 10 hãng xe có giá trung bình cao nhất 

    with col1:
        st.subheader("Top 10 hãng xe có giá trung bình cao nhất")
        df_sum_brand = (
            df[["brand", "price"]].groupby("brand").mean().sort_values("price", ascending=False).head(10)
        )
        fig2 = px.bar(
            df_sum_brand, x=df_sum_brand.index, y="price",
            labels={"x": "Hãng xe", "price": "Giá trung bình ($)"},
            color="price", color_continuous_scale="Blues",
        )
        st.plotly_chart(fig2, use_container_width=True)

        #  3. Số lượng xe vs giá trung bình (top 10 theo số lượng) 

    with col2:
        st.subheader("Số lượng xe & giá trung bình (Top 10 hãng phổ biến)")
        brand_stats = (
            df.groupby("brand").agg(count=("price", "count"), avg_price=("price", "mean"))
            .sort_values("count", ascending=False).head(10)
        )
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(
            go.Bar(x=brand_stats.index, y=brand_stats["count"], name="Số lượng xe", marker_color="#2563eb"),
            secondary_y=False,
        )
        fig3.add_trace(
            go.Scatter(x=brand_stats.index, y=brand_stats["avg_price"], name="Giá trung bình ($)",
                       mode="lines+markers", line=dict(color="#ef4444", width=3)),
            secondary_y=True,
        )
        fig3.update_yaxes(title_text="Số lượng xe", secondary_y=False)
        fig3.update_yaxes(title_text="Giá trung bình ($)", secondary_y=True)
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()

        #  4. Boxplot outlier 

    st.subheader("Phân phối giá & số dặm đã đi (kiểm tra outlier)")
    col3, col4 = st.columns(2)
    with col3:
        fig4 = px.box(df, y="price", points="outliers", title="Phân phối giá xe (Price)")
        st.plotly_chart(fig4, use_container_width=True)
    with col4:
        fig5 = px.box(df, y="milage", points="outliers", title="Phân phối số dặm đã đi (Milage)")
        st.plotly_chart(fig5, use_container_width=True)

    st.divider()

        #  5. Biểu đồ tròn 

    st.subheader("Cơ cấu tỉ trọng dữ liệu")
    col5, col6 = st.columns(2)
    with col5:
        fuel_counts = df["fuel_type"].value_counts()
        fig6 = px.pie(values=fuel_counts.values, names=fuel_counts.index,
                      title="Tỉ lệ loại nhiên liệu", hole=0.35)
        st.plotly_chart(fig6, use_container_width=True)
    with col6:
        df_10y = df[df["model_year"] >= CURRENT_YEAR - 10]
        top10_10y = df_10y["brand"].value_counts().head(10)
        fig7 = px.pie(values=top10_10y.values, names=top10_10y.index,
                      title="Top 10 hãng xe phổ biến (10 năm gần nhất)", hole=0.35)
        st.plotly_chart(fig7, use_container_width=True)

    st.divider()

        #  6. Ma trận tương quan 
        
    st.subheader("Mức độ tương quan giữa các đặc trưng số và giá xe")
    numeric_cols = ["price", "milage", "age", "engine_hp", "engine_liter", "engine_cylinder"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    corr = df[numeric_cols].corr()
    fig8 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
                     title="Ma trận tương quan (Correlation Matrix)")
    st.plotly_chart(fig8, use_container_width=True)

    # Trang dự đoán giá xe

def page_prediction(df, rare_models, artifacts):
    st.header("Dự đoán giá xe")
    st.caption("Nhập thông tin xe để mô hình Random Forest dự đoán mức giá phù hợp.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            brand = st.selectbox("Hãng xe (brand)", sorted(df["brand"].unique()))
            models_for_brand = sorted(df[df["brand"] == brand]["model"].unique())
            model_name = st.selectbox("Dòng xe (model)", models_for_brand if models_for_brand else sorted(df["model"].unique()))
            model_year = st.number_input("Năm sản xuất", min_value=1980, max_value=CURRENT_YEAR, value=2020, step=1)
            milage = st.number_input("Số dặm đã đi (milage)", min_value=0, value=30000, step=1000)

        with c2:
            fuel_type = st.selectbox("Loại nhiên liệu", sorted(df["fuel_type"].unique()))
            transmission = st.selectbox("Hộp số", sorted(df["transmission"].unique()))
            accident = st.selectbox("Lịch sử tai nạn", sorted(df["accident"].unique()))
            clean_title = st.selectbox("Giấy tờ sạch (clean title)", sorted(df["clean_title"].unique()))

        with c3:
            ext_col = st.selectbox("Màu ngoại thất", sorted(df["ext_col"].unique()))
            int_col = st.selectbox("Màu nội thất", sorted(df["int_col"].unique()))
            engine_hp = st.number_input("Mã lực động cơ (HP)", min_value=0.0, value=float(round(df["engine_hp"].median(), 1)))
            colE1, colE2 = st.columns(2)
            with colE1:
                engine_liter = st.number_input("Dung tích (L)", min_value=0.0, value=float(round(df["engine_liter"].median(), 1)), step=0.1)
            with colE2:
                engine_cylinder = st.number_input("Số xy-lanh", min_value=0, value=int(df["engine_cylinder"].median()), step=1)

        submitted = st.form_submit_button("🔮 Dự đoán giá xe", use_container_width=True)

    if submitted:
        input_data = {
            "brand": brand,
            "model": model_name,
            "model_year": model_year,
            "milage": milage,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "ext_col": ext_col,
            "int_col": int_col,
            "accident": accident,
            "clean_title": clean_title,
            "engine_hp": engine_hp,
            "engine_liter": engine_liter,
            "engine_cylinder": engine_cylinder,
        }
        predicted_price = predict_price(input_data, rare_models, artifacts)

        st.markdown(
            f"""
            <div class="result-box">
                <div style="font-size:1rem; opacity:0.85;">Giá xe dự đoán</div>
                <h1>${predicted_price:,.0f}</h1>
                <div style="opacity:0.85;">{brand} {model_name} · {model_year} · {milage:,} mi.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        similar = df[(df["brand"] == brand)]
        if len(similar) > 0:
            st.caption(
                f"Giá trung bình thực tế của các xe hãng {brand} trong dữ liệu: "
                f"${similar['price'].mean():,.0f} (n={len(similar)})"
            )

# Trang đánh giá mô hình

def page_evaluation(artifacts):
    st.header("Đánh giá mô hình")

    m = artifacts["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (Sai số tuyệt đối trung bình)", f"${m['MAE']:,.0f}")
    c2.metric("RMSE", f"${m['RMSE']:,.0f}")
    c3.metric("R² Score", f"{m['R2']:.3f}")

    st.divider()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Giá dự đoán so với giá thực tế")
        y_test, y_pred = artifacts["y_test"], artifacts["y_pred"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers", name="Dữ liệu dự đoán",
            marker=dict(color="#2563eb", opacity=0.55, size=6),
        ))
        line_min, line_max = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        fig.add_trace(go.Scatter(
            x=[line_min, line_max], y=[line_min, line_max], mode="lines",
            name="Đường lý tưởng (y = x)", line=dict(color="#ef4444", dash="dash"),
        ))
        fig.update_layout(xaxis_title="Giá thực tế ($)", yaxis_title="Giá dự đoán ($)")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Mức độ ảnh hưởng của đặc trưng")
        top_features = artifacts["feature_importance"].head(12).sort_values()
        fig_imp = px.bar(
            x=top_features.values, y=top_features.index, orientation="h",
            labels={"x": "Mức độ quan trọng", "y": ""},
            color=top_features.values, color_continuous_scale="Purples",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "Mô hình sử dụng **Random Forest Regressor** huấn luyện trên `log1p(price)`, "
        "với target encoding cho dòng xe (model), one-hot encoding cho các biến phân loại "
        "và chuẩn hoá (StandardScaler) cho các biến số."
    )


# 5. Main
def main():
    st.markdown('<div class="main-title">Phân tích và dự đoán giá xe cũ</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Phân tích chuyên sâu & dự đoán giá xe ô tô đã qua sử dụng bằng Machine Learning</div>', unsafe_allow_html=True)
    st.write("")

    try:
        df_raw = load_data()
    except Exception as e:
        st.error(
            "Chưa đọc được dữ liệu!"
        )
        st.code(str(e))
        st.stop()

    required_cols = {
        "brand", "model", "model_year", "milage", "fuel_type", "engine",
        "transmission", "ext_col", "int_col", "accident", "clean_title", "price",
    }
    missing_cols = required_cols - set(df_raw.columns)
    if missing_cols:
        st.error(f"Lỗi file csv thiếu các cột: {sorted(missing_cols)}")
        st.stop()

    df_clean, rare_models = preprocess(df_raw)
    artifacts = train_model(df_clean)

    st.sidebar.title("Điều hướng")
    page = st.sidebar.radio(
        "Chọn trang",
        ["Tổng quan dữ liệu", "Biểu đồ phân tích", "Dự đoán giá xe", "Đánh giá mô hình"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.metric("Số lượng xe (sau xử lý)", f"{len(df_clean):,}")
    st.sidebar.metric("R² của mô hình", f"{artifacts['metrics']['R2']:.3f}")
    st.sidebar.caption("Model: Random Forest Regressor · 100 cây")

    if page == "Tổng quan dữ liệu":
        page_overview(df_raw, df_clean)
    elif page == "Biểu đồ phân tích":
        page_charts(df_clean)
    elif page == "Dự đoán giá xe":
        page_prediction(df_clean, rare_models, artifacts)
    else:
        page_evaluation(artifacts)


if __name__ == "__main__":
    main()