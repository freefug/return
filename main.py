import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측 회귀 모델 평가", layout="wide")
st.title("🧪 서울 연평균 기온 선형회귀 모델 평가 & 기간별 비교")

# 데이터 불러오기 및 전처리 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 데이터 처리 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    df = df.dropna(subset=["평균기온"])
    
    # 연도별 관측일수 및 연평균 기온 집계
    yearly = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 필터링: 2025년 이하 & 관측일수 300일 이상
    filtered = yearly[(yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)].copy()
    filtered["지난연수"] = filtered["연도"] - 1908
    return filtered

yearly_df = load_data()

# ---------------------------------------------------------
# 데이터 분할 (Train / Test)
# ---------------------------------------------------------
# Test: 최근 20년 (2006 ~ 2025년)
test_df = yearly_df[(yearly_df["연도"] >= 2006) & (yearly_df["연도"] <= 2025)].copy()

# Train 1: 최근 50년 학습 (1956 ~ 2005년)
train_50_df = yearly_df[(yearly_df["연도"] >= 1956) & (yearly_df["연도"] <= 2005)].copy()

# Train 2: 최근 100년 학습 (1906 ~ 2005년)
train_100_df = yearly_df[(yearly_df["연도"] >= 1906) & (yearly_df["연도"] <= 2005)].copy()

# ---------------------------------------------------------
# 선형 회귀 모델 학습 함수
# ---------------------------------------------------------
def train_eval_model(train_data, test_data):
    X_train = train_data["지난연수"]
    y_train = train_data["평균기온"]
    
    # 1차 회귀선 피팅 (y = slope * X + intercept)
    slope, intercept = np.polyfit(X_train, y_train, 1)
    
    # 테스트 데이터 예측
    X_test = test_data["지난연수"]
    y_test = test_data["평균기온"]
    y_pred = slope * X_test + intercept
    
    # 평가 지표 계산
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "slope": slope,
        "intercept": intercept,
        "rate_100y": slope * 100,
        "mae": mae,
        "mse": mse,
        "r2": r2,
        "y_pred": y_pred
    }

# 모델 실행
res_50 = train_eval_model(train_50_df, test_df)
res_100 = train_eval_model(train_100_df, test_df)

# ---------------------------------------------------------
# 대시보드 화면 구성
# ---------------------------------------------------------
st.subheader("📌 데이터 분할 및 학습 조건")
st.markdown(f"""
- **공통 테스트 데이터 (Test Set)**: 최근 20년 (`2006년 ~ 2025년`, 총 **{len(test_df)}개** 관측 데이터)
- **학습 모델 A**: 최근 50년 데이터 (`1956년 ~ 2005년`, 총 **{len(train_50_df)}개** 데이터로 학습)
- **학습 모델 B**: 최근 100년 데이터 (`1906년 ~ 2005년`, 총 **{len(train_100_df)}개** 데이터로 학습)
""")

st.divider()

# 1. 모델 기울기 및 예측 성능 메트릭 비교
st.subheader("📊 최근 20년(2006~2025) 테스트 데이터 예측 성능 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔵 모델 A (최근 50년 학습: 1956~2005)")
    st.metric("100년당 기온 상승 폭", f"+{res_50['rate_100y']:.2f} °C / 100년")
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("MAE", f"{res_50['mae']:.4f}")
    m_col2.metric("MSE", f"{res_50['mse']:.4f}")
    m_col3.metric("R² Score", f"{res_50['r2']:.4f}")

with col2:
    st.markdown("### 🔴 모델 B (최근 100년 학습: 1906~2005)")
    st.metric(
        "100년당 기온 상승 폭", 
        f"+{res_100['rate_100y']:.2f} °C / 100년",
        delta=f"{res_100['rate_100y'] - res_50['rate_100y']:.2f} °C (50년 대비)"
    )
    
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric("MAE", f"{res_100['mae']:.4f}")
    m_col2.metric("MSE", f"{res_100['mse']:.4f}")
    m_col3.metric("R² Score", f"{res_100['r2']:.4f}")

st.divider()

# 2. 성능 차이 분석 요약
st.subheader("💡 학습 데이터 기간에 따른 기울기와 성능 비교 분석")

st.info(f"""
- **기울기 비교**: 
  - 최근 50년(1956~2005년)으로 학습했을 때의 상승 폭(**+{res_50['rate_100y']:.2f}°C/100년**)이 최근 100년(1906~2005년) 데이터로 학습했을 때(**+{res_100['rate_100y']:.2f}°C/100년**)보다 가파릅니다. 이는 20세기 후반으로 갈수록 지구온난화 및 도시화로 인한 온난화 속도가 빨라졌음을 의미합니다.
- **예측 성능(MAE, MSE, R²) 비교**:
  - 최근 20년(2006~2025년)의 기온은 지속적으로 급상승하는 추세입니다.
  - 이에 따라 최근 50년 학습 모델이 더 높은 상승 기울기를 반영하고 있으므로, 테스트 구간 예측 시 **오차(MAE, MSE)가 더 작고 예측 정확도($R^2$)가 우수**하게 나타납니다.
""")

st.divider()

# 3. Plotly 시각화 (학습 데이터, 테스트 데이터, 회귀선)
st.subheader("📈 회귀선 및 테스트 데이터 연장 예측 시각화")

fig = go.Figure()

# 실제 관측 데이터 구분 표기
# Train 100년 영역 (가장 오래된 데이터)
fig.add_trace(
    go.Scatter(
        x=yearly_df[yearly_df["연도"] < 1956]["연도"],
        y=yearly_df[yearly_df["연도"] < 1956]["평균기온"],
        mode="markers",
        name="과거 데이터 (1906~1955)",
        marker=dict(color="gray", opacity=0.5, size=6)
    )
)

# Train 50년 영역
fig.add_trace(
    go.Scatter(
        x=train_50_df["연도"],
        y=train_50_df["평균기온"],
        mode="markers",
        name="학습 데이터 (1956~2005)",
        marker=dict(color="blue", opacity=0.7, size=7)
    )
)

# Test 데이터 영역
fig.add_trace(
    go.Scatter(
        x=test_df["연도"],
        y=test_df["평균기온"],
        mode="markers",
        name="테스트 데이터 (2006~2025)",
        marker=dict(color="green", size=9, symbol="diamond")
    )
)

# 회귀 직선 표시 (1906년 ~ 2025년 연장)
x_line_years = np.arange(1906, 2026)
x_line_passed = x_line_years - 1908

# 모델 A (최근 50년 학습) 회귀선
y_line_50 = res_50["slope"] * x_line_passed + res_50["intercept"]
fig.add_trace(
    go.Scatter(
        x=x_line_years,
        y=y_line_50,
        mode="lines",
        name="모델 A 회귀선 (최근 50년 학습)",
        line=dict(color="blue", width=2.5)
    )
)

# 모델 B (최근 100년 학습) 회귀선
y_line_100 = res_100["slope"] * x_line_passed + res_100["intercept"]
fig.add_trace(
    go.Scatter(
        x=x_line_years,
        y=y_line_100,
        mode="lines",
        name="모델 B 회귀선 (최근 100년 학습)",
        line=dict(color="red", width=2.5, dash="dash")
    )
)

fig.update_layout(
    xaxis=dict(title="연도", dtick=10),
    yaxis=dict(title="평균기온 (°C)"),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_white",
    height=550
)

st.plotly_chart(fig, use_container_width=True)
