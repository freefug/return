import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측기")

# 데이터 불러오기 함수
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
    df = pd.read_csv(url, encoding="utf-8")
    
    # 날짜 컬럼을 datetime 형식으로 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 평균기온 결측치 제거
    df = df.dropna(subset=["평균기온"])
    
    # 연도별 관측일수 및 평균기온 계산
    yearly = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 필터링 조건 적용: 2025년 이하 & 관측일수 300일 이상
    filtered_yearly = yearly[(yearly["연도"] <= 2025) & (yearly["관측일수"] >= 300)].copy()
    
    return filtered_yearly

# 데이터 로드
yearly_df = load_data()

# 1. 전체 기간 선형 회귀 계산 (독립 변수: 1908년부터 지난 연수)
yearly_df["지난연수"] = yearly_df["연도"] - 1908
X_full = yearly_df["지난연수"]
y_full = yearly_df["평균기온"]
slope_full, intercept_full = np.polyfit(X_full, y_full, 1)

# 2. 최근 20년 선형 회귀 계산
max_year = int(yearly_df["연도"].max())
recent_20_df = yearly_df[yearly_df["연도"] >= (max_year - 19)].copy()
X_recent = recent_20_df["지난연수"]
y_recent = recent_20_df["평균기온"]
slope_recent, intercept_recent = np.polyfit(X_recent, y_recent, 1)

# 100년당 기온 상승량 계산 (기울기 * 100)
rate_100y_full = slope_full * 100
rate_100y_recent = slope_recent * 100

# 데이터 기본 통계치
start_year = int(yearly_df["연도"].min())
end_year = int(yearly_df["연도"].max())
total_years = len(yearly_df)
corr = yearly_df["연도"].corr(yearly_df["평균기온"])

# 기본 정보 서술
st.markdown(f"""
- **학습에 사용된 연도 개수**: {total_years}개 해
- **데이터 분석 기간**: {start_year}년 ~ {end_year}년
- **연도와 평균기온 간 상관계수**: `{corr:.4f}`
""")

st.divider()

# 📈 100년당 상승률 나란히 비교 표시
st.subheader("🔥 기온 상승 속도 비교 (100년당 상승 폭)")
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"🌐 전체 기간 ({start_year}~{end_year}년) 상승률",
        value=f"+{rate_100y_full:.2f} °C / 100년",
        help="전체 observation 기간 동안의 연평균 상승 폭입니다."
    )

with col2:
    recent_start_year = int(recent_20_df["연도"].min())
    st.metric(
        label=f"⚡ 최근 20년 ({recent_start_year}~{end_year}년) 상승률",
        value=f"+{rate_100y_recent:.2f} °C / 100년",
        delta=f"{rate_100y_recent - rate_100y_full:+.2f} °C/100년 (전체 대비)",
        help="최근 20년간의 데이터만 사용하여 계산한 가속화된 상승 폭입니다."
    )

st.divider()

# 연도 선택 슬라이더
target_year = st.slider(
    "예상 기온을 확인할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 예측 기온 계산 (전체 기간 모델 및 최근 20년 모델)
pred_years_passed = target_year - 1908
predicted_temp_full = slope_full * pred_years_passed + intercept_full
predicted_temp_recent = slope_recent * pred_years_passed + intercept_recent

# 예측 결과 비교 표시
pred_col1, pred_col2 = st.columns(2)
with pred_col1:
    st.metric(
        label=f"🎯 {target_year}년 예상 기온 (전체 기간 추세)",
        value=f"{predicted_temp_full:.2f} °C"
    )
with pred_col2:
    st.metric(
        label=f"🚀 {target_year}년 예상 기온 (최근 20년 추세)",
        value=f"{predicted_temp_recent:.2f} °C"
    )

st.divider()

# Plotly 시각화
fig = px.scatter(
    yearly_df,
    x="연도",
    y="평균기온",
    title="서울 연도별 평균기온 및 회귀 추세선 비교",
    labels={"연도": "연도", "평균기온": "평균기온 (°C)"},
    hover_data={"연도": True, "평균기온": ":.2f"}
)

# 회귀 직선 그리기를 위한 연도 범위 (1900년 ~ 2100년)
reg_x_years = np.array([1900, 2100])
reg_x_passed = reg_x_years - 1908

# 1. 전체 기간 회귀선 (레드)
reg_y_full = slope_full * reg_x_passed + intercept_full
fig.add_trace(
    go.Scatter(
        x=reg_x_years,
        y=reg_y_full,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(color="red", width=2)
    )
)

# 2. 최근 20년 회귀선 (오렌지 점선)
reg_y_recent = slope_recent * reg_x_passed + intercept_recent
fig.add_trace(
    go.Scatter(
        x=reg_x_years,
        y=reg_y_recent,
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(color="orange", width=2, dash="dash")
    )
)

# 3. 선택 연도 예측 지점 표시
fig.add_trace(
    go.Scatter(
        x=[target_year],
        y=[predicted_temp_full],
        mode="markers",
        name=f"선택 해 ({target_year}년, 전체 추세)",
        marker=dict(color="gold", size=14, symbol="star")
    )
)

fig.add_trace(
    go.Scatter(
        x=[target_year],
        y=[predicted_temp_recent],
        mode="markers",
        name=f"선택 해 ({target_year}년, 최근 20년 추세)",
        marker=dict(color="darkorange", size=14, symbol="diamond")
    )
)

# Layout 설정
fig.update_layout(
    xaxis=dict(title="연도", dtick=20),
    yaxis=dict(title="평균기온 (°C)"),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)
