import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.colors as pc

st.set_page_config(layout="wide")

# ETF 데이터 (배당률 포함)
etfs = {
    "TIGER 미국나스닥100커버드콜": {"country": "미국", "dividend_yield": 8.98},
    "PLUS K리츠": {"country": "한국", "dividend_yield": 5.50},
    "TIGER 200커버드콜ATM": {"country": "한국", "dividend_yield": 6.59},
    "RISE 200고배당커버드콜ATM": {"country": "한국", "dividend_yield": 6.20},
    "TIGER 리츠부동산인프라": {"country": "한국", "dividend_yield": 6.28},
    "KODEX 미국채10년물(H)": {"country": "미국", "dividend_yield": 2.0},
    "KODEX 미국달러선물": {"country": "미국", "dividend_yield": 1.5},
    "KODEX 인버스": {"country": "한국", "dividend_yield": 0.5},
    "KODEX 골드선물(H)": {"country": "미국", "dividend_yield": 0.3},
    "KODEX 미국S&P500TR": {"country": "미국", "dividend_yield": 1.8},
    "Invesco QQQ Trust ETF": {"country": "미국", "dividend_yield": 0.6},
    "TIGER 글로벌리튬&2차전지SOLACTIVE": {"country": "글로벌", "dividend_yield": 0.5},
    "Schwab U.S. Dividend Equity ETF™ (SCHD)": {
        "country": "미국",
        "dividend_yield": 3.5,
    },
    "JPMorgan Equity Premium Income ETF (JEPI)": {
        "country": "미국",
        "dividend_yield": 9.2,
    },
}
# 주요 지표 설정
indicators = {
    "채권 및 금리": {"TLT": "^TYX"},
    "통화": {"USD/KRW": "KRW=X"},
    "금융": {"DXY": "DX-Y.NYB"},
    "주식 시장": {"SPY": "SPY", "QQQ": "QQQ"},
    "변동성": {"VIX": "^VIX"},
}

# 데이터 가져오기
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")


@st.cache_data
def load_data():
    data = pd.DataFrame()
    for category, tickers in indicators.items():
        for name, ticker in tickers.items():
            df = yf.download(ticker, start=start_date, end=end_date)["Adj Close"]
            data[name] = df
    return data.ffill().bfill()


data = load_data()

# 데이터 정규화
normalized_data = data / data.iloc[0] * 100

st.title("Economic Indicators Analysis and Investment Strategy")

# 정규화된 데이터 시각화
st.subheader("Normalized Economic Indicators Trend (2020-Present)")

fig = make_subplots(
    rows=3, cols=2, subplot_titles=list(indicators.keys()), vertical_spacing=0.1
)

for i, (category, tickers) in enumerate(indicators.items()):
    row = i // 2 + 1
    col = i % 2 + 1
    for name in tickers:
        fig.add_trace(
            go.Scatter(x=normalized_data.index, y=normalized_data[name], name=name),
            row=row,
            col=col,
        )
    fig.update_xaxes(title_text="Date", row=row, col=col)
    fig.update_yaxes(title_text="Normalized Value", row=row, col=col)

fig.update_layout(
    height=1200, width=1000, title_text="Normalized Economic Indicators by Category"
)
st.plotly_chart(fig)

# ETF 티커 매핑
etf_tickers = {
    "KODEX 미국채10년물(H)": "152380.KS",
    "KODEX 미국달러선물": "261240.KS",
    "KODEX 인버스": "114800.KS",
    "KODEX 골드선물(H)": "132030.KS",
    "KODEX 미국S&P500TR": "360750.KS",
    "Invesco QQQ Trust ETF": "QQQ",
    "TIGER 글로벌리튬&2차전지SOLACTIVE": "456680.KS",
    "KODEX 인도Nifty50": "275280.KS",
    "KODEX 미국반도체MV": "426410.KS",
    "KODEX 미국나스닥100TR": "379800.KS",
    "KODEX 미국30년국채선물(H)": "280930.KS",
    "KODEX 미국에너지TOP10": "457030.KS",
    "KODEX 미국배당귀족커버드콜": "457690.KS",
    "KODEX 글로벌반비마전로제TOP2 Plus": "462390.KS",
    "TIGER 미국테크TOP10 INDXX": "364980.KS",
    "TIGER 차이나전기차SOLACTIVE": "371460.KS",
    "TIGER 미국필라델피아반도체나스닥": "364660.KS",
    "TIGER 미국나스닥100": "133690.KS",
    "TIGER 미국S&P500": "360750.KS",
    "SOL 미국30년국채선물(H)": "439870.KS",
}

st.title("ETF 투자 분석 도구")

# 사용자 입력: ETF 선택 또는 직접 입력
st.subheader("ETF 선택")
etf_option = st.radio("ETF 선택 방식:", ["목록에서 선택", "직접 입력"])

if etf_option == "목록에서 선택":
    selected_ticker = st.selectbox(
        "투자할 ETF를 선택하세요:",
        list(etf_tickers.keys()),
        format_func=lambda x: f"{etf_tickers[x]} ({x})",
    )
    selected_ticker = etf_tickers[selected_ticker]
else:
    selected_ticker = st.text_input("ETF 티커를 입력하세요 (예: 379810.KS):")

# ETF 이름 찾기
try:
    etf_info = yf.Ticker(selected_ticker)
    selected_etf_name = etf_info.info["longName"]
except:
    selected_etf_name = "알 수 없는 ETF"

st.write(f"선택된 ETF: {selected_etf_name} ({selected_ticker})")


@st.cache_data
def load_etf_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        return None


def predict_returns(data, forecast_period=30):
    X = np.arange(len(data)).reshape(-1, 1)
    y = data.values
    model = LinearRegression().fit(X, y)
    future_X = np.arange(len(data), len(data) + forecast_period).reshape(-1, 1)
    future_prices = model.predict(future_X)
    return pd.Series(
        future_prices,
        index=pd.date_range(
            start=data.index[-1] + timedelta(days=1), periods=forecast_period
        ),
    )


def calculate_expected_return(data, forecast):
    return (forecast.iloc[-1] - data.iloc[-1]) / data.iloc[-1]


# 날짜 범위 설정
end_date = datetime.now()
start_date = end_date - timedelta(days=1460)  # 4년

# 데이터 로드
etf_data = load_etf_data(selected_ticker, start_date, end_date)

if etf_data is not None and not etf_data.empty:
    # 그래프 그리기
    st.subheader(f"{selected_etf_name} 최근 4년 추이 및 예측")

    # 예측 수행
    forecast = predict_returns(etf_data["Close"])
    expected_return = calculate_expected_return(etf_data["Close"], forecast)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3]
    )

    fig.add_trace(
        go.Scatter(x=etf_data.index, y=etf_data["Close"], name="실제 종가"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast.values,
            name="예측 종가",
            line=dict(dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=etf_data.index, y=etf_data["Volume"], name="거래량"), row=2, col=1
    )

    fig.update_layout(
        height=800, title_text=f"{selected_etf_name} 가격, 예측 및 거래량"
    )
    fig.update_xaxes(title_text="날짜", row=2, col=1)
    fig.update_yaxes(title_text="가격", row=1, col=1)
    fig.update_yaxes(title_text="거래량", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # 예상 수익률 표시
    st.subheader("예상 수익률")
    st.write(f"30일 예상 수익률: {expected_return:.2%}")

    # 기본 통계 정보 표시
    st.subheader("기본 통계 정보")
    stats = etf_data["Close"].describe()
    st.write(stats)

else:
    st.warning(
        "선택한 ETF의 데이터를 불러올 수 없습니다. 다른 ETF를 선택하거나 티커를 확인해주세요."
    )

# 실제 배당률 데이터 (2024년 데이터 기준)
# 실제 배당률 데이터 (2024년 데이터 기준)
actual_dividend_yields = {
    "TIGER 미국나스닥100커버드콜": 8.98,
    "PLUS K리츠": 5.50,
    "TIGER 200커버드콜ATM": 6.59,
    "RISE 200고배당커버드콜ATM": 6.20,
    "TIGER 리츠부동산인프라": 6.28,
    "KODEX 미국S&P500배당귀족": 4.43,
    "TIGER 유로스탁스배당30": 4.32,
    "KODEX 일본부동산리츠(H)": 6.48,
    "KODEX 미국배당귀족커버드콜": 6.58,
    "KOSEF 고배당": 4.84,
    "PLUS 고배당주": 7.11,
    "TIGER 미국배당귀족커버드콜": 7.91,
    "PLUS 고배당저변동50": 5.28,
    "TIMEFOLIO Korea플러스배당": 5.31,
}


# 초기 보유금액 설정
initial_balance = 87911630

# 현재 날짜 설정 (예: 2024년 10월 1일부터 시작)
start_date = datetime(2024, 10, 1)
current_date = datetime.now()


# 총 계좌 잔액
total_account_balance = 87911630

# 투자 기간 설정
months = ["2024-10", "2024-11", "2024-12", "2025-01", "2025-02", "2025-03"]

# 초기 투자 전략
initial_investment_strategy = {
    "2024-10": {
        "KODEX 미국채10년물(H)": 16000000,
        "KODEX 미국달러선물": 12000000,
        "KODEX 인버스": 8000000,
        "KODEX 미국S&P500TR": 16000000,
        "Invesco QQQ Trust ETF": 16000000,
        "TIGER 글로벌리튬&2차전지SOLACTIVE": 12000000,
    },
    "2024-11": {
        "KODEX 미국채10년물(H)": 12000000,
        "KODEX 미국달러선물": 12000000,
        "KODEX 인버스": 4000000,
        "KODEX 미국S&P500TR": 20000000,
        "Invesco QQQ Trust ETF": 20000000,
        "TIGER 글로벌리튬&2차전지SOLACTIVE": 12000000,
    },
    "2024-12": {
        "KODEX 미국채10년물(H)": 12000000,
        "KODEX 미국달러선물": 8000000,
        "KODEX 골드선물(H)": 8000000,
        "KODEX 미국S&P500TR": 20000000,
        "Invesco QQQ Trust ETF": 20000000,
        "TIGER 글로벌리튬&2차전지SOLACTIVE": 12000000,
    },
    "2025-01": {
        "KODEX 미국채10년물(H)": 16000000,
        "KODEX 미국달러선물": 8000000,
        "KODEX 골드선물(H)": 8000000,
        "KODEX 미국S&P500TR": 16000000,
        "Invesco QQQ Trust ETF": 16000000,
        "TIGER 글로벌리튬&2차전지SOLACTIVE": 16000000,
    },
    "2025-02": {
        "KODEX 미국채10년물(H)": 20000000,
        "KODEX 미국달러선물": 8000000,
        "KODEX 골드선물(H)": 12000000,
        "KODEX 미국S&P500TR": 12000000,
        "Invesco QQQ Trust ETF": 12000000,
        "TIGER 글로벌리튬&2차전지SOLACTIVE": 16000000,
    },
    "2025-03": {
        "KODEX 미국채10년물(H)": 24000000,
        "KODEX 미국달러선물": 8000000,
        "KODEX 골드선물(H)": 12000000,
        "KODEX 미국S&P500TR": 12000000,
        "Invesco QQQ Trust ETF": 12000000,
        "TIGER 글로벌리튬&2차전지SOLACTIVE": 12000000,
    },
}

st.title("Enhanced ETF Portfolio Management")

# 총 보유금액 표시
st.metric("총 보유금액", f"{total_account_balance:,.0f}원")


# 투자 전략 설정
st.subheader("월별 ETF 투자 금액 설정 (단위: 만원)")

# 탭 생성
tabs = st.tabs(months)

adjusted_strategy = {}
total_investment = 0

for i, (month, tab) in enumerate(zip(months, tabs)):
    with tab:
        st.write(f"### {month} 투자 계획")
        adjusted_strategy[month] = {}

        etf_count = 0
        for j, (etf, info) in enumerate(etfs.items()):
            if etf_count >= 10:
                st.warning("투자 항목이 10개를 초과했습니다. 추가 항목은 무시됩니다.")
                break

            color = (
                "lightblue"
                if info["country"] == "한국"
                else "lightgreen" if info["country"] == "미국" else "lightyellow"
            )
            initial_value = (
                float(initial_investment_strategy.get(month, {}).get(etf, 0)) / 10000
            )  # 만원 단위로 변환
            amount = st.number_input(
                f"{j+1}. {etf} ({info['country']}, 배당률: {info['dividend_yield']:.2f}%)",
                value=int(initial_value),
                min_value=0,
                max_value=int(initial_balance / 10000),
                step=100,  # 100만원 단위로 조정 가능
                key=f"{etf}_{month}",
                help=f"배경색: {color}",
            )

            if amount > 0:
                adjusted_strategy[month][etf] = (
                    amount * 10000
                )  # 원 단위로 변환하여 저장
                etf_count += 1

        month_total = sum(adjusted_strategy[month].values())
        total_investment += month_total

        if month_total > initial_balance:
            st.error(
                "총 투자 금액이 초기 보유금액을 초과했습니다. 투자 금액을 조정해주세요."
            )

# 총 현금보유금(MMF) 계산
total_cash_balance = initial_balance - total_investment

# 현금 보유금 이자 계산 (연 3%)
days_passed = (current_date - start_date).days
cash_interest = total_cash_balance * (0.03 * days_passed / 365)

# 투자 수익 계산
investment_profit = sum(
    amount * (etfs[etf]["dividend_yield"] / 100 * days_passed / 365)
    for month in adjusted_strategy
    for etf, amount in adjusted_strategy[month].items()
)

# 현재 가치 금액 계산
current_value = (
    total_investment + investment_profit + total_cash_balance + cash_interest
)
# 현금 보유금(MMF) 계산
cash_balance = initial_balance - total_investment
# 금액 표시
col1, col2, col3 = st.columns(3)
col1.metric("초기 보유금액", f"{initial_balance:,.0f}원")
col2.metric("총 투자금액", f"{total_investment:,.0f}원")
col3.metric("현금보유금(MMF)", f"{cash_balance:,.0f}원")

# 포트폴리오 분석
st.subheader("포트폴리오 분석")

expected_dividend = sum(
    amount * (etfs[etf]["dividend_yield"] / 100 / 12)
    for month in adjusted_strategy
    for etf, amount in adjusted_strategy[month].items()
)

st.write(f"예상 연간 배당금: {expected_dividend*12:,.0f}원")
st.write(f"예상 연간 배당 수익률: {expected_dividend*12/total_investment*100:.2f}%")

# ETF별 투자 비중 그래프
st.subheader("ETF별 투자 비중")

etf_totals = {}
for month in adjusted_strategy:
    for etf, amount in adjusted_strategy[month].items():
        etf_totals[etf] = etf_totals.get(etf, 0) + amount

# 현금(MMF) 추가
etf_totals["현금(MMF)"] = cash_balance

total_investment = sum(etf_totals.values())

# 비율이 1% 미만인 ETF를 'Others'로 그룹화
others_threshold = 0.01
others_sum = 0
etf_totals_filtered = {}

for etf, amount in etf_totals.items():
    if amount / total_investment >= others_threshold:
        etf_totals_filtered[etf] = amount
    else:
        others_sum += amount

if others_sum > 0:
    etf_totals_filtered["Others"] = others_sum

etf_totals_filtered = dict(
    sorted(etf_totals_filtered.items(), key=lambda item: item[1], reverse=True)
)

# 색상 팔레트 생성
colors = pc.qualitative.Plotly * 2

# 각 ETF에 대한 텍스트 생성
etf_texts = []
for etf, amount in etf_totals_filtered.items():
    percentage = amount / total_investment * 100
    amount_in_ten_thousand = amount / 10000  # 만원 단위로 변환
    text = f"{etf}: {percentage:.2f}% ({amount_in_ten_thousand:.0f}만원)"
    etf_texts.append(text)

# 라벨과 값 리스트 생성
labels = list(etf_totals_filtered.keys())
values = list(etf_totals_filtered.values())

fig_pie = go.Figure(
    data=[
        go.Pie(
            labels=labels,
            values=values,
            text=etf_texts,
            textinfo="text",
            hoverinfo="label+percent+value",
            marker=dict(
                colors=colors[: len(etf_totals_filtered)],
                line=dict(color="#000000", width=2),
            ),
        )
    ]
)

fig_pie.update_layout(
    height=600,
    title="ETF별 총 투자 금액 비중 (현금 포함, 단위: 만원)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig_pie)


# 월별 투자 금액 그래프
st.subheader("월별 ETF 투자 금액")

fig_monthly = go.Figure()


def get_color(etf):
    if etf == "현금(MMF)":
        return "lightgray"
    elif etf in etfs:
        if etfs[etf]["country"] == "한국":
            return "lightblue"
        elif etfs[etf]["country"] == "미국":
            return "lightgreen"
        else:
            return "lightyellow"
    else:
        return "lightpink"  # 예외 처리를 위한 기본 색상


for etf in etf_totals.keys():
    if etf == "현금(MMF)":
        monthly_investments = [
            (total_account_balance - sum(adjusted_strategy[month].values())) / 10000
            for month in months
        ]
    else:
        monthly_investments = [
            adjusted_strategy[month].get(etf, 0) / 10000 for month in months
        ]

    fig_monthly.add_trace(
        go.Bar(x=months, y=monthly_investments, name=etf, marker_color=get_color(etf))
    )

fig_monthly.update_layout(
    barmode="stack",
    title="월별 ETF 투자 금액 (현금 포함)",
    xaxis_title="월",
    yaxis_title="투자 금액 (만원)",
    height=600,
)

# y축 눈금을 만원 단위로 표시
fig_monthly.update_yaxes(tickformat=",d", title_text="투자 금액 (만원)")

st.plotly_chart(fig_monthly)
