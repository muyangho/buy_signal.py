import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import datetime
import re
import numpy as np

# ==========================================
# 1. 데이터 로드 및 유틸리티
# ==========================================
@st.cache_data(ttl=86400)
def load_krx_dict():
    try:
        df_krx = fdr.StockListing('KRX')
        name_to_code = {row['Name']: row['Code'] for _, row in df_krx.iterrows()}
        code_to_name = {row['Code']: row['Name'] for _, row in df_krx.iterrows()}
        return name_to_code, code_to_name
    except:
        return {}, {}

def parse_input(raw_text):
    tokens = re.split(r'[\s,]+', raw_text.strip())
    return [t for t in tokens if t]

def resolve_ticker(token, market, name_to_code, code_to_name):
    token_upper = token.upper()
    if market == "한국 주식 (KRX)":
        aliases = {"네이버": "NAVER", "기아차": "기아", "카카오페이": "카카오페이", "엔씨": "엔씨소프트"}
        search_name = aliases.get(token, token)
        search_name_upper = aliases.get(token_upper, token_upper)

        if search_name in name_to_code:
            return f"{search_name} ({name_to_code[search_name]})", name_to_code[search_name]
        elif search_name_upper in name_to_code:
            return f"{search_name_upper} ({name_to_code[search_name_upper]})", name_to_code[search_name_upper]
        elif token in code_to_name:
            return f"{code_to_name[token]} ({token})", token
        else:
            return f"{token} (검색 실패)", None
    else: 
        return f"{token_upper} ({token_upper})", token_upper

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(window=period).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

# ==========================================
# 2. 강력해진 매수 시그널 및 필터 로직 (복구 완료)
# ==========================================
def analyze_buy_signals_and_grid(df):
    if df.empty or len(df) < 30:
        return [], [], {}, 0
        
    signals = []
    warnings = []
    
    current_price = df['Close'].iloc[-1]
    prev_close = df['Close'].iloc[-2]
    today_high = df['High'].iloc[-1]
    last_vol = df['Volume'].iloc[-1]
    
    ma_5 = df['Close'].rolling(window=5).mean()
    ma_20 = df['Close'].rolling(window=20).mean()
    avg_vol_20 = df['Volume'].rolling(window=20).mean().iloc[-2]
    
    # [1] 핵심 매수 시그널 포착
    if current_price > prev_close * 1.02 and last_vol > (avg_vol_20 * 1.5):
        signals.append("🚀 [수급 유입] 대량 거래량 동반 상승")
        
    if ma_5.iloc[-1] > ma_20.iloc[-1] and ma_5.iloc[-2] <= ma_20.iloc[-2]:
        signals.append("📈 [추세 전환] 5일선이 20일선 상향 돌파 (골든크로스)")
    elif current_price > ma_20.iloc[-1] and prev_close <= ma_20.iloc[-2]:
        signals.append("🔼 [지지선 회복] 주가가 20일선 상향 돌파")
        
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9, adjust=False).mean()
    
    if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
        signals.append("🟢 [모멘텀 강화] MACD 골든크로스 발생")

    # 차트 시그널 점수 계산 (시그널 개수당 1점)
    chart_score = len(signals)

    # [2] 진입 필터 (휩쏘 및 과열 방지)
    if signals: 
        if today_high > 0 and (today_high - current_price) / today_high > 0.015:
            warnings.append("⚠️ [윗꼬리 경고] 고점 대비 밀리는 중. 분할 매수 필수.")
        if ma_5.iloc[-1] > 0 and (current_price / ma_5.iloc[-1]) > 1.04:
            warnings.append("⚠️ [단기 과열] 5일선 이격도 높음. 눌림목 대기.")
        rsi_series = calculate_rsi(df['Close'])
        if rsi_series.iloc[-1] > 70:
            warnings.append("⚠️ [RSI 과매수] 단기 매수세 과열 상태 (RSI 70+).")

    # [3] ATR 기반 거미줄 타점
    atr_series = calculate_atr(df)
    current_atr = atr_series.iloc[-1]
    
    grid_levels = {
        "1차 (정찰병)": current_price,
        "2차 (눌림목)": round(current_price - (current_atr * 1.0), 2),
        "3차 (투매구간)": round(current_price - (current_atr * 2.5), 2),
    }

    return signals, warnings, grid_levels, chart_score

# ==========================================
# 3. Streamlit UI 구성
# ==========================================
st.set_page_config(layout="wide")
st.title("🦅 실전 타점 지휘소 (수급+차트 통합 픽스본)")
st.markdown("차트의 **기술적 시그널(거래량/돌파/MACD)**과 수동으로 확인한 **MTS 수급 상태**를 합산해 진짜 등급을 매깁니다.")

name_to_code, code_to_name = load_krx_dict()
market_selection = st.radio("분석할 시장을 선택하세요:", ["한국 주식 (KRX)", "미국 주식 (US)"], horizontal=True)

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("1. 관종 입력")
    with st.form("input_form"):
        placeholder_text = "예시: 삼성전자, 000660" if "한국" in market_selection else "예시: NVDA, TSLA"
        user_input = st.text_area("종목 입력", placeholder=placeholder_text, height=68)
        st.caption("※ 여러 종목을 입력할 경우, 우측의 수급 체크박스는 '가장 사고 싶은 1순위 종목' 기준으로 체크해 주세요.")
        submitted = st.form_submit_button("차트 타점 및 등급 계산")

with col2:
    st.subheader("2. 실시간 수급/호가 체크 (MTS 확인)")
    st.markdown("스마트폰 MTS를 켜고 1순위 타겟 종목의 장중 상태를 체크하세요.")
    
    if market_selection == "한국 주식 (KRX)":
        c1 = st.checkbox("🟢 **[프로그램 매매]** 현재 '순매수' 우위")
        c2 = st.checkbox("🟢 **[외인/기관]** 당일 가집계 '양매수' 또는 '매도 축소'")
        c3 = st.checkbox("🟢 **[체결강도]** 현재 체결강도 100% 이상")
        c4 = st.checkbox("🟢 **[거래량]** 1분봉상 직전 음봉을 잡아먹는 양봉 발생")
    else:
        c1 = st.checkbox("🟢 **[프리마켓/선물]** 나스닥 지수 선물이 상승(초록불) 중")
        c2 = st.checkbox("🟢 **[거래량 급증]** 1분봉상 평균 거래량 대비 2배 이상 터짐")
        c3 = st.checkbox("🟢 **[호가창]** 매수 잔량보다 매도 잔량이 더 많음")
        c4 = st.checkbox("🟢 **[섹터 동조화]** 동일 섹터 주식들도 다같이 오르는 중")

    mts_score = sum([c1, c2, c3, c4])

if submitted and user_input:
    tokens = parse_input(user_input)
    if not tokens:
        st.warning("종목을 입력해주세요.")
    else:
        st.divider()
        st.subheader("🎯 최종 진입 시나리오 판독 결과")
        
        end_date = datetime.datetime.today()
        start_date = end_date - datetime.timedelta(days=90)
        
        for token in tokens:
            display_name, search_code = resolve_ticker(token, market_selection, name_to_code, code_to_name)
            
            if search_code is None:
                st.error(f"{display_name} - 종목을 찾을 수 없습니다.")
                continue
                
            try:
                df = fdr.DataReader(search_code, start_date, end_date)
                signals, warnings, grid_levels, chart_score = analyze_buy_signals_and_grid(df)
                
                # 핵심 변경점: 총점 = 차트 시그널 점수(최대 3~4점) + MTS 수급 점수(최대 4점)
                total_score = chart_score + mts_score
                
                st.markdown(f"### 🏷️ {display_name}")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    # 차트에서 잡힌 시그널 출력
                    st.markdown("**[포착된 기술적 시그널]**")
                    if signals:
                        for sig in signals:
                            st.write(sig)
                    else:
                        st.write("관망 (차트상 뚜렷한 매수 시그널 없음)")
                        
                    if warnings:
                        st.markdown("**[장중 리스크 경고]**")
                        for warn in warnings:
                            st.write(warn)
                            
                    st.markdown("**[🕸️ 분할 매수 타점]**")
                    if market_selection == "한국 주식 (KRX)":
                        for step, price in grid_levels.items():
                            st.write(f"- {step}: **{int(price):,}원**")
                    else:
                        for step, price in grid_levels.items():
                            st.write(f"- {step}: **${price:,.2f}**")
                            
                with res_col2:
                    st.markdown(f"**[종합 스코어: {total_score}점]** (차트 {chart_score} + 수급 {mts_score})")
                    
                    # 총점에 따른 입체적 등급 분류
                    if chart_score == 0:
                        st.error("🛑 **[매수 금지]** 차트상 시그널이 전혀 없습니다. 수급이 좋아 보여도 개미 털기일 확률이 높으니 2차 타점까지 기다리세요.")
                    elif total_score >= 6:
                        st.success("🔥 **[S등급 / 강력 매수]** 차트 돌파와 수급이 완벽하게 일치합니다. 현재가 부근에서 비중 50% 이상 과감하게 1차 진입해도 좋습니다.")
                    elif total_score >= 4:
                        st.info("👍 **[A등급 / 분할 매수]** 준수한 타점입니다. 현재가에서 비중 20~30% 정찰병 진입 후, 2차 타점에 낚싯대를 걸어두세요.")
                    elif total_score >= 2:
                        st.warning("⚠️ **[B등급 / 휩쏘 주의]** 차트 시그널은 떴으나 수급이 약합니다. 윗꼬리를 달고 내려올 수 있으니 무조건 2차(눌림목) 타점까지 기다리세요.")
                    else:
                        st.error("🛑 **[C등급 / 관망]** 시그널이 매우 약합니다. 매수를 보류하세요.")
                st.markdown("---")
                
            except Exception as e:
                st.error(f"{display_name} - 데이터 호출 실패")