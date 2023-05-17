import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


def PredLir(df, n):
    days = int(n)
    df = df[['Close']]
    df['Prediction'] = df[['Close']].shift(-days)

    x = np.array(df.drop(['Prediction'], axis=1))
    x = x[:-days]

    y = np.array(df['Prediction'])
    y = y[:-days]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = LinearRegression()
    model.fit(x_train, y_train)
    model_conf = model.score(x_test, y_test)

    st.write("Linear Regression Prediction accuracy: %{}".format(round((model_conf*100), 2)))

    x_forecast = np.array(df.drop(['Prediction'], axis=1))[-days:]
    y_pred = model.predict(x_forecast)

    a = np.array(df['Close'].values)
    b = np.array(y_pred)
    v = np.hstack((a, b))

    plt.figure(figsize=(16, 8))
    plt.ylabel('Price')
    plt.plot(v, label='Predicted Price', linewidth=4)
    plt.plot(a, label='Current Price', linewidth=4)
    plt.legend()

    st.pyplot(plt.gcf())


def PredSvr(df, n):
    days = int(n)
    df = df[['Close']]
    df['Prediction'] = df[['Close']].shift(-days)

    x = np.array(df.drop(['Prediction'], axis=1))
    x = x[:-days]

    y = np.array(df['Prediction'])
    y = y[:-days]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    model = SVR()
    model.fit(x_train, y_train)
    model_conf = model.score(x_test, y_test)

    st.write("Support Vector Regression Prediction accuracy: %{}".format(round((model_conf*100), 2)))

    x_forecast = np.array(df.drop(['Prediction'], axis=1))[-days:]
    y_pred = model.predict(x_forecast)
    
    a = np.array(df['Close'].values)
    b = np.array(y_pred)
    v = np.hstack((a, b))

    plt.figure(figsize=(16, 8))
    plt.ylabel('Price')
    plt.plot(v, label='Predicted Price', linewidth=4)
    plt.plot(a, label='Current Price', linewidth=4)
    plt.legend()

    st.pyplot(plt.gcf())


def Candel(df):
    fig = go.Figure(
        data = [
            go.Candlestick(
                x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Close'],
                increasing_line_color = 'green',
                decreasing_line_color = 'red'
            )
        ]
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


def CalcMFI(df, period):
    typical_price = (df['Close'] + df['High'] + df['Low'])/3
    money_flow = typical_price * df['Volume']

    positive_flow = []
    negative_flow = []

    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow[i-1])
            negative_flow.append(0)
        elif typical_price[i] < typical_price[i-1]:
            positive_flow.append(0)
            negative_flow.append(money_flow[i-1])
        else:
            positive_flow.append(0)
            negative_flow.append(0)

    positive_mf = []
    negative_mf = []

    for i in range(period-1, len(positive_flow)):
        positive_mf.append(sum(positive_flow[i+1-period:i+1]))

    for i in range(period-1, len(negative_flow)):
        negative_mf.append(sum(negative_flow[i+1-period:i+1]))

    mfi = 100*(np.array(positive_mf)/(np.array(positive_mf)+np.array(negative_mf)))

    new_data = pd.DataFrame()
    new_data['MFI'] = mfi

    plt.style.use('fivethirtyeight')
    new_fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    ax1.plot(df['Close'], label="Close Price")
    ax2.plot(new_data['MFI'], label='MFI', color='g', alpha=0.5)
    ax2.axhline(20, linestyle='--', color='r')
    ax2.axhline(80, linestyle='--', color='r')

    st.pyplot(new_fig)


def CalcRSI(df, period):
    delta = df['Close'].diff(1)
    delta.dropna()

    up = delta.copy()
    down = delta.copy()

    up[up<0] = 0
    down[down>0] = 0

    avg_gain = up.rolling(window=period).mean()
    avg_loss = abs(down.rolling(window=period).mean())
    RS = avg_gain / avg_loss
    RSI = 100.0 - (100.0 / (1.0 + RS))

    new_data = pd.DataFrame()
    new_data['Close'] = df['Close']
    new_data['RSI'] = RSI

    plt.style.use('fivethirtyeight')
    new_fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))
    ax1.plot(new_data['Close'], label="Close Price")
    ax2.plot(new_data['RSI'], label='RSI', color='g', alpha=0.5)
    ax2.axhline(20, linestyle='--', color='r')
    ax2.axhline(80, linestyle='--', color='r')

    st.pyplot(new_fig)


def CalcMACD(df, p_first_val, p_second_val):
    shortEMA = df.Close.ewm(span=p_first_val, adjust=False).mean()
    longEMA = df.Close.ewm(span=p_second_val, adjust=False).mean()
    MACD = shortEMA - longEMA
    bs_signal = MACD.ewm(span=9, adjust=False).mean()

    sma_data = pd.DataFrame()
    sma_data['Price'] = df['Close']
    sma_data['MACD'] = MACD
    sma_data['bs_signal'] = bs_signal

    def signal(data):
        buySignal = []
        sellSignal = []
        f = -1
        for i in range(len(data)):
            if data['MACD'][i] > data['bs_signal'][i]:
                sellSignal.append(np.nan)
                if f != 1:
                    buySignal.append(data['Price'][i])
                    f = 1
                else:
                    buySignal.append(np.nan)
            elif data['MACD'][i] < data['bs_signal'][i]:
                buySignal.append(np.nan)
                if f != 0:
                    sellSignal.append(data['Price'][i])
                    f = 0
                else:
                    sellSignal.append(np.nan)
            else:
                buySignal.append(np.nan)
                sellSignal.append(np.nan)
        return buySignal, sellSignal
    
    buy_sell = signal(sma_data)
    sma_data['Buy Signal'] = buy_sell[0]
    sma_data['Sell Signal'] = buy_sell[1]

    plt.figure(figsize=(16, 8))
    plt.plot(sma_data['Price'], label="Close Price", alpha=0.4)
    plt.scatter(sma_data.index, sma_data['Buy Signal'], label="Buy", marker="^", color="g", linewidths=5)
    plt.scatter(sma_data.index, sma_data['Sell Signal'], label="Sell", marker="v", color="r", linewidths=5)
    plt.legend()

    st.pyplot(plt.gcf())


def CalcMA(df, p_first_val, p_second_val):
    ma_1 = pd.DataFrame()
    ma_1['AM'] = df['Close'].rolling(window=p_first_val).mean()

    ma_2 = pd.DataFrame()
    ma_2['AM'] = df['Close'].rolling(window=p_second_val).mean()

    sma_data = pd.DataFrame()
    sma_data['Price'] = df['Close']
    sma_data['MA_1'] = ma_1['AM']
    sma_data['MA_2'] = ma_2['AM']

    def signal(data):
        buySignal = []
        sellSignal = []
        f = -1
        for i in range(len(data)):
            if data['MA_1'][i] > data['MA_2'][i]:
                if f != 1:
                    buySignal.append(data['Price'][i])
                    sellSignal.append(np.nan)
                    f = 1
                else:
                    buySignal.append(np.nan)
                    sellSignal.append(np.nan)
            elif data['MA_1'][i] < data['MA_2'][i]:
                if f != 0:
                    buySignal.append(np.nan)
                    sellSignal.append(data['Price'][i])
                    f = 0
                else:
                    buySignal.append(np.nan)
                    sellSignal.append(np.nan)
            else:
                buySignal.append(np.nan)
                sellSignal.append(np.nan)
                
        return buySignal, sellSignal

    buy_sell = signal(sma_data)
    sma_data['Buy Signal'] = buy_sell[0]
    sma_data['Sell Signal'] = buy_sell[1]

    plt.figure(figsize=(16, 8))
    plt.plot(sma_data['Price'], label="Close Price", alpha=0.4)
    plt.scatter(sma_data.index, sma_data['Buy Signal'], label="Buy", marker="^", color="g", linewidths=5)
    plt.scatter(sma_data.index, sma_data['Sell Signal'], label="Sell", marker="v", color="r", linewidths=5)
    plt.legend()

    st.pyplot(plt.gcf())


st.set_page_config(
    page_title="Sanalyze",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': '''
        # Sanalyze\n
        Stock market analyze app\n
        Author: Alien-Max\n
        Homepage: https://github.com/alien-max/Sanalyze
        '''
    }
)


periods = {
    '1D': '1d',
    '5D': '5d',
    '1M': '1mo',
    '3M': '3mo',
    '6M': '6mo',
    '1Y': '1y',
    '2Y': '2y',
    '5Y': '5y',
    '10Y': '10y',
    'Max': 'max'
}

def get_input():
    stock_symbol = st.sidebar.text_input("Symbol:", "AAPL")
    numb = st.sidebar.selectbox("Time Period:", options=periods, index=2)
    return stock_symbol, numb


def get_data(symbol, n):
    data = yf.Ticker(symbol)
    if data:
        df = data.history(period=periods[n])
        info = data.info
        return df, info
    else:
        return 0, None


symbol, n = get_input()
df, info = get_data(symbol.upper(), n)

if df is not None:
    try:
        st.header(info['longName'])
    except:
        pass
    try:
        st.write(info['country'])
    except:
        pass
    try:
        st.write(info['sector'])
    except:
        pass
    try:
        st.write(info['website'])
    except:
        pass

    st.plotly_chart(Candel(df), use_container_width=True)

    pred = st.sidebar.selectbox("Prediction", options=('LIR Prediction', 'SVR Prediction'))
    if pred == 'LIR Prediction':
        st.write("----")
        st.header("Prediction")
        num_days = st.text_input("How many days do you want to be predicted?", 1)
        PredLir(df, num_days)
    elif pred == 'SVR Prediction':
        st.write("----")
        st.header("Prediction")
        num_days = st.text_input("How many days do you want to be predicted?", 1)
        PredSvr(df, num_days)


    ind = st.sidebar.selectbox("Indicators", options=('MFI', 'RSI', 'MACD', 'MAVG'))
    if ind == 'MFI':
        day_period = {
            '1 D': 1,
            '3 D': 3,
            '7 D': 7,
            '14 D': 14,
            '30 D': 30,
            '60 D': 60,
            '100 D': 100,
            '180 D': 180,
            '1 Year': 365
        }
        st.write("----")
        st.header("MFI Indicator")
        period_selection = st.selectbox("Time Period", options=day_period, index=2)
        period = day_period[period_selection]
        CalcMFI(df, period)

    elif ind == 'RSI':
        day_period = {
            '1 D': 1,
            '3 D': 3,
            '7 D': 7,
            '14 D': 14,
            '30 D': 30,
            '60 D': 60,
            '100 D': 100,
            '180 D': 180,
            '1 Year': 365
        }
        st.write("----")
        st.header("RSI Indicator")
        period_selection = st.selectbox("Time Period", options=day_period, index=2)
        period = day_period[period_selection]
        CalcRSI(df, period)

    elif ind == 'MACD':
        st.write("----")
        st.header("MACD Indicator")
        sma_period = {
            '12-26 D': (12, 26),
            '30-100 D': (30, 100),
            '50-200 D': (50, 200),
        }
        sma_period_selection = st.selectbox("Time Period", options=sma_period)
        p_first_val = sma_period[sma_period_selection][0]
        p_second_val = sma_period[sma_period_selection][1]
        CalcMACD(df, p_first_val, p_second_val)

    elif ind == 'MAVG':
        st.write("----")
        st.header("MACD Indicator")
        sma_period = {
            '12-26 D': (12, 26),
            '30-100 D': (30, 100),
            '50-200 D': (50, 200),
        }
        sma_period_selection = st.selectbox("Time Period", options=sma_period)
        p_first_val = sma_period[sma_period_selection][0]
        p_second_val = sma_period[sma_period_selection][1]
        CalcMA(df, p_first_val, p_second_val)

else:
    st.error("Symbol is Wrong !!!")
