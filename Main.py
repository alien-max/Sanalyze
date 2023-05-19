import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def Candel(df):
    fig = go.Figure(
        data = [
            go.Candlestick(
                x = df.index,
                open = df['Open'],
                high = df['High'],
                low = df['Low'],
                close = df['Close'],
                increasing_line_color= 'green',
                decreasing_line_color= 'red'
            )
        ]
    )
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


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
    dc = 1
    for i in y_pred:
        st.write("Day {} Predicted price: {}".format(dc, round(i, 2)))
        dc += 1

    sh_df = df.tail(30)
    a = np.array(sh_df['Close'].values)
    b = np.array(y_pred)
    v = np.hstack((a, b))

    plt.figure(figsize=(16, 8))
    plt.ylabel('Price')
    plt.plot(v, label='Predicted Price', linewidth=4)
    plt.plot(a, label='Current Price', linewidth=4)
    plt.legend()

    st.pyplot(plt.gcf())


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

    plt.figure(figsize=(16, 4))
    plt.plot(mfi, color='b', alpha=0.7)
    plt.axhline(20, linestyle='--', color='r')
    plt.axhline(80, linestyle='--', color='r')

    st.pyplot(plt.gcf())


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

    plt.figure(figsize=(16, 4))
    plt.plot(RSI, color='g', alpha=0.7)
    plt.axhline(20, linestyle='--', color='r')
    plt.axhline(80, linestyle='--', color='r')

    st.pyplot(plt.gcf())


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


def get_input():
    stock_symbol = st.sidebar.text_input("Symbol:", "AAPL")
    return stock_symbol


def get_data(symbol):
    data = yf.Ticker(symbol)
    if data:
        df = data.history(period='max')
        s_df = yf.download(symbol, period='1d', interval='15m')
        info = data.info
        return df, s_df, info
    else:
        return 0, None


symbol = get_input()
df, s_df, info = get_data(symbol.upper())

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

    st.plotly_chart(Candel(s_df), use_container_width=True)

    st.write("----")
    st.header("Prediction")
    num_days = st.text_input("How many days do you want to be predicted?", 1)
    PredLir(df, num_days)

    ind = st.sidebar.selectbox("Indicators", options=('MACD', 'MAVG'))
    if ind == 'MACD':
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
        CalcMACD(df.tail(3*p_second_val), p_first_val, p_second_val)
        CalcMFI(df.tail(3*p_second_val), p_first_val)
        CalcRSI(df.tail(3*p_second_val), p_first_val)

    elif ind == 'MAVG':
        st.write("----")
        st.header("MAVG Indicator")
        sma_period = {
            '12-26 D': (12, 26),
            '30-100 D': (30, 100),
            '50-200 D': (50, 200),
        }
        sma_period_selection = st.selectbox("Time Period", options=sma_period)
        p_first_val = sma_period[sma_period_selection][0]
        p_second_val = sma_period[sma_period_selection][1]
        CalcMA(df.tail(3*p_second_val), p_first_val, p_second_val)
        CalcMFI(df.tail(3*p_second_val), p_first_val)
        CalcRSI(df.tail(3*p_second_val), p_first_val)

else:
    st.error("Symbol is Wrong !!!")
