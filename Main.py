import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from utils import *


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

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    st.sidebar.title(f'Welcome *{name}*')
    symbol = get_input()
    df, s_df, info = get_data(symbol.upper())
    if df is not None:
        st.header(info['longName'])
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

        authenticator.logout('Logout', 'sidebar')
    else:
        st.error("Symbol is Wrong !!!")

elif authentication_status is False:
    st.error('Username/password is incorrect')
elif authentication_status is None:
    st.warning('Please enter your username and password')


