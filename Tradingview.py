import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta, date
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import appdirs as ad
CACHE_DIR = ".cache"
# Force appdirs to say that the cache dir is .cache
ad.user_cache_dir = lambda *args: CACHE_DIR
# Create the cache dir if it doesn't exist
Path(CACHE_DIR).mkdir(exist_ok=True)
import yfinance as yf
from sklearn.linear_model import LinearRegression
from streamlit import set_page_config
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
# Set page configuration for full width
set_page_config(layout="wide")

# Developed by Sreenivas, Data Scientist
st.header('prediction models')
#st.title('Developed by Sreenivas, Data Scientist')

# Definitions
today = date.today()
comm_dict = {'^NSEBANK':'NSEBANK','^CNX100':'CNX100','^INDIAVIX':'INDIAVIX',
             'LIX15.NS':'LIX15','^NSEMDCP50':'NSEMDCP50','^NSEI':'NSEI',
             '^NSMIDCP':'NSMIDCP','^CNXDIVOP':'CNXDIVOP',
             'NIFTYQUALITY30.NS':'NIFTYQUALITY30','NV20.NS':'NV20',
             '^TV.NS':'TV','^CNXAUTO':'CNXAUTO','^CNXPSUBANK':'CNXPSUBANK',
             'NIFTYPVTBANK.NS':'NIFTYPVTBANK','^CNXFMCG':'CNXFMCG',
             '^CNXMETAL':'CNXMETAL','^CNXPSE':'CNXPSE','^CNX200':'CNX200',
             '^CNXENERGY':'CNXENERGY','^CNXPHARMA':'CNXPHARMA',
             '^CNXMNC':'CNXMNC','CPSE.NS':'CPSE','^CRSLDX':'CRSLDX',
             '^CRSMID':'CRSMID','^NSEDIV':'NSEDIV','NI15.NS':'NI15',
             '^CNXIT':'CNXIT','NIFTYTR2XLEV.NS':'NIFTYTR2XLEV',
             'NIFTYPR2XLEV.NS':'NIFTYPR2XLEV','NIFTYMIDLIQ15.NS':'NIFTYMIDLIQ15',
             '^CNXINFRA':'CNXINFRA','^CNXCONSUM':'CNXCONSUM',
             '^CNXSERVICE':'CNXSERVICE','NIFTYTR1XINV.NS':'NIFTYTR1XINV',
             '^CNXCMDT':'CNXCMDT','^CNXREALTY':'CNXREALTY','CL=F':'Crude_Oil','PLN=X':'PLN/USD',
             '^CNXMEDIA':'CNXMEDIA','^CNXFIN':'CNXFIN','^CNXSC':'CNXSC',
             'NIFTYPR1XINV.NS':'NIFTYPR1XINV'}


# Data Retrieval
def comm_f(comm):
    global df_c1
    for label, name in comm_dict.items():
        if name == comm:
            df_c = pd.DataFrame(yf.download(f'{label}', start='2000-09-01', end = today,interval='1d'))
            df_c1 = df_c.reset_index()
           
    return df_c1 

# Historical Data                    
def comm_data(comm):
    global Tab_his1
    shape_test=[]
    sh = df_c1.shape[0]
    start_date = df_c1.Date.min()
    end_date = df_c1.Date.max()
    close_max = "{:.2f}".format(df_c1['Close'].max())
    close_min = "{:.2f}".format(df_c1['Close'].min())
    last_close = "{:.2f}".format(df_c1['Close'].iloc[-1])
    v = (comm, sh, start_date,end_date,close_max,close_min,last_close)
    shape_test.append(v)
    Tab_length = pd.DataFrame(shape_test, columns= ['Name','Rows', 'Start_Date', 'End_Date','Close_max','Close_min','Last_close'])   
    Tab_his = Tab_length[['Start_Date','End_Date','Close_max','Close_min','Last_close']]
    Tab_his['Start_Date'] = Tab_his['Start_Date'].dt.strftime('%Y-%m-%d')
    Tab_his['End_Date'] = Tab_his['End_Date'].dt.strftime('%Y-%m-%d')
    Tab_his1 = Tab_his.T
    Tab_his1.rename(columns={0: "Details"}, inplace=True)
    
    return Tab_his1

st.sidebar.header('Commodities, Indexies, Currencies & Bonds')
comm = st.sidebar.selectbox('What do you want to analyze today?', list(comm_dict.values()))
comm_f(comm)
st.sidebar.write('You selected:', comm)
st.sidebar.dataframe(comm_data(comm))

# Add this to the sidebar selectbox options
comm_options = list(comm_dict.values())
comm_options.sort()
comm_options.insert(0, 'Select...')
comm = st.sidebar.selectbox('What do you want to analyze today?', comm_options)
if comm != 'Select...':
    comm_f(comm)
    st.sidebar.write('You selected:', comm)
    st.sidebar.dataframe(comm_data(comm))
    
# 15-minute charts will be inserted here
def t1_f(char1):
    global tf_c1
    for label, name in comm_dict.items():
        if name == char1:
            box = yf.Ticker(label)
            tf_c = pd.DataFrame(box.history(period='1d', interval="1m"))
            tf_c1 = tf_c[-100:]
    return tf_c1 

def t2_f(char2):
    global tf_c2
    for label, name in comm_dict.items():
        if name == char2:        
            box = yf.Ticker(label)
            tf_c = pd.DataFrame(box.history(period='1d', interval="1m"))
            tf_c2 = tf_c[-100:]
    return tf_c2 


col1, col2 = st.columns([0.47, 0.53])
with col1:
    box = list(comm_dict.values())
    char1 = st.selectbox('Daily trading dynamics', box, index= box.index('Crude_Oil'),key = "<char1>")
    t1_f(char1)
    data_x1 = tf_c1.index
    fig_char1 = px.line(tf_c1, x=data_x1, y=['Open','High','Low','Close'],color_discrete_map={
                 'Open':'yellow','High':'red','Low':'blue','Close':'green'}, width=500, height=400) 
    fig_char1.update_layout(showlegend=False)
    fig_char1.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_char1) #use_container_width=True
with col2:
    char2 = st.selectbox('Daily trading dynamics', box, index=box.index('PLN/USD'),key = "<char2>")
    t2_f(char2)
    data_x2 = tf_c2.index
    fig_char2 = px.line(tf_c2, x=data_x2, y=['Open','High','Low','Close'],color_discrete_map={
                 'Open':'yellow','High':'red','Low':'blue','Close':'green'}, width=500, height=400) 
    fig_char2.update_layout(showlegend=True)
    fig_char2.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_char2)

# Definition of moving average chart 
st.subheader('Buy and sell signals generated by short and long rolling averages')
st.subheader(f'for {comm} Prices from NYSE')

xy = (list(df_c1.index)[-1] + 1)  
col3, col4, col5 = st.columns([0.4, 0.3, 0.3])
with col3:
    oil_p = st.slider('How long prices history you need?', 0, xy, 200, key = "<commodities>")
with col4:
    nums = st.number_input('Enter the number of days for short average',value=10, key = "<m30>")
with col5:
    numl = st.number_input('Enter the number of days for long average',value=30, key = "<m35>")
    
def roll_avr(nums,numl):
    global df_c_XDays
    # Calculate short-term and long-term moving averages
    df_c1['Short_SMA']= df_c1['Close'].rolling(window=nums).mean()
    df_c1['Long_SMA']= df_c1['Close'].rolling(window=numl).mean()
    
    # Generate buy and sell signals
    df_c1['Buy_Signal'] = (df_c1['Short_SMA'] > df_c1['Long_SMA']).astype(int).diff()
    df_c1['Sell_Signal'] = (df_c1['Short_SMA'] < df_c1['Long_SMA']).astype(int).diff()
     
    df_c_XDays = df_c1.iloc[xy - oil_p:xy]
      
    fig1 = px.line(df_c_XDays, x='Date', y=['Close','Short_SMA','Long_SMA'], color_discrete_map={'Close':'#d62728',
                  'Short_SMA': '#f0f921','Long_SMA':'#0d0887'}, width=1000, height=500)
    fig1.add_trace(go.Scatter(x=df_c_XDays[df_c_XDays['Buy_Signal'] == 1].Date, y=df_c_XDays[df_c_XDays['Buy_Signal'] == 1]['Short_SMA'], name='Buy_Signal', mode='markers', 
                             marker=dict(color='green', size=15, symbol='triangle-up')))
    fig1.add_trace(go.Scatter(x=df_c_XDays[df_c_XDays['Sell_Signal'] == 1].Date, y=df_c_XDays[df_c_XDays['Sell_Signal'] == 1]['Short_SMA'], name='Sell_Signal',
                              mode='markers', marker=dict(color='red', size=15, symbol='triangle-down')))
    buy_signals = df_c_XDays[df_c_XDays['Buy_Signal'] == 1]
    for i in buy_signals.index:
        fig1.add_hline(y=buy_signals.loc[i, 'Short_SMA'], line_width=0.5, line_dash="dash", line_color="black")

    sell_signals = df_c_XDays[df_c_XDays['Sell_Signal'] == 1]
    for i in sell_signals.index:
      fig1.add_hline(y=sell_signals.loc[i, 'Short_SMA'], line_width=0.5, line_dash="dash", line_color="black")
    
    fig1.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig1, use_container_width=True)

roll_avr(nums,numl)

# Definition of stochastic oscillator chart 
st.subheader('Buy and sell signals generated by Stochastic oscillator')
st.subheader(f'for {comm} Prices from NYSE')

xyx = (list(df_c1.index)[-1] + 1)  
col6, col7, col8 = st.columns([0.4, 0.3, 0.3])
with col6:
    cut_p = st.slider('How long prices history you need?', 0, xyx, 200, key = "<commodities1>")
with col7:
    K_num = st.number_input('Enter the number of days for %K parameter',value=14, key = "<k14>")
with col8:
    D_num = st.number_input('Enter the number of days for %D parameter',value=14, key = "<d14>")

# Calculation of %K and %D for stochastic oscillator
def stoch_oscil(K_num,D_num):
    low_min  = df_c1['Low'].rolling(window = K_num).min()
    high_max = df_c1['High'].rolling(window = D_num).max()
    df_c1['%K'] = (100*(df_c1['Close'] - low_min) / (high_max - low_min)).fillna(0)
    df_c1['%D'] = df_c1['%K'].rolling(window = 3).mean()

    # Generating buy/sell signals
    df_c1['Buy_Signal'] = np.where((df_c1['%K'] < 20) & (df_c1['%K'] > df_c1['%D']), df_c1['Close'], np.nan)
    df_c1['Sell_Signal'] = np.where((df_c1['%K'] > 80) & (df_c1['%K'] < df_c1['%D']), df_c1['Close'], np.nan)

    df_cx_d = df_c1.iloc[xyx - cut_p:xyx]

    fig2 = px.line(df_cx_d,x='Date', y=['Close'],color_discrete_map={'Close':'dodgerblue'}, width=1000, height=500)
    #'Close':'#d62728',,'%K': '#f0f921','%D':'#0d0887'
    fig2.add_trace(go.Scatter(x=df_cx_d['Date'], y=df_cx_d['Buy_Signal'], mode='markers', name='Buy Signal', marker=dict(color='green', size=15, symbol='triangle-up')))
    fig2.add_trace(go.Scatter(x=df_cx_d['Date'], y=df_cx_d['Sell_Signal'], mode='markers', name='Sell Signal', marker=dict(color='red', size=15, symbol='triangle-down')))

    # Add horizontal lines for buy and sell signals
    buy_signals = df_cx_d.dropna(subset=['Buy_Signal'])
    for i in buy_signals.index:
       fig2.add_hline(y=buy_signals.loc[i, 'Buy_Signal'], line_width=0.5, line_dash="dash", line_color="black")

    sell_signals = df_cx_d.dropna(subset=['Sell_Signal'])
    for i in sell_signals.index:
     fig2.add_hline(y=sell_signals.loc[i, 'Sell_Signal'], line_width=0.5, line_dash="dash", line_color="black")

    fig2.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig2, use_container_width=True)

stoch_oscil(K_num,D_num)

# Arima model - trend forecast
def Arima_f(comm, size_a):
    data = np.asarray(df_c1['Close'][-300:]).reshape(-1, 1)
    p = 10
    d = 0
    q = 5
    n = size_a

    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit(method_kwargs={'maxiter': 3000})
    model_fit = model.fit(method_kwargs={'xtol': 1e-6})
    fore_arima = model_fit.forecast(steps=n)  
    
    arima_dates = [datetime.today() + timedelta(days=i) for i in range(0, size_a)]
    arima_pred_df = pd.DataFrame({'Date': arima_dates, 'Predicted Close': fore_arima})
    arima_pred_df['Date'] = arima_pred_df['Date'].dt.strftime('%Y-%m-%d')
    arima_df = pd.DataFrame(df_c1[['Date','High','Close']][-500:])
    arima_df['Date'] = arima_df['Date'].dt.strftime('%Y-%m-%d')
    arima_chart_df = pd.concat([arima_df, arima_pred_df], ignore_index=True)
    x_ar = (list(arima_chart_df.index)[-1] + 1)
    arima_chart_dff = arima_chart_df.iloc[x_ar - 30:x_ar]
    
    fig_ar = px.line(arima_chart_dff, x='Date', y=['High', 'Close', 'Predicted Close'], color_discrete_map={
                  'High': 'yellow', 'Close': 'black', 'Predicted Close': 'red'}, width=1000, height=500)
    fig_ar.add_vline(x = today,line_width=3, line_dash="dash", line_color="green")
    fig_ar.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(fig_ar, use_container_width=True)      
    
# Definition of turnover chart
def vol_chart(comm):
    volc = ['Crude_Oil','Gold','NASDAQ','SP_500','Copper','Silver','Natural Gas','Platinum','Rice Futures',
            'Soy Futures','KC HRW Wheat Futures']
    vol_com = {'Crude_Oil':1000,'Gold':1000,'NASDAQ':1000,'SP_500':1000,'Copper':25,'Silver':5000,'Natural Gas':1000,
               'Platinum':50,'Rice Futures':100,'Soy Futures':50,'KC HRW Wheat Futures':50}
    
    com = vol_com.get(comm,0)
    
    vol = df_c1['Volume'].iloc[-1] * df_c1['Close'].iloc[-1] / com
    vol1 = "{:.2f}".format(vol)
    
    st.sidebar.write('The volume for the last transaction is', vol1)
    
    x = (list(df_c1.index)[-1] + 1) 
    vol_xd = df_c1.iloc[x - 30:x]
    
    figv = px.line(vol_xd, x='Date', y=['Volume'], color_discrete_map={'Volume': '#9467bd'}, width=1000, height=500)
    figv.add_vline(x = today,line_width=3, line_dash="dash", line_color="green")
    figv.update_layout(xaxis=None, yaxis=None)
    st.plotly_chart(figv, use_container_width=True)

# LSTM model
def LSTM_f(comm,size_a):
    data = df_c1.filter(['Close'])
    dataset = data.values
    training_data_len = int(np.ceil( len(dataset) * .95 ))

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    train_data = scaled_data[0:int(training_data_len), : ]
    x_train=[]
    y_train = []
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    test_data = scaled_data[training_data_len - 60: , : ]

    x_test = []
    y_test =  dataset[training_data_len : , : ]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])

    x_test = np.array(x_test)

    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
    
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    
    x = (list(valid.index)[-1] + 1) 
    valid1 = valid.iloc[x - size_a:x]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train['Close'], name='Train',mode='lines'))
    fig.add_trace(go.Scatter(x=valid.index, y=valid['Close'], name='Validation',mode='lines'))
    fig.add_trace(go.Scatter(x=valid1.index, y=valid1['Predictions'], name='Predictions',mode='lines'))

    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

def model_type_f():
    model_t = st.radio('Select the model', ('ARIMA','LSTM'))
    return model_t

col9, col10 = st.columns([0.5, 0.5])
with col9:
    comm_options.insert(0, 'Select...')
    comm1 = st.selectbox('Select the commodity for ARIMA model forecast', comm_options)
    if comm1 != 'Select...':
        Arima_f(comm1,30)
with col10:
    size_a1 = st.number_input('Enter the number of days for the ARIMA forecast', value=30)
    if size_a1 > 0:
        Arima_f(comm1,size_a1)

col11, col12 = st.columns([0.5, 0.5])
with col11:
    comm_options.insert(0, 'Select...')
    comm2 = st.selectbox('Select the commodity for LSTM model forecast', comm_options)
    if comm2 != 'Select...':
        LSTM_f(comm2,30)
