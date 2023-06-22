import streamlit as st
from datetime import date
import yfinance
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START="2003-01-01"

TODAY=date.today().strftime("%Y-%m-%d")

st.title("StocK prediction app")

stocks=("AAPL","MSFT", "NVDA")

selected_stocks=st.selectbox("select stock", stocks)

n_years=st.slider("Years of predictions:", 1,10)

period=n_years*365
@st.cache
def load_data(stock):
    
    data=yfinance.download(stock, START,TODAY)
    data.reset_index(inplace=True)
    return data

data=load_data(selected_stocks)

st.subheader("stock data")
st.write(data.tail())

#plot
fig1 =go.Figure()

fig1.add_trace(go.Scatter(x=data["Date"], y=data["Open"],name="stock_open"))
fig1.add_trace(go.Scatter(x=data["Date"], y=data["Close"],name="stock_close"))

fig1.layout.update(title_text="Time Series data", xaxis_rangeslider_visible=True)
st.plotly_chart(fig1)
    
#predict future prices

df_train=data[["Date", "Close"]]
df_train=df_train.rename(columns={"Date":"ds","Close":"y"})


m=Prophet()
m.fit(df_train)
future=m.make_future_dataframe(periods=period)
forecast=m.predict(future)

#plot forcast

st.subheader("Forecast data")

fig2=plot_plotly(m, forecast)

st.plotly_chart(fig2)
