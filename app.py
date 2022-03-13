import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import warnings
import streamlit.components.v1 as components
import time
import plotly.figure_factory as ff
# 

warnings.filterwarnings('ignore')
 
pd.options.display.float_format = '${:,.2f}'.format

today = datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'

eth_df = yf.download('ETH-USD',start_date, today)

# eth_df.tail()

# eth_df.info()

#eth_df.isnull().sum()

#eth_df.columns

eth_df.reset_index(inplace=True)
# eth_df.columns

df = eth_df[["Date", "Open"]]

new_names = {
    "Date": "ds", 
    "Open": "y",
}

df.rename(columns=new_names, inplace=True)

# df.tail()

# plot the open price

x = df["ds"]
y = df["y"]
# st.area_chart(y)
fig = go.Figure()

bb=fig.add_trace(go.Scatter(x=x, y=y))
# st.write("# History of Ethereum")

# Set title
fig.update_layout(
    title_text="Time series plot of Ethereum Open Price",
)
fig2 = go.Figure(data=[go.Candlestick(x=eth_df['Date'],
                open=eth_df['Open'],
                high=eth_df['High'],
                low=eth_df['Low'],
                close=eth_df['Close'])])
# st.write("# History of Ethereum with Candlesticks")

fig2.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    )
)

fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
        type="date",
    )
)

m = Prophet(
    seasonality_mode="multiplicative" 
)

m.fit(df)

future = m.make_future_dataframe(periods = 365)
future.tail()

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

next_day = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')


aa=plot_plotly(m, forecast)
#st.write("# Ethereum Coin Price Prediction")
# st.write(aa)



plot_components_plotly(m, forecast)
rad=st.sidebar.radio("Menu",["History of Ethereum","History of Ethereum with Candlesticks","Ethereum Price Prediction","Custom","About Us"])
if rad == "History of Ethereum":
  progress = st.progress(0)
  for i in range(0,2):
    time.sleep(0.2)
    progress.progress((i+1)*100-100)
  st.write("# History of Ethereum")
  st.write(bb)
  #
  # col1 = st.columns(1)
  from datetime import date
  today = date.today()
  # d1 = today.strftime("%Y-%m-%d")
  # yesterday = today - timedelta(days = 1)
  # d2 = yesterday.strftime("%Y-%m-%d")
  # for i in forecast.index:
  #     if(today == forecast['ds'][i]):
  #       col1.metric("Change", forecast['yhat'][i], forecast['yhat'][i]-forecast['yhat'][i-1])
  col1= st.columns(1)
  for i in forecast.index:
    if(today == forecast['ds'][i]):
      r1=forecast['yhat'][i]
      r2=r1/100
  
      st.write(r1," Dollars")
  #r1=int(r1)
  #r2=int(r2)
  #col1.metric("ETH",r1,r2)
  #st.write(forecast)


elif rad == "History of Ethereum with Candlesticks":
  progress = st.progress(0)
  for i in range(0,2):
    time.sleep(0.2)
    progress.progress((i+1)*100-100)
  st.write("# History of Ethereum with Candlesticks")
  st.write(fig2)
elif rad == "Ethereum Price Prediction":
  progress = st.progress(0)
  for i in range(0,2):
    time.sleep(0.2)
    progress.progress((i+1)*100-100)
  st.write("# Ethereum Price Prediction")
  st.write(aa)
  # hist_data = [m, forecast]
  # group_labels = ["History","Predicted"]
  # fig = ff.create_distplot(hist_data, group_labels, bin_size=[10, 25])
  # st.plotly_chart(fig, use_container_width=True)
elif rad == "Custom":
  progress = st.progress(0)
  for i in range(0,2):
    time.sleep(0.2)
    progress.progress((i+1)*100-100)
  
  
  
  strdate=st.date_input("Enter Date")
  #datetimeobj=datetime.strptime(strdate,"%y-%m-%d")
  #
  for i in forecast.index:
      if(strdate ==forecast['ds'][i]):
        st.write("## Ethereum Price on",forecast['ds'][i],"is",forecast['yhat'][i],"United States Dollar","and",forecast['yhat'][i]*76.1,"in Indian Rupee")
      # j=0
      # if st.checkbox("Show Center Information data"):
      #     for i in forecast.index:
      #       st.write(forecast['ds'][i])
      #       st.write(forecast['yhat'][i])
      #       strdate += timedelta(days=1)
      #       j+=1
      #       if(j>10):
      #         break
else:
  st.balloons()
  progress = st.progress(0)
  for i in range(0,2):
    time.sleep(0.2)
    progress.progress((i+1)*100-100)
  
  st.success("Thank you for knowing about us")
  
  st.write("### We are Computer Science Undergraduates from Saveetha Engineering College")
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: gray;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with Streamlit by <br>SYV</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
 
