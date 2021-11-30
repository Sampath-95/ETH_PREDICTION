import yfinance as yf
from datetime import datetime
from datetime import timedelta
import plotly.graph_objects as go
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
# # pd.options.display.float_format = '${:,.2f}'.format
today = datetime.today().strftime('%Y-%m-%d')
start_date = '2016-01-01'
# 
eth_df = yf.download('ETH-USD',start_date, today)
 
eth_df.tail()
# 
eth_df.info()
# 
eth_df.isnull().sum()
# 
eth_df.columns
# 
eth_df.reset_index(inplace=True)
eth_df.columns
# 
df = eth_df[["Date", "Open"]]
# 
new_names = {
    "Date": "ds", 
    "Open": "y",
}
# 
df.rename(columns=new_names, inplace=True)
# 
df.tail()
# 
#plot the open price
# 
x = df["ds"]
y = df["y"]
# 
fig = go.Figure()
# 
fig.add_trace(go.Scatter(x=x, y=y))
# 
# # Set title
fig.update_layout(
    title_text="Time series plot of Ethereum Open Price",
)
# 
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
 
forecast[forecast['ds'] == next_day]['yhat'].item()
 
plot_plotly(m, forecast)
 
plot_components_plotly(m, forecast)
st.line_chart(m,forecast)
