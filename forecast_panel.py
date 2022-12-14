# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 12:50:27 2022

@author: serhiy
"""

import streamlit as st
import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from prophet.diagnostics import performance_metrics

import plotly.express as px
import plotly.graph_objects as go



from sklearn.metrics import mean_absolute_percentage_error
import itertools

import streamlit as st

st.set_page_config(layout="wide")
#hide_menu_style = """
#        <style>
#        #MainMenu {visibility: hidden;}
#        </style>
#        """
#st.markdown(hide_menu_style, unsafe_allow_html=True)


@st.cache()
def load_iqvia(filepath):
    mdf = pd.read_csv(filepath)
    mdf['Date'] = mdf['Date'].str.replace('_1', '')
    mdf = mdf.rename(columns = {' Vol': 'Vol'})
    mdf['date'] = pd.to_datetime(mdf["Date"], format='%Y%m%d')
    return mdf

@st.cache(allow_output_mutation=True)
def pie_plot(mdf):
    fig = px.pie(mdf,
                 values = mdf['Product Class'].value_counts().values,
                 names = mdf['Product Class'].value_counts().index,
                 height= 500)
    #fig.update_layout(template='solar', paper_bgcolor='#002b36')
    return fig
@st.cache(allow_output_mutation=True)
def line_plot(mdf):
    fig = px.line(mdf.groupby(by=['Product Class', 'date']).sum().reset_index(),
                  x = 'date',
                  y = 'Vol',
                  color = 'Product Class',
                  )
    #fig.update_layout(template='solar', paper_bgcolor='#002b36')
    return fig

def forecast_plot(forecast, mape):
    _fcst = forecast[['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper', 'yhat']].merge(part_df[1:-1], on = 'ds', how = 'outer')
    fig = go.Figure([
    go.Scatter(
        name='Historical data',
        x=_fcst['ds'],
        y=_fcst['y'],
        mode='lines',
        line=dict(color='rgb(31,119,180)'),
    ),
    go.Scatter(
        name='Model',
        x=_fcst['ds'],
        y=_fcst['yhat'],
        mode='lines',
        line=dict(color='rgb(214,39,40)'),
    ),
    go.Scatter(
        name='Model upper error',
        x=_fcst['ds'],
        y=_fcst['yhat_upper'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Model lower error',
        x=_fcst['ds'],
        y=_fcst['yhat_lower'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(214,39,40, 0.1)',
        fill='tonexty',
        showlegend=False
    ),
    go.Scatter(
        name='Trend',
        x=_fcst['ds'],
        y=_fcst['trend'],
        mode='lines',
        line=dict(color='rgb(255,127,14)'),
    ),
    go.Scatter(
        name='Trend upper error',
        x=_fcst['ds'],
        y=_fcst['trend_upper'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Trend lower error',
        x=_fcst['ds'],
        y=_fcst['trend_lower'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(255,127,14, 0.1)',
        fill='tonexty',
        showlegend=False
    )
    ])
    fig.update_layout(
        yaxis_title=predict_type,
        title='Model prediction / Mean Average Percentage Error: ' + str(mape) + '%',
        hovermode="x"
    )
    return fig

mdf = load_iqvia("iqvia_data/New Analysis (1).csv")
pie_chart = pie_plot(mdf)
line_chart = line_plot(mdf)

st.title('IQVIA Forecast panel')

c1, c2 = st.columns(((1, 2)), gap = 'small')
with c1:
    st.header("Volume of available data chart")
    st.plotly_chart(pie_chart, use_container_width=True)
with c2:
    st.header("Time series by each category chart")
    st.plotly_chart(line_chart, use_container_width=True)

but1, but2 = st.columns(2)
with but1:    
    product_class = st.selectbox(label = 'Choose product category to predict',
                                 options = mdf['Product Class'].unique())
with but2:
    predict_type = st.selectbox(label = 'Choose value to predict',
                                options = ['Units', 'Vol'])

gr_part_data = mdf.loc[mdf['Product Class'] == product_class]
gr_part_data = gr_part_data.groupby(by='date').agg({'Vol': 'sum',
                                                    'Units': 'sum',
                                                    'UPC': 'count'}).reset_index()
part_df = gr_part_data.rename(columns = {'date':'ds', predict_type: 'y'})[['ds', 'y']]
part_df = part_df.loc[part_df['ds'] > '2020-05-10']
#part_df = part_df.loc[~((part_df['ds'].dt.month == 1) & (part_df['ds'].dt.day.isin([1,2])))]

part_df_ = part_df.copy()
part_df_.index = part_df['ds']
part_df_ = part_df_.drop(columns = 'ds')
part_df = part_df.resample('2W', on='ds').sum().reset_index()


split_int = len(part_df) - 6
df_train = part_df[1:split_int]
df_val = part_df[split_int:-1]

prophet_basic = Prophet(yearly_seasonality = True,
                        seasonality_prior_scale = 1,
                        changepoint_prior_scale = 0.4,
                        holidays_prior_scale = 0.15,
                       seasonality_mode = 'multiplicative')
prophet_basic.add_country_holidays('CA')
prophet_basic.fit(part_df[1:-1])

future= prophet_basic.make_future_dataframe(periods=12, freq = '2W')
forecast=prophet_basic.predict(future)

mape = mean_absolute_percentage_error(part_df.y, forecast.yhat[:len(part_df.y)]) * 100
st.plotly_chart(forecast_plot(forecast, mape) , use_container_width= True)

with st.expander(label = 'Prophet components'):
    st.pyplot(prophet_basic.plot_components(forecast))
#plt.savefig('fcst_iqvia.png', dpi = 150)

