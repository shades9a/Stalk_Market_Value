import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense

app = dash.Dash()
server = app.server

scaler=MinMaxScaler(feature_range=(0,1))

# --------- GM Stock ------------

df_gm = pd.read_csv("data/GM.csv")

df_gm["Date"]=pd.to_datetime(df_gm.Date,format="%Y-%m-%d")
df_gm.index=df_gm['Date']


data=df_gm.sort_index(ascending=True,axis=0)
new_data=pd.DataFrame(index=range(0,len(df_gm)),columns=['Date','Close'])

for i in range(0,len(data)):
    new_data["Date"][i]=data['Date'][i]
    new_data["Close"][i]=data["Close"][i]

new_data.index=new_data.Date
new_data.drop("Date",axis=1,inplace=True)

dataset=new_data.values

train=dataset[0:987,:]
valid=dataset[987:,:]

scaler=MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)

x_train,y_train=[],[]

for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model=load_model("jupyter_notebooks/gm_lstm_model.h5")

inputs=new_data[len(new_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_data[:987]
valid=new_data[987:]
valid['Predictions']=closing_price

# ------- MSFT ------

df_msft = pd.read_csv("data/MSFT.csv")

df_msft["Date"]=pd.to_datetime(df_msft.Date,format="%Y-%m-%d")
df_msft.index=df_msft['Date']


msft_data=df_msft.sort_index(ascending=True,axis=0)
new_msft_data=pd.DataFrame(index=range(0,len(df_msft)),columns=['Date','Close'])

for i in range(0,len(msft_data)):
    new_msft_data["Date"][i]=msft_data['Date'][i]
    new_msft_data["Close"][i]=msft_data["Close"][i]

new_msft_data.index=new_msft_data.Date
new_msft_data.drop("Date",axis=1,inplace=True)

msft_dataset=new_msft_data.values

msft_train=msft_dataset[0:987,:]
msft_valid=msft_dataset[987:,:]

msft_scaler=MinMaxScaler(feature_range=(0,1))
scaled_msft_data=msft_scaler.fit_transform(msft_dataset)

x_msft_train,y_msft_train=[],[]

for i in range(60,len(train)):
    x_msft_train.append(scaled_msft_data[i-60:i,0])
    y_msft_train.append(scaled_msft_data[i,0])
    
x_msft_train,y_msft_train=np.array(x_msft_train),np.array(y_msft_train)

x_msft_train=np.reshape(x_msft_train,(x_msft_train.shape[0],x_msft_train.shape[1],1))

msft_model=load_model("jupyter_notebooks/msft_lstm_model.h5")

msft_inputs=new_msft_data[len(new_msft_data)-len(valid)-60:].values
msft_inputs=msft_inputs.reshape(-1,1)
msft_inputs=msft_scaler.transform(msft_inputs)

X_msft_test=[]
for i in range(60,msft_inputs.shape[0]):
    X_msft_test.append(msft_inputs[i-60:i,0])
X_msft_test=np.array(X_msft_test)

X_msft_test=np.reshape(X_msft_test,(X_msft_test.shape[0],X_msft_test.shape[1],1))
closing_price_msft=msft_model.predict(X_msft_test)
closing_price_msft=msft_scaler.inverse_transform(closing_price_msft)

msft_train=new_msft_data[:987]
msft_valid=new_msft_data[987:]
msft_valid['Predictions']=closing_price_msft

# -------- Tesla ----------

df_tsla = pd.read_csv("data/TSLA.csv")

df_tsla["Date"]=pd.to_datetime(df_tsla.Date,format="%Y-%m-%d")
df_tsla.index=df_tsla['Date']


tsla_data=df_tsla.sort_index(ascending=True,axis=0)
new_tsla_data=pd.DataFrame(index=range(0,len(df_tsla)),columns=['Date','Close'])

for i in range(0,len(tsla_data)):
    new_tsla_data["Date"][i]=tsla_data['Date'][i]
    new_tsla_data["Close"][i]=tsla_data["Close"][i]

new_tsla_data.index=new_tsla_data.Date
new_tsla_data.drop("Date",axis=1,inplace=True)

tsla_dataset=new_tsla_data.values

tsla_train=tsla_dataset[0:987,:]
tsla_valid=tsla_dataset[987:,:]

tsla_scaler=MinMaxScaler(feature_range=(0,1))
scaled_tsla_data=tsla_scaler.fit_transform(tsla_dataset)

x_tsla_train,y_tsla_train=[],[]

for i in range(60,len(train)):
    x_tsla_train.append(scaled_tsla_data[i-60:i,0])
    y_tsla_train.append(scaled_tsla_data[i,0])
    
x_tsla_train,y_tsla_train=np.array(x_tsla_train),np.array(y_tsla_train)

x_tsla_train=np.reshape(x_tsla_train,(x_tsla_train.shape[0],x_tsla_train.shape[1],1))

tsla_model=load_model("jupyter_notebooks/tsla_lstm_model.h5")

tsla_inputs=new_tsla_data[len(new_tsla_data)-len(valid)-60:].values
tsla_inputs=tsla_inputs.reshape(-1,1)
tsla_inputs=tsla_scaler.transform(tsla_inputs)

X_tsla_test=[]
for i in range(60,tsla_inputs.shape[0]):
    X_tsla_test.append(tsla_inputs[i-60:i,0])
X_tsla_test=np.array(X_tsla_test)

X_tsla_test=np.reshape(X_tsla_test,(X_tsla_test.shape[0],X_tsla_test.shape[1],1))
closing_price_tsla=tsla_model.predict(X_tsla_test)
closing_price_tsla=tsla_scaler.inverse_transform(closing_price_tsla)

tsla_train=new_tsla_data[:987]
tsla_valid=new_tsla_data[987:]
tsla_valid['Predictions']=closing_price_tsla

# -------- Twitter --------

df_twtr = pd.read_csv("data/TWTR.csv")

df_twtr["Date"]=pd.to_datetime(df_twtr.Date,format="%Y-%m-%d")
df_twtr.index=df_twtr['Date']


twtr_data=df_twtr.sort_index(ascending=True,axis=0)
new_twtr_data=pd.DataFrame(index=range(0,len(df_twtr)),columns=['Date','Close'])

for i in range(0,len(twtr_data)):
    new_twtr_data["Date"][i]=twtr_data['Date'][i]
    new_twtr_data["Close"][i]=twtr_data["Close"][i]

new_twtr_data.index=new_twtr_data.Date
new_twtr_data.drop("Date",axis=1,inplace=True)

twtr_dataset=new_twtr_data.values

twtr_train=twtr_dataset[0:987,:]
twtr_valid=twtr_dataset[987:,:]

twtr_scaler=MinMaxScaler(feature_range=(0,1))
scaled_twtr_data=twtr_scaler.fit_transform(twtr_dataset)

x_twtr_train,y_twtr_train=[],[]

for i in range(60,len(train)):
    x_twtr_train.append(scaled_twtr_data[i-60:i,0])
    y_twtr_train.append(scaled_twtr_data[i,0])
    
x_twtr_train,y_twtr_train=np.array(x_twtr_train),np.array(y_twtr_train)

x_twtr_train=np.reshape(x_twtr_train,(x_twtr_train.shape[0],x_twtr_train.shape[1],1))

twtr_model=load_model("jupyter_notebooks/twtr_lstm_model.h5")

twtr_inputs=new_twtr_data[len(new_twtr_data)-len(valid)-60:].values
twtr_inputs=twtr_inputs.reshape(-1,1)
twtr_inputs=twtr_scaler.transform(twtr_inputs)

X_twtr_test=[]
for i in range(60,twtr_inputs.shape[0]):
    X_twtr_test.append(twtr_inputs[i-60:i,0])
X_twtr_test=np.array(X_twtr_test)

X_twtr_test=np.reshape(X_twtr_test,(X_twtr_test.shape[0],X_twtr_test.shape[1],1))
closing_price_twtr=twtr_model.predict(X_twtr_test)
closing_price_twtr=twtr_scaler.inverse_transform(closing_price_twtr)

twtr_train=new_twtr_data[:987]
twtr_valid=new_twtr_data[987:]
twtr_valid['Predictions']=closing_price_twtr

# ----- All Stocks ------

df= pd.read_csv("./stock_data.csv")

# ----- App HTML ---------
app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='General Motors Stock Data',children=[
			html.Div([
				html.H2("GM Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="GM Data",
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=df_gm["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM GM Predicted Closing Price",style={"textAlign": "center"}),
				dcc.Graph(
					id="GM Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=valid["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])
        ]),

        dcc.Tab(label='Microsoft Stock Data',children=[
			html.Div([
				html.H2("MSFT Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="MSFT Data",
					figure={
						"data":[
							go.Scatter(
								x=msft_valid.index,
								y=df_msft["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM MSFT Predicted Closing Price",style={"textAlign": "center"}),
				dcc.Graph(
					id="MSFT Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=msft_valid.index,
								y=msft_valid["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])
        ]),

        dcc.Tab(label='Tesla Stock Data',children=[
			html.Div([
				html.H2("TSLA Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="TSLA Data",
					figure={
						"data":[
							go.Scatter(
								x=tsla_valid.index,
								y=df_tsla["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM TSLA Predicted Closing Price",style={"textAlign": "center"}),
				dcc.Graph(
					id="TSLA Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=tsla_valid.index,
								y=tsla_valid["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])
        ]),

		dcc.Tab(label='Twitter Stock Data',children=[
			html.Div([
				html.H2("TWTR Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="TWTR Data",
					figure={
						"data":[
							go.Scatter(
								x=twtr_valid.index,
								y=df_twtr["Close"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				),
				html.H2("LSTM TWTR Predicted Closing Price",style={"textAlign": "center"}),
				dcc.Graph(
					id="TWTR Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=twtr_valid.index,
								y=twtr_valid["Predictions"],
								mode='markers'
							)

						],
						"layout":go.Layout(
							title='scatter plot',
							xaxis={'title':'Date'},
							yaxis={'title':'Closing Rate'}
						)
					}

				)				
			])
        ]),

        dcc.Tab(label='Actual Stock Data', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'General Motors', 'value': 'GM'},
                                      {'label': 'Microsoft','value': 'MSFT'}, 
                                      {'label': 'Tesla', 'value': 'TSLA'}, 
                                      {'label': 'Twitter','value': 'TWTR'}], 
                             multi=True,value=['MSFT'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'General Motors', 'value': 'GM'},
                                      {'label': 'Microsoft','value': 'MSFT'}, 
                                      {'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Twitter','value': 'TWTR'}], 
                             multi=True,value=['MSFT'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])


    ])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"GM": "General Motors","MSFT": "Microsoft","TSLA": "Tesla","TWTR": "Twitter",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["ticker"] == stock]["Date"],
                     y=df[df["ticker"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["ticker"] == stock]["Date"],
                     y=df[df["ticker"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"GM": "General Motors","MSFT": "Microsoft","TSLA": "Tesla","TWTR": "Twitter",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["ticker"] == stock]["Date"],
                     y=df[df["ticker"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure



if __name__=='__main__':
	app.run_server(debug=True)