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



df_ford = pd.read_csv("./Ford.csv")

df_ford["Date"]=pd.to_datetime(df_ford.Date,format="%Y-%m-%d")
df_ford.index=df_ford['Date']


fordData=df_ford.sort_index(ascending=True,axis=0)
ford_data=pd.DataFrame(index=range(0,len(df_ford)),columns=['Date','Close'])

for i in range(0,len(fordData)):
    ford_data["Date"][i]=fordData['Date'][i]
    ford_data["Close"][i]=fordData["Close"][i]

ford_data.index=ford_data.Date
ford_data.drop("Date",axis=1,inplace=True)

dataset=ford_data.values

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

model=load_model("Ford_lstm_model.h5")

inputs=ford_data[len(ford_data)-len(valid)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)

X_test=[]
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=ford_data[:987]
valid=ford_data[987:]
valid['FordPredictions']=closing_price
##--------------------------------------------------------------------------
df_Amazon = pd.read_csv("./AMZN.csv")

df_Amazon["Date"]=pd.to_datetime(df_Amazon.Date,format="%Y-%m-%d")
df_Amazon.index=df_Amazon['Date']


amazonData=df_Amazon.sort_index(ascending=True,axis=0)
amazon_data=pd.DataFrame(index=range(0,len(df_Amazon)),columns=['Date','Close'])

for i in range(0,len(amazonData)):
    amazon_data["Date"][i]=amazonData['Date'][i]
    amazon_data["Close"][i]=amazonData["Close"][i]

amazon_data.index=amazon_data.Date
amazon_data.drop("Date",axis=1,inplace=True)

dataset1=amazon_data.values

train1=dataset1[0:987,:]
valid1=dataset1[987:,:]

scaler1=MinMaxScaler(feature_range=(0,1))
scaled_data1=scaler1.fit_transform(dataset1)

x_train1,y_train1=[],[]

for i in range(60,len(train)):
    x_train1.append(scaled_data1[i-60:i,0])
    y_train1.append(scaled_data1[i,0])
    
x_train1,y_train1=np.array(x_train1),np.array(y_train1)

x_train1=np.reshape(x_train1,(x_train1.shape[0],x_train1.shape[1],1))

model1=load_model("Amazon_lstm_model.h5")

inputs1=amazon_data[len(amazon_data)-len(valid1)-60:].values
inputs1=inputs1.reshape(-1,1)
inputs1=scaler1.transform(inputs1)

X_test1=[]
for i in range(60,inputs1.shape[0]):
    X_test1.append(inputs1[i-60:i,0])
X_test1=np.array(X_test1)

X_test1=np.reshape(X_test1,(X_test1.shape[0],X_test1.shape[1],1))
closing_price1=model.predict(X_test1)
closing_price1=scaler.inverse_transform(closing_price1)

train1=amazon_data[:987]
valid1=amazon_data[987:]
valid1['AmazonPredictions']=closing_price1


df= pd.read_csv("./stock_data.csv")

app.layout = html.Div([
   
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
   
    dcc.Tabs(id="tabs", children=[
       
        dcc.Tab(label='Ford Stock Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Ford Data",
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=df_ford["Close"],
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
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Ford Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=valid.index,
								y=valid["FordPredictions"],
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

        dcc.Tab(label='Amazon Stock Data',children=[
			html.Div([
				html.H2("Actual closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Amazon Data",
					figure={
						"data":[
							go.Scatter(
								x=valid1.index,
								y=df_Amazon["Close"],
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
				html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
				dcc.Graph(
					id="Amazon Predicted Data",
					figure={
						"data":[
							go.Scatter(
								x=valid1.index,
								y=valid1["AmazonPredictions"],
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
        dcc.Tab(label='Facebook Stock Data', children=[
            html.Div([
                html.H1("Stocks High vs Lows", 
                        style={'textAlign': 'center'}),
              
                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'}, 
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Stocks Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'Facebook', 'value': 'FB'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['FB'],
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
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
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
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","FB": "Facebook","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
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