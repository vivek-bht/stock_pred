import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
import pandas as pd
from datetime import datetime, timedelta
import Prediction as Pred
import numpy as np
import subprocess

checker = True
max_date = datetime.now() + timedelta(days=30)

def run_python_file(file_path):
    try:
        subprocess.run(["python", file_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to run {file_path}. Return code: {e.returncode}")
    except FileNotFoundError:
        print("Error: Python interpreter not found. Please make sure Python is installed.")

filtered_data = pd.read_csv('Nifty50.csv')
stock_data = {
    'Nifty 50': filtered_data['Nifty 50']
}

# Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Stock Market Data Prediction Dashboard"),
    html.Label("Select Stock:"),
    dcc.Dropdown(
        id='stock-dropdown',
        options=[{'label': key, 'value': key} for key in stock_data.keys()],
        value='Nifty 50'
    ),
    html.Label("Select Date:"),
    dcc.DatePickerSingle(
        id='date-picker',
        min_date_allowed=datetime(2007, 9, 17),
        max_date_allowed=max_date,
        initial_visible_month=datetime(2024, 3, 3),
        date=datetime(2024, 3, 3),
    ),
    dcc.Graph(id='prediction-graph'),
    html.Div(id='predicted-price')
])

# Callback to update graph and predicted price
@app.callback(
    [Output('prediction-graph', 'figure'),
     Output('predicted-price', 'children')],
    [Input('stock-dropdown', 'value'),
     Input('date-picker', 'date')]
)
def update_graph(stock, selected_date):
    global filtered_data
    # Filter data based on selected stock
    df = pd.DataFrame(stock_data)
    df = df[[stock]]
    
    # Convert selected date to datetime
    selected_date = datetime.strptime(selected_date.split('T')[0], '%Y-%m-%d')
    initialDate = datetime.strptime(filtered_data['Date'][0], '%Y-%m-%d') + timedelta(days=1)
    
    # Get predicted price for selected date
    predicted_price_df = Pred.Get_values(selected_date)
    predicted_price = np.array(predicted_price_df).flatten()
    dateRangeINI = pd.date_range(initialDate, selected_date)
    dateRangeINI_formatted = [date.strftime('%Y-%m-%d') for date in dateRangeINI]
    dateRange = np.array(dateRangeINI_formatted)
    
    trace_actual = go.Scatter(x=filtered_data['Date'], y=df[stock], mode='lines', name='Actual')
    trace_predicted = go.Scatter(x=dateRange, y=predicted_price, mode='lines', name='Predicted')
    
    layout = go.Layout(title=f'{stock} Data', xaxis=dict(title='Date'), yaxis=dict(title='Price'))
    fig = go.Figure(data=[trace_actual, trace_predicted], layout=layout)
    if len(predicted_price) == 1:
        price_text = f"The predicted price for {stock} on {selected_date.strftime('%d-%m-%Y')} is Rs. {predicted_price[0]}."
        return fig, html.P(price_text, style={'textAlign': 'center'})
    else:
        difference = predicted_price[-1] - predicted_price[0]
        
        # Determine button and text based on difference
        button_color = 'red' if difference < 0 else 'green'
        button_text = 'Short the stock' if difference < 0 else 'Buy the stock'
        market_text = f"Market is expected to {'fall' if difference < 0 else 'rise'} by {int(abs(difference))} points."
        
        # Create button HTML
        button_html = html.Button(button_text, style={'backgroundColor': button_color, 'color': 'black', 'fontSize': '30px'})
        market_text_html = html.P(market_text, style={'textAlign': 'center'})
        center_div = html.Div([button_html, html.Br(), market_text_html], style={'textAlign': 'center'})
        
        return fig, center_div

# Run the app
if __name__ == '__main__':
    if checker:
        file_path = r"C:\Users\acer\Desktop\codes\python\data.py"
        run_python_file(file_path)
        checker = False
    app.run_server(debug=True)
