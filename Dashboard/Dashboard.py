import json

import dash
from dash import no_update
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
from datetime import date
import waitress

import tools as tl
import data_offline as offline

data_offline = pd.read_csv("data.csv", header=[0,1], index_col=0)
data_offline.index = data_offline.index.astype(dtype='datetime64[ns]')

print('test pour voir si tout recommence')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

tickers = ['PKI','KR','VNO','NEM','BIIB','HFC','NFLX','COG','NLOK','FLIR','SEE','CLX','ABMD','AAL','DPZ','CPB','TSLA','GILD','LB','CTXS','MRO']

dash_app.layout = html.Div([

    html.H1('Our financial dash'),

    html.Div([
        
        html.H6('Select a date range : '),

        dcc.DatePickerRange(
            id='date_range',
            min_date_allowed=date(2010, 1, 1),
            max_date_allowed=date(2020, 12, 31),
            initial_visible_month=date(2010, 1, 1),
            start_date=date(2012,1,1),
            end_date=date(2015,12,31)
        )]
    ),

    html.Div([
        
        html.H6('Select a rolling window (in months) : '),

        dcc.Slider(
        id="window",
        min=1,
        max=24,
        step=1,
        value=12,
        marks={i: '{}'.format(i) for i in range(1, 24)}
        )]
    ),

    html.H6('Choose optimizer parameters : '),

    html.Div([
        html.Label('Diversification ratio'),
        dcc.RadioItems(
            id='div_ratio',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block'}
        ),

        html.Label('Max expected return'),
        dcc.RadioItems(
            id='max_exp_return',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block'}
        ),

        html.Label('Long'),
        dcc.RadioItems(
            id='long',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block'}
        ),

        html.Label('Short'),
        dcc.RadioItems(
            id='short',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block'}
        ),

        html.Label('Beta eq (not compatible with max turnover)'),
        dcc.RadioItems(
            id='beta_eq',
            options=[
                {'label': 'Yes', 'value': 1},
                {'label': 'No', 'value': 0},
            ],
            value=1,
            labelStyle={'display': 'inline-block'}
        ),

        html.Label('Risk parameter'),
        dcc.Input(
            id='risk',
            type='number',
            min=0,
            value=10
        ),

        html.Label('Method'),
        dcc.Input(
            id='method',
            type='text',
            value='SLSQP'
        ),

        html.Label('Transition cost'),
        dcc.Input(
            id='trans_cost1',
            type='number',
            value=0.01
        ),
        dcc.Input(
            id='trans_cost2',
            type='number',
            value=2
        ),

        html.Label('Max turnover'),
        dcc.Input(
            id='max_turn1',
            type='number',
            value=None
        ),
        dcc.Input(
            id='max_turn2',
            type='number',
            value=None
        ),

        html.Label('RSI 1'),
        dcc.Input(
            id='RSI1',
            type='number',
            value=None
        ),

        html.Label('RSI 2'),
        dcc.Input(
            id='RSI21',
            type='number',
            value=-10
        ),
        dcc.Input(
            id='RSI22',
            type='number',
            value=1
        ),

    ], style={'columnCount': 2}),
    

    dcc.Graph(
        id='returns_fig'
    ),
    dcc.Graph(id="weights_figure"),
    dcc.Graph(id="portfolio-returns_figure"),
    dcc.Graph(id="diversification-ratio-historic_figure"),

    html.Div(id='returns', style={'display': 'none'}),
    html.Div(id='weights', style={'display': 'none'}),
    html.Div(id='diversification_ratio_historic', style={'display': 'none'}),
    html.Div(id='portfolio_returns', style={'display': 'none'})
])

print("ready for callbacks")

@dash_app.callback(
    Output("returns", "children"),
    [Input("date_range", "start_date"),
     Input("date_range", "end_date")])
def build_returns(start_date, end_date):
    data = offline.get_data_offline(data_offline, tickers, start_date, end_date)
    returns = data["Returns"]
    returns_json = returns.to_json(date_format='iso', orient='split')
    print("data selection done")
    return returns_json

@dash_app.callback(
    Output('returns_fig', 'figure'),
    Input("returns", "children"))
def display_returns(returns_json):
    returns = pd.read_json(returns_json, orient='split')
    print("data collected")
    cum_returns = tl.cumulated_returns(returns)
    fig = px.line(cum_returns, x = cum_returns.index, y = list(cum_returns.columns.values), title = 'Returns for the tickers')
    print("data displayed")
    return fig

@dash_app.callback(
    [Output("weights", "children"),
     Output("diversification_ratio_historic", "children")],
    [Input("returns", "children"),
     Input("window", "value"),
     Input("div_ratio", "value"),
     Input("max_exp_return", "value"),
     Input("long", "value"),
     Input("short", "value"),
     Input("beta_eq", "value"),
     Input("risk", "value"),
     Input("method", "value"),
     Input("trans_cost1", "value"),
     Input("trans_cost2", "value"),
     Input("max_turn1", "value"),
     Input("max_turn2", "value"),
     Input("RSI1", "value"),
     Input("RSI21", "value"),
     Input("RSI22", "value")])
def build_weights_div_historic(returns_json, window, div_ratio, max_exp_return, long, short, beta_eq, risk, method, trans_cost1, trans_cost2, max_turn1, max_turn2, RSI1, RSI21, RSI22):
    returns = pd.read_json(returns_json, orient='split')
    print("start calculate weights")
    weights, diversification_ratio_historic = tl.weights_rolling_window(returns, nb_months=window, diversification=div_ratio, risk_parameter=risk, max_expected_return=max_exp_return, long=long, short=short, beta_eq=beta_eq, method=method, transition_cost=[trans_cost1,trans_cost2], maxturnover = [max_turn1,max_turn2], display=False, RSI1=RSI1, RSI2 = [RSI21, RSI22])
    print("weights calculated")
    return weights.to_json(date_format='iso', orient='split'), diversification_ratio_historic.to_json(date_format='iso', orient='split')

@dash_app.callback(
    Output("portfolio_returns", "children"),
    [Input("weights", "children"),
     Input("returns", "children"),
     Input("date_range", "start_date")])
def build_portfolio_returns(weights_json, returns_json, start_date):
    if weights_json is not None:
        weights = pd.read_json(weights_json, orient='split')
        returns = pd.read_json(returns_json, orient='split')

        dr_portfolio_return = tl.portfolio_return(weights, returns[weights.index[0]:])
        dr_portfolio_return.columns = ["Diversification Ratio Optimized Portfolio"]

        ew_portfolio_return = tl.equaly_weighted_portfolio_return(returns[start_date:])
        ew_portfolio_return.columns = ["Equaly Weighted Portfolio"]

        dr_portfolio_cumreturn = tl.cumulated_returns(dr_portfolio_return)

        ew_portfolio_cumreturns = tl.cumulated_returns(ew_portfolio_return[dr_portfolio_return.index[0]:])

        portfolio_returns = pd.concat([dr_portfolio_cumreturn,ew_portfolio_cumreturns], axis=1)

        print("cumreturns of the portfolio calculated")
        print("--------------------------------------")

        return portfolio_returns.to_json(date_format='iso', orient='split')
    
    else:
        return {}

@dash_app.callback(
    Output("weights_figure", "figure"),
    Input("weights", "children"))
def display_weights(weights_json):
    weights = pd.read_json(weights_json, orient='split')

    fig = px.line(weights, x=weights.index, y=list(weights.columns), title="Weights of the portfolio")
    return fig

@dash_app.callback(
    Output("portfolio-returns_figure", "figure"),
    [Input("portfolio_returns", "children")])
def display_portfolio_returns(portfolio_returns_json):
    portfolio_returns = pd.read_json(portfolio_returns_json, orient='split')

    fig = px.line(portfolio_returns, x=portfolio_returns.index, y=list(portfolio_returns.columns.values), title="Cumulated return of the portfolio with and without optimization")
    return fig

@dash_app.callback(
    Output("diversification-ratio-historic_figure", "figure"),
    [Input("diversification_ratio_historic", "children")])
def display_diversification_ratio_historic(diversification_ratio_historic_json):
    diversification_ratio_historic = pd.read_json(diversification_ratio_historic_json, orient='split')

    fig = px.line(diversification_ratio_historic, x=diversification_ratio_historic.index, y=list(diversification_ratio_historic.columns.values), title="Comparison of the diversification ratio with and without the optimization")
    return fig

if __name__ == '__main__':
    # dash_app.run_server(debug=True)
    #production setup: waitress
    waitress.serve(dash_app.server, 
    threads=8, # !!!CALL IT DASH APP OTHERWISE ERROR with waitress
    host='0.0.0.0',
    port='8050')