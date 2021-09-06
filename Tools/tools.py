# Package created by : Yoann Launay*, Antoine Bigeard*, Etienne Cornu*, Maxime Seince*, Tina Aubrun*, Murat Can Acun*
# *CentraleSupélec, Paris-Saclay University.
# March-April 2021.

import yfinance as yf
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
import scipy.optimize as opt
import random as rd
import data_offline as offline

data_offline_market = pd.read_csv("data_market.csv", header=[0,1], index_col=0)
data_offline_market.index = data_offline_market.index.astype(dtype='datetime64[ns]')

def SP500():
    """Return all the tickers of the S&P50 in a list"""
    return ['MMM', 'ABT', 'ABBV', 'ABMD', 'ACN', 'ATVI', 'ADBE', 'AMD', 'AAP', 'AES', 'AFL', 'A', 'APD', 'AKAM', 'ALK',
            'ALB', 'ARE', 'ALXN', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AAL',
            'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'ANTM', 'AON', 'AOS',
            'APA', 'AAPL', 'AMAT', 'APTV', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY',
            'BKR', 'BLL', 'BAC', 'BK', 'BAX', 'BDX', 'BRK.B', 'BBY', 'BIO', 'BIIB', 'BLK', 'BA', 'BKNG', 'BWA', 'BXP',
            'BSX', 'BMY', 'AVGO', 'BR', 'BF.B', 'CHRW', 'COG', 'CDNS', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR',
            'CTLT', 'CAT', 'CBOE', 'CBRE', 'CDW', 'CE', 'CNC', 'CNP', 'CERN', 'CF', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB',
            'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CTXS', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA',
            'CMA', 'CAG', 'COP', 'ED', 'STZ', 'COO', 'CPRT', 'GLW', 'CTVA', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI',
            'DHR', 'DRI', 'DVA', 'DE', 'DAL', 'XRAY', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DISCA', 'DISCK', 'DISH',
            'DG', 'DLTR', 'D', 'DPZ', 'DOV', 'DOW', 'DTE', 'DUK', 'DRE', 'DD', 'DXC', 'EMN', 'ETN', 'EBAY', 'ECL',
            'EIX', 'EW', 'EA', 'EMR', 'ENPH', 'ETR', 'EOG', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ETSY', 'EVRG', 'ES',
            'RE', 'EXC', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FB', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FE', 'FRC',
            'FISV', 'FLT', 'FLIR', 'FLS', 'FMC', 'F', 'FTNT', 'FTV', 'FBHS', 'FOXA', 'FOX', 'BEN', 'FCX', 'GPS', 'GRMN',
            'IT', 'GD', 'GE', 'GIS', 'GM', 'GPC', 'GILD', 'GL', 'GPN', 'GS', 'GWW', 'HAL', 'HBI', 'HIG', 'HAS', 'HCA',
            'PEAK', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HFC', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUM',
            'HBAN', 'HII', 'IEX', 'IDXX', 'INFO', 'ITW', 'ILMN', 'INCY', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF',
            'INTU', 'ISRG', 'IVZ', 'IPGP', 'IQV', 'IRM', 'JKHY', 'J', 'JBHT', 'SJM', 'JNJ', 'JCI', 'JPM', 'JNPR', 'KSU',
            'K', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KLAC', 'KHC', 'KR', 'LB', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LEG',
            'LDOS', 'LEN', 'LLY', 'LNC', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LUMN', 'LYB', 'MTB', 'MRO', 'MPC',
            'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MKC', 'MXIM', 'MCD', 'MCK', 'MDT', 'MRK', 'MET', 'MTD', 'MGM',
            'MCHP', 'MU', 'MSFT', 'MAA', 'MHK', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI',
            'NDAQ', 'NTAP', 'NFLX', 'NWL', 'NEM', 'NWSA', 'NWS', 'NEE', 'NLSN', 'NKE', 'NI', 'NSC', 'NTRS', 'NOC',
            'NLOK', 'NCLH', 'NOV', 'NRG', 'NUE', 'NVDA', 'NVR', 'ORLY', 'OXY', 'ODFL', 'OMC', 'OKE', 'ORCL', 'OTIS',
            'PCAR', 'PKG', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PBCT', 'PEP', 'PKI', 'PRGO', 'PFE', 'PM', 'PSX', 'PNW',
            'PXD', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO',
            'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RHI', 'ROK', 'ROL',
            'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SEE', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SLG',
            'SNA', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STE', 'SYK', 'SIVB', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW',
            'TTWO', 'TPR', 'TGT', 'TEL', 'TDY', 'TFX', 'TER', 'TSLA', 'TXN', 'TXT', 'TMO', 'TJX', 'TSCO', 'TT', 'TDG',
            'TRV', 'TRMB', 'TFC', 'TWTR', 'TYL', 'TSN', 'UDR', 'ULTA', 'USB', 'UAA', 'UA', 'UNP', 'UAL', 'UNH', 'UPS',
            'URI', 'UHS', 'UNM', 'VLO', 'VAR', 'VTR', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VFC', 'VIAC', 'VTRS', 'V', 'VNT',
            'VNO', 'VMC', 'WRB', 'WAB', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WU',
            'WRK', 'WY', 'WHR', 'WMB', 'WLTW', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZION', 'ZTS']





def get_returns(assets=["TSLA", "AMZN"], price="Close", start_date="2017-01-01", end_date="2017-04-30"):
    ''' Function to get wanted returns and put them in the right shape in order to optimize'''
    
    returns = []

    for asset in assets:
        yf.pdr_override()

        # download dataframe using pandas_datareader
        opt = pdr.get_data_yahoo(asset, start=start_date, end=end_date)

        opt_prices = opt[[price]]
        
        # compute returns
        opt_returns = opt_prices.pct_change()

        returns.append(opt_returns)

    nb_returns = len(returns)

    returns = pd.concat(returns, axis=1).T
    
    # delete non-available first value of returns
    del returns[returns.columns[0].strftime("%Y-%m-%d")]

    return returns, nb_returns


def get_data(tickers, start="2017-01-01", end="2018-12-31"):
    '''tickers: list of strings represnting the assets selected, start_date: start date of prices selected, end_date: end date of prices selected
    Select the prices and returns of selected assets for the time period selected
    Return a DataFrame containing the prices (data["Prices"]) and the returns (data["Returns"]) for selected assets'''
    prices = []
    for ticker in tickers:
        price = pdr.get_data_yahoo(ticker, start, end)[
            "Close"]  # Get the prices for all ticker in tickers, return a DataFrame
        prices.append(price)
    prices = pd.concat(prices, axis=1)  # Concatenate the prices in a sigle DataFrame

    returns = prices.pct_change()  # Compute the returns for each assets

    prices.columns = pd.MultiIndex.from_tuples([("Prices", ticker) for ticker in tickers])
    returns.columns = pd.MultiIndex.from_tuples([("Returns", ticker) for ticker in tickers])

    data = pd.concat([prices, returns], axis=1)  # Concatanate prices and returns in a single DataFrame

    return data





def suggested_assets(r, indexes, start_date="2020-03-01", end_date="2021-03-01"):
    ''' Low correlated assets research, returns a suggested list of r assests which are chosen with relative low correlation among those in indexes'''
    Returns, nb_return = get_returns(assets=indexes, price="Close", start_date=start_date, end_date=end_date)
    C = np.corrcoef(Returns)
    selectedIndex = [rd.randint(0, len(indexes) - 1)] # 1st asset is randomly selected
    ind = 0
    # select r most uncorrelated in a L1 sense :
    for i in range(r): 
        temp = np.argmin(C, axis=0) # gives the lowest correlated asset for each asset
        minscore = np.inf
        minscorer = None
        for j in range(len(C)):
            if j not in selectedIndex:
                tempscore = 0
                for el in selectedIndex:
                    tempscore += abs(C[j, el])
                if tempscore < minscore:
                    minscore = tempscore
                    minscorer = j
        selectedIndex.append(minscorer)
    finalindexes = [indexes[i] for i in selectedIndex]
    return (finalindexes)


### FINANCIAL TOOLS :


def diversification_ratio(w, V):
    '''returns the diversification ratio given weights w and covariance matrix V'''
    w = np.array(w)
    V = np.array(V)
    """
    Compute the diversication ratio of the portfolio from the weights w of all assets and the covariance matrix V
    """
    w_vol = np.dot(w.T, np.sqrt(np.diag(V)))

    port_vol = np.sqrt(abs(np.dot(np.dot(w.T, V), w))) #abs stops non-positive numbers errors

    diversification_ratio = w_vol / port_vol

    return -diversification_ratio


'''
Function to optimize the portfolio
'''


def Betas2(returns, market="^GSPC"):
    '''returns betas between two dates from a list of index, all included in a given market index'''
    startdate = returns.T.columns[0].strftime("%Y-%m-%d")
    enddate = returns.T.columns[-1].strftime("%Y-%m-%d")
    # add the market to the data and clean NaN values
    returnM = get_data([market], start=startdate, end=enddate)['Returns']
    Returns = pd.concat([returns, returnM], axis=1)
    Returns = Returns.fillna(0)
    
    #compute beta of each asset as normalized covariances with the market
    V = Returns.cov()
    V = np.array(V)
    betas = V[-1, :-1]
    betas = betas / V[-1, -1]
    
    return (betas)

def Betas(returns, market="^GSPC"):
    '''returns betas between two dates from a list of index, all included in a given market index'''
    startdate = returns.T.columns[0].strftime("%Y-%m-%d")
    enddate = returns.T.columns[-1].strftime("%Y-%m-%d")
    returnM = offline.get_data_offline(data_offline_market,[market], startdate, enddate)['Returns']
    returnM_json = returnM.to_json(date_format='iso', orient='split')
    returnM = pd.read_json(returnM_json, orient='split')
    Returns = pd.concat([returns, returnM], axis=1)
    Returns = Returns.fillna(0)
    V = Returns.cov()
    V = np.array(V)
    betas = V[-1, :-1]
    betas = betas / V[-1, -1]
    return (betas)


def RelativeStrengthIndex(returns, last=-14):
    '''Compute RSI momentum of the returns for the last 'last' days'''
    # max period we train on to compute the RSI
    Returns = returns.to_numpy().T
    H = []  # growth rates
    B = []  # decline rates
    for asset in Returns:
        H.append(np.mean([elt for elt in asset[last:] if elt > 0])) # mean of the growth rates on the period
        B.append(np.mean([abs(elt) for elt in asset[last:] if elt < 0])) # mean of the decline rates on the period
    RSI = [100 - 100 / (1 + H[i] / B[i]) for i in range(len(H))] #adapted from wikipedia formulas
    return (RSI)


### OPTIMIZER :

def optimize_portfolio(returns, nb_returns, diversification=True, risk_parameter = 1, max_expected_return=True, w0=None, long=True,
                       short=True, method='SLSQP',
                       market="^GSPC", beta_eq=False, transition_cost=[None, 1], maxturnover=[None, 1],
                       specificportfolio=None, RSI1=None, RSI2=None):
    '''
    Returns the optimal portfolio trained on returns data (see get_returns, nb_returns exit formats). Different options are available but optional :
    - diversification [True(default)/False] : add maximization of the diversification ratio or not
    - risk_parameter [float] : risk multiplier of the diversification (risk) ratio
    - max_expected_return [True(default)/False] : add maximization of the expected return (according to the given training returns) or not
    - w0 [List] : initial guess for the optimizer. The uniform portfolio is used by default. Useful for transaction costs penalization and max turn over (see below).
    - long [True (default) /False] : authorizes non-negative portfolio
    - short [True (default) /False] : authorizes non-positive portfolio
    - method [String] : specifies the solver which is used
    - market [String [SP500 by default]] : index of the market where assets come from (for beta constraint use)
    - beta_eq [True /False (default)] : specifies if we constrain the portfolio's beta to be equal to 0 (independant from the chosen market)
    - transition_cost [2-list] : Changing it implies adding a penalization on objective function related to the transition cost. First element is the price of transaction. Second element is the chosen norm to compute the turnover from w0.
    - maxturnover [2-list] : Changing it implies adding a constraint on the turnover which has to be inferior to the first element. Second element is the chosen norm to compute the turnover from w0.
    - specificportfolio [String] : to use a solver with very specific parametres (Test zone in the code)
    - RSI1 [None (default) or non-positive integer] : if a non-positive integer is given, max period we train on to compute the RSI and adapt long or short for each asset.
    - RSI2 [None (default) or [last, non-negative price]] : if a price is given, corresponding to the max amount we win (RSI=30) or loose (RSI = 70), add a score depending on the RSI momentum to the objective function, last parameter being the number of days we compute RSI
    '''
    # pre-Parameters
    V = returns.cov()

    #Compute Momentum
    if RSI1 != None:
        RSIS = RelativeStrengthIndex(returns, last=RSI1)
        #print('RSIs:', RSIS)
    if RSI2 != None:
        RSIS = RelativeStrengthIndex(returns, last=RSI2[0])
        #print('RSIs:', RSIS)

    # short or long
    if long and not short:
        cons_slsqp = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        cons_cobyla = [{'type': 'ineq', 'fun': lambda w: np.sum(w) - 1},
                       {'type': 'ineq', 'fun': lambda w: -np.sum(w) + 1}]
        bnds = ((0, 1),) * nb_returns
        if type(w0) != list:
            w0 = [1 / nb_returns] * nb_returns
    elif short and not long:
        cons_slsqp = [{'type': 'eq', 'fun': lambda w: np.sum(w) + 1}]
        cons_cobyla = [{'type': 'ineq', 'fun': lambda w: np.sum(w) + 1},
                       {'type': 'ineq', 'fun': lambda w: -np.sum(w) - 1}]
        bnds = ((-1, 0),) * nb_returns
        if type(w0) != list:
            w0 = [-1 / nb_returns] * nb_returns
    elif short and long:
        cons_slsqp = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        cons_cobyla = [{'type': 'ineq', 'fun': lambda w: np.sum(w) - 1},
                       {'type': 'ineq', 'fun': lambda w: -np.sum(w) + 1}]
        bnds = ((-1, 1),) * nb_returns
        if type(w0) != list:
            w0 = [1 / nb_returns] * nb_returns
            
    # Add Momentum consideration as constraint
    if RSI1 != None:
        cons_slsqp = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        cons_cobyla = [{'type': 'ineq', 'fun': lambda w: np.sum(w) - 1},
                       {'type': 'ineq', 'fun': lambda w: -np.sum(w) + 1}]
        RSIS = RelativeStrengthIndex(returns, last=RSI1)
        bnds = []
        for RSI in RSIS:
            if RSI <= 30:
                bnds.append((0, 1))
            elif RSI >= 70:
                bnds.append((-1, 0))
            else:
                bnds.append((-1, 1))
        bnds = tuple(bnds)
        if type(w0) != list:
            w0 = [1 / nb_returns] * nb_returns

    # Add Beta = 0 constraint
    if beta_eq:
        B = Betas(returns, market=market)
        cons_slsqp.append({'type': 'eq', 'fun': lambda w: np.dot(w.T, B)})
        cons_cobyla.append({'type': 'ineq', 'fun': lambda w: np.dot(w.T, B)})
        cons_cobyla.append({'type': 'ineq', 'fun': lambda w: -np.dot(w.T, B)})
    
    # Add max turnover constraint
    if maxturnover[0] != None:
        if type(w0) == list:
            cons_slsqp.append(
                {'type': 'ineq', 'fun': lambda w: -np.linalg.norm(w - w0, maxturnover[1]) + maxturnover[0]})
            cons_cobyla.append(
                {'type': 'ineq', 'fun': lambda w: -np.linalg.norm(w - w0, maxturnover[1]) + maxturnover[0]})
        else:
            print("Please enter w0, maxturnover aborted")

    # Objective function definition
    
    def objective_function(w):
        temp = 0
        
        # Add max diversification ratio
        if diversification:
            temp += risk_parameter*diversification_ratio(w, V)
            
        # Add transition costs penalization
        if transition_cost[0] != None:
            if type(w0) == list:
                temp += transition_cost[0] * np.linalg.norm(w - w0, transition_cost[1])
            else:
                print("Please enter w0, transition_cost aborted")
           
        # Add max expected return objective function
        if max_expected_return:
            returns_over_the_period = np.prod((1+returns), axis=0)-1
            temp -= np.dot(w, returns_over_the_period)
            
        # Add Momentum consideration as penalization
        if RSI2 != None:
            cents = RSI2[1]
            score = 0  # score to maximize
            for i in range(len(RSIS)):
                rsi = RSIS[i]
                if rsi <= 30:
                    score += w[i] * rsi * cents / 30
                if rsi >= 70:
                    score -= w[i] * (100 - rsi) * cents / 30
            temp -= score
            
        return (temp)

    # Area to define prototypes of portfolio
    if specificportfolio == 'long_or_short_div_ratio':
        if w0 == None:
            w0 = [-1 / nb_returns] * nb_returns + [1 / nb_returns] * nb_returns

        def double_diversification_ratio(w):
            n = len(w) // 2
            wshort = w[0:n]
            wlong = w[n:]
            w_vol_short = np.dot(wshort.T, np.sqrt(np.diag(V)))
            port_vol_short = np.sqrt(np.dot(np.dot(wshort.T, V), wshort))
            w_vol_long = np.dot(wlong.T, np.sqrt(np.diag(V)))
            port_vol_long = np.sqrt(np.dot(np.dot(wlong.T, V), wlong))
            diversification_ratio = (w_vol_short / port_vol_short) + (w_vol_long / port_vol_long)
            return -diversification_ratio

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bnds = ((-1, 0),) * nb_returns + ((0, 1),) * nb_returns
        optimized = opt.minimize(double_diversification_ratio, w0, method='SLSQP', bounds=bnds, constraints=cons)
        return optimized
    
    # Optimizing with the chosen optimizer
    if method == 'SLSQP':
        # Optimize with COBYLA
        optimized = opt.minimize(objective_function, w0, method='SLSQP', bounds=bnds, constraints=cons_slsqp)

    if method == 'COBYLA':
        # Optimize with COBYLA
        optimized = opt.minimize(objective_function, w0, method='COBYLA', bounds=bnds, constraints=cons_cobyla)

    return optimized


### ROLLING SIMULATION :


def get_previous_months_returns(returns,date,nb_months):
    """
    Select the returns from the nb_months (int) prior to date
    """
    range_max = date
    range_min = range_max - pd.DateOffset(months=nb_months)
    return returns[range_min:range_max]

def weights_rolling_window(returns,nb_months=6, diversification=True, risk_parameter = 10, max_expected_return=True, long=True, short=True, beta_eq=False, method='SLSQP', market="^GSPC",  transition_cost=[None, 1], maxturnover=[None, 1], specificportfolio=None, RSI1=None, RSI2=None, display = False):
  
    """
    Actualize the optimized weights every months, considering the data from the nb_months previous months.
    returns : DataFrame, nb_months : int
    Return a DataFrame containing all the weights, and a dataFrame containing all the diversification ratios
    """
    start_date = returns.index[0] + pd.DateOffset(months=nb_months)
    end_date = returns.index[-1]
    iterations = (end_date.year - start_date.year)*12 + (end_date.month - start_date.month) + 1
    nb_returns = len(returns.columns)
    weights = pd.DataFrame([]).reindex_like(returns)[start_date:]
    diversification_ratio_historic = pd.DataFrame(index=weights.index)
    diversification_ratio_historic["Optimized Portfolio"] = ""
    diversification_ratio_historic["Equally Weighted Portfolio"] = ""
    date = start_date
    w = []
    maxturnovers = []
    transitioncosts = []
    
    for i in range(iterations) :
        past_returns = get_previous_months_returns(returns,date,nb_months) #Get the returns of the past months
        if transition_cost[0] != None or maxturnover[0] != None:
            if w!=[]:
                optimized = optimize_portfolio(past_returns, nb_returns,w0 = w, long=long, short=short, method=method, beta_eq=beta_eq, specificportfolio=specificportfolio, max_expected_return=max_expected_return, diversification=diversification, risk_parameter = risk_parameter,RSI1=RSI1, RSI2=RSI2, market=market,  transition_cost=transition_cost, maxturnover=maxturnover) #Optimize the weights considering the selected returns
            else:
                optimized = optimize_portfolio(past_returns, nb_returns, long=long, short=short, method=method, beta_eq=beta_eq, specificportfolio=specificportfolio, max_expected_return=max_expected_return, diversification=diversification,risk_parameter = risk_parameter,RSI1=RSI1, RSI2=RSI2, market=market,  transition_cost=transition_cost, maxturnover=maxturnover)
        else:
            optimized = optimize_portfolio(past_returns, nb_returns, long=long, short=short, method=method, beta_eq=beta_eq, specificportfolio=specificportfolio, max_expected_return=max_expected_return, diversification=diversification,risk_parameter = risk_parameter,RSI1=RSI1, RSI2=RSI2, market=market,  transition_cost=transition_cost, maxturnover=maxturnover) #Optimize the weights considering the selected returns
              
        if not optimized.success: #if unfeasible problem 
                print('Optimization status '+str(i)+' : ', optimized.success)          
        if w != []:
            if maxturnover[0]!=None:
                maxturnovers.append(np.linalg.norm(w - optimized.x, maxturnover[1]))
            if transition_cost[0]!=None:
                transitioncosts.append(transition_cost[0] * np.linalg.norm(w - optimized.x, transition_cost[1]))
            
        w = optimized.x
        newdate = date + pd.DateOffset(months=1)
        V = past_returns.cov()
        diversification_ratio_historic["Optimized Portfolio"][date:newdate] = -diversification_ratio(w,V)
        diversification_ratio_historic["Equally Weighted Portfolio"][date:newdate] = -diversification_ratio(np.repeat(1/nb_returns, nb_returns),V)
        for j in range(nb_returns) :
            weights[weights.columns[j]][date:newdate] = w[j]
        date = newdate
        
    #performances plots:
    if display:
        if maxturnover[0] != None:
            pd.DataFrame(maxturnovers,columns=["turnover (proportion of the portfolio)"]).plot()
        if transition_cost[0] != None:
            pd.DataFrame(transitioncosts,columns=["Transition costs (dollar per dollar transited)"]).plot()
        
    return weights, diversification_ratio_historic


### RESULTS :

def portfolio_return(weights,returns):
    """
    Given weights (DataFrame or series) and returns of assets (DataFrame), returns the return (DataFrame) of the portfolio over the time
    """
    if isinstance(weights, pd.DataFrame):
        start_date = returns.index[0]
        ret = pd.DataFrame((weights*returns[start_date:]).sum(axis=1))
        ret.columns = ["Portfolio"]
        return ret
    return weights.T @ returns

def portfolio_volatility(weights, covmat):
    return weights.T @ covmat @ weights

def equaly_weighted_portfolio_return(returns):
    """
    Considering the returns of several assets (DataFrame), return the return (DataFrame) of the portfolio with equal weight on all the assets
    """
    nb_returns = returns.shape[1]
    weights = pd.DataFrame([]).reindex_like(returns)
    w = 1/nb_returns
    for asset in weights :
        weights[asset] = w
    return portfolio_return(weights, returns)

def cumulated_returns(returns):
    """
    Compute the cumulated returns from the returns. Return a DataFrame
    """
    return (1+returns).cumprod()-1