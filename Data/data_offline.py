import tools as tl
import pandas as pd
import matplotlib.pyplot as plt

tickers = ['PKI','KR','VNO','NEM','BIIB','HFC','NFLX','COG','NLOK','FLIR','SEE','CLX','ABMD','AAL','DPZ','CPB','TSLA','GILD','LB','CTXS','MRO']

def build_df(tickers, start_date, end_date, outfile="data.csv"):
    data = tl.get_data(tickers, start_date, end_date)
    print("data collected")
    print(data)
    data.to_csv(outfile)

def get_data_offline(data_offline, tickers, start_date, end_date):
    """tickers: list of strings represnting the assets selected, start_date: start date of prices selected, end_date: end date of prices selected
    Select the prices and returns of selected assets for the time period selected
    Return a DataFrame containing the prices (data["Prices"]) and the returns (data["Returns"]) for selected assets
    """
    prices = data_offline["Prices"][tickers]
    prices = prices.loc[prices.index >= start_date]
    prices = prices.loc[prices.index <= end_date]

    returns = data_offline["Returns"][tickers]
    returns = returns.loc[returns.index >= start_date]
    returns = returns.loc[returns.index <= end_date]

    prices.columns = pd.MultiIndex.from_tuples([("Prices", ticker) for ticker in tickers])
    returns.columns = pd.MultiIndex.from_tuples([("Returns", ticker) for ticker in tickers])

    data = pd.concat([prices, returns], axis=1)

    return data

if __name__ == "__main__":
    build_df(tickers, "2010-01-01", "2020-12-31", outfile="data.csv")
    # data_offline = pd.read_csv("data.csv", header=[0,1], index_col=0)
    # data_offline.index = data_offline.index.astype(dtype='datetime64[ns]')

    # returns = data_offline["Returns"]
    # # returns.plot()
    # # plt.show()

    # prices = data_offline["Prices"]

    # data = get_data_offline(data_offline, tickers, "2012-01-01", "2015-12-31")
    # print(data)
    # returns = data["Returns"]
    # returns.plot()
    # plt.show()