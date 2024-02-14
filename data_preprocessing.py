import pandas as pd
import datetime, dateutil
from finta import TA
import yfinance as yf

def preprocess_data(ticker):

  drop_cols=["Open", "High", "Low", "Close", "Adj Close", "Volume", "Security", "GICS Sector",
            "GICS Sub-Industry", "News - Volume", "News - Store Openings", "News - Product Recalls", "News - Layoffs"]
  snes_data = pd.read_csv("data.csv").drop(columns=drop_cols, axis=1)
  snes_data.drop(snes_data[snes_data['Date'] == '2020-09-30'].index, axis=0, inplace=True)

  # Starting and ending dates are same regardless of whether ticker in S&P 500
  snes_start = snes_data.iloc[0]['Date']
  snes_end = snes_data.iloc[len(snes_data)-1]['Date']
  datetime_end = dateutil.parser.parse(snes_end, fuzzy=True).date()

  ticker_in_snp500 = (len(snes_data[snes_data['Symbol'] == ticker].index) != 0)

  if not ticker_in_snp500:
    print(f"Sentiment and news data not found: {ticker} is not in the S&P 500 index fund")
  else:
    #snes_data.dropna(how="all", inplace=True)
    print("SNES Data start date: ", snes_start)
    print("SNES Data end date: ", snes_end)
    snes_data = snes_data[snes_data['Symbol'] == ticker].drop(columns=['Symbol'], axis=1).reset_index(drop=True)

  yf_end = datetime_end + datetime.timedelta(days=1)
  data = yf.download(ticker, start=snes_start, end=yf_end).drop(["Close"], axis=1)
  data = data.rename(columns={"Adj Close": "Close"})
  data.index = data.index.date

  macd_data = TA.MACD(data, 12)

  # data['MACD_Line'] = macd_data['MACD']
  data['Signal_Line'] = macd_data['SIGNAL']
  data['Histogram'] = macd_data['MACD'] - data['Signal_Line']
  data['BB_Upper'] = TA.BBANDS(data, period=20)['BB_UPPER']
  data['BB_Lower'] = TA.BBANDS(data, period=20)['BB_LOWER']
  data['Stoch_price_positions'] = TA.STOCH(data)
  data['ATR'] = TA.ATR(data)

  data['RSI'] = TA.RSI(data)
  data['OBV'] = TA.OBV(data)

  snp500_close = yf.download("^GSPC", start=snes_start, end=yf_end)['Adj Close']
  data['snp500_close'] = snp500_close

  data.drop(columns=['Open', "High", "Low"], axis=1, inplace=True)
  data.fillna(0, inplace=True)
  #data.reset_index(drop=True, inplace=True)

  if ticker_in_snp500:

    assert len(data) == len(snes_data)
    # Below for loop confirms that all dates match between yfinance API price data and SNES dataset
    for idx in range(len(data)):
      data_date = data.index[idx].strftime("%Y-%m-%d")
      if data_date != snes_data['Date'][idx]:
        print(f"{data_date} in data doesn't MATCH in snes_data")
        if len(snes_data[snes_data['Date'] == data_date].index) == 0:
          print(f"{data_date} does not EXIST in snes_data")

    snes_data.drop(columns=['Date'], axis=1, inplace=True)

    data = pd.concat([data, snes_data.set_index(data.index)], axis=1)
    # print(data.head())

  total_data = len(data)
  data_cutoff = int(total_data*0.85)
  train_df = data.iloc[:data_cutoff,:]
  test_df = data.iloc[data_cutoff:,:]

  return data, train_df, test_df, ticker_in_snp500