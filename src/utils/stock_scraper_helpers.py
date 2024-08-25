import os
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

def load_features(features_file):
    """Load the full features DataFrame and extract Ticker and Filing Date columns."""
    features_df = pd.read_excel(features_file)
    features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date'], dayfirst=True)
    ticker_filing_dates = features_df[['Ticker', 'Filing Date']]
    earliest_date = ticker_filing_dates['Filing Date'].min()
    return ticker_filing_dates, earliest_date

def download_daily_stock_data(ticker, filing_date, max_days):
    """Download daily stock data for a given ticker between filing_date and 20 business days after."""
    end_date = pd.to_datetime(filing_date, dayfirst=True) + BDay(max_days + 5)
    try:
        stock_data = yf.download(ticker, start=filing_date, end=end_date, interval='1d', progress=False)
        if stock_data.empty:
            print(f"No data found for {ticker} between {filing_date} and {end_date}.")
            return None
        return stock_data
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        return None

def create_stock_data_dict(results):
    """Create a dictionary of stock data aligned to filing dates."""
    stock_data_dict = {}

    for ticker, filing_date, max_days, ticker_data in results:
        if ticker_data is not None:
            # Slice the first 20 business days of data
            if 'Close' in ticker_data.columns:
                closing_data = ticker_data['Close'][:max_days].reset_index(drop=True)
                # Ensure that all data is stored correctly
                stock_data_dict[(ticker, filing_date.strftime('%d/%m/%Y %H:%M'))] = closing_data.T
            else:
                print(f"Ticker data for {ticker} does not have a 'Close' column. Data: {ticker_data.head()}")

    return stock_data_dict

def save_to_excel(stock_data_df, return_df, output_file):
    """Save the stock data and returns sheets to an Excel file."""
    stock_data_dir = os.path.dirname(output_file)
    os.makedirs(stock_data_dir, exist_ok=True)
    
    with pd.ExcelWriter(output_file) as writer:
        stock_data_df.to_excel(writer, sheet_name='Stock Data', index=False)
        return_df.to_excel(writer, sheet_name='Returns', index=False)
    
    print(f"Data successfully saved to {output_file}.")

def download_data_wrapper(info):
    """Wrapper function to pass to multiprocessing."""
    ticker, filing_date, max_days = info
    stock_data = download_daily_stock_data(ticker, filing_date, max_days)
    return (ticker, filing_date, max_days, stock_data)
