import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def process_gt_pair(prediction_file):
    model_name = os.path.basename(os.path.dirname(prediction_file))
    df = pd.read_excel(prediction_file)
    columns = df.columns

    # Gather all pairs of prediction and ground truth columns
    pred_gt_pairs = [(pred_col, gt_col) for pred_col in columns if 'Pred_' in pred_col for gt_col in columns if f'GT_{pred_col.split("Pred_")[1]}' == gt_col]

    results = []
    for pair in pred_gt_pairs:
        target_name = pair[0].split('_')[1]
        results.append((df, model_name, target_name, pair))

    return results

def is_binary(series):
    """Check if a column is already binary."""
    unique_values = series.dropna().unique()
    return set(unique_values).issubset({0, 1})

def threshold_binary(data, threshold=0):
    """Convert floats to binary based on a threshold."""
    return (data >= threshold).astype(int)


def gather_prediction_gt_pairs(predictions_dir):
    """Gather all prediction-gt column pairs from prediction sheets."""
    prediction_files = []
    for root, _, files in os.walk(predictions_dir):
        for file in files:
            if file.endswith('.xlsx'):
                prediction_files.append(os.path.join(root, file))

    prediction_gt_pairs = []
    
    # Using Pool for multiprocessing
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap(process_gt_pair, prediction_files), total=len(prediction_files), desc="- Gathering Pred-GT pairs"):
            prediction_gt_pairs.extend(result)

    return prediction_gt_pairs


def load_stock_data(stock_data_file):
    """
    Load the stock data from the 'Returns' and 'Alpha' sheets, with specific structure:
    1. First row as column titles.
    2. First column as tickers, second column as filing dates (both as indices).
    3. All other content converted to float.
    """
    # Load the Returns sheet
    returns_df = pd.read_excel(stock_data_file, sheet_name="Returns", index_col=[0, 1])
    
    # Convert all other columns (Day 1, Day 2, ..., Day 20) to float
    returns_df = returns_df.apply(pd.to_numeric, errors='coerce')
    
    # Load the Alpha sheet with the same structure
    alpha_df = pd.read_excel(stock_data_file, sheet_name="Alpha", index_col=[0, 1])
    
    # Convert all other columns to float
    alpha_df = alpha_df.apply(pd.to_numeric, errors='coerce')

    # Return as a dictionary for access in further operations
    stock_data = {
        'Returns': returns_df,
        'Alpha': alpha_df
    }
    
    return stock_data

def process_prediction_pair(args):
    """Process a single prediction-gt pair in parallel."""
    df, stock_data, model_name, pred_col, gt_col = args
    results = []

    # Ensure both prediction and GT columns are binary (threshold float columns if necessary)
    if not is_binary(df[pred_col]):
        df[pred_col] = threshold_binary(df[pred_col])
    if not is_binary(df[gt_col]):
        df[gt_col] = threshold_binary(df[gt_col])

    # Get the signal name by removing the "Pred_" or "GT_" part
    signal = pred_col.replace("Pred_", "").replace("GT_", "")
    
    # If the signal contains '_stop_', invert the boolean values
    if '_stop_' in signal:
        df[pred_col] = 1 - df[pred_col]  # Invert prediction values
        df[gt_col] = 1 - df[gt_col]  # Invert ground truth values

    # Filter by the prediction signal being 1
    pred_filter = df[df[pred_col] == 1]['Ticker']
    
    # Filter by the GT signal being 1
    gt_filter = df[df[gt_col] == 1]['Ticker']

    # Load the stock data for Day 20
    returns_df = stock_data['Returns']
    alpha_df = stock_data['Alpha']
    
    # Filter stock data based on the tickers in the prediction and GT signals
    pred_returns = returns_df.loc[pred_filter, 'Day 20 Stock']
    pred_alphas = alpha_df.loc[pred_filter, 'Day 20 Alpha']
    gt_returns = returns_df.loc[gt_filter, 'Day 20 Stock']
    gt_alphas = alpha_df.loc[gt_filter, 'Day 20 Alpha']

    # Calculate mean return and alpha for both pred and gt
    mean_pred_return = pred_returns.mean() if not pred_returns.empty else np.nan
    mean_pred_alpha = pred_alphas.mean() if not pred_alphas.empty else np.nan
    mean_gt_return = gt_returns.mean() if not gt_returns.empty else np.nan
    mean_gt_alpha = gt_alphas.mean() if not gt_alphas.empty else np.nan

    # Calculate mean return and alpha for both pred and gt
    median_pred_return = pred_returns.median() if not pred_returns.empty else np.nan
    median_pred_alpha = pred_alphas.median() if not pred_alphas.empty else np.nan
    median_gt_return = gt_returns.median() if not gt_returns.empty else np.nan
    median_gt_alpha = gt_alphas.median() if not gt_alphas.empty else np.nan

    result = {
        'Model': model_name,
        'Signal': signal,
        'Mean_Pred_Return (in %)': np.round(mean_pred_return*100,2),
        'Median_Pred_Return (in %)': np.round(median_pred_return*100,2),
        'Mean_Pred_Alpha (in %)': np.round(mean_pred_alpha*100,2),
        'Median_Pred_Alpha (in %)': np.round(median_pred_alpha*100,2),
        'Mean_GT_Return (in %)': np.round(mean_gt_return*100,2),
        'Median_GT_Return (in %)': np.round(median_gt_return*100,2),
        'Mean_GT_Alpha (in %)': np.round(mean_gt_alpha*100,2),
        'Median_GT_Alpha (in %)': np.round(median_gt_alpha*100,2)
    }

    results.append(result)

    return results

def save_results(all_results, output_file):
    """Save all the backtest results into a single Excel file."""
    final_df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_excel(output_file, index=False)
    print(f"- Backtest results saved to {output_file}")
