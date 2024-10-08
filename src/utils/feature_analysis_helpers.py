import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold

def load_feature_data(file_path):
    """Load the feature data from an Excel file and extract Ticker and Filing Date."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    file_path = os.path.join(data_dir, file_path)
    if os.path.exists(file_path):
        try:
            data = pd.read_excel(file_path)
            # Ensure 'Filing Date' is in datetime format with both date and time
            data['Filing Date'] = pd.to_datetime(data['Filing Date'], dayfirst=True, errors='coerce')
            ticker_filing_dates = data[['Ticker', 'Filing Date']].copy()  # Create a copy to avoid SettingWithCopyWarning
            ticker_filing_dates['Filing Date'] = ticker_filing_dates['Filing Date'].dt.strftime('%d/%m/%Y %H:%M')
            print(f"Sheet successfully loaded from {file_path}.")
            return data, ticker_filing_dates
        except Exception as e:
            print(f"Failed to load sheet from {file_path}: {e}")
            return None, None
    else:
        print(f"File '{file_path}' does not exist.")
        return None, None


def save_feature_data(data, ticker_filing_dates, file_path):
    """Save the processed feature data along with Ticker and Filing Date to an Excel file."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    file_path = os.path.join(data_dir, file_path)
    # Ensure 'Filing Date' is treated as datetime
    ticker_filing_dates['Filing Date'] = pd.to_datetime(ticker_filing_dates['Filing Date'], dayfirst=True, errors='coerce')
    # Drop any rows where 'Filing Date' could not be converted
    ticker_filing_dates.dropna(subset=['Filing Date'], inplace=True)
    # Format the 'Filing Date' to 'dd/mm/yyyy hh:mm'
    ticker_filing_dates['Filing Date'] = ticker_filing_dates['Filing Date'].dt.strftime('%d/%m/%Y %H:%M')
    # Concatenate Ticker and Filing Date back to the data
    final_data = pd.concat([ticker_filing_dates, data], axis=1)

    if not final_data.empty:
        try:
            final_data.to_excel(file_path, index=False)
            print(f"Data successfully saved to {file_path}.")
        except Exception as e:
            print(f"Failed to save data to Excel: {e}")
    else:
        print("No data to save.")

def identify_feature_types(data):
    """Identify and return categorical and continuous features."""
    categorical_feature_names = ["CEO", "CFO", "COO", "Dir", "Pres", "VP", "TenPercent", 
                                 "CDL_DOJI", "CDL_HAMMER", "CDL_ENGULFING", 
                                 "Sector_Basic Materials", "Sector_Communication Services", 
                                 "Sector_Consumer Cyclical", "Sector_Consumer Defensive", 
                                 "Sector_Energy", "Sector_Financial Services", 
                                 "Sector_Healthcare", "Sector_Industrials", "Sector_Real Estate", 
                                 "Sector_Technology", "Sector_Utilities"]

    categorical_features = data[[feature for feature in categorical_feature_names if feature in data.columns]]
    continuous_feature_names = [feature for feature in data.columns if feature not in categorical_feature_names]
    continuous_features = data[continuous_feature_names]

    return categorical_features, continuous_features

def filter_low_variance_features(data, continuous_features, categorical_features, variance_threshold=0.02, categorical_threshold=0.02):
    """Filter out low variance continuous features and rare categorical features."""
    normalized_features = ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'ATR_14', 'SAR', 
                           'OBV', 'Cumulative_Alpha', 'Rolling_Alpha_30', 'Jensen_Alpha',
                           'Operating_Cash_Flow_to_Market_Cap', 'Investing_Cash_Flow_to_Market_Cap', 
                           'Financing_Cash_Flow_to_Market_Cap', 'Net_Income_to_Market_Cap', 
                           'Total_Assets_to_Market_Cap', 'Total_Liabilities_to_Market_Cap', 
                           'Total_Equity_to_Market_Cap', 'Average_Volume_to_Market_Cap', 'Free_Cash_Flow_to_Market_Cap', 
                           '52_Week_High_Normalized', '52_Week_Low_Normalized']

    non_normalized_continuous_features = [col for col in continuous_features.columns if col not in normalized_features]

    if non_normalized_continuous_features:
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(continuous_features[non_normalized_continuous_features])
        low_variance_features = continuous_features[non_normalized_continuous_features].columns[~selector.get_support()]

        data.drop(columns=low_variance_features, inplace=True)
        continuous_features = continuous_features.drop(columns=low_variance_features, axis=1)
        print(f"Dropped {len(low_variance_features)} low variance continuous features: {low_variance_features.tolist()}")

    rare_categorical_features = []
    for col in categorical_features.columns:
        min_class_freq = min(data[col].mean(), 1 - data[col].mean())

        if min_class_freq < categorical_threshold:
            rare_categorical_features.append(col)

    if rare_categorical_features:
        data.drop(columns=rare_categorical_features, inplace=True)
        categorical_features = categorical_features.drop(columns=rare_categorical_features, axis=1)
    
    print(f"Dropped {len(rare_categorical_features)} rare categorical features: {rare_categorical_features}")

    return data

def clip_continuous_features(data, continuous_features, lower=0.01, upper=0.99):
    """Clip continuous features at the specified lower and upper percentiles."""
    for column in continuous_features:
        if column in data.columns:
            lower_bound = data[column].quantile(lower)
            upper_bound = data[column].quantile(upper)
            data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    print("Clipped continuous features at the 1st and 99th percentiles.")
    return data

def normalize_continuous_features(data, continuous_features):
    """Apply Min-Max Normalization to continuous features."""
    for column in continuous_features:
        min_value = data[column].min()
        max_value = data[column].max()

        if max_value - min_value == 0:
            print(f"Warning: {column} has zero variance. Skipping normalization.")
            continue

        data[column] = (data[column] - min_value) / (max_value - min_value)
    
    print("Applied Min-Max Normalization to continuous features.")
    return data

def calculate_correlations(data, continuous_features):
    """Calculate Pearson and Spearman correlations and return the combined correlation matrix."""
    pearson_corr = data.corr(method='pearson').abs()
    spearman_corr = data.corr(method='spearman').abs()

    corr = pd.DataFrame(index=data.columns, columns=data.columns)

    for col1 in data.columns:
        for col2 in data.columns:
            if col1 == col2:
                corr.at[col1, col2] = np.nan
            elif col1 in continuous_features.columns and col2 in continuous_features.columns:
                corr.at[col1, col2] = pearson_corr.at[col1, col2]
            else:
                corr.at[col1, col2] = spearman_corr.at[col1, col2]

    return corr

def drop_highly_correlated_features(data, corr_matrix, threshold=0.9):
    """Drop features with correlation higher than the threshold."""
    upper_triangle_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    sorted_corr = corr_matrix.where(upper_triangle_mask).unstack().dropna().sort_values(ascending=False)

    # Identify features to drop based on the threshold
    to_drop = []
    for feature in sorted_corr.index:
        if sorted_corr[feature] > threshold:
            to_drop.append(feature[1])  # Select the second level of the MultiIndex, which corresponds to the columns

    # Drop duplicates in `to_drop` to avoid redundant drops
    to_drop = list(set(to_drop))

    # Drop the identified columns from the data
    data.drop(columns=to_drop, axis=1, inplace=True)

    # Recalculate correlation matrix after dropping columns
    updated_corr_matrix = data.corr()

    print(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")
    return data, updated_corr_matrix


def plot_correlation_heatmap(corr_matrix, output_dir):
    """Plot a heatmap of the correlation matrix and save it as a PNG file."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    file_path = os.path.join(data_dir, output_dir)
    
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(20, 20), dpi=300)
    sns.heatmap(corr_matrix.astype(float), mask=mask, annot=False, cmap='coolwarm', square=True, cbar_kws={"shrink": .75})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    file_path = os.path.join(file_path, 'feature_correlation_heatmap.png')
    plt.savefig(file_path)
    plt.close()
    print("Feature correlation heatmap saved at", file_path)

def plot_sorted_correlations(corr_matrix, output_dir):
    """Plot sorted correlations as a barplot and save it as a PNG file."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    file_path = os.path.join(data_dir, output_dir)
    
    upper_triangle_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    sorted_corr = corr_matrix.where(upper_triangle_mask).unstack().dropna().sort_values(ascending=False)

    sorted_corr = sorted_corr[sorted_corr > 0.8]

    plt.figure(figsize=(14, 10), dpi=300)
    sorted_corr.plot(kind='bar', width=0.8)
    plt.title('Sorted Feature Correlations Above 0.8')
    plt.ylim(0.8, 1.0)
    plt.yticks(np.arange(0.8, 1.05, 0.05))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    file_path = os.path.join(file_path, 'sorted_correlations.png')
    plt.savefig(file_path, dpi=300)
    plt.close()
    print("Sorted feature correlation barplot saved at", file_path)