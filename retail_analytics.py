"""
Retail Market Basket Analysis & Sales Revenue Forecasting
Author: Vansh Gandhi
Description:
    End-to-end Machine Learning pipeline analyzing UK retail transaction data:
    1. Data Cleaning & Feature Engineering
    2. Exploratory Data Analysis (EDA) & Key Metrics
    3. Unsupervised Learning: Market Basket Analysis via Apriori & Association Rules
    4. Supervised Learning: Daily Revenue Time-Series Forecasting via SARIMA
"""

import os
import gc
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 5)


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw Excel/CSV transaction data and performs essential preprocessing:
    - Removes missing product descriptions and customer identifiers
    - Filters out cancelled orders (Invoice beginning with 'C')
    - Filters out non-positive quantities
    - Converts timestamps and numeric columns
    - Engineers 'Revenue' feature (Quantity * Price)
    """
    print(f"[*] Loading data from: {file_path}")
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")

    print(f"    Raw records loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    # Drop CustomerID if present
    df.drop('CustomerID', axis=1, inplace=True, errors='ignore')

    # Drop missing item names
    item_col = 'Itemname' if 'Itemname' in df.columns else 'Description'
    df.dropna(axis=0, subset=[item_col], inplace=True)
    df[item_col] = df[item_col].astype(str).str.strip()

    # Filter out cancellations
    bill_col = 'BillNo' if 'BillNo' in df.columns else 'InvoiceNo'
    df = df[~df[bill_col].astype(str).str.startswith('C')]

    # Filter positive quantities
    df = df[df['Quantity'] > 0]

    # Convert data types
    date_col = 'Date' if 'Date' in df.columns else 'InvoiceDate'
    df['Date'] = pd.to_datetime(df[date_col])
    df['Quantity'] = pd.to_numeric(df['Quantity'], downcast='integer')

    price_col = 'Price' if 'Price' in df.columns else 'UnitPrice'
    df['Price'] = pd.to_numeric(df[price_col], downcast='float')

    # Compute Revenue
    df['Revenue'] = df['Quantity'] * df['Price']

    # Standardize column naming
    df['BillNo'] = df[bill_col]
    df['Itemname'] = df[item_col]

    print(f"[✓] Data preprocessing complete. Cleaned records: {df.shape[0]:,}")
    return df


def calculate_business_metrics(df: pd.DataFrame) -> dict:
    """
    Computes key financial and operational KPIs from the transaction data.
    """
    metrics = {
        "total_revenue": df['Revenue'].sum(),
        "total_orders": df['BillNo'].nunique(),
        "total_items_sold": int(df['Quantity'].sum()),
        "total_unique_products": df['Itemname'].nunique(),
        "avg_order_value": df.groupby('BillNo')['Revenue'].sum().mean(),
        "median_order_value": df.groupby('BillNo')['Revenue'].sum().median()
    }

    print("\n" + "=" * 45)
    print("           KEY BUSINESS METRICS")
    print("=" * 45)
    print(f"Total Revenue:            ${metrics['total_revenue']:,.2f}")
    print(f"Total Unique Orders:       {metrics['total_orders']:,}")
    print(f"Total Items Sold:          {metrics['total_items_sold']:,}")
    print(f"Total Unique Products:     {metrics['total_unique_products']:,}")
    print(f"Average Order Value (AOV): ${metrics['avg_order_value']:,.2f}")
    print(f"Median Order Value:        ${metrics['median_order_value']:,.2f}")
    print("=" * 45 + "\n")

    return metrics


def run_market_basket_analysis(df: pd.DataFrame, country: str = "United Kingdom",
                               min_support: float = 0.02, min_lift: float = 1.0) -> pd.DataFrame:
    """
    Unsupervised Learning: Generates association rules using the Apriori algorithm.
    Focuses on transactions from the specified target market.
    """
    print(f"[*] Running Market Basket Analysis for: {country} (min_support={min_support}, min_lift={min_lift})")
    df_geo = df[df['Country'] == country] if 'Country' in df.columns else df

    transactions = df_geo.groupby('BillNo')['Itemname'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    basket_sets = pd.DataFrame(te_ary, columns=te.columns_)

    del transactions, df_geo
    gc.collect()

    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    if frequent_itemsets.empty:
        print("[!] No frequent itemsets found with current support threshold.")
        return pd.DataFrame()

    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_lift)
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    rules = rules.sort_values(by='lift', ascending=False)

    print(f"[✓] Generated {len(rules)} association rules. Top 5 affinity pairings:")
    for _, r in rules.head(5).iterrows():
        print(f"    - IF [{r['antecedents']}] THEN [{r['consequents']}] (Lift: {r['lift']:.2f}, Conf: {r['confidence']:.2%})")

    return rules


def train_sarima_forecasting(df: pd.DataFrame, test_days: int = 30, forecast_days: int = 30):
    """
    Supervised Learning: Fits a SARIMA time-series model on daily aggregated revenue.
    Validates on the held-out test_days, then retrains to forecast forecast_days ahead.
    """
    print(f"\n[*] Preparing daily revenue time series for SARIMA forecasting...")
    daily_sales = df.set_index('Date').resample('D')['Revenue'].sum().to_frame()

    # Train / Test split
    train_data = daily_sales[:-test_days]
    test_data = daily_sales[-test_days:]

    print(f"    Training window: {train_data.index.min().date()} to {train_data.index.max().date()}")
    print(f"    Evaluation window: {test_data.index.min().date()} to {test_data.index.max().date()}")

    # Fit SARIMA on training split
    sarima_order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)
    print(f"[*] Training SARIMA{sarima_order}x{seasonal_order} on training set...")
    val_model = SARIMAX(train_data['Revenue'], order=sarima_order, seasonal_order=seasonal_order)
    val_results = val_model.fit(disp=False)

    # Evaluate validation predictions
    val_pred = val_results.get_prediction(start=test_data.index[0], end=test_data.index[-1]).predicted_mean
    mae = np.mean(np.abs(val_pred.values - test_data['Revenue'].values))
    print(f"[✓] Validation MAE: ${mae:,.2f}")

    # Retrain on full dataset for future projection
    print(f"[*] Retraining SARIMA on full dataset to forecast next {forecast_days} days...")
    full_model = SARIMAX(daily_sales['Revenue'], order=sarima_order, seasonal_order=seasonal_order)
    full_results = full_model.fit(disp=False)
    forecast = full_results.get_forecast(steps=forecast_days)

    print(f"[✓] Forecast completed for {forecast_days} days into the future.")
    return daily_sales, val_pred, forecast


def main():
    parser = argparse.ArgumentParser(description="Retail Market Basket Analysis & Sales Forecasting")
    parser.add_argument("--data", type=str, required=True, help="Path to transactional dataset (.xlsx or .csv)")
    parser.add_argument("--country", type=str, default="United Kingdom", help="Target country for Apriori analysis")
    parser.add_argument("--min_support", type=float, default=0.02, help="Minimum support for Apriori")
    parser.add_argument("--min_lift", type=float, default=1.0, help="Minimum lift for association rules")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data file '{args.data}' not found.")
        return

    df = load_and_preprocess_data(args.data)
    calculate_business_metrics(df)
    run_market_basket_analysis(df, country=args.country, min_support=args.min_support, min_lift=args.min_lift)
    train_sarima_forecasting(df)


if __name__ == "__main__":
    main()
