import kagglehub
import pandas as pd
from math import nan
import os

# --- 1. Data Loading ---
print("--- 1. load data ---")
path = kagglehub.dataset_download("suchintikasarkar/sentiment-analysis-for-mental-health")
file_name = "Combined Data.csv"
file_path = os.path.join(path, file_name)

print("Path to dataset files:", path)
df = pd.read_csv(file_path)

# --- 1. Dirty data injecting ---

# Create dirty dataFrame
dirty_data = {
    'statement': [
        123456,  # Value Error - integer
        3.14159,  # Value Error - float
        None,  # None
        nan,
        "",  # Empty
    ],
    'status': ['Anxiety'] * 5  # All dirty tags are set to Anxiety
}

# Merge dirty dataFrame
df_dirty = pd.DataFrame(dirty_data)
df = pd.concat([df, df_dirty], ignore_index=True)

print(f"Shape after injection: {df.shape}")
print("Example of injected dirty data (last 5 lines):")
print(df.tail(10))

# --- 2. Data cleaning ---

initial_shape = df.shape

# 1. Drop the irrelevant index column
df = df.drop(columns=['Unnamed: 0'])

# 2. Drop missing values
df.dropna(subset=['statement'], inplace=True)
filter_empty = (df['statement'] != '')
df = df[filter_empty]

# 3. Drop duplicates
df.drop_duplicates(subset=['statement'], inplace=True)

# 4. Convert all text to string
df['statement'] = df['statement'].astype(str)

# 5. Drop outliers
filter_outliers = df['statement'].str.contains('[a-zA-Z]')
df = df[filter_outliers]

print(f"Data cleaning completed")
print(f"Original shape: {initial_shape} -> Shape after cleaning: {df.shape}")
print(f"Number of dropping: {initial_shape[0] - df.shape[0]}")
print(df.tail())