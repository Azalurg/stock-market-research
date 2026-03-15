import os
import pandas as pd

# Directory containing the CSV files
csv_directory = '../data'

# List all CSV files in directory
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Create empty DataFrame to store combined data
combined_df = pd.DataFrame()

# Load each file, select Date and Close columns, rename Close to filename, and merge
for file in csv_files:
    file_path = os.path.join(csv_directory, file)
    df = pd.read_csv(file_path, usecols=['Date', 'Close'])
    df.rename(columns={'Close': file}, inplace=True)
    if combined_df.empty:
        combined_df = df
    else:
        combined_df = pd.merge(combined_df, df, on='Date', how='outer')

# Fill missing values with 0.0
combined_df = combined_df.fillna(0.0)

cols_to_round = combined_df.columns.difference(['Date'])
combined_df[cols_to_round] = combined_df[cols_to_round].round(3)

# Save combined DataFrame to new CSV
combined_df.to_csv('combined.csv', index=False)