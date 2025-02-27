import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap

def charts(visuals):
    # Load the data from the CSV files
    file_paths = {
        "Control": "whole_image_features/Control_features.csv",
        "Phenotype_1": "whole_image_features/Phenotype 1_features.csv",
        "Phenotype_2": "whole_image_features/Phenotype 2_features.csv",
        "Images": "whole_image_features/Images_features.csv"
    }

    # Check if all files exist
    for name, path in file_paths.items():
        if not os.path.exists(path):
            print(f"Warning: {path} does not exist. Skipping...")
            # Continue with files that do exist rather than returning
            file_paths.pop(name)

    # If no files exist, return early
    if not file_paths:
        print("No CSV files found. Exiting...")
        return

    # Read the CSV files
    dataframes = {}
    for name, path in file_paths.items():
        try:
            df = pd.read_csv(path)

            # Process branch_lengths and projection_lengths columns
            if 'branch_lengths' in df.columns:
                # Convert string representation of lists to actual lists
                df['branch_lengths'] = df['branch_lengths'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                # Calculate mean of each list
                df['branch_lengths_mean'] = df['branch_lengths'].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else 0)

            if 'projection_lengths' in df.columns:
                # Convert string representation of lists to actual lists
                df['projection_lengths'] = df['projection_lengths'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                # Calculate mean of each list
                df['projection_lengths_mean'] = df['projection_lengths'].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else 0)

            # Drop non-numeric columns that aren't needed for analysis
            columns_to_drop = ['file_name', 'object_label']

            # Also drop original branch_lengths and projection_lengths as they're now represented as mean values
            if 'branch_lengths' in df.columns:
                columns_to_drop.append('branch_lengths')
            if 'projection_lengths' in df.columns:
                columns_to_drop.append('projection_lengths')

            # Drop columns that exist
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

            # Store the processed dataframe
            dataframes[name] = df

        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue  # Continue with other files instead of returning

    # Process each DataFrame to calculate statistics for all data
    data_stats = {}

    for name, df in dataframes.items():
        if df.empty:
            print(f"Warning: {name} dataframe is empty")
            continue

        # Calculate statistics for numeric columns only
        numeric_df = df.select_dtypes(include=['number'])
        if numeric_df.empty:
            print(f"Warning: No numeric columns in {name} dataframe")
            continue

        # Calculate statistics for all values in each column (no filtering)
        column_stats = {}

        for col in numeric_df.columns:
            # Get all values for this column
            values = numeric_df[col].dropna()

            if not values.empty:
                # Calculate mean and standard deviation
                column_stats[col] = (values.mean(), values.std(ddof=1))
            else:
                column_stats[col] = (0, 0)

        data_stats[name] = column_stats

    # Prepare data for plotting
    features = set()
    for name in dataframes:
        if name in data_stats:
            features.update(data_stats[name].keys())

    # Create the 'charts' directory if it does not exist
    charts_dir = "charts"
    os.makedirs(charts_dir, exist_ok=True)

    # Plot each feature
    for feature in features:
        plt.figure(figsize=(10, 6))

        # Set positions for bars
        index = np.arange(len(dataframes))

        # Collect data for this feature
        means = []
        stds = []

        for name in dataframes:
            if name in data_stats and feature in data_stats[name]:
                mean, std = data_stats[name][feature]
            else:
                mean, std = 0, 0

            means.append(mean)
            stds.append(std)

        # Create the bar plot for all data
        plt.bar(index, means, 0.6, alpha=0.8, color='blue')

        # Add error bars that don't go below zero
        for i, (mean, std) in enumerate(zip(means, stds)):
            lower_error = min(mean, std)  # Limit lower error to the mean value (avoid negative)
            upper_error = std
            plt.errorbar(index[i], mean, yerr=[[lower_error], [upper_error]],
                        fmt='none', ecolor='black', capsize=5)

        # Add labels and title
        plt.xlabel('Condition')
        plt.ylabel('Value')
        plt.title(f'Feature: {feature}')
        plt.xticks(index, data_stats.keys())

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(charts_dir, f"{feature}.png"))

        # Show or close the plot
        if visuals:
            plt.show()
        else:
            plt.close()

    print(f"Generated charts in {charts_dir} directory")
