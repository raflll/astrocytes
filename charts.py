import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap

def charts(visuals):
    # Load the data
    file_paths = {
        "Control": "extracted_features/Control_features.csv",
        "Phenotype_1": "extracted_features/Phenotype 1_features.csv",
        "Phenotype_2": "extracted_features/Phenotype 2_features.csv"
    }

    # Reload the data
    dataframes = {name: pd.read_csv(path) for name, path in file_paths.items()}

    # Remove 'image_filename' column if it exists
    for name in dataframes:
        if 'image_filename' in dataframes[name].columns:
            dataframes[name] = dataframes[name].drop(columns=['image_filename'])

    # Process each DataFrame
    high_res_data = {}
    low_res_data = {}

    for name, df in dataframes.items():
        means = df.mean(numeric_only=True)  # Compute mean for each feature
        high_res = df[df > means]  # Values above mean (high resolution)
        low_res = df[df <= means]  # Values below mean (low resolution)

        # Compute average and standard deviation for both high and low resolution values
        high_res_data[name] = (high_res.mean(numeric_only=True), high_res.std(numeric_only=True))
        low_res_data[name] = (low_res.mean(numeric_only=True), low_res.std(numeric_only=True))

    # Convert the processed data into DataFrames for visualization
    high_res_df = pd.DataFrame({k: v[0] for k, v in high_res_data.items()})
    high_res_std = pd.DataFrame({k: v[1] for k, v in high_res_data.items()})

    low_res_df = pd.DataFrame({k: v[0] for k, v in low_res_data.items()})
    low_res_std = pd.DataFrame({k: v[1] for k, v in low_res_data.items()})

    # Create the 'charts' directory if it does not exist
    charts_dir = "charts"
    os.makedirs(charts_dir, exist_ok=True)

    # Save each feature plot to the 'charts' directory
    for feature in high_res_df.index:
        plt.figure(figsize=(8, 6))

        # Plot mean values with error bars representing standard deviation
        plt.bar(high_res_df.columns, high_res_df.loc[feature], yerr=high_res_std.loc[feature], capsize=5, label="High Res", alpha=0.7, color='blue')
        plt.bar(low_res_df.columns, low_res_df.loc[feature], yerr=low_res_std.loc[feature], capsize=5, label="Low Res", alpha=0.7, color='orange')

        plt.title(f"Feature: {feature}")
        plt.ylabel("Average Value")
        plt.xlabel("Condition")
        plt.legend()
        plt.xticks(rotation=45)

        # Save the plot
        plot_path = os.path.join(charts_dir, f"{feature}.png")
        plt.savefig(plot_path)

        # Show plot if visuals
        if visuals: plt.show()
        else: plt.close()

    # Confirm the saved files
    os.listdir(charts_dir)

