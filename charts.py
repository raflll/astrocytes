import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import shap

# Force matplotlib to use a thread-safe backend from the very beginning
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that's thread-safe
import matplotlib.pyplot as plt

# Disable interactive mode completely
plt.ioff()

def charts(visuals=False):
    """
    Generate colorful charts from feature CSV files.
    This function is designed to be thread-safe and avoid matplotlib GUI issues.

    Args:
        visuals (bool): Deprecated parameter, kept for backwards compatibility.
                        No interactive visualization in threaded environment.

    Returns:
        str: Path to the directory where charts are saved.
    """
    try:
        # Load the data from the CSV files
        file_paths = {
            "Control": "whole_image_features/Control_features.csv",
            "Phenotype_1": "whole_image_features/Phenotype 1_features.csv",
            "Phenotype_2": "whole_image_features/Phenotype 2_features.csv",
            "Images": "whole_image_features/Images_features.csv"
        }

        # Define a colorful palette for the bars
        colors = ['#FF5733', '#33FF57', '#3357FF', '#F033FF']  # Vibrant colors

        # Create the 'charts' directory if it does not exist
        charts_dir = "charts"
        os.makedirs(charts_dir, exist_ok=True)

        # Check if all files exist
        valid_paths = {}
        for name, path in file_paths.items():
            if os.path.exists(path):
                valid_paths[name] = path
            else:
                print(f"Warning: {path} does not exist. Skipping...")

        # If no files exist, return early
        if not valid_paths:
            print("No CSV files found. Exiting...")
            return charts_dir

        # Read the CSV files
        dataframes = {}
        for name, path in valid_paths.items():
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

        # Set a colorful style for matplotlib
        plt.style.use('ggplot')

        # Process each feature one by one
        for feature in features:
            # Explicitly create a new figure for each feature to avoid any state leakage
            fig = plt.figure(figsize=(14, 8))

            try:
                # Create a gradient background
                ax = plt.gca()
                ax.set_facecolor('#f8f9fa')  # Light background color

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

                # Calculate y-axis limits to avoid squished appearance
                if means:
                    ymax = max(means) * 1.25  # Allow 25% headroom above the tallest bar
                    plt.ylim(0, ymax)

                # Create the bar plot with different colors for each bar
                bars = plt.bar(index, means, 0.6, alpha=0.85,
                            color=colors[:len(dataframes)],
                            edgecolor='black', linewidth=1.2)

                # Add colorful error bars
                for i, (mean, std) in enumerate(zip(means, stds)):
                    lower_error = min(mean, std)  # Limit lower error to the mean value (avoid negative)
                    upper_error = std
                    plt.errorbar(index[i], mean, yerr=[[lower_error], [upper_error]],
                                fmt='none', ecolor='#404040', capsize=6, capthick=2,
                                elinewidth=2)

                # Add value labels inside the bars at the bottom
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., 0.01,
                            f'{height:.2f}', ha='center', va='bottom',
                            fontweight='bold', fontsize=10, color='black')

                # Add a grid for better readability with custom style
                plt.grid(axis='y', linestyle='--', alpha=0.7)

                # Add labels and title with improved styling
                plt.xlabel('Condition', fontsize=12, fontweight='bold')
                plt.ylabel('Value', fontsize=12, fontweight='bold')
                plt.title(f'Feature: {feature}', fontsize=16, fontweight='bold', pad=20)
                plt.xticks(index, data_stats.keys(), fontsize=11, rotation=0)

                # Add a subtle box around the plot
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                for spine in ax.spines.values():
                    spine.set_color('#888888')
                    spine.set_linewidth(1)

                # Ensure there's enough room for the title and labels
                plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])

                # Save the plot with higher DPI for better quality
                chart_path = os.path.join(charts_dir, f"{feature}.png")
                plt.savefig(chart_path, dpi=150)
                print(f"Created chart: {chart_path}")

                # Always explicitly close the figure to release memory
                plt.close(fig)

            except Exception as e:
                print(f"Error creating chart for {feature}: {e}")
                # Always close the figure, even in case of error
                plt.close(fig)
                continue

        print(f"Generated colorful charts in {charts_dir} directory")
        return charts_dir

    except Exception as e:
        print(f"An error occurred during chart generation: {e}")
        # Make sure we close any open plots in case of exception
        plt.close('all')
        return "charts"  # Return the default directory even if an error occurred
