import pandas as pd
import os
import numpy as np

# Force matplotlib to use a thread-safe backend from the very beginning
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend that's thread-safe
import matplotlib.pyplot as plt

# Disable interactive mode completely
plt.ioff()

def charts(visuals=False):
    """
    Generate charts with a style similar to plot.py from feature CSV files.
    This function is designed to be thread-safe and avoid matplotlib GUI issues.

    Args:
        visuals (bool): Deprecated parameter, kept for backwards compatibility.
                        No interactive visualization in threaded environment.

    Returns:
        str: Path to the directory where charts are saved.
    """
    try:
        # Create the 'charts' directory if it does not exist
        charts_dir = "charts"
        os.makedirs(charts_dir, exist_ok=True)

        # Find all feature CSV files in the extracted_features directory
        features_dir = "extracted_features"
        if not os.path.exists(features_dir):
            print(f"Warning: {features_dir} directory does not exist. Exiting...")
            return charts_dir

        # Get all CSV files in the directory
        csv_files = [f for f in os.listdir(features_dir) if f.endswith('_features.csv')]
        if not csv_files:
            print("No feature CSV files found. Exiting...")
            return charts_dir

        # Create a dictionary mapping folder names to their CSV paths
        file_paths = {}
        for csv_file in csv_files:
            # Extract folder name from CSV filename (remove '_features.csv')
            folder_name = csv_file[:-13]  # Remove '_features.csv'
            # Standardize folder name by replacing spaces with underscores
            folder_name = folder_name.replace(' ', '_')
            file_paths[folder_name] = os.path.join(features_dir, csv_file)

        # Read the CSV files
        dataframes = {}
        for name, path in file_paths.items():
            try:
                df = pd.read_csv(path)

                # Drop branch-related features and roundness
                columns_to_drop = [
                    'branch_lengths', 'num_branches', 'total_branch_length',
                    'avg_branch_length', 'branch_density', 'roundness'
                ]
                df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

                # Process projection_lengths column
                if 'projection_lengths' in df.columns:
                    # Convert string representation of lists to actual lists
                    df['projection_lengths'] = df['projection_lengths'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                    # Calculate mean of each list
                    df['projection_lengths_mean'] = df['projection_lengths'].apply(lambda x: np.mean(x) if isinstance(x, list) and len(x) > 0 else 0)
                    # Drop the original projection_lengths column
                    df = df.drop(columns=['projection_lengths'])

                # Drop non-numeric columns that aren't needed for analysis
                columns_to_drop = ['file_name', 'object_label']
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
                    column_stats[col] = {
                        'mean': values.mean(),
                        'std': values.std(ddof=1),
                        'sem': values.std(ddof=1) / np.sqrt(len(values)),
                        'values': values.tolist()  # Store all individual values for scatter plots
                    }
                else:
                    column_stats[col] = {
                        'mean': 0,
                        'std': 0,
                        'sem': 0,
                        'values': []
                    }

            data_stats[name] = column_stats

        # Prepare data for plotting
        features = set()
        for name in dataframes:
            if name in data_stats:
                features.update(data_stats[name].keys())

        # Set matplotlib style to be minimalist like in plot.py
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial']
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False

        # Process each feature one by one
        for feature in features:
            # Explicitly create a new figure for each feature to avoid any state leakage
            fig, ax = plt.subplots(figsize=(6, 5))

            try:
                # Set background color to white
                ax.set_facecolor('white')
                fig.patch.set_facecolor('white')

                # Set positions for the groups with some spacing
                num_groups = len(dataframes)
                positions = np.linspace(0.2, 0.2 + (num_groups-1)*0.5, num_groups)
                width = 0.2

                # For jitter in scatter plot
                jitter_amount = 0.08
                np.random.seed(42)  # For reproducibility

                # For tracking y max value to place significance markers
                y_max = 0

                # Colors for each condition (alternating between white and black for scatter points)
                scatter_colors = ['white', 'black', 'white', 'black']
                scatter_edge_colors = ['black', 'black', 'black', 'black']

                # Plot each group's data
                for i, (name, stats) in enumerate(data_stats.items()):
                    if feature in stats:
                        feat_stats = stats[feature]
                        mean_val = feat_stats['mean']
                        sem_val = feat_stats['sem']
                        values = feat_stats['values']

                        # Update y_max
                        if values:
                            y_max = max(y_max, max(values))

                        # Create jittered x positions
                        jittered_x = np.ones(len(values)) * positions[i] + np.random.uniform(-jitter_amount, jitter_amount, len(values))

                        # Plot individual data points with jitter
                        scatter_color_idx = min(i, len(scatter_colors)-1)
                        edge_color_idx = min(i, len(scatter_edge_colors)-1)
                        ax.scatter(jittered_x, values, color=scatter_colors[scatter_color_idx],
                                  edgecolor=scatter_edge_colors[edge_color_idx], s=50, zorder=1)

                        # Plot mean lines (blue) with higher zorder to appear in front
                        ax.plot([positions[i] - width/2, positions[i] + width/2],
                               [mean_val, mean_val], color='blue', linewidth=2, zorder=3)

                        # Plot error bars (SEM) with higher zorder to appear in front
                        ax.errorbar(positions[i], mean_val, yerr=sem_val, color='blue',
                                   linewidth=2, capsize=4, capthick=2, zorder=3)

                # Add significance markers if we have more than one group
                if len(data_stats) >= 2:
                    # Add some padding to the top for the significance marker
                    y_pos = y_max * 1.15

                    # Draw the line connecting the groups and the star
                    ax.plot([positions[0], positions[-1]], [y_pos, y_pos], color='black', linewidth=1)

                    # If there are more than 2 groups, place star in middle, otherwise above the two
                    middle_pos = (positions[0] + positions[-1]) / 2
                    ax.text(middle_pos, y_pos * 1.05, '*', ha='center', va='center', fontsize=14)

                # Set axis limits with room for significance markers
                y_padding = y_max * 0.3  # 30% padding at top
                ax.set_ylim(0, y_max + y_padding)

                # Set nice y-axis ticks (4-5 ticks)
                max_y_rounded = np.ceil(y_max * 4) / 4  # Round to nearest 0.25
                ax.set_yticks(np.linspace(0, max_y_rounded, 4))
                ax.set_yticklabels([f'{x:.2f}' for x in np.linspace(0, max_y_rounded, 4)])

                # Set x-axis parameters
                ax.set_xticks(positions)
                ax.set_xticklabels(list(data_stats.keys()), rotation=45)
                ax.set_xlim(min(positions) - 0.3, max(positions) + 0.3)

                # Set axis labels
                ax.set_ylabel(f'{feature}', fontsize=12)

                # Make only left and bottom spines visible
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['left'].set_linewidth(1)
                ax.spines['bottom'].set_linewidth(1)

                # Ensure there's enough room for the title and labels
                plt.tight_layout()

                # Save the plot
                chart_path = os.path.join(charts_dir, f"{feature}.png")
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                print(f"Created chart: {chart_path}")

                # Always explicitly close the figure to release memory
                plt.close(fig)

            except Exception as e:
                print(f"Error creating chart for {feature}: {e}")
                # Always close the figure, even in case of error
                plt.close(fig)
                continue

        print(f"Generated charts in {charts_dir} directory")
        return charts_dir

    except Exception as e:
        print(f"An error occurred during chart generation: {e}")
        # Make sure we close any open plots in case of exception
        plt.close('all')
        return "charts"  # Return the default directory even if an error occurred
