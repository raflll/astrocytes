import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os

def load_data():
    """
    Load data from CSV files in the extracted_features directory.
    Returns a dictionary of dataframes and their labels.
    """
    dataframes = {}
    
    # Find all feature CSV files in the extracted_features directory
    features_dir = "extracted_features"
    if not os.path.exists(features_dir):
        print(f"Warning: {features_dir} directory does not exist!")
        return None

    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(features_dir) if f.endswith('_features.csv')]
    if not csv_files:
        print("No feature CSV files found!")
        return None

    # Load each CSV file and assign labels
    for i, csv_file in enumerate(csv_files):
        csv_path = os.path.join(features_dir, csv_file)
        try:
            df = pd.read_csv(csv_path)
            # Extract folder name from CSV filename (remove '_features.csv')
            folder_name = csv_file[:-13]  # Remove '_features.csv'
            dataframes[folder_name] = df
            dataframes[folder_name]["label"] = i
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            continue

    return dataframes

def preprocess_data(df):
    # First, remove non-numeric columns and save label column
    X = df.drop(columns=["file_name", "object_label", "label"])

    # Drop branch-related features and roundness
    columns_to_drop = [
        'branch_lengths', 'num_branches', 'total_branch_length',
        'avg_branch_length', 'branch_density', 'roundness'
    ]
    X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])

    # Handle list columns by converting them to string length (if they exist)
    for col in X.columns:
        if X[col].apply(lambda x: isinstance(x, list)).any():
            X[col] = X[col].apply(lambda x: len(x) if isinstance(x, list) else x)

    # Ensure all columns are numeric
    numeric_cols = X.select_dtypes(include=['number']).columns
    X = X[numeric_cols]

    # Get the target variable
    y = df["label"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply log transformation to numeric columns (prevents errors with string columns)
    X_train_log = np.log1p(X_train)
    X_test_log = np.log1p(X_test)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_log)
    X_test_scaled = scaler.transform(X_test_log)

    return X, y, X_train_scaled, X_test_scaled, y_train, y_test, X.columns

# trains selected model and returns: accuracy, f1 score, regular feature importance, and shap values
def train_and_evaluate(model_name, X_train, X_test, y_train, y_test, feature_names):
    if model_name == "LR":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Ridge":
        model = RidgeClassifier(alpha=1)
    elif model_name == "ENet":
        model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.3, max_iter=10000)
    elif model_name == "RF":
        model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42, oob_score=True)
    else:
        return "Undefined model", 0, 0, None, None

    # NOTE: these were the columns to drop based on Ethan's binarize (only applicable to Ridge and Logistic Regression)
    if model_name == "Ridge":
        drop_cols = ["num_edges", "total_branch_length", "avg_branch_length", "avg_projection_length", "length_width_ratio"]
    elif model_name == "LR":
        drop_cols = ["num_edges", "total_branch_length", "avg_branch_length", "avg_projection_length", "length_width_ratio", "circularity"]
    else:
        drop_cols = []

    # Filter out columns that don't exist in the dataset
    drop_cols = [col for col in drop_cols if col in feature_names]

    # Create DataFrames for SHAP analysis
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Drop specified columns
    X_train_filtered = X_train_df.drop(columns=drop_cols, errors='ignore')
    X_test_filtered = X_test_df.drop(columns=drop_cols, errors='ignore')

    # Fit the model
    model.fit(X_train_filtered, y_train)
    y_pred = model.predict(X_test_filtered)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Get feature importance based on model type
    if hasattr(model, "coef_"):
        if len(model.coef_.shape) > 1 and model.coef_.shape[0] > 1:
            # For multi-class models
            feature_importance = np.mean(np.abs(model.coef_), axis=0)
        else:
            # For binary classification
            feature_importance = model.coef_[0] if model.coef_.shape[0] > 1 else model.coef_
    else:
        feature_importance = model.feature_importances_

    try:
        # Compute SHAP values based on model type
        if isinstance(model, RandomForestClassifier):
            # TreeExplainer for Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_filtered)
        else:
            # KernelExplainer for linear models
            # Limit to 100 background samples for performance
            background_samples = shap.sample(X_train_filtered, min(100, len(X_train_filtered)))

            if hasattr(model, 'decision_function'):
                explainer = shap.KernelExplainer(model.decision_function, background_samples)
            else:
                # For multi-class models
                explainer = shap.KernelExplainer(model.predict_proba, background_samples)

            shap_values = explainer.shap_values(X_test_filtered.iloc[:100])  # Limit test samples for performance
    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
        shap_values = None

    return accuracy, f1, feature_importance, shap_values

# Computes VIF (Variance Inflation Factor)
def compute_vif(X):
    # Only compute VIF for numeric columns
    X_numeric = X.select_dtypes(include=['number'])

    # Drop columns with zero variance
    variance = X_numeric.var()
    cols_to_drop = variance[variance <= 1e-10].index.tolist()
    X_filtered = X_numeric.drop(columns=cols_to_drop)

    # Compute VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_filtered.columns

    try:
        vif_data["VIF"] = [variance_inflation_factor(X_filtered.values, i)
                           for i in range(X_filtered.shape[1])]
    except Exception as e:
        print(f"VIF calculation failed: {str(e)}")
        vif_data["VIF"] = np.nan

    return vif_data

def train_model(model_name, visuals):
    dataframes = load_data()

    if dataframes is None:
        print("Dataframes not loaded properly. Exiting train_model.")
        return

    df = pd.concat(dataframes.values(), ignore_index=True)
    df = df[df["file_name"].str.contains("-ch2")]
    # df = df[df["num_projections"] != 0] # optional, filtering out no projections

    X, y, X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

    # Convert X_train to DataFrame for VIF calculation
    X_train_df = pd.DataFrame(X_train, columns=feature_names)

    try:
        vif_df = compute_vif(X_train_df)
        vif_df = vif_df.sort_values(by="VIF", ascending=False)
        print("Variance Inflation Factors:")
        print(vif_df)
    except Exception as e:
        print(f"Error computing VIF: {str(e)}")

    try:
        print(f"\nTraining {model_name} model...")
        accuracy, f1, feature_importance, shap_values = train_and_evaluate(
            model_name, X_train, X_test, y_train, y_test, feature_names)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": feature_importance
        }).sort_values(by="Importance", ascending=False)

        print("\nTop 10 important features:")
        print(importance_df.head(10))

        # Plotting feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df["Feature"].head(15), importance_df["Importance"].head(15), align="center")
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title(f"Feature Importance - {model_name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{model_name}_feature_importance.png")
        if visuals: plt.show()

        # Plot SHAP values if available
        if shap_values is not None:
            if isinstance(shap_values, list):
                # For multi-class models
                for i, class_values in enumerate(shap_values):
                    plt.figure(figsize=(10, 8))
                    shap.summary_plot(class_values, pd.DataFrame(X_test, columns=feature_names),
                                    show=False, plot_size=(10, 8))
                    plt.title(f"SHAP values for class {i}")
                    plt.tight_layout()
                    plt.savefig(f"{model_name}_shap_class_{i}.png")
                    if visuals: plt.show()
            else:
                # For binary classification or regression
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, pd.DataFrame(X_test, columns=feature_names),
                                show=False, plot_size=(10, 8))
                plt.title(f"SHAP values for {model_name}")
                plt.tight_layout()
                plt.savefig(f"{model_name}_shap.png")
                if visuals: plt.show()
    except Exception as e:
        print(f"Error during model training and evaluation: {str(e)}")
