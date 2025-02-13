from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Reload the newly uploaded datasets
updated_file_paths = {
    "Phenotype_2": "extracted_features/Phenotype 2_features.csv",
    "Phenotype_1": "extracted_features/Phenotype 1_features.csv",
    "Control": "extracted_features/Control_features.csv",
}

# Load the new datasets
updated_dataframes = {key: pd.read_csv(path) for key, path in updated_file_paths.items()}

# Assign labels to each dataset
updated_dataframes["Phenotype_1"]["label"] = 1
updated_dataframes["Phenotype_2"]["label"] = 2
updated_dataframes["Control"]["label"] = 0

# Combine datasets
df_updated_combined = pd.concat(updated_dataframes.values(), ignore_index=True)

# Drop non-numeric column (if exists)
if "image_filename" in df_updated_combined.columns:
    df_updated_combined = df_updated_combined.drop(columns=["image_filename"])

# Separate features and target label
X_updated = df_updated_combined.drop(columns=["label"])
y_updated = df_updated_combined["label"]

# Train Random Forest model with updated features
rf_classifier_updated = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42)
rf_classifier_updated.fit(X_updated, y_updated)

# Compute feature importance
feature_importances_updated = rf_classifier_updated.feature_importances_

# Create a DataFrame for visualization
importance_df_updated = pd.DataFrame({
    "Feature": X_updated.columns,
    "Importance": feature_importances_updated
}).sort_values(by="Importance", ascending=False)

# Plot feature importances
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.barh(importance_df_updated["Feature"], importance_df_updated["Importance"], align="center")
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Updated Random Forest Classifier")
plt.gca().invert_yaxis()
plt.show()

# Display feature importance values
importance_df_updated
