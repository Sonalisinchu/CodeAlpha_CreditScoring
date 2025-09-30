# ===============================
# Credit Score Project: German Credit Scoring
# Full Step-by-Step Code with EDA, Visualization, and Excel Output
# ===============================

# Step 1: Import Libraries
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ===============================
# Step 2: Load Dataset
dataset_path = r"C:\Users\LENOVO\OneDrive\Desktop\CodeAlpha_CreditScoring\data\german.csv"
df = pd.read_csv(dataset_path, sep=';')
df.columns = df.columns.str.strip()  # remove extra spaces
print("Dataset loaded successfully!\n")
print(df.head())

# ===============================
# Step 3: Exploratory Data Analysis (EDA)
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Create output folder for plots
os.makedirs(r"C:\Users\LENOVO\OneDrive\Desktop\CodeAlpha_CreditScoring\outputs\plots", exist_ok=True)

# Count plot for target
plt.figure(figsize=(6,4))
sns.countplot(x='Creditability', data=df)
plt.title('Creditability Distribution (0 = Bad, 1 = Good)')
plt.savefig(r"C:\Users\LENOVO\OneDrive\Desktop\CodeAlpha_CreditScoring\outputs\plots\creditability_distribution.png")
plt.show()

# Correlation heatmap (numeric features)
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig(r"C:\Users\LENOVO\OneDrive\Desktop\CodeAlpha_CreditScoring\outputs\plots\correlation_heatmap.png")
plt.show()

# Histograms for numeric features
df.hist(figsize=(12,10), bins=15)
plt.tight_layout()
plt.savefig(r"C:\Users\LENOVO\OneDrive\Desktop\CodeAlpha_CreditScoring\outputs\plots\numeric_feature_histograms.png")
plt.show()

# ===============================
# Step 4: Preprocessing

# Categorical columns
categorical_cols = [
    'Account_Balance',
    'Payment_Status_of_Previous_Credit',
    'Purpose',
    'Value_Savings_Stocks',
    'Length_of_current_employment',
    'Sex_Marital_Status',
    'Guarantors',
    'Most_valuable_available_asset',
    'Type_of_apartment',
    'Occupation',
    'Telephone',
    'Foreign_Worker'
]

# Numeric columns
numeric_cols = [
    'Duration_of_Credit_monthly',
    'Credit_Amount',
    'Instalment_per_cent',
    'Duration_in_Current_address',
    'Age_years',
    'Concurrent_Credits',
    'No_of_Credits_at_this_Bank',
    'No_of_dependents'
]

# Target column
target_col = 'Creditability'

# Encode categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("\nCategorical features encoded!")

# Scale numeric features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print("Numeric features scaled!")

# ===============================
# Step 5: Split Features and Target
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Data split into train/test sets!")

# ===============================
# Step 6: Train & Evaluate Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

best_model_name = None
best_accuracy = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n===== {name} =====")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Track best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# ===============================
# Step 7: Save Best Model
models_folder = r"C:\Users\LENOVO\OneDrive\Desktop\CodeAlpha_CreditScoring\models"
os.makedirs(models_folder, exist_ok=True)
model_path = os.path.join(models_folder, f"{best_model_name.replace(' ', '_')}_model.pkl")
joblib.dump(best_model, model_path)
print(f"\nBest model ({best_model_name}) saved at {model_path}!")

# ===============================
# Step 8: Save Predictions and Metrics to Excel
outputs_folder = r"C:\Users\LENOVO\OneDrive\Desktop\CodeAlpha_CreditScoring\outputs"
os.makedirs(outputs_folder, exist_ok=True)

# Predictions
output_df = X_test.copy()
output_df['Actual'] = y_test
output_df['Predicted'] = best_model.predict(X_test)
predictions_path = os.path.join(outputs_folder, "credit_predictions.xlsx")
output_df.to_excel(predictions_path, index=False)
print(f"Predictions saved to {predictions_path}!")

# Metrics
report_dict = classification_report(y_test, output_df['Predicted'], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
metrics_path = os.path.join(outputs_folder, "credit_metrics.xlsx")
report_df.to_excel(metrics_path, index=True)
print(f"Evaluation metrics saved to {metrics_path}!")

# ===============================
# Step 9: Show Example Predictions
print("\nExample predictions on first 5 test samples:")
example_pred = best_model.predict(X_test.iloc[:5])
print("Predictions:", example_pred)
print("Actual:", y_test.iloc[:5].values)

# ===============================
# Step 10: Print Best Model Result Clearly
print("\n===============================")
print(f"Best Model: {best_model_name}")
print(f"Best Model Accuracy: {best_accuracy:.2f}")
print("===============================")
