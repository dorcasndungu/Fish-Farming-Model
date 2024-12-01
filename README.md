# Fish-Farming-Model: Predicting Fish Growth and Water Quality for Kenyan Fish Farmers

## Overview
The goal of this project is to build a predictive model that helps Kenyan fish farmers improve their livelihoods through optimized fish farming practices. This aligns with **SDG 8 - Decent Work and Economic Growth**, by enabling farmers to enhance fish production. The model predicts fish **length** and **weight** based on water quality parameters such as temperature, turbidity, dissolved oxygen, pH, ammonia, and nitrate levels. Additionally, the model classifies the water quality as **Good, Moderate, or Poor**. The model offers a user-friendly interface for farmers to input real-time data and receive predictions on fish growth and water quality classification.
# Project Proposal - SDG 8: Decent Work and Economic Growth

## 1. SDG Selection
For this project, we focused on **SDG 8 - Decent Work and Economic Growth**, which aims to promote inclusive and sustainable economic growth, employment, and decent work for all. This SDG is crucial for fish farmers, as sustainable aquaculture practices can provide increased income opportunities and improve livelihoods, contributing to economic growth in local communities.

## 2. Specific Problem Addressed
The specific problem addressed in this project is improving **fish farming conditions** by predicting the ideal environmental factors (temperature, dissolved oxygen, pH, turbidity, ammonia, and nitrate) to maximize fish growth. Fish farming plays an important role in food production and income generation, and optimizing farming conditions is key to enhancing productivity.

Our goal is to develop a **machine learning model** that can be used by **automated pond management systems** to recommend actions for maintaining water quality. This can help fish farmers improve fish production, thereby increasing their economic returns and contributing to **decent work** and **economic growth**.

## 3. Planned ML Application
The ML application will use historical environmental data to predict fish growth metrics such as **fish length** and **fish weight** based on water quality parameters. These predictions will assist fish farmers in making data-driven decisions on how to adjust conditions for optimal fish farming. Additionally, the model will classify the water quality into three categories: **Good**, **Moderate**, and **Poor**, providing fish farmers with actionable insights on how to improve water quality and, in turn, fish production.

The model will also be integrated with a **UI interface**, where fish farmers can input water quality parameters manually to receive real-time predictions for fish growth (length and weight) and water quality classification. This allows farmers to interact with the system easily and make informed decisions on-site.
## 4. Dataset Information
The datasets used in this project are generated from **freshwater aquaponics catfish ponds** equipped with various water quality sensors. The datasets were collected automatically at 5-second intervals from June to mid-October 2021. The dataset includes the number of fish in the pond, along with the target variables: fish length (cm) and weight (g). All attributes are continuous, and the model uses water quality parameters (temperature, turbidity, dissolved oxygen, pH, ammonia, nitrate) for prediction.
##  5. Overview of ML Techniques
We selected **Random Forest Regressor** models to predict both fish length and fish weight based on various environmental factors. Additionally, a **Multi-Output Random Forest Regressor** model will be used to predict both target variables simultaneously. These models were chosen due to their ability to handle non-linear relationships and provide feature importance insights, which are valuable in identifying the most significant environmental factors for fish growth.

For the water quality classification, we used a **Random Forest Classifier** to categorize the water quality into three levels: **Good**, **Moderate**, and **Poor**. This model provides a simple yet effective way to assist farmers in managing their ponds based on real-time water conditions.

## 6. Tasks Assigned to Members
- **Dorcas Mwihaki**: Conducted initial research on SDG 8 and the specific problem of optimizing fish farming conditions. Coordinated the dataset and model research.
- **Lee Ngari**: Handled data preprocessing, and feature engineering, ensuring the data was clean and ready for model training. Led the evaluation of model performance.

## 7. UI Interface for Manual Predictions
A user-friendly **UI interface** will be developed to allow fish farmers to input water quality parameters manually and receive predictions for both fish growth (length and weight) and water quality classification (Good, Moderate, Poor). This will empower farmers to make quick, informed decisions about their farming practices without needing technical expertise.

![WhatsApp Image 2024-11-29 at 10 05 45 PM](https://github.com/user-attachments/assets/f8a216c8-d4e1-4186-98c9-2d1abf268ba0)

## Design and Development
This will involve **data preprocessing and initial model training**
**Importing libraries, loading and exploring the dataset**
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
```

```
# Load the dataset
data = pd.read_csv('IoTPond8.csv')
```

```
# General Dataset Info
print("Dataset Info:")
print(data.info())
data.describe()
```
**Data cleaning**
Drop the unnamed column that is full of missing values since its information is unclear.
Check duplicates
Set unrealistic values within the range.

```
# Check Missing Values
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)

#drop three columns unnamed 11, 12, 13
data = data.drop(columns=['Unnamed: 11'])

# Check Duplicates
duplicates = data.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicates}")
```
```
# Define realistic ranges for each column
reasonable_ranges = {
    'Temperature (C)': (0, 40),
    'Turbidity(NTU)': (0, 100),
    'Dissolved Oxygen (mg/L)': (0, 14),
    'PH': (0, 14),
    'Ammonia (mg/L)': (0, 1),
    'Nitrate (mg/L)': (0, 0.1),
    'Total_length (cm)': (0, 100),
    'Weight (g)': (0, 2000)
}

for col, (min_val, max_val) in reasonable_ranges.items():
    if col in data.columns:
        # Clip values to the range
        data[col] = data[col].clip(lower=min_val, upper=max_val)

```
```
# Confirm if there are still missing values.
missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values)
```
```
#Extract date and time from the data
# Data Type Conversions 
data['created_at'] = pd.to_datetime(data['created_at'], errors='coerce')

#  Extract Date and Time Components
data['Date'] = data['created_at'].dt.date
data['Time'] = data['created_at'].dt.time
data['Hour'] = data['created_at'].dt.hour
data['Day'] = data['created_at'].dt.day
data['Month'] = data['created_at'].dt.month
```
```
# Visualize the data
# 1. Distribution of Numerical Variables
numerical_cols = ['Temperature (C)', 'Turbidity(NTU)', 'Dissolved Oxygen (mg/L)',
                  'PH', 'Ammonia (mg/L)', 'Nitrate (mg/L)', 'Total_length (cm)', 'Weight (g)']

for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[col], kde=True, bins=30, color='blue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# 2. Correlation Heatmap
#Shows the relationship between the attributes
plt.figure(figsize=(10, 8))
correlation_matrix = data[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
```
**Observations from Data Visualization**

**Water Quality Parameters:**

Temperature: The temperature remained stable around 24-25°C, which is suitable for fish farming.

Dissolved Oxygen: Levels were consistent, primarily ranging from 13 to 15 mg/L, ensuring healthy aquatic conditions for fish.

pH: The pH values were slightly alkaline (8.2-8.4), which is ideal for species like tilapia.

Turbidity: There was some fluctuation in turbidity, with most values falling below 100 NTU, indicating clear water, though higher values may indicate occasional disturbances.

**Fish Metrics:**

No significant relationship was observed between temperature or dissolved oxygen and fish length.
A slight positive correlation was observed between pH and fish weight, suggesting better growth under slightly alkaline conditions.
The ammonia levels did not show a clear pattern with fish weight, though ammonia is generally a critical factor for fish health.

**Trends Over Time:**

Temperature remained relatively stable over time, with only minor fluctuations.
Dissolved Oxygen, pH, and Turbidity displayed minor fluctuations, with dissolved oxygen levels remaining adequate for the health of the fish.

```
# Boxplots for Outliers
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=data[col], color='lightblue')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)
    plt.show()
```
# Model Training

```
X = data[['Temperature (C)', 'Turbidity(NTU)', 'Dissolved Oxygen (mg/L)',
                  'PH', 'Ammonia (mg/L)', 'Nitrate (mg/L)']]
y_length = data['Total_length (cm)']
y_weight = data['Weight (g)']

X_train, X_test, y_train_length, y_test_length = train_test_split(X, y_length, test_size=0.2, random_state=42)
_, _, y_train_weight, y_test_weight = train_test_split(X, y_weight, test_size=0.2, random_state=42)
```
```
#feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
Ensures all attributes contribute equally to the model.
```
# Build Random Forest Regressor for Fish Length Prediction
length_model = RandomForestRegressor(n_estimators=100, random_state=42)
length_model.fit(X_train_scaled, y_train_length)

# Predictions for Fish Length
y_pred_length = length_model.predict(X_test_scaled)

# Evaluate Fish Length Model
print("\nFish Length Prediction:")
print(f"Mean Squared Error: {mean_squared_error(y_test_length, y_pred_length):.2f}")
print(f"R2 Score: {r2_score(y_test_length, y_pred_length):.2f}")

# Build Random Forest Regressor for Fish Weight Prediction
weight_model = RandomForestRegressor(n_estimators=100, random_state=42)
weight_model.fit(X_train_scaled, y_train_weight)

# Predictions for Fish Weight
y_pred_weight = weight_model.predict(X_test_scaled)

# Evaluate Fish Weight Model
print("\nFish Weight Prediction:")
print(f"Mean Squared Error: {mean_squared_error(y_test_weight, y_pred_weight):.2f}")
print(f"R2 Score: {r2_score(y_test_weight, y_pred_weight):.2f}")
```
The MSE of 0.20 indicates very small error between predicted and actual fish lengths, meaning the model's predictions are very close to the actual values.


The R² score of 1.00 suggests that the model explains 100% of the variance in fish length, indicating perfect prediction. This implies that the model has almost no error in predicting fish length, which is an excellent result.

The R² score of 0.99 indicates that the model explains 99% of the variance in fish weight, suggesting that the model is very effective at predicting weight, with a small amount of error.

```
# Combine Feature Importance
feature_importance_length = pd.Series(length_model.feature_importances_, index=X.columns)
feature_importance_weight = pd.Series(weight_model.feature_importances_, index=X.columns)

# Generate x-axis positions
x = np.arange(len(feature_importance_length))

# Plot side-by-side bars
plt.figure(figsize=(12, 6))
bar_width = 0.4  # Width of each bar

plt.bar(x - bar_width / 2, feature_importance_length.values, width=bar_width, color="blue", alpha=0.7, label="Fish Length")
plt.bar(x + bar_width / 2, feature_importance_weight.values, width=bar_width, color="orange", alpha=0.7, label="Fish Weight")

# Customize the plot
plt.xticks(x, feature_importance_length.index, rotation=45)
plt.title("Feature Importance for Fish Growth Prediction")
plt.ylabel("Importance Score")
plt.legend()
plt.tight_layout()
plt.show()
```
PH had the most  impact on the model from the feature importance
**MULTI OUTPUT RANDOM FOREST**

```
# Target Variables (combined)
y = data[['Total_length (cm)', 'Weight (g)']]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Multi-Output Random Forest Regressor
multi_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
multi_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = multi_model.predict(X_test_scaled)

# Evaluate Performance for Each Target
print("\nFish Length Prediction:")
print(f"Mean Squared Error: {mean_squared_error(y_test['Total_length (cm)'], y_pred[:, 0]):.2f}")
print(f"R2 Score: {r2_score(y_test['Total_length (cm)'], y_pred[:, 0]):.2f}")

print("\nFish Weight Prediction:")
print(f"Mean Squared Error: {mean_squared_error(y_test['Weight (g)'], y_pred[:, 1]):.2f}")
print(f"R2 Score: {r2_score(y_test['Weight (g)'], y_pred[:, 1]):.2f}")

```
## WATER QUALITY
```
# Logic for defining water quality
def classify_water_quality(row):
    if (18 <= row['Temperature (C)'] <= 26 and
        row['Dissolved Oxygen (mg/L)'] > 5 and
        6.5 <= row['PH'] <= 8.5 and
        row['Ammonia (mg/L)'] < 0.05 and
        row['Nitrate (mg/L)'] < 0.1 and
        row['Turbidity(NTU)'] < 10):
        return 'Good'
    elif ((15 <= row['Temperature (C)'] <= 18 or 26 < row['Temperature (C)'] <= 30) or
          (3 <= row['Dissolved Oxygen (mg/L)'] <= 5) or
          (5.5 <= row['PH'] < 6.5 or 8.5 < row['PH'] <= 9) or
          (0.05 <= row['Ammonia (mg/L)'] <= 0.1) or
          (0.1 <= row['Nitrate (mg/L)'] <= 0.5) or
          (10 <= row['Turbidity(NTU)'] <= 50)):
        return 'Moderate'
    else:
        return 'Poor'


data['Water_Quality'] = data.apply(classify_water_quality, axis=1)
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Step 1: Define Features and Target
X = data[['Temperature (C)', 'Dissolved Oxygen (mg/L)', 'PH', 'Ammonia (mg/L)', 'Nitrate (mg/L)', 'Turbidity(NTU)']]
y = data['Water_Quality']

# Step 2: Encode Target Labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 4: Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Extract Feature Importances
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Step 6: Visualize Feature Importances
plt.figure(figsize=(8, 6))
plt.bar(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance for Water Quality Classification')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Step 7: Print Feature Importances
print("\nFeature Importances:")
print(feature_importances)
```
```
# Step 8: Model Evaluation
# Predictions
y_pred = clf.predict(X_test)
```
```
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```
```
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()
```
![pred](https://github.com/user-attachments/assets/eae58273-cc9c-4966-b164-58adb812f47d)

The Confusion Matrix shows that most predictions were accurate.
