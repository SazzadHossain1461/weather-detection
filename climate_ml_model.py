import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('Temp_and_rain.csv')

print("=" * 60)
print("CLIMATE DATA ANALYSIS & ML MODEL")
print("=" * 60)
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nBasic Statistics:\n{df.describe()}")

# ===================== DATA PREPROCESSING =====================
print("\n" + "=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Check for missing values
print(f"\nMissing Values:\n{df.isnull().sum()}")

# Create additional temporal features
df['Year_Month'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
df['Season'] = df['Month'].apply(lambda x: 
    'Winter' if x in [12, 1, 2] else
    'Spring' if x in [3, 4, 5] else
    'Summer' if x in [6, 7, 8] else 'Fall')

# Encode season
season_map = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
df['Season_Encoded'] = df['Season'].map(season_map)

# Create lagged features (previous month's temperature and rainfall)
df['Temp_Lag1'] = df['tem'].shift(1)
df['Rain_Lag1'] = df['rain'].shift(1)

# Drop rows with NaN values (from lagged features)
df = df.dropna()

print(f"\nDataset after feature engineering: {df.shape}")
print(f"\nNew features created: Year_Month, Season, Season_Encoded, Temp_Lag1, Rain_Lag1")

# ===================== EXPLORATORY DATA ANALYSIS =====================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Correlation analysis
correlation = df[['tem', 'rain', 'Month', 'Season_Encoded', 'Temp_Lag1', 'Rain_Lag1']].corr()
print(f"\nCorrelation Matrix:\n{correlation}")

# Temperature and rainfall by month
monthly_stats = df.groupby('Month').agg({
    'tem': ['mean', 'std', 'min', 'max'],
    'rain': ['mean', 'std', 'min', 'max']
}).round(2)
print(f"\nMonthly Statistics:\n{monthly_stats}")

# ===================== MODEL BUILDING =====================
print("\n" + "=" * 60)
print("MODEL BUILDING & TRAINING")
print("=" * 60)

# Prepare features and targets
X = df[['Month', 'Season_Encoded', 'Temp_Lag1', 'Rain_Lag1']]
y_temp = df['tem']
y_rain = df['rain']

# Split data (80-20)
X_train, X_test, y_temp_train, y_temp_test = train_test_split(
    X, y_temp, test_size=0.2, random_state=42
)
_, _, y_rain_train, y_rain_test = train_test_split(
    X, y_rain, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# ===================== TEMPERATURE PREDICTION =====================
print("\n" + "-" * 60)
print("TEMPERATURE PREDICTION MODELS")
print("-" * 60)

models_temp = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results_temp = {}

for name, model in models_temp.items():
    # Train
    model.fit(X_train_scaled, y_temp_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    mse = mean_squared_error(y_temp_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_temp_test, y_pred)
    r2 = r2_score(y_temp_test, y_pred)
    
    results_temp[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f}°C")
    print(f"  MAE:  {mae:.4f}°C")
    print(f"  R²:   {r2:.4f}")

# ===================== RAINFALL PREDICTION =====================
print("\n" + "-" * 60)
print("RAINFALL PREDICTION MODELS")
print("-" * 60)

models_rain = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results_rain = {}
best_model_rain = None
best_score = -float('inf')

for name, model in models_rain.items():
    # Train
    model.fit(X_train_scaled, y_rain_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    mse = mean_squared_error(y_rain_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_rain_test, y_pred)
    r2 = r2_score(y_rain_test, y_pred)
    
    results_rain[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    
    if r2 > best_score:
        best_score = r2
        best_model_rain = (name, model)
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:.4f} mm")
    print(f"  MAE:  {mae:.4f} mm")
    print(f"  R²:   {r2:.4f}")

# ===================== FEATURE IMPORTANCE =====================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

rf_temp = models_temp['Random Forest']
rf_rain = models_rain['Random Forest']

feature_names = ['Month', 'Season', 'Temp_Lag1', 'Rain_Lag1']

print("\nTemperature Model - Feature Importance:")
for name, importance in zip(feature_names, rf_temp.feature_importances_):
    print(f"  {name}: {importance:.4f}")

print("\nRainfall Model - Feature Importance:")
for name, importance in zip(feature_names, rf_rain.feature_importances_):
    print(f"  {name}: {importance:.4f}")

# ===================== SAMPLE PREDICTIONS =====================
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS (First 10 Test Samples)")
print("=" * 60)

best_temp_model = models_temp['Random Forest']
y_pred_temp = best_temp_model.predict(X_test_scaled)
y_pred_rain = best_model_rain[1].predict(X_test_scaled)

sample_df = pd.DataFrame({
    'Actual_Temp': y_temp_test.values[:10],
    'Pred_Temp': y_pred_temp[:10],
    'Actual_Rain': y_rain_test.values[:10],
    'Pred_Rain': y_pred_rain[:10],
    'Temp_Error': np.abs(y_temp_test.values[:10] - y_pred_temp[:10]),
    'Rain_Error': np.abs(y_rain_test.values[:10] - y_pred_rain[:10])
})

print("\n" + sample_df.to_string(index=False))

# ===================== FUTURE PREDICTIONS =====================
print("\n" + "=" * 60)
print("FUTURE PREDICTIONS (Next 6 Months)")
print("=" * 60)

# Get last row for lagged features
last_temp = df['tem'].iloc[-1]
last_rain = df['rain'].iloc[-1]

future_predictions = []
for month in range(1, 7):
    # Create feature vector
    features = np.array([[
        month,
        season_map['Winter' if month in [12, 1, 2] else 'Spring' if month in [3, 4, 5] else 'Summer' if month in [6, 7, 8] else 'Fall'],
        last_temp,
        last_rain
    ]])
    
    features_scaled = scaler.transform(features)
    
    pred_temp = best_temp_model.predict(features_scaled)[0]
    pred_rain = best_model_rain[1].predict(features_scaled)[0]
    
    future_predictions.append({
        'Month': month,
        'Predicted_Temp': pred_temp,
        'Predicted_Rain': pred_rain
    })
    
    # Update last values for next iteration
    last_temp = pred_temp
    last_rain = pred_rain

future_df = pd.DataFrame(future_predictions)
print("\n" + future_df.to_string(index=False))

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)