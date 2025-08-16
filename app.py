# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import joblib
# # from sklearn.preprocessing import MinMaxScaler
# # from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# # import math
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from io import BytesIO
# # import warnings
# # warnings.filterwarnings('ignore')

# # # Set page config
# # st.set_page_config(
# #     page_title="ML Model Robustness Testing",
# #     page_icon="🧪",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # Custom CSS
# # st.markdown("""
# # <style>
# #     .main-header {
# #         font-size: 2.5rem;
# #         color: #1f77b4;
# #         text-align: center;
# #         margin-bottom: 2rem;
# #     }
# #     .sub-header {
# #         font-size: 1.5rem;
# #         color: #2c3e50;
# #         margin: 1rem 0;
# #     }
# #     .metric-card {
# #         background-color: #f8f9fa;
# #         padding: 1rem;
# #         border-radius: 0.5rem;
# #         border-left: 4px solid #1f77b4;
# #         margin: 0.5rem 0;
# #     }
# #     .pass-status {
# #         color: #28a745;
# #         font-weight: bold;
# #     }
# #     .fail-status {
# #         color: #dc3545;
# #         font-weight: bold;
# #     }
# # </style>
# # """, unsafe_allow_html=True)

# # # LSTM Model Class
# # class GlucoseLSTM(nn.Module):
# #     def __init__(self, input_dim, hidden_dim=64):
# #         super(GlucoseLSTM, self).__init__()
# #         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
# #         self.fc = nn.Linear(hidden_dim, 1)
    
# #     def forward(self, x):
# #         _, (h_n, _) = self.lstm(x)
# #         return self.fc(h_n[-1])

# # # Helper Functions
# # def create_sequences(data, seq_length):
# #     X, y = [], []
# #     for i in range(len(data) - seq_length):
# #         X.append(data[i:i+seq_length])
# #         y.append(data[i+seq_length][0])  # glucose
# #     return np.array(X), np.array(y)

# # def load_and_preprocess_timeseries(df):
# #     """Preprocess time series data"""
# #     df['time'] = pd.to_datetime(df['time'])
# #     df = df.sort_values('time').reset_index(drop=True)
    
# #     features = ['glucose', 'calories', 'heart_rate', 'steps',
# #                 'basal_rate', 'bolus_volume_delivered', 'carb_input']
    
# #     scaler = MinMaxScaler()
# #     scaled_data = scaler.fit_transform(df[features])
    
# #     sequence_length = 10
# #     X, y = create_sequences(scaled_data, sequence_length)
# #     split = int(0.7 * len(X))
# #     X_train, X_test = X[:split], X[split:]
# #     y_train, y_test = y[:split], y[split:]
    
# #     return X_train, X_test, y_train, y_test, features, scaler

# # def evaluate_timeseries_scenario(model, X, y, modify_fn=None):
# #     """Evaluate time series model with scenario"""
# #     X_mod = modify_fn(X) if modify_fn else X
# #     with torch.no_grad():
# #         preds = model(torch.tensor(X_mod, dtype=torch.float32)).squeeze().numpy()
# #     mse = mean_squared_error(y, preds)
# #     rmse = math.sqrt(mse)
# #     mae = mean_absolute_error(y, preds)
# #     return mse, rmse, mae

# # def evaluate_regression_scenario(model, X_mod, y_true):
# #     """Evaluate regression model with scenario"""
# #     preds = model.predict(X_mod)
# #     mse = mean_squared_error(y_true, preds)
# #     rmse = np.sqrt(mse)
# #     r2 = r2_score(y_true, preds)
# #     return mse, rmse, r2

# # # Main App
# # def main():
# #     st.markdown('<h1 class="main-header">🧪 ML Model Robustness Testing Suite</h1>', unsafe_allow_html=True)
    
# #     # Sidebar
# #     st.sidebar.title("🔧 Configuration")
# #     model_type = st.sidebar.selectbox("Select Model Type", ["Time Series (LSTM)", "Regression (MLR)"])
    
# #     if model_type == "Time Series (LSTM)":
# #         timeseries_interface()
# #     else:
# #         regression_interface()

# # def timeseries_interface():
# #     st.markdown('<h2 class="sub-header">📈 Time Series LSTM Model Testing</h2>', unsafe_allow_html=True)
    
# #     col1, col2 = st.columns([1, 1])
    
# #     with col1:
# #         st.subheader("📁 Upload Model & Dataset")
# #         model_file = st.file_uploader("Upload LSTM Model (.pth)", type=['pth'], key="lstm_model")
# #         data_file = st.file_uploader("Upload Dataset (.csv)", type=['csv'], key="lstm_data")
    
# #     with col2:
# #         st.subheader("⚙️ Test Configuration")
# #         threshold_pct = st.slider("Performance Degradation Threshold (%)", 5, 50, 20)
        
# #         # Scenario parameters
# #         st.markdown("**Scenario Parameters:**")
# #         noise_low = st.slider("Low Noise Level", 0.001, 0.1, 0.01, step=0.001)
# #         noise_high = st.slider("High Noise Level", 0.01, 0.2, 0.05, step=0.01)
# #         spike_value = st.slider("Glucose Spike Value", 0.05, 0.5, 0.1, step=0.05)
# #         heart_rate_increase = st.slider("Heart Rate Increase (%)", 10, 50, 20, step=5)
# #         adversarial_shift = st.slider("Adversarial Shift", 0.01, 0.2, 0.05, step=0.01)
# #         calorie_increase = st.slider("Calorie Increase", 0.1, 0.5, 0.2, step=0.1)
    
# #     if model_file and data_file:
# #         try:
# #             # Load data
# #             df = pd.read_csv(data_file, delimiter=';')
# #             X_train, X_test, y_train, y_test, features, scaler = load_and_preprocess_timeseries(df)
            
# #             # Load model
# #             input_dim = X_train.shape[2]
# #             model = GlucoseLSTM(input_dim)
# #             model.load_state_dict(torch.load(model_file, map_location='cpu'))
# #             model.eval()
            
# #             st.success("✅ Model and dataset loaded successfully!")
            
# #             # Display data info
# #             col1, col2, col3, col4 = st.columns(4)
# #             with col1:
# #                 st.metric("Training Samples", X_train.shape[0])
# #             with col2:
# #                 st.metric("Test Samples", X_test.shape[0])
# #             with col3:
# #                 st.metric("Features", len(features))
# #             with col4:
# #                 st.metric("Sequence Length", X_train.shape[1])
            
# #             if st.button("🚀 Run Robustness Tests", key="run_lstm_tests"):
# #                 with st.spinner("Running robustness tests..."):
# #                     # Define scenarios with user parameters
# #                     scenarios = [
# #                         ("Baseline", None),
# #                         ("Gaussian noise (low)", lambda X: X + np.random.normal(0, noise_low, X.shape)),
# #                         ("Gaussian noise (high)", lambda X: X + np.random.normal(0, noise_high, X.shape)),
# #                         ("Zero-out steps feature", lambda X: np.where(
# #                             np.arange(X.shape[2]) == features.index('steps'), 0, X)),
# #                         ("Drop carb_input values", lambda X: np.where(
# #                             np.arange(X.shape[2]) == features.index('carb_input'), 0, X)),
# #                         ("Increase heart rate by {}%".format(heart_rate_increase), 
# #                          lambda X: X * (1 + (np.arange(X.shape[2]) == features.index('heart_rate')) * (heart_rate_increase/100))),
# #                         ("Sudden spike in glucose (+{})".format(spike_value), 
# #                          lambda X: np.where(np.arange(X.shape[2]) == features.index('glucose'), X + spike_value, X)),
# #                         ("Simulate missing basal_rate (set to mean)", 
# #                          lambda X: np.where(np.arange(X.shape[2]) == features.index('basal_rate'), 
# #                                            np.mean(X[:,:,features.index('basal_rate')]), X)),
# #                         ("Adversarial shift: all features +{}".format(adversarial_shift), 
# #                          lambda X: X + adversarial_shift),
# #                         ("Extreme calorie values (+{})".format(calorie_increase), 
# #                          lambda X: np.where(np.arange(X.shape[2]) == features.index('calories'), X + calorie_increase, X))
# #                     ]
                    
# #                     # Run scenarios
# #                     results = []
# #                     baseline_mse, baseline_rmse, baseline_mae = evaluate_timeseries_scenario(model, X_test, y_test, None)
                    
# #                     for name, mod_fn in scenarios:
# #                         mse, rmse, mae = evaluate_timeseries_scenario(model, X_test, y_test, mod_fn)
# #                         delta_percent = ((rmse - baseline_rmse) / baseline_rmse) * 100
# #                         status = "PASS" if abs(delta_percent) <= threshold_pct else "FAIL"
# #                         results.append({
# #                             "Scenario": name,
# #                             "MSE": f"{mse:.6e}",
# #                             "RMSE": f"{rmse:.6f}",
# #                             "MAE": f"{mae:.6f}",
# #                             "Delta %": f"{delta_percent:.2f}",
# #                             "Status": status
# #                         })
                    
# #                     # Display results
# #                     st.markdown('<h3 class="sub-header">📊 Test Results</h3>', unsafe_allow_html=True)
                    
# #                     df_results = pd.DataFrame(results)
                    
# #                     # Color code the results
# #                     def highlight_status(val):
# #                         if val == "PASS":
# #                             return 'background-color: #d4edda; color: #155724'
# #                         elif val == "FAIL":
# #                             return 'background-color: #f8d7da; color: #721c24'
# #                         return ''
                    
# #                     styled_df = df_results.style.applymap(highlight_status, subset=['Status'])
# #                     st.dataframe(styled_df, use_container_width=True)
                    
# #                     # Summary metrics
# #                     passed_tests = sum(1 for r in results if r["Status"] == "PASS")
# #                     total_tests = len(results)
                    
# #                     col1, col2, col3 = st.columns(3)
# #                     with col1:
# #                         st.metric("Total Tests", total_tests)
# #                     with col2:
# #                         st.metric("Passed Tests", passed_tests, f"{passed_tests}/{total_tests}")
# #                     with col3:
# #                         pass_rate = (passed_tests / total_tests) * 100
# #                         st.metric("Pass Rate", f"{pass_rate:.1f}%")
                    
# #                     # Visualization
# #                     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
# #                     # RMSE comparison
# #                     scenarios_names = [r["Scenario"] for r in results]
# #                     rmse_values = [float(r["RMSE"]) for r in results]
# #                     colors = ['green' if r["Status"] == "PASS" else 'red' for r in results]
                    
# #                     ax1.barh(scenarios_names, rmse_values, color=colors, alpha=0.7)
# #                     ax1.axvline(baseline_rmse, color='blue', linestyle='--', label='Baseline RMSE')
# #                     ax1.set_xlabel('RMSE')
# #                     ax1.set_title('RMSE by Scenario')
# #                     ax1.legend()
                    
# #                     # Pass/Fail pie chart
# #                     pass_fail_counts = df_results['Status'].value_counts()
# #                     ax2.pie(pass_fail_counts.values, labels=pass_fail_counts.index, autopct='%1.1f%%',
# #                            colors=['lightgreen', 'lightcoral'])
# #                     ax2.set_title('Test Results Distribution')
                    
# #                     st.pyplot(fig)
        
# #         except Exception as e:
# #             st.error(f"❌ Error: {str(e)}")

# # def regression_interface():
# #     st.markdown('<h2 class="sub-header">📊 Regression Model Testing</h2>', unsafe_allow_html=True)
    
# #     col1, col2 = st.columns([1, 1])
    
# #     with col1:
# #         st.subheader("📁 Upload Model & Dataset")
# #         model_file = st.file_uploader("Upload Regression Model (.pkl)", type=['pkl'], key="reg_model")
# #         data_file = st.file_uploader("Upload Dataset (.xlsx or .csv)", type=['xlsx', 'csv'], key="reg_data")
    
# #     with col2:
# #         st.subheader("⚙️ Test Configuration")
# #         threshold_pct = st.slider("Performance Degradation Threshold (%)", 5, 50, 10)
        
# #         # Scenario parameters
# #         st.markdown("**Scenario Parameters:**")
# #         noise_low = st.slider("Low Noise Standard Deviation", 0.01, 0.2, 0.05, step=0.01)
# #         noise_high = st.slider("High Noise Standard Deviation", 0.1, 1.0, 0.2, step=0.1)
# #         drift_up_pct = st.slider("Upward Feature Drift (%)", 5, 30, 10, step=5)
# #         drift_down_pct = st.slider("Downward Feature Drift (%)", 5, 30, 10, step=5)
# #         missing_low_pct = st.slider("Low Missing Values (%)", 5, 25, 10, step=5)
# #         missing_high_pct = st.slider("High Missing Values (%)", 20, 50, 30, step=5)
# #         outlier_pct = st.slider("Outlier Injection (%)", 1, 10, 5, step=1)
# #         outlier_multiplier = st.slider("Outlier Multiplier", 2, 20, 10, step=1)
# #         scaling_multiplier = st.slider("Feature Scaling Multiplier", 10, 1000, 100, step=10)
    
# #     if model_file and data_file:
# #         try:
# #             # Load model
# #             model = joblib.load(model_file)
            
# #             # Load data
# #             if data_file.name.endswith('.xlsx'):
# #                 df = pd.read_excel(data_file)
# #             else:
# #                 df = pd.read_csv(data_file)
            
# #             # Clean column names
# #             df.columns = df.columns.str.strip()
            
# #             # Auto-detect target column (assuming it's the last column or contains 'strength', 'target', etc.)
# #             target_candidates = [col for col in df.columns if any(word in col.lower() 
# #                                for word in ['strength', 'target', 'compressive', 'mpa'])]
            
# #             if target_candidates:
# #                 target_column = st.selectbox("Select Target Column", target_candidates, index=0)
# #             else:
# #                 target_column = st.selectbox("Select Target Column", df.columns, index=-1)
            
# #             X = df.drop(columns=[target_column])
# #             y = df[target_column]
            
# #             # Train-test split
# #             from sklearn.model_selection import train_test_split
# #             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
# #             st.success("✅ Model and dataset loaded successfully!")
            
# #             # Display data info
# #             col1, col2, col3, col4 = st.columns(4)
# #             with col1:
# #                 st.metric("Training Samples", X_train.shape[0])
# #             with col2:
# #                 st.metric("Test Samples", X_test.shape[0])
# #             with col3:
# #                 st.metric("Features", X.shape[1])
# #             with col4:
# #                 st.metric("Missing Values", df.isnull().sum().sum())
            
# #             # Show feature importance if available
# #             if hasattr(model, 'coef_'):
# #                 st.subheader("📈 Feature Coefficients")
# #                 coef_df = pd.DataFrame({
# #                     "Feature": X.columns,
# #                     "Coefficient": model.coef_
# #                 })
# #                 coef_df['Abs_Coefficient'] = abs(coef_df['Coefficient'])
# #                 coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
                
# #                 fig, ax = plt.subplots(figsize=(10, 6))
# #                 ax.barh(coef_df['Feature'], coef_df['Coefficient'])
# #                 ax.set_xlabel('Coefficient Value')
# #                 ax.set_title('Feature Coefficients')
# #                 st.pyplot(fig)
            
# #             if st.button("🚀 Run Robustness Tests", key="run_reg_tests"):
# #                 with st.spinner("Running robustness tests..."):
# #                     results = []
                    
# #                     # Baseline
# #                     baseline_mse, baseline_rmse, baseline_r2 = evaluate_regression_scenario(model, X_test, y_test)
# #                     results.append({
# #                         "Scenario": "Baseline",
# #                         "MSE": f"{baseline_mse:.6e}",
# #                         "RMSE": f"{baseline_rmse:.6f}",
# #                         "R² Score": f"{baseline_r2:.6f}",
# #                         "Delta %": "0.00",
# #                         "Status": "BASELINE"
# #                     })
                    
# #                     # Test scenarios with user parameters
# #                     test_scenarios = [
# #                         ("Gaussian noise (low)", lambda X: X + np.random.normal(0, noise_low, X.shape)),
# #                         ("Gaussian noise (high)", lambda X: X + np.random.normal(0, noise_high, X.shape)),
# #                         (f"Feature drift (+{drift_up_pct}%)", lambda X: X * (1 + drift_up_pct/100)),
# #                         (f"Feature drift (-{drift_down_pct}%)", lambda X: X * (1 - drift_down_pct/100)),
# #                     ]
                    
# #                     # Missing values scenarios
# #                     for pct, name in [(missing_low_pct, "low"), (missing_high_pct, "high")]:
# #                         def create_missing_scenario(percentage):
# #                             def scenario_fn(X):
# #                                 X_missing = X.copy()
# #                                 mask = np.random.rand(*X_missing.shape) < (percentage/100)
# #                                 X_missing[mask] = np.nan
# #                                 return X_missing.fillna(X.mean())
# #                             return scenario_fn
# #                         test_scenarios.append((f"{pct}% missing values ({name})", create_missing_scenario(pct)))
                    
# #                     # Outlier injection
# #                     def outlier_scenario(X):
# #                         X_outliers = X.copy()
# #                         n_outliers = int((outlier_pct/100) * len(X_outliers))
# #                         rows = np.random.choice(X_outliers.index, n_outliers, replace=False)
# #                         X_outliers.loc[rows] = X_outliers.loc[rows] * outlier_multiplier
# #                         return X_outliers
# #                     test_scenarios.append((f"{outlier_pct}% outliers (×{outlier_multiplier})", outlier_scenario))
                    
# #                     # Most important feature missing
# #                     if hasattr(model, 'coef_'):
# #                         important_feature = coef_df.sort_values('Abs_Coefficient', ascending=False).iloc[0]["Feature"]
# #                         def important_feature_scenario(X):
# #                             X_single_missing = X.copy()
# #                             X_single_missing[important_feature] = X_single_missing[important_feature].mean()
# #                             return X_single_missing
# #                         test_scenarios.append((f"Missing feature: {important_feature}", important_feature_scenario))
                    
# #                     # Feature scaling shift
# #                     def scaling_scenario(X):
# #                         X_scaled = X.copy()
# #                         X_scaled.iloc[:, 0] = X_scaled.iloc[:, 0] * scaling_multiplier
# #                         return X_scaled
# #                     test_scenarios.append((f"Feature scaling shift (×{scaling_multiplier})", scaling_scenario))
                    
# #                     # Run all scenarios
# #                     for name, scenario_fn in test_scenarios:
# #                         try:
# #                             X_modified = scenario_fn(X_test)
# #                             mse, rmse, r2 = evaluate_regression_scenario(model, X_modified, y_test)
# #                             delta = ((mse - baseline_mse) / baseline_mse) * 100
# #                             status = "PASS" if delta <= threshold_pct else "FAIL"
                            
# #                             results.append({
# #                                 "Scenario": name,
# #                                 "MSE": f"{mse:.6e}",
# #                                 "RMSE": f"{rmse:.6f}",
# #                                 "R² Score": f"{r2:.6f}",
# #                                 "Delta %": f"{delta:.2f}",
# #                                 "Status": status
# #                             })
# #                         except Exception as e:
# #                             st.warning(f"⚠️ Error in scenario '{name}': {str(e)}")
                    
# #                     # Display results
# #                     st.markdown('<h3 class="sub-header">📊 Test Results</h3>', unsafe_allow_html=True)
                    
# #                     df_results = pd.DataFrame(results)
                    
# #                     # Color code the results
# #                     def highlight_status(val):
# #                         if val == "PASS":
# #                             return 'background-color: #d4edda; color: #155724'
# #                         elif val == "FAIL":
# #                             return 'background-color: #f8d7da; color: #721c24'
# #                         elif val == "BASELINE":
# #                             return 'background-color: #cce5ff; color: #004085'
# #                         return ''
                    
# #                     styled_df = df_results.style.applymap(highlight_status, subset=['Status'])
# #                     st.dataframe(styled_df, use_container_width=True)
                    
# #                     # Summary metrics (excluding baseline)
# #                     test_results = [r for r in results if r["Status"] != "BASELINE"]
# #                     passed_tests = sum(1 for r in test_results if r["Status"] == "PASS")
# #                     total_tests = len(test_results)
                    
# #                     col1, col2, col3 = st.columns(3)
# #                     with col1:
# #                         st.metric("Total Tests", total_tests)
# #                     with col2:
# #                         st.metric("Passed Tests", passed_tests, f"{passed_tests}/{total_tests}")
# #                     with col3:
# #                         if total_tests > 0:
# #                             pass_rate = (passed_tests / total_tests) * 100
# #                             st.metric("Pass Rate", f"{pass_rate:.1f}%")
                    
# #                     # Visualization
# #                     if total_tests > 0:
# #                         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
# #                         # RMSE comparison
# #                         test_names = [r["Scenario"] for r in test_results]
# #                         rmse_values = [float(r["RMSE"]) for r in test_results]
# #                         colors = ['green' if r["Status"] == "PASS" else 'red' for r in test_results]
                        
# #                         ax1.barh(test_names, rmse_values, color=colors, alpha=0.7)
# #                         ax1.axvline(baseline_rmse, color='blue', linestyle='--', label='Baseline RMSE')
# #                         ax1.set_xlabel('RMSE')
# #                         ax1.set_title('RMSE by Scenario')
# #                         ax1.legend()
# #                         plt.setp(ax1.get_yticklabels(), rotation=0, ha="right")
                        
# #                         # Pass/Fail pie chart
# #                         status_counts = pd.Series([r["Status"] for r in test_results]).value_counts()
# #                         if len(status_counts) > 0:
# #                             ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
# #                                    colors=['lightgreen' if x == 'PASS' else 'lightcoral' for x in status_counts.index])
# #                             ax2.set_title('Test Results Distribution')
                        
# #                         st.pyplot(fig)
        
# #         except Exception as e:
# #             st.error(f"❌ Error: {str(e)}")
# #             st.error("Please make sure your model file is compatible and the dataset format is correct.")

# # if __name__ == "__main__":
# #     main()


# import streamlit as st
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import joblib
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import math
# import matplotlib.pyplot as plt
# import seaborn as sns
# from io import BytesIO
# import warnings
# import zipfile
# import os
# import tempfile
# import shutil
# from PIL import Image
# import cv2
# warnings.filterwarnings('ignore')

# # Try importing TensorFlow/Keras
# try:
#     import tensorflow as tf
#     from tensorflow.keras.models import load_model
#     from tensorflow.keras.preprocessing import image
#     from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#     TF_AVAILABLE = True
# except ImportError:
#     TF_AVAILABLE = False
#     st.warning("TensorFlow not available. CNN model testing will be disabled.")

# # Set page config
# st.set_page_config(
#     page_title="ML Model Robustness Testing Suite",
#     page_icon="🧪",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .sub-header {
#         font-size: 1.5rem;
#         color: #2c3e50;
#         margin: 1rem 0;
#     }
#     .metric-card {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 0.5rem;
#         border-left: 4px solid #1f77b4;
#         margin: 0.5rem 0;
#     }
#     .pass-status {
#         color: #28a745;
#         font-weight: bold;
#     }
#     .fail-status {
#         color: #dc3545;
#         font-weight: bold;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Model Classes
# class GlucoseLSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim=64):
#         super(GlucoseLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, 1)
    
#     def forward(self, x):
#         _, (h_n, _) = self.lstm(x)
#         return self.fc(h_n[-1])

# # Helper Functions for Time Series
# def create_sequences(data, seq_length):
#     X, y = [], []
#     for i in range(len(data) - seq_length):
#         X.append(data[i:i+seq_length])
#         y.append(data[i+seq_length][0])  # glucose
#     return np.array(X), np.array(y)

# def load_and_preprocess_timeseries(df):
#     """Preprocess time series data"""
#     df['time'] = pd.to_datetime(df['time'])
#     df = df.sort_values('time').reset_index(drop=True)
    
#     features = ['glucose', 'calories', 'heart_rate', 'steps',
#                 'basal_rate', 'bolus_volume_delivered', 'carb_input']
    
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(df[features])
    
#     sequence_length = 10
#     X, y = create_sequences(scaled_data, sequence_length)
#     split = int(0.7 * len(X))
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]
    
#     return X_train, X_test, y_train, y_test, features, scaler

# def evaluate_timeseries_scenario(model, X, y, modify_fn=None):
#     """Evaluate time series model with scenario"""
#     X_mod = modify_fn(X) if modify_fn else X
#     with torch.no_grad():
#         preds = model(torch.tensor(X_mod, dtype=torch.float32)).squeeze().numpy()
#     mse = mean_squared_error(y, preds)
#     rmse = math.sqrt(mse)
#     mae = mean_absolute_error(y, preds)
#     return mse, rmse, mae

# def evaluate_regression_scenario(model, X_mod, y_true):
#     """Evaluate regression model with scenario"""
#     preds = model.predict(X_mod)
#     mse = mean_squared_error(y_true, preds)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_true, preds)
#     return mse, rmse, r2

# # Helper Functions for CNN
# def extract_dataset_from_zip(zip_file):
#     """Extract dataset from uploaded zip file"""
#     temp_dir = tempfile.mkdtemp()
    
#     with zipfile.ZipFile(zip_file, 'r') as zip_ref:
#         zip_ref.extractall(temp_dir)
    
#     # Find dataset directory with train/test/validation structure
#     dataset_path = None
#     for root, dirs, files in os.walk(temp_dir):
#         if 'train' in dirs and 'test' in dirs and 'validation' in dirs:
#             dataset_path = root
#             break
#         elif any('train' in d for d in dirs) and any('test' in d for d in dirs):
#             dataset_path = root
#             break
    
#     return dataset_path, temp_dir

# def load_images_from_directory(directory_path, target_size=(224, 224), max_images=100):
#     """Load images from directory with their labels"""
#     images = []
#     labels = []
#     class_names = []
    
#     for class_name in os.listdir(directory_path):
#         class_path = os.path.join(directory_path, class_name)
#         if os.path.isdir(class_path):
#             class_names.append(class_name)
#             class_idx = len(class_names) - 1
            
#             image_files = [f for f in os.listdir(class_path) 
#                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
#             # Limit images per class to prevent memory issues
#             for img_file in image_files[:max_images//len(os.listdir(directory_path))]:
#                 img_path = os.path.join(class_path, img_file)
#                 try:
#                     img = Image.open(img_path).convert('RGB')
#                     img = img.resize(target_size)
#                     img_array = np.array(img) / 255.0
#                     images.append(img_array)
#                     labels.append(class_idx)
#                 except Exception as e:
#                     continue
    
#     return np.array(images), np.array(labels), class_names

# def apply_image_augmentation(images, augmentation_type):
#     """Apply various augmentations to images"""
#     augmented_images = images.copy()
    
#     if augmentation_type == "gaussian_noise":
#         noise = np.random.normal(0, 0.05, images.shape)
#         augmented_images = np.clip(images + noise, 0, 1)
    
#     elif augmentation_type == "brightness_increase":
#         augmented_images = np.clip(images * 1.3, 0, 1)
    
#     elif augmentation_type == "brightness_decrease":
#         augmented_images = np.clip(images * 0.7, 0, 1)
    
#     elif augmentation_type == "contrast_change":
#         augmented_images = np.clip((images - 0.5) * 1.5 + 0.5, 0, 1)
    
#     elif augmentation_type == "blur":
#         for i, img in enumerate(images):
#             img_uint8 = (img * 255).astype(np.uint8)
#             blurred = cv2.GaussianBlur(img_uint8, (5, 5), 0)
#             augmented_images[i] = blurred / 255.0
    
#     elif augmentation_type == "horizontal_flip":
#         augmented_images = np.flip(images, axis=2)
    
#     elif augmentation_type == "rotation":
#         # Simple rotation simulation by flipping
#         augmented_images = np.flip(images, axis=1)
    
#     elif augmentation_type == "saturation_change":
#         # Convert to HSV and modify saturation
#         for i, img in enumerate(images):
#             img_uint8 = (img * 255).astype(np.uint8)
#             hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
#             hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
#             rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
#             augmented_images[i] = rgb / 255.0
    
#     return augmented_images

# def evaluate_cnn_scenario(model, images, labels, class_names, augmentation_type=None):
#     """Evaluate CNN model with different scenarios"""
#     if augmentation_type:
#         test_images = apply_image_augmentation(images, augmentation_type)
#     else:
#         test_images = images
    
#     predictions = model.predict(test_images, verbose=0)
#     predicted_classes = np.argmax(predictions, axis=1)
    
#     accuracy = accuracy_score(labels, predicted_classes)
    
#     # Calculate per-class metrics
#     report = classification_report(labels, predicted_classes, 
#                                  target_names=class_names, 
#                                  output_dict=True, 
#                                  zero_division=0)
    
#     return accuracy, report, predicted_classes

# # Main App
# def main():
#     st.markdown('<h1 class="main-header">🧪 ML Model Robustness Testing Suite</h1>', unsafe_allow_html=True)
    
#     # Sidebar
#     st.sidebar.title("🔧 Configuration")
#     model_types = ["Time Series (LSTM)", "Regression (MLR)"]
#     if TF_AVAILABLE:
#         model_types.append("CNN Image Classification")
    
#     model_type = st.sidebar.selectbox("Select Model Type", model_types)
    
#     if model_type == "Time Series (LSTM)":
#         timeseries_interface()
#     elif model_type == "Regression (MLR)":
#         regression_interface()
#     elif model_type == "CNN Image Classification" and TF_AVAILABLE:
#         cnn_interface()

# def timeseries_interface():
#     st.markdown('<h2 class="sub-header">📈 Time Series LSTM Model Testing</h2>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.subheader("📁 Upload Model & Dataset")
#         model_file = st.file_uploader("Upload LSTM Model (.pth)", type=['pth'], key="lstm_model")
#         data_file = st.file_uploader("Upload Dataset (.csv)", type=['csv'], key="lstm_data")
    
#     with col2:
#         st.subheader("⚙️ Test Configuration")
#         threshold_pct = st.slider("Performance Degradation Threshold (%)", 5, 50, 20)
        
#         # Scenario parameters
#         st.markdown("**Scenario Parameters:**")
#         noise_low = st.slider("Low Noise Level", 0.001, 0.1, 0.01, step=0.001)
#         noise_high = st.slider("High Noise Level", 0.01, 0.2, 0.05, step=0.01)
#         spike_value = st.slider("Glucose Spike Value", 0.05, 0.5, 0.1, step=0.05)
#         heart_rate_increase = st.slider("Heart Rate Increase (%)", 10, 50, 20, step=5)
#         adversarial_shift = st.slider("Adversarial Shift", 0.01, 0.2, 0.05, step=0.01)
#         calorie_increase = st.slider("Calorie Increase", 0.1, 0.5, 0.2, step=0.1)
    
#     if model_file and data_file:
#         try:
#             # Load data
#             df = pd.read_csv(data_file, delimiter=';')
#             X_train, X_test, y_train, y_test, features, scaler = load_and_preprocess_timeseries(df)
            
#             # Load model
#             input_dim = X_train.shape[2]
#             model = GlucoseLSTM(input_dim)
#             model.load_state_dict(torch.load(model_file, map_location='cpu'))
#             model.eval()
            
#             st.success("✅ Model and dataset loaded successfully!")
            
#             # Display data info
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Training Samples", X_train.shape[0])
#             with col2:
#                 st.metric("Test Samples", X_test.shape[0])
#             with col3:
#                 st.metric("Features", len(features))
#             with col4:
#                 st.metric("Sequence Length", X_train.shape[1])
            
#             if st.button("🚀 Run Robustness Tests", key="run_lstm_tests"):
#                 with st.spinner("Running robustness tests..."):
#                     # Define scenarios with user parameters
#                     scenarios = [
#                         ("Baseline", None),
#                         ("Gaussian noise (low)", lambda X: X + np.random.normal(0, noise_low, X.shape)),
#                         ("Gaussian noise (high)", lambda X: X + np.random.normal(0, noise_high, X.shape)),
#                         ("Zero-out steps feature", lambda X: np.where(
#                             np.arange(X.shape[2]) == features.index('steps'), 0, X)),
#                         ("Drop carb_input values", lambda X: np.where(
#                             np.arange(X.shape[2]) == features.index('carb_input'), 0, X)),
#                         ("Increase heart rate by {}%".format(heart_rate_increase), 
#                          lambda X: X * (1 + (np.arange(X.shape[2]) == features.index('heart_rate')) * (heart_rate_increase/100))),
#                         ("Sudden spike in glucose (+{})".format(spike_value), 
#                          lambda X: np.where(np.arange(X.shape[2]) == features.index('glucose'), X + spike_value, X)),
#                         ("Simulate missing basal_rate (set to mean)", 
#                          lambda X: np.where(np.arange(X.shape[2]) == features.index('basal_rate'), 
#                                            np.mean(X[:,:,features.index('basal_rate')]), X)),
#                         ("Adversarial shift: all features +{}".format(adversarial_shift), 
#                          lambda X: X + adversarial_shift),
#                         ("Extreme calorie values (+{})".format(calorie_increase), 
#                          lambda X: np.where(np.arange(X.shape[2]) == features.index('calories'), X + calorie_increase, X))
#                     ]
                    
#                     # Run scenarios
#                     results = []
#                     baseline_mse, baseline_rmse, baseline_mae = evaluate_timeseries_scenario(model, X_test, y_test, None)
                    
#                     for name, mod_fn in scenarios:
#                         mse, rmse, mae = evaluate_timeseries_scenario(model, X_test, y_test, mod_fn)
#                         delta_percent = ((rmse - baseline_rmse) / baseline_rmse) * 100
#                         status = "PASS" if abs(delta_percent) <= threshold_pct else "FAIL"
#                         results.append({
#                             "Scenario": name,
#                             "MSE": f"{mse:.6e}",
#                             "RMSE": f"{rmse:.6f}",
#                             "MAE": f"{mae:.6f}",
#                             "Delta %": f"{delta_percent:.2f}",
#                             "Status": status
#                         })
                    
#                     display_results(results, baseline_rmse, "RMSE")
        
#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")

# def regression_interface():
#     st.markdown('<h2 class="sub-header">📊 Regression Model Testing</h2>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.subheader("📁 Upload Model & Dataset")
#         model_file = st.file_uploader("Upload Regression Model (.pkl)", type=['pkl'], key="reg_model")
#         data_file = st.file_uploader("Upload Dataset (.xlsx or .csv)", type=['xlsx', 'csv'], key="reg_data")
    
#     with col2:
#         st.subheader("⚙️ Test Configuration")
#         threshold_pct = st.slider("Performance Degradation Threshold (%)", 5, 50, 10)
        
#         # Scenario parameters
#         st.markdown("**Scenario Parameters:**")
#         noise_low = st.slider("Low Noise Standard Deviation", 0.01, 0.2, 0.05, step=0.01)
#         noise_high = st.slider("High Noise Standard Deviation", 0.1, 1.0, 0.2, step=0.1)
#         drift_up_pct = st.slider("Upward Feature Drift (%)", 5, 30, 10, step=5)
#         drift_down_pct = st.slider("Downward Feature Drift (%)", 5, 30, 10, step=5)
#         missing_low_pct = st.slider("Low Missing Values (%)", 5, 25, 10, step=5)
#         missing_high_pct = st.slider("High Missing Values (%)", 20, 50, 30, step=5)
#         outlier_pct = st.slider("Outlier Injection (%)", 1, 10, 5, step=1)
#         outlier_multiplier = st.slider("Outlier Multiplier", 2, 20, 10, step=1)
#         scaling_multiplier = st.slider("Feature Scaling Multiplier", 10, 1000, 100, step=10)
    
#     if model_file and data_file:
#         try:
#             # Load model
#             model = joblib.load(model_file)
            
#             # Load data
#             if data_file.name.endswith('.xlsx'):
#                 df = pd.read_excel(data_file)
#             else:
#                 df = pd.read_csv(data_file)
            
#             # Clean column names
#             df.columns = df.columns.str.strip()
            
#             # Auto-detect target column
#             target_candidates = [col for col in df.columns if any(word in col.lower() 
#                                for word in ['strength', 'target', 'compressive', 'mpa'])]
            
#             if target_candidates:
#                 target_column = st.selectbox("Select Target Column", target_candidates, index=0)
#             else:
#                 target_column = st.selectbox("Select Target Column", df.columns, index=-1)
            
#             X = df.drop(columns=[target_column])
#             y = df[target_column]
            
#             # Train-test split
#             from sklearn.model_selection import train_test_split
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
#             st.success("✅ Model and dataset loaded successfully!")
            
#             # Display data info
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Training Samples", X_train.shape[0])
#             with col2:
#                 st.metric("Test Samples", X_test.shape[0])
#             with col3:
#                 st.metric("Features", X.shape[1])
#             with col4:
#                 st.metric("Missing Values", df.isnull().sum().sum())
            
#             if st.button("🚀 Run Robustness Tests", key="run_reg_tests"):
#                 with st.spinner("Running robustness tests..."):
#                     results = []
                    
#                     # Baseline
#                     baseline_mse, baseline_rmse, baseline_r2 = evaluate_regression_scenario(model, X_test, y_test)
#                     results.append({
#                         "Scenario": "Baseline",
#                         "MSE": f"{baseline_mse:.6e}",
#                         "RMSE": f"{baseline_rmse:.6f}",
#                         "R² Score": f"{baseline_r2:.6f}",
#                         "Delta %": "0.00",
#                         "Status": "BASELINE"
#                     })
                    
#                     # Test scenarios with user parameters
#                     test_scenarios = [
#                         ("Gaussian noise (low)", lambda X: X + np.random.normal(0, noise_low, X.shape)),
#                         ("Gaussian noise (high)", lambda X: X + np.random.normal(0, noise_high, X.shape)),
#                         (f"Feature drift (+{drift_up_pct}%)", lambda X: X * (1 + drift_up_pct/100)),
#                         (f"Feature drift (-{drift_down_pct}%)", lambda X: X * (1 - drift_down_pct/100)),
#                     ]
                    
#                     # Missing values scenarios
#                     for pct, name in [(missing_low_pct, "low"), (missing_high_pct, "high")]:
#                         def create_missing_scenario(percentage):
#                             def scenario_fn(X):
#                                 X_missing = X.copy()
#                                 mask = np.random.rand(*X_missing.shape) < (percentage/100)
#                                 X_missing[mask] = np.nan
#                                 return X_missing.fillna(X.mean())
#                             return scenario_fn
#                         test_scenarios.append((f"{pct}% missing values ({name})", create_missing_scenario(pct)))
                    
#                     # Outlier injection
#                     def outlier_scenario(X):
#                         X_outliers = X.copy()
#                         n_outliers = int((outlier_pct/100) * len(X_outliers))
#                         rows = np.random.choice(X_outliers.index, n_outliers, replace=False)
#                         X_outliers.loc[rows] = X_outliers.loc[rows] * outlier_multiplier
#                         return X_outliers
#                     test_scenarios.append((f"{outlier_pct}% outliers (×{outlier_multiplier})", outlier_scenario))
                    
#                     # Feature scaling shift
#                     def scaling_scenario(X):
#                         X_scaled = X.copy()
#                         X_scaled.iloc[:, 0] = X_scaled.iloc[:, 0] * scaling_multiplier
#                         return X_scaled
#                     test_scenarios.append((f"Feature scaling shift (×{scaling_multiplier})", scaling_scenario))
                    
#                     # Run all scenarios
#                     for name, scenario_fn in test_scenarios:
#                         try:
#                             X_modified = scenario_fn(X_test)
#                             mse, rmse, r2 = evaluate_regression_scenario(model, X_modified, y_test)
#                             delta = ((mse - baseline_mse) / baseline_mse) * 100
#                             status = "PASS" if delta <= threshold_pct else "FAIL"
                            
#                             results.append({
#                                 "Scenario": name,
#                                 "MSE": f"{mse:.6e}",
#                                 "RMSE": f"{rmse:.6f}",
#                                 "R² Score": f"{r2:.6f}",
#                                 "Delta %": f"{delta:.2f}",
#                                 "Status": status
#                             })
#                         except Exception as e:
#                             st.warning(f"⚠️ Error in scenario '{name}': {str(e)}")
                    
#                     display_results(results, baseline_rmse, "RMSE")
        
#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")

# # def cnn_interface():
# #     st.markdown('<h2 class="sub-header">🖼️ CNN Image Classification Model Testing</h2>', unsafe_allow_html=True)
    
# #     col1, col2 = st.columns([1, 1])
    
# #     with col1:
# #         st.subheader("📁 Upload Model & Dataset")
# #         model_file = st.file_uploader("Upload CNN Model (.h5)", type=['h5'], key="cnn_model")
# #         dataset_zip = st.file_uploader("Upload Dataset (.zip)", type=['zip'], key="cnn_data")
        
# #         st.subheader("📊 Dataset Structure Expected")
# #         st.code("""
# # dataset.zip
# # ├── train/
# # │   ├── class1/
# # │   └── class2/
# # ├── test/
# # │   ├── class1/
# # │   └── class2/
# # └── validation/
# #     ├── class1/
# #     └── class2/
# #         """)
    
# #     with col2:
# #         st.subheader("⚙️ Test Configuration")
# #         threshold_pct = st.slider("Accuracy Drop Threshold (%)", 1, 20, 5)
# #         max_test_images = st.slider("Max Test Images per Class", 10, 100, 50)
        
# #         st.markdown("**Test Scenarios:**")
# #         st.write("1. Gaussian Noise")
# #         st.write("2. Brightness Changes")
# #         st.write("3. Contrast Changes") 
# #         st.write("4. Blur Effect")
# #         st.write("5. Horizontal Flip")
# #         st.write("6. Rotation Simulation")
# #         st.write("7. Saturation Changes")
# #         st.write("8. Combined Augmentations")
# #         st.write("9. Extreme Brightness")
# #         st.write("10. Heavy Blur")
    
# #     if model_file and dataset_zip and TF_AVAILABLE:
# #         try:
# #             # Load model
# #             model = load_model(model_file)
# #             st.success("✅ CNN Model loaded successfully!")
            
# #             # Extract dataset
# #             with st.spinner("Extracting dataset..."):
# #                 dataset_path, temp_dir = extract_dataset_from_zip(dataset_zip)
                
# #                 if not dataset_path:
# #                     st.error("❌ Could not find valid dataset structure in ZIP file")
# #                     return
                
# #                 st.success(f"✅ Dataset extracted to: {dataset_path}")
            
# #             # Load test images
# #             test_path = os.path.join(dataset_path, 'test')
# #             if not os.path.exists(test_path):
# #                 # Try 'validation' if 'test' doesn't exist
# #                 test_path = os.path.join(dataset_path, 'validation')
            
# #             if os.path.exists(test_path):
# #                 with st.spinner("Loading test images..."):
# #                     test_images, test_labels, class_names = load_images_from_directory(
# #                         test_path, target_size=(224, 224), max_images=max_test_images
# #                     )
                
# #                 if len(test_images) == 0:
# #                     st.error("❌ No images found in test directory")
# #                     return
                
# #                 st.success(f"✅ Loaded {len(test_images)} test images")
                
# #                 # Display dataset info
# #                 col1, col2, col3 = st.columns(3)
# #                 with col1:
# #                     st.metric("Test Images", len(test_images))
# #                 with col2:
# #                     st.metric("Classes", len(class_names))
# #                 with col3:
# #                     st.metric("Image Size", "224x224x3")
                
# #                 st.write(f"**Classes found:** {', '.join(class_names)}")
                
# #                 if st.button("🚀 Run CNN Robustness Tests", key="run_cnn_tests"):
# #                     with st.spinner("Running CNN robustness tests..."):
# #                         results = []
                        
# #                         # Baseline
# #                         baseline_accuracy, baseline_report, _ = evaluate_cnn_scenario(
# #                             model, test_images, test_labels, class_names
# #                         )
                        
# #                         results.append({
# #                             "Scenario": "Baseline",
# #                             "Accuracy": f"{baseline_accuracy:.4f}",
# #                             "Precision": f"{baseline_report['macro avg']['precision']:.4f}",
# #                             "Recall": f"{baseline_report['macro avg']['recall']:.4f}",
# #                             "F1-Score": f"{baseline_report['macro avg']['f1-score']:.4f}",
# #                             "Delta %": "0.00",
# #                             "Status": "BASELINE"
# #                         })
                        
# #                         # Test scenarios
# #                         test_scenarios = [
# #                             ("Gaussian Noise", "gaussian_noise"),
# #                             ("Brightness Increase", "brightness_increase"),
# #                             ("Brightness Decrease", "brightness_decrease"),
# #                             ("Contrast Change", "contrast_change"),
# #                             ("Blur Effect", "blur"),
# #                             ("Horizontal Flip", "horizontal_flip"),
# #                             ("Rotation Simulation", "rotation"),
# #                             ("Saturation Change", "saturation_change")
# #                         ]
                        
# #                         for scenario_name, augmentation_type in test_scenarios:
# #                             try:
# #                                 accuracy, report, _ = evaluate_cnn_scenario(
# #                                     model, test_images, test_labels, class_names, augmentation_type
# #                                 )
                                
# #                                 delta = ((baseline_accuracy - accuracy) / baseline_accuracy) * 100
# #                                 status = "PASS" if delta <= threshold_pct else "FAIL"
                                
# #                                 results.append({
# #                                     "Scenario": scenario_name,
# #                                     "Accuracy": f"{accuracy:.4f}",
# #                                     "Precision": f"{report['macro avg']['precision']:.4f}",
# #                                     "Recall": f"{report['macro avg']['recall']:.4f}",
# #                                     "F1-Score": f"{report['macro avg']['f1-score']:.4f}",
# #                                     "Delta %": f"{delta:.2f}",
# #                                     "Status": status
# #                                 })
                                
# #                             except Exception as e:
# #                                 st.warning(f"⚠️ Error in scenario '{scenario_name}': {str(e)}")
                        
# #                         # Additional extreme scenarios
# #                         extreme_scenarios = [
# #                             ("Extreme Brightness", lambda imgs: np.clip(imgs * 2.0, 0, 1)),
# #                             ("Heavy Blur", lambda imgs: np.array([cv2.GaussianBlur((img*255).astype(np.uint8), (15, 15), 0)/255.0 for img in imgs]))
# #                         ]
                        
# #                         for scenario_name, transform_fn in extreme_scenarios:
# #                             try:
# #                                 transformed_images = transform_fn(test_images)
# #                                 predictions = model.predict(transformed_images, verbose=0)
# #                                 predicted_classes = np.argmax(predictions, axis=1)
# #                                 accuracy = accuracy_score(test_labels, predicted_classes)
                                
# #                                 report = classification_report(test_labels, predicted_classes, 
# #                                                              target_names=class_names, 
# #                                                              output_dict=True, 
# #                                                              zero_division=0)
                                
# #                                 delta = ((baseline_accuracy - accuracy) / baseline_accuracy) * 100
# #                                 status = "PASS" if delta <= threshold_pct else "FAIL"
                                
# #                                 results.append({
# #                                     "Scenario": scenario_name,
# #                                     "Accuracy": f"{accuracy:.4f}",
# #                                     "Precision": f"{report['macro avg']['precision']:.4f}",
# #                                     "Recall": f"{report['macro avg']['recall']:.4f}",
# #                                     "F1-Score": f"{report['macro avg']['f1-score']:.4f}",
# #                                     "Delta %": f"{delta:.2f}",
# #                                     "Status": status
# #                                 })
                                
# #                             except Exception as e:
# #                                 st.warning(f"⚠️ Error in scenario '{scenario_name}': {str(e)}")
                        
# #                         display_cnn_results(results, baseline_accuracy, class_names, test_labels, model, test_images)
                        
# #                         # Cleanup
# #                         shutil.rmtree(temp_dir)
# #             else:
# #                 st.error("❌ Test directory not found in dataset")
        
# #         except Exception as e:
# #             st.error(f"❌ Error: {str(e)}")
# #             import traceback
# #             st.error(traceback.format_exc())
# def cnn_interface():
#     st.markdown('<h2 class="sub-header">🖼️ CNN Image Classification Model Testing</h2>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.subheader("📁 Upload Model & Dataset")
#         model_file = st.file_uploader("Upload CNN Model (.h5)", type=['h5'], key="cnn_model")
#         dataset_zip = st.file_uploader("Upload Dataset (.zip)", type=['zip'], key="cnn_data")
        
#         st.subheader("📊 Dataset Structure Expected")
#         st.code("""
# dataset.zip
# ├── train/
# │   ├── class1/
# │   └── class2/
# ├── test/
# │   ├── class1/
# │   └── class2/
# └── validation/
#     ├── class1/
#     └── class2/
#         """)
    
#     with col2:
#         st.subheader("⚙️ Test Configuration")
#         threshold_pct = st.slider("Accuracy Drop Threshold (%)", 1, 20, 5)
#         max_test_images = st.slider("Max Test Images per Class", 10, 100, 50)
        
#         st.markdown("**Test Scenarios:**")
#         st.write("1. Gaussian Noise")
#         st.write("2. Brightness Changes")
#         st.write("3. Contrast Changes") 
#         st.write("4. Blur Effect")
#         st.write("5. Horizontal Flip")
#         st.write("6. Rotation Simulation")
#         st.write("7. Saturation Changes")
#         st.write("8. Combined Augmentations")
#         st.write("9. Extreme Brightness")
#         st.write("10. Heavy Blur")
    
#     if model_file and dataset_zip and TF_AVAILABLE:
#         try:
#             # FIX: Save uploaded model to temporary file and load it
#             with st.spinner("Loading CNN model..."):
#                 # Create temporary file for the model
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_model:
#                     tmp_model.write(model_file.read())
#                     tmp_model_path = tmp_model.name
                
#                 # Load model from temporary file
#                 model = load_model(tmp_model_path)
                
#                 # Clean up temporary file
#                 os.unlink(tmp_model_path)
                
#             st.success("✅ CNN Model loaded successfully!")
            
#             # Extract dataset
#             with st.spinner("Extracting dataset..."):
#                 dataset_path, temp_dir = extract_dataset_from_zip(dataset_zip)
                
#                 if not dataset_path:
#                     st.error("❌ Could not find valid dataset structure in ZIP file")
#                     return
                
#                 st.success(f"✅ Dataset extracted successfully!")
            
#             # Load test images
#             test_path = os.path.join(dataset_path, 'test')
#             if not os.path.exists(test_path):
#                 # Try 'validation' if 'test' doesn't exist
#                 test_path = os.path.join(dataset_path, 'validation')
            
#             if os.path.exists(test_path):
#                 with st.spinner("Loading test images..."):
#                     test_images, test_labels, class_names = load_images_from_directory(
#                         test_path, target_size=(224, 224), max_images=max_test_images
#                     )
                
#                 if len(test_images) == 0:
#                     st.error("❌ No images found in test directory")
#                     # Clean up
#                     if 'temp_dir' in locals():
#                         shutil.rmtree(temp_dir)
#                     return
                
#                 st.success(f"✅ Loaded {len(test_images)} test images")
                
#                 # Display dataset info
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Test Images", len(test_images))
#                 with col2:
#                     st.metric("Classes", len(class_names))
#                 with col3:
#                     st.metric("Image Size", "224x224x3")
                
#                 st.write(f"**Classes found:** {', '.join(class_names)}")
                
#                 if st.button("🚀 Run CNN Robustness Tests", key="run_cnn_tests"):
#                     with st.spinner("Running CNN robustness tests..."):
#                         results = []
                        
#                         # Baseline
#                         baseline_accuracy, baseline_report, _ = evaluate_cnn_scenario(
#                             model, test_images, test_labels, class_names
#                         )
                        
#                         results.append({
#                             "Scenario": "Baseline",
#                             "Accuracy": f"{baseline_accuracy:.4f}",
#                             "Precision": f"{baseline_report['macro avg']['precision']:.4f}",
#                             "Recall": f"{baseline_report['macro avg']['recall']:.4f}",
#                             "F1-Score": f"{baseline_report['macro avg']['f1-score']:.4f}",
#                             "Delta %": "0.00",
#                             "Status": "BASELINE"
#                         })
                        
#                         # Test scenarios
#                         test_scenarios = [
#                             ("Gaussian Noise", "gaussian_noise"),
#                             ("Brightness Increase", "brightness_increase"),
#                             ("Brightness Decrease", "brightness_decrease"),
#                             ("Contrast Change", "contrast_change"),
#                             ("Blur Effect", "blur"),
#                             ("Horizontal Flip", "horizontal_flip"),
#                             ("Rotation Simulation", "rotation"),
#                             ("Saturation Change", "saturation_change")
#                         ]
                        
#                         for scenario_name, augmentation_type in test_scenarios:
#                             try:
#                                 accuracy, report, _ = evaluate_cnn_scenario(
#                                     model, test_images, test_labels, class_names, augmentation_type
#                                 )
                                
#                                 delta = ((baseline_accuracy - accuracy) / baseline_accuracy) * 100
#                                 status = "PASS" if delta <= threshold_pct else "FAIL"
                                
#                                 results.append({
#                                     "Scenario": scenario_name,
#                                     "Accuracy": f"{accuracy:.4f}",
#                                     "Precision": f"{report['macro avg']['precision']:.4f}",
#                                     "Recall": f"{report['macro avg']['recall']:.4f}",
#                                     "F1-Score": f"{report['macro avg']['f1-score']:.4f}",
#                                     "Delta %": f"{delta:.2f}",
#                                     "Status": status
#                                 })
                                
#                             except Exception as e:
#                                 st.warning(f"⚠️ Error in scenario '{scenario_name}': {str(e)}")
                        
#                         # Additional extreme scenarios
#                         extreme_scenarios = [
#                             ("Extreme Brightness", lambda imgs: np.clip(imgs * 2.0, 0, 1)),
#                             ("Heavy Blur", lambda imgs: np.array([cv2.GaussianBlur((img*255).astype(np.uint8), (15, 15), 0)/255.0 for img in imgs]))
#                         ]
                        
#                         for scenario_name, transform_fn in extreme_scenarios:
#                             try:
#                                 transformed_images = transform_fn(test_images)
#                                 predictions = model.predict(transformed_images, verbose=0)
#                                 predicted_classes = np.argmax(predictions, axis=1)
#                                 accuracy = accuracy_score(test_labels, predicted_classes)
                                
#                                 report = classification_report(test_labels, predicted_classes, 
#                                                              target_names=class_names, 
#                                                              output_dict=True, 
#                                                              zero_division=0)
                                
#                                 delta = ((baseline_accuracy - accuracy) / baseline_accuracy) * 100
#                                 status = "PASS" if delta <= threshold_pct else "FAIL"
                                
#                                 results.append({
#                                     "Scenario": scenario_name,
#                                     "Accuracy": f"{accuracy:.4f}",
#                                     "Precision": f"{report['macro avg']['precision']:.4f}",
#                                     "Recall": f"{report['macro avg']['recall']:.4f}",
#                                     "F1-Score": f"{report['macro avg']['f1-score']:.4f}",
#                                     "Delta %": f"{delta:.2f}",
#                                     "Status": status
#                                 })
                                
#                             except Exception as e:
#                                 st.warning(f"⚠️ Error in scenario '{scenario_name}': {str(e)}")
                        
#                         display_cnn_results(results, baseline_accuracy, class_names, test_labels, model, test_images)
                        
#                     # Cleanup
#                     shutil.rmtree(temp_dir)
#             else:
#                 st.error("❌ Test directory not found in dataset")
#                 # Clean up
#                 if 'temp_dir' in locals():
#                     shutil.rmtree(temp_dir)
        
#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")
#             import traceback
#             st.error(traceback.format_exc())
#             # Clean up on error
#             if 'temp_dir' in locals():
#                 shutil.rmtree(temp_dir)

# def display_results(results, baseline_value, metric_name):
#     """Display results for time series and regression models"""
#     st.markdown('<h3 class="sub-header">📊 Test Results</h3>', unsafe_allow_html=True)
    
#     df_results = pd.DataFrame(results)
    
#     # Color code the results
#     def highlight_status(val):
#         if val == "PASS":
#             return 'background-color: #d4edda; color: #155724'
#         elif val == "FAIL":
#             return 'background-color: #f8d7da; color: #721c24'
#         elif val == "BASELINE":
#             return 'background-color: #cce5ff; color: #004085'
#         return ''
    
#     styled_df = df_results.style.applymap(highlight_status, subset=['Status'])
#     st.dataframe(styled_df, use_container_width=True)
    
#     # Summary metrics (excluding baseline)
#     test_results = [r for r in results if r["Status"] != "BASELINE"]
#     passed_tests = sum(1 for r in test_results if r["Status"] == "PASS")
#     total_tests = len(test_results)
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Tests", total_tests)
#     with col2:
#         st.metric("Passed Tests", passed_tests, f"{passed_tests}/{total_tests}")
#     with col3:
#         if total_tests > 0:
#             pass_rate = (passed_tests / total_tests) * 100
#             st.metric("Pass Rate", f"{pass_rate:.1f}%")
    
#     # Visualization
#     if total_tests > 0:
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
#         # Metric comparison
#         test_names = [r["Scenario"] for r in test_results]
#         if metric_name == "RMSE":
#             metric_values = [float(r["RMSE"]) for r in test_results]
#         else:
#             metric_values = [float(r["MSE"]) for r in test_results]
#         colors = ['green' if r["Status"] == "PASS" else 'red' for r in test_results]
        
#         ax1.barh(test_names, metric_values, color=colors, alpha=0.7)
#         ax1.axvline(baseline_value, color='blue', linestyle='--', label=f'Baseline {metric_name}')
#         ax1.set_xlabel(metric_name)
#         ax1.set_title(f'{metric_name} by Scenario')
#         ax1.legend()
        
#         # Pass/Fail pie chart
#         status_counts = pd.Series([r["Status"] for r in test_results]).value_counts()
#         if len(status_counts) > 0:
#             ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
#                    colors=['lightgreen' if x == 'PASS' else 'lightcoral' for x in status_counts.index])
#             ax2.set_title('Test Results Distribution')
        
#         st.pyplot(fig)

# def display_cnn_results(results, baseline_accuracy, class_names, test_labels, model, test_images):
#     """Display results for CNN models with additional visualizations"""
#     st.markdown('<h3 class="sub-header">📊 CNN Test Results</h3>', unsafe_allow_html=True)
    
#     df_results = pd.DataFrame(results)
    
#     # Color code the results
#     def highlight_status(val):
#         if val == "PASS":
#             return 'background-color: #d4edda; color: #155724'
#         elif val == "FAIL":
#             return 'background-color: #f8d7da; color: #721c24'
#         elif val == "BASELINE":
#             return 'background-color: #cce5ff; color: #004085'
#         return ''
    
#     styled_df = df_results.style.applymap(highlight_status, subset=['Status'])
#     st.dataframe(styled_df, use_container_width=True)
    
#     # Summary metrics (excluding baseline)
#     test_results = [r for r in results if r["Status"] != "BASELINE"]
#     passed_tests = sum(1 for r in test_results if r["Status"] == "PASS")
#     total_tests = len(test_results)
    
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Total Tests", total_tests)
#     with col2:
#         st.metric("Passed Tests", passed_tests, f"{passed_tests}/{total_tests}")
#     with col3:
#         if total_tests > 0:
#             pass_rate = (passed_tests / total_tests) * 100
#             st.metric("Pass Rate", f"{pass_rate:.1f}%")
    
#     # Visualization
#     if total_tests > 0:
#         fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
#         # Accuracy comparison
#         test_names = [r["Scenario"] for r in test_results]
#         accuracies = [float(r["Accuracy"]) for r in test_results]
#         colors = ['green' if r["Status"] == "PASS" else 'red' for r in test_results]
        
#         ax1.barh(test_names, accuracies, color=colors, alpha=0.7)
#         ax1.axvline(baseline_accuracy, color='blue', linestyle='--', label='Baseline Accuracy')
#         ax1.set_xlabel('Accuracy')
#         ax1.set_title('Accuracy by Scenario')
#         ax1.legend()
#         plt.setp(ax1.get_yticklabels(), rotation=0, ha="right")
        
#         # Pass/Fail pie chart
#         status_counts = pd.Series([r["Status"] for r in test_results]).value_counts()
#         if len(status_counts) > 0:
#             ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
#                    colors=['lightgreen' if x == 'PASS' else 'lightcoral' for x in status_counts.index])
#             ax2.set_title('Test Results Distribution')
        
#         # F1-Score comparison
#         f1_scores = [float(r["F1-Score"]) for r in test_results]
#         ax3.barh(test_names, f1_scores, color=colors, alpha=0.7)
#         baseline_f1 = float(results[0]["F1-Score"])  # Baseline is first result
#         ax3.axvline(baseline_f1, color='blue', linestyle='--', label='Baseline F1-Score')
#         ax3.set_xlabel('F1-Score')
#         ax3.set_title('F1-Score by Scenario')
#         ax3.legend()
#         plt.setp(ax3.get_yticklabels(), rotation=0, ha="right")
        
#         # Delta percentage
#         deltas = [float(r["Delta %"]) for r in test_results]
#         ax4.barh(test_names, deltas, color=colors, alpha=0.7)
#         ax4.set_xlabel('Accuracy Drop (%)')
#         ax4.set_title('Performance Degradation by Scenario')
#         plt.setp(ax4.get_yticklabels(), rotation=0, ha="right")
        
#         plt.tight_layout()
#         st.pyplot(fig)
        
#         # Confusion Matrix for baseline
#         st.subheader("🔍 Baseline Confusion Matrix")
#         baseline_predictions = model.predict(test_images, verbose=0)
#         baseline_pred_classes = np.argmax(baseline_predictions, axis=1)
        
#         cm = confusion_matrix(test_labels, baseline_pred_classes)
#         fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
#                    xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
#         ax_cm.set_title('Baseline Confusion Matrix')
#         ax_cm.set_xlabel('Predicted Label')
#         ax_cm.set_ylabel('True Label')
#         st.pyplot(fig_cm)

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import warnings
import zipfile
import os
import tempfile
import shutil
from PIL import Image
import cv2
warnings.filterwarnings('ignore')

# Try importing TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("TensorFlow not available. CNN model testing will be disabled.")

# Set page config
st.set_page_config(
    page_title="ML Model Robustness Testing Suite",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .pass-status {
        color: #28a745;
        font-weight: bold;
    }
    .fail-status {
        color: #dc3545;
        font-weight: bold;
    }
    .parameter-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Model Classes
class GlucoseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GlucoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# Helper Functions for Time Series
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])  # glucose
    return np.array(X), np.array(y)

def load_and_preprocess_timeseries(df):
    """Preprocess time series data"""
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    features = ['glucose', 'calories', 'heart_rate', 'steps',
                'basal_rate', 'bolus_volume_delivered', 'carb_input']
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    sequence_length = 10
    X, y = create_sequences(scaled_data, sequence_length)
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    return X_train, X_test, y_train, y_test, features, scaler

def evaluate_timeseries_scenario(model, X, y, modify_fn=None):
    """Evaluate time series model with scenario"""
    X_mod = modify_fn(X) if modify_fn else X
    with torch.no_grad():
        preds = model(torch.tensor(X_mod, dtype=torch.float32)).squeeze().numpy()
    mse = mean_squared_error(y, preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    return mse, rmse, mae

def evaluate_regression_scenario(model, X_mod, y_true):
    """Evaluate regression model with scenario"""
    preds = model.predict(X_mod)
    mse = mean_squared_error(y_true, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, preds)
    return mse, rmse, r2

# Helper Functions for CNN
def extract_dataset_from_zip(zip_file):
    """Extract dataset from uploaded zip file"""
    temp_dir = tempfile.mkdtemp()
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find dataset directory with train/test/validation structure
    dataset_path = None
    for root, dirs, files in os.walk(temp_dir):
        if 'train' in dirs and 'test' in dirs and 'validation' in dirs:
            dataset_path = root
            break
        elif any('train' in d for d in dirs) and any('test' in d for d in dirs):
            dataset_path = root
            break
    
    return dataset_path, temp_dir

def load_images_from_directory(directory_path, target_size=(224, 224), max_images=100):
    """Load images from directory with their labels"""
    images = []
    labels = []
    class_names = []
    
    for class_name in os.listdir(directory_path):
        class_path = os.path.join(directory_path, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            class_idx = len(class_names) - 1
            
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Limit images per class to prevent memory issues
            for img_file in image_files[:max_images//len(os.listdir(directory_path))]:
                img_path = os.path.join(class_path, img_file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(target_size)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(class_idx)
                except Exception as e:
                    continue
    
    return np.array(images), np.array(labels), class_names

def apply_image_augmentation(images, augmentation_type, **params):
    """Apply various augmentations to images with dynamic parameters"""
    augmented_images = images.copy()
    
    if augmentation_type == "gaussian_noise":
        noise_std = params.get('noise_std', 0.05)
        noise = np.random.normal(0, noise_std, images.shape)
        augmented_images = np.clip(images + noise, 0, 1)
    
    elif augmentation_type == "brightness_increase":
        brightness_factor = params.get('brightness_factor', 1.3)
        augmented_images = np.clip(images * brightness_factor, 0, 1)
    
    elif augmentation_type == "brightness_decrease":
        brightness_factor = params.get('brightness_factor', 0.7)
        augmented_images = np.clip(images * brightness_factor, 0, 1)
    
    elif augmentation_type == "contrast_change":
        contrast_factor = params.get('contrast_factor', 1.5)
        augmented_images = np.clip((images - 0.5) * contrast_factor + 0.5, 0, 1)
    
    elif augmentation_type == "blur":
        blur_kernel = params.get('blur_kernel', 5)
        for i, img in enumerate(images):
            img_uint8 = (img * 255).astype(np.uint8)
            blurred = cv2.GaussianBlur(img_uint8, (blur_kernel, blur_kernel), 0)
            augmented_images[i] = blurred / 255.0
    
    elif augmentation_type == "horizontal_flip":
        augmented_images = np.flip(images, axis=2)
    
    elif augmentation_type == "rotation":
        # Simple rotation simulation by flipping
        augmented_images = np.flip(images, axis=1)
    
    elif augmentation_type == "saturation_change":
        saturation_factor = params.get('saturation_factor', 1.5)
        for i, img in enumerate(images):
            img_uint8 = (img * 255).astype(np.uint8)
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            augmented_images[i] = rgb / 255.0
    
    elif augmentation_type == "color_shift":
        color_shift = params.get('color_shift', 0.1)
        shift = np.random.uniform(-color_shift, color_shift, (1, 1, 3))
        augmented_images = np.clip(images + shift, 0, 1)
    
    elif augmentation_type == "jpeg_compression":
        quality = params.get('quality', 30)
        for i, img in enumerate(images):
            img_uint8 = (img * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8)
            buffer = BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed_img = Image.open(buffer)
            augmented_images[i] = np.array(compressed_img) / 255.0
    
    return augmented_images

def evaluate_cnn_scenario(model, images, labels, class_names, augmentation_type=None, **params):
    """Evaluate CNN model with different scenarios"""
    if augmentation_type:
        test_images = apply_image_augmentation(images, augmentation_type, **params)
    else:
        test_images = images
    
    predictions = model.predict(test_images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predicted_classes)
    
    # Calculate per-class metrics
    report = classification_report(labels, predicted_classes, 
                                 target_names=class_names, 
                                 output_dict=True, 
                                 zero_division=0)
    
    return accuracy, report, predicted_classes

def get_timeseries_parameters():
    """Get dynamic parameters for time series testing"""
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Dynamic Test Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Noise Parameters**")
        noise_low = st.slider("Low Noise Level", 0.001, 0.2, 0.01, step=0.001, key="ts_noise_low")
        noise_high = st.slider("High Noise Level", 0.01, 0.5, 0.05, step=0.01, key="ts_noise_high")
        
        st.markdown("**Feature Modification**")
        spike_value = st.slider("Glucose Spike Value", 0.01, 1.0, 0.1, step=0.01, key="ts_spike")
        heart_rate_increase = st.slider("Heart Rate Increase (%)", 5, 100, 20, step=5, key="ts_hr")
        calorie_increase = st.slider("Calorie Increase", 0.1, 1.0, 0.2, step=0.1, key="ts_cal")
    
    with col2:
        st.markdown("**Adversarial Parameters**")
        adversarial_shift = st.slider("Adversarial Shift", 0.01, 0.3, 0.05, step=0.01, key="ts_adv")
        feature_dropout_prob = st.slider("Feature Dropout Probability", 0.1, 0.9, 0.3, step=0.1, key="ts_dropout")
        
        st.markdown("**Extreme Test Parameters**")
        extreme_noise = st.slider("Extreme Noise Level", 0.1, 1.0, 0.3, step=0.1, key="ts_extreme")
        temporal_shift = st.slider("Temporal Shift (steps)", 1, 5, 2, step=1, key="ts_temporal")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'noise_low': noise_low,
        'noise_high': noise_high,
        'spike_value': spike_value,
        'heart_rate_increase': heart_rate_increase,
        'calorie_increase': calorie_increase,
        'adversarial_shift': adversarial_shift,
        'feature_dropout_prob': feature_dropout_prob,
        'extreme_noise': extreme_noise,
        'temporal_shift': temporal_shift
    }

# def get_regression_parameters():
#     """Get dynamic parameters for regression testing"""
#     st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
#     st.markdown("### 🎛️ Dynamic Test Parameters")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("**Noise Parameters**")
#         noise_low = st.slider("Low Noise Std Dev", 0.01, 0.5, 0.05, step=0.01, key="reg_noise_low")
#         noise_high = st.slider("High Noise Std Dev", 0.1, 2.0, 0.2, step=0.1, key="reg_noise_high")
        
#         st.markdown("**Feature Drift**")
#         drift_up_pct = st.slider("Upward Drift (%)", 5, 50, 10, step=5, key="reg_drift_up")
#         drift_down_pct = st.slider("Downward Drift (%)", 5, 50, 10, step=5, key="reg_drift_down")
#         nonlinear_drift = st.slider("Nonlinear Drift Factor", 0.1, 2.0, 0.5, step=0.1, key="reg_nonlinear")
    
#     with col2:
#         st.markdown("**Missing Values**")
#         missing_low_pct = st.slider("Low Missing Values (%)", 5, 30, 10, step=5, key="reg_missing_low")
#         missing_high_pct = st.slider("High Missing Values (%)", 20, 70, 30, step=5, key="reg_missing_high")
        
#         st.markdown("**Outlier Parameters**")
#         outlier_pct = st.slider("Outlier Injection (%)", 1, 20, 5, step=1, key="reg_outlier_pct")
#         outlier_multiplier = st.slider("Outlier Multiplier", 2, 50, 10, step=1, key="reg_outlier_mult")
        
#         st.markdown("**Scaling Parameters**")
#         scaling_multiplier = st.slider("Feature Scaling Factor", 0.1, 100, 10, step=0.1, key="reg_scaling")
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     return {
#         'noise_low': noise_low,
#         'noise_high': noise_high,
#         'drift_up_pct': drift_up_pct,
#         'drift_down_pct': drift_down_pct,
#         'nonlinear_drift': nonlinear_drift,
#         'missing_low_pct': missing_low_pct,
#         'missing_high_pct': missing_high_pct,
#         'outlier_pct': outlier_pct,
#         'outlier_multiplier': outlier_multiplier,
#         'scaling_multiplier': scaling_multiplier
#     }
def get_regression_parameters():
    """Get dynamic parameters for regression testing - FIXED VERSION"""
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Dynamic Test Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Noise Parameters**")
        # Fix: All parameters are float type
        noise_low = st.slider("Low Noise Std Dev", 
                             min_value=0.01, max_value=0.50, value=0.05, step=0.01, 
                             key="reg_noise_low")
        noise_high = st.slider("High Noise Std Dev", 
                              min_value=0.10, max_value=2.00, value=0.20, step=0.10, 
                              key="reg_noise_high")
        
        st.markdown("**Feature Drift**")
        # Fix: All parameters are int type
        drift_up_pct = st.slider("Upward Drift (%)", 
                                 min_value=5, max_value=50, value=10, step=5, 
                                 key="reg_drift_up")
        drift_down_pct = st.slider("Downward Drift (%)", 
                                  min_value=5, max_value=50, value=10, step=5, 
                                  key="reg_drift_down")
        nonlinear_drift = st.slider("Nonlinear Drift Factor", 
                                   min_value=0.1, max_value=2.0, value=0.5, step=0.1, 
                                   key="reg_nonlinear")
    
    with col2:
        st.markdown("**Missing Values**")
        # Fix: All parameters are int type
        missing_low_pct = st.slider("Low Missing Values (%)", 
                                   min_value=5, max_value=30, value=10, step=5, 
                                   key="reg_missing_low")
        missing_high_pct = st.slider("High Missing Values (%)", 
                                    min_value=20, max_value=70, value=30, step=5, 
                                    key="reg_missing_high")
        
        st.markdown("**Outlier Parameters**")
        # Fix: All parameters are int type
        outlier_pct = st.slider("Outlier Injection (%)", 
                               min_value=1, max_value=20, value=5, step=1, 
                               key="reg_outlier_pct")
        outlier_multiplier = st.slider("Outlier Multiplier", 
                                      min_value=2, max_value=50, value=10, step=1, 
                                      key="reg_outlier_mult")
        
        st.markdown("**Scaling Parameters**")
        # Fix: All parameters are float type
        scaling_multiplier = st.slider("Feature Scaling Factor", 
                                      min_value=0.1, max_value=100.0, value=10.0, step=0.1, 
                                      key="reg_scaling")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'noise_low': noise_low,
        'noise_high': noise_high,
        'drift_up_pct': drift_up_pct,
        'drift_down_pct': drift_down_pct,
        'nonlinear_drift': nonlinear_drift,
        'missing_low_pct': missing_low_pct,
        'missing_high_pct': missing_high_pct,
        'outlier_pct': outlier_pct,
        'outlier_multiplier': outlier_multiplier,
        'scaling_multiplier': scaling_multiplier
    }
    
def get_cnn_parameters():
    """Get dynamic parameters for CNN testing"""
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.markdown("### 🎛️ Dynamic Test Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Noise & Quality**")
        noise_std = st.slider("Gaussian Noise Std Dev", 0.01, 0.3, 0.05, step=0.01, key="cnn_noise")
        jpeg_quality = st.slider("JPEG Compression Quality", 10, 90, 30, step=10, key="cnn_jpeg")
        
        st.markdown("**Brightness & Contrast**")
        brightness_increase = st.slider("Brightness Increase Factor", 1.1, 3.0, 1.3, step=0.1, key="cnn_bright_inc")
        brightness_decrease = st.slider("Brightness Decrease Factor", 0.1, 0.9, 0.7, step=0.1, key="cnn_bright_dec")
        contrast_factor = st.slider("Contrast Change Factor", 0.5, 3.0, 1.5, step=0.1, key="cnn_contrast")
        
        st.markdown("**Blur & Effects**")
        blur_kernel = st.slider("Blur Kernel Size", 3, 21, 5, step=2, key="cnn_blur")
    
    with col2:
        st.markdown("**Color Parameters**")
        saturation_factor = st.slider("Saturation Factor", 0.5, 3.0, 1.5, step=0.1, key="cnn_saturation")
        color_shift = st.slider("Color Shift Amount", 0.05, 0.5, 0.1, step=0.05, key="cnn_color_shift")
        
        st.markdown("**Extreme Test Parameters**")
        extreme_brightness = st.slider("Extreme Brightness Factor", 2.0, 5.0, 2.5, step=0.5, key="cnn_extreme_bright")
        heavy_blur_kernel = st.slider("Heavy Blur Kernel", 11, 31, 15, step=2, key="cnn_heavy_blur")
        extreme_noise = st.slider("Extreme Noise Std Dev", 0.1, 0.5, 0.2, step=0.05, key="cnn_extreme_noise")
        
        st.markdown("**Combination Parameters**")
        combo_intensity = st.slider("Combination Test Intensity", 0.3, 1.0, 0.5, step=0.1, key="cnn_combo")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'noise_std': noise_std,
        'jpeg_quality': jpeg_quality,
        'brightness_increase': brightness_increase,
        'brightness_decrease': brightness_decrease,
        'contrast_factor': contrast_factor,
        'blur_kernel': blur_kernel,
        'saturation_factor': saturation_factor,
        'color_shift': color_shift,
        'extreme_brightness': extreme_brightness,
        'heavy_blur_kernel': heavy_blur_kernel,
        'extreme_noise': extreme_noise,
        'combo_intensity': combo_intensity
    }

# Main App
def main():
    st.markdown('<h1 class="main-header">🧪 ML Model Robustness Testing Suite</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    model_types = ["Time Series (LSTM)", "Regression (MLR)"]
    if TF_AVAILABLE:
        model_types.append("CNN Image Classification")
    
    model_type = st.sidebar.selectbox("Select Model Type", model_types)
    
    if model_type == "Time Series (LSTM)":
        timeseries_interface()
    elif model_type == "Regression (MLR)":
        regression_interface()
    elif model_type == "CNN Image Classification" and TF_AVAILABLE:
        cnn_interface()

def timeseries_interface():
    st.markdown('<h2 class="sub-header">📈 Time Series LSTM Model Testing</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Model & Dataset")
        model_file = st.file_uploader("Upload LSTM Model (.pth)", type=['pth'], key="lstm_model")
        data_file = st.file_uploader("Upload Dataset (.csv)", type=['csv'], key="lstm_data")
    
    with col2:
        st.subheader("⚙️ Test Configuration")
        threshold_pct = st.slider("Performance Degradation Threshold (%)", 5, 50, 20, key="ts_threshold")
    
    # Dynamic parameters
    params = get_timeseries_parameters()
    
    if model_file and data_file:
        try:
            # Load data
            df = pd.read_csv(data_file, delimiter=';')
            X_train, X_test, y_train, y_test, features, scaler = load_and_preprocess_timeseries(df)
            
            # Load model
            input_dim = X_train.shape[2]
            model = GlucoseLSTM(input_dim)
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            model.eval()
            
            st.success("✅ Model and dataset loaded successfully!")
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Samples", X_train.shape[0])
            with col2:
                st.metric("Test Samples", X_test.shape[0])
            with col3:
                st.metric("Features", len(features))
            with col4:
                st.metric("Sequence Length", X_train.shape[1])
            
            if st.button("🚀 Run Robustness Tests", key="run_lstm_tests"):
                with st.spinner("Running robustness tests..."):
                    # Define scenarios with dynamic parameters
                    scenarios = [
                        ("Baseline", None),
                        ("Gaussian noise (low)", lambda X: X + np.random.normal(0, params['noise_low'], X.shape)),
                        ("Gaussian noise (high)", lambda X: X + np.random.normal(0, params['noise_high'], X.shape)),
                        ("Extreme noise", lambda X: X + np.random.normal(0, params['extreme_noise'], X.shape)),
                        ("Zero-out steps feature", lambda X: np.where(
                            np.arange(X.shape[2]) == features.index('steps'), 0, X)),
                        ("Drop carb_input values", lambda X: np.where(
                            np.arange(X.shape[2]) == features.index('carb_input'), 0, X)),
                        (f"Increase heart rate by {params['heart_rate_increase']}%", 
                         lambda X: X * (1 + (np.arange(X.shape[2]) == features.index('heart_rate')) * (params['heart_rate_increase']/100))),
                        (f"Glucose spike (+{params['spike_value']})", 
                         lambda X: np.where(np.arange(X.shape[2]) == features.index('glucose'), X + params['spike_value'], X)),
                        ("Simulate missing basal_rate (set to mean)", 
                         lambda X: np.where(np.arange(X.shape[2]) == features.index('basal_rate'), 
                                           np.mean(X[:,:,features.index('basal_rate')]), X)),
                        (f"Adversarial shift: all features +{params['adversarial_shift']}", 
                         lambda X: X + params['adversarial_shift']),
                        (f"Calorie increase (+{params['calorie_increase']})", 
                         lambda X: np.where(np.arange(X.shape[2]) == features.index('calories'), X + params['calorie_increase'], X)),
                        (f"Feature dropout (p={params['feature_dropout_prob']})",
                         lambda X: X * (np.random.rand(*X.shape) > params['feature_dropout_prob'])),
                        (f"Temporal shift ({params['temporal_shift']} steps)",
                         lambda X: np.roll(X, params['temporal_shift'], axis=1))
                    ]
                    
                    # Run scenarios
                    results = []
                    baseline_mse, baseline_rmse, baseline_mae = evaluate_timeseries_scenario(model, X_test, y_test, None)
                    
                    for name, mod_fn in scenarios:
                        mse, rmse, mae = evaluate_timeseries_scenario(model, X_test, y_test, mod_fn)
                        delta_percent = ((rmse - baseline_rmse) / baseline_rmse) * 100
                        status = "PASS" if abs(delta_percent) <= threshold_pct else "FAIL"
                        results.append({
                            "Scenario": name,
                            "MSE": f"{mse:.6e}",
                            "RMSE": f"{rmse:.6f}",
                            "MAE": f"{mae:.6f}",
                            "Delta %": f"{delta_percent:.2f}",
                            "Status": status
                        })
                    
                    display_results(results, baseline_rmse, "RMSE")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def regression_interface():
    st.markdown('<h2 class="sub-header">📊 Regression Model Testing</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Model & Dataset")
        model_file = st.file_uploader("Upload Regression Model (.pkl)", type=['pkl'], key="reg_model")
        data_file = st.file_uploader("Upload Dataset (.xlsx or .csv)", type=['xlsx', 'csv'], key="reg_data")
    
    with col2:
        st.subheader("⚙️ Test Configuration")
        threshold_pct = st.slider("Performance Degradation Threshold (%)", 5, 50, 10, key="reg_threshold")
    
    # Dynamic parameters
    params = get_regression_parameters()
    
    if model_file and data_file:
        try:
            # Load model
            model = joblib.load(model_file)
            
            # Load data
            if data_file.name.endswith('.xlsx'):
                df = pd.read_excel(data_file)
            else:
                df = pd.read_csv(data_file)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Auto-detect target column
            target_candidates = [col for col in df.columns if any(word in col.lower() 
                               for word in ['strength', 'target', 'compressive', 'mpa'])]
            
            if target_candidates:
                target_column = st.selectbox("Select Target Column", target_candidates, index=0)
            else:
                target_column = st.selectbox("Select Target Column", df.columns, index=-1)
            
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Train-test split
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.success("✅ Model and dataset loaded successfully!")
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Samples", X_train.shape[0])
            with col2:
                st.metric("Test Samples", X_test.shape[0])
            with col3:
                st.metric("Features", X.shape[1])
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            if st.button("🚀 Run Robustness Tests", key="run_reg_tests"):
                with st.spinner("Running robustness tests..."):
                    results = []
                    
                    # Baseline
                    baseline_mse, baseline_rmse, baseline_r2 = evaluate_regression_scenario(model, X_test, y_test)
                    results.append({
                        "Scenario": "Baseline",
                        "MSE": f"{baseline_mse:.6e}",
                        "RMSE": f"{baseline_rmse:.6f}",
                        "R² Score": f"{baseline_r2:.6f}",
                        "Delta %": "0.00",
                        "Status": "BASELINE"
                    })
                    
                    # Test scenarios with dynamic parameters
                    test_scenarios = [
                        (f"Gaussian noise (low, σ={params['noise_low']})", 
                         lambda X: X + np.random.normal(0, params['noise_low'], X.shape)),
                        (f"Gaussian noise (high, σ={params['noise_high']})", 
                         lambda X: X + np.random.normal(0, params['noise_high'], X.shape)),
                        (f"Feature drift (+{params['drift_up_pct']}%)", 
                         lambda X: X * (1 + params['drift_up_pct']/100)),
                        (f"Feature drift (-{params['drift_down_pct']}%)", 
                         lambda X: X * (1 - params['drift_down_pct']/100)),
                        (f"Nonlinear drift (factor={params['nonlinear_drift']})", 
                         lambda X: X * (1 + params['nonlinear_drift'] * np.sin(np.arange(X.shape[0]).reshape(-1, 1)))),
                    ]
                    
                    # Missing values scenarios with dynamic parameters
                    for pct, name in [(params['missing_low_pct'], "low"), (params['missing_high_pct'], "high")]:
                        def create_missing_scenario(percentage):
                            def scenario_fn(X):
                                X_missing = X.copy()
                                mask = np.random.rand(*X_missing.shape) < (percentage/100)
                                X_missing[mask] = np.nan
                                return X_missing.fillna(X.mean())
                            return scenario_fn
                        test_scenarios.append((f"{pct}% missing values ({name})", create_missing_scenario(pct)))
                    
                    # Outlier injection with dynamic parameters
                    def outlier_scenario(X):
                        X_outliers = X.copy()
                        n_outliers = int((params['outlier_pct']/100) * len(X_outliers))
                        rows = np.random.choice(X_outliers.index, n_outliers, replace=False)
                        X_outliers.loc[rows] = X_outliers.loc[rows] * params['outlier_multiplier']
                        return X_outliers
                    test_scenarios.append((f"{params['outlier_pct']}% outliers (×{params['outlier_multiplier']})", outlier_scenario))
                    
                    # Feature scaling shift with dynamic parameters
                    def scaling_scenario(X):
                        X_scaled = X.copy()
                        X_scaled.iloc[:, 0] = X_scaled.iloc[:, 0] * params['scaling_multiplier']
                        return X_scaled
                    test_scenarios.append((f"Feature scaling shift (×{params['scaling_multiplier']})", scaling_scenario))
                    
                    # Additional dynamic scenarios
                    def polynomial_transformation(X):
                        X_poly = X.copy()
                        X_poly.iloc[:, 0] = X_poly.iloc[:, 0] ** 2
                        return X_poly
                    test_scenarios.append(("Polynomial transformation (x²)", polynomial_transformation))
                    
                    def logarithmic_transformation(X):
                        X_log = X.copy()
                        X_log = np.log1p(np.abs(X_log))  # log(1+|x|) to handle negatives
                        return X_log
                    test_scenarios.append(("Logarithmic transformation", logarithmic_transformation))
                    
                    # Run all scenarios
                    for name, scenario_fn in test_scenarios:
                        try:
                            X_modified = scenario_fn(X_test)
                            mse, rmse, r2 = evaluate_regression_scenario(model, X_modified, y_test)
                            delta = ((mse - baseline_mse) / baseline_mse) * 100
                            status = "PASS" if delta <= threshold_pct else "FAIL"
                            
                            results.append({
                                "Scenario": name,
                                "MSE": f"{mse:.6e}",
                                "RMSE": f"{rmse:.6f}",
                                "R² Score": f"{r2:.6f}",
                                "Delta %": f"{delta:.2f}",
                                "Status": status
                            })
                        except Exception as e:
                            st.warning(f"⚠️ Error in scenario '{name}': {str(e)}")
                    
                    display_results(results, baseline_rmse, "RMSE")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def cnn_interface():
    st.markdown('<h2 class="sub-header">🖼️ CNN Image Classification Model Testing</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Model & Dataset")
        model_file = st.file_uploader("Upload CNN Model (.h5)", type=['h5'], key="cnn_model")
        dataset_zip = st.file_uploader("Upload Dataset (.zip)", type=['zip'], key="cnn_data")
        
        st.subheader("📊 Dataset Structure Expected")
        
    with col2:
        st.subheader("⚙️ Test Configuration")
        threshold_pct = st.slider("Accuracy Drop Threshold (%)", 1, 20, 5, key="cnn_threshold")
        max_test_images = st.slider("Max Test Images per Class", 10, 100, 50, key="cnn_max_images")
    
    # Dynamic parameters
    params = get_cnn_parameters()
    
    if model_file and dataset_zip and TF_AVAILABLE:
        try:
            # Load model from uploaded file
            with st.spinner("Loading CNN model..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_model:
                    tmp_model.write(model_file.read())
                    tmp_model_path = tmp_model.name
                
                model = load_model(tmp_model_path)
                os.unlink(tmp_model_path)
                
            st.success("✅ CNN Model loaded successfully!")
            
            # Extract dataset
            with st.spinner("Extracting dataset..."):
                dataset_path, temp_dir = extract_dataset_from_zip(dataset_zip)
                
                if not dataset_path:
                    st.error("❌ Could not find valid dataset structure in ZIP file")
                    return
                
                st.success("✅ Dataset extracted successfully!")
            
            # Load test images
            test_path = os.path.join(dataset_path, 'test')
            if not os.path.exists(test_path):
                test_path = os.path.join(dataset_path, 'validation')
            
            if os.path.exists(test_path):
                with st.spinner("Loading test images..."):
                    test_images, test_labels, class_names = load_images_from_directory(
                        test_path, target_size=(224, 224), max_images=max_test_images
                    )
                
                if len(test_images) == 0:
                    st.error("❌ No images found in test directory")
                    if 'temp_dir' in locals():
                        shutil.rmtree(temp_dir)
                    return
                
                st.success(f"✅ Loaded {len(test_images)} test images")
                
                # Display dataset info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Test Images", len(test_images))
                with col2:
                    st.metric("Classes", len(class_names))
                with col3:
                    st.metric("Image Size", "224x224x3")
                
                st.write(f"**Classes found:** {', '.join(class_names)}")
                
                if st.button("🚀 Run CNN Robustness Tests", key="run_cnn_tests"):
                    with st.spinner("Running CNN robustness tests..."):
                        results = []
                        
                        # Baseline
                        baseline_accuracy, baseline_report, _ = evaluate_cnn_scenario(
                            model, test_images, test_labels, class_names
                        )
                        
                        results.append({
                            "Scenario": "Baseline",
                            "Accuracy": f"{baseline_accuracy:.4f}",
                            "Precision": f"{baseline_report['macro avg']['precision']:.4f}",
                            "Recall": f"{baseline_report['macro avg']['recall']:.4f}",
                            "F1-Score": f"{baseline_report['macro avg']['f1-score']:.4f}",
                            "Delta %": "0.00",
                            "Status": "BASELINE"
                        })
                        
                        # Test scenarios with dynamic parameters
                        test_scenarios = [
                            (f"Gaussian Noise (σ={params['noise_std']})", "gaussian_noise", 
                             {"noise_std": params['noise_std']}),
                            (f"Brightness Increase (×{params['brightness_increase']})", "brightness_increase", 
                             {"brightness_factor": params['brightness_increase']}),
                            (f"Brightness Decrease (×{params['brightness_decrease']})", "brightness_decrease", 
                             {"brightness_factor": params['brightness_decrease']}),
                            (f"Contrast Change (×{params['contrast_factor']})", "contrast_change", 
                             {"contrast_factor": params['contrast_factor']}),
                            (f"Blur Effect (kernel={params['blur_kernel']})", "blur", 
                             {"blur_kernel": params['blur_kernel']}),
                            ("Horizontal Flip", "horizontal_flip", {}),
                            ("Rotation Simulation", "rotation", {}),
                            (f"Saturation Change (×{params['saturation_factor']})", "saturation_change", 
                             {"saturation_factor": params['saturation_factor']}),
                            (f"Color Shift (±{params['color_shift']})", "color_shift", 
                             {"color_shift": params['color_shift']}),
                            (f"JPEG Compression (Q={params['jpeg_quality']})", "jpeg_compression", 
                             {"quality": params['jpeg_quality']}),
                        ]
                        
                        for scenario_name, augmentation_type, aug_params in test_scenarios:
                            try:
                                accuracy, report, _ = evaluate_cnn_scenario(
                                    model, test_images, test_labels, class_names, 
                                    augmentation_type, **aug_params
                                )
                                
                                delta = ((baseline_accuracy - accuracy) / baseline_accuracy) * 100
                                status = "PASS" if delta <= threshold_pct else "FAIL"
                                
                                results.append({
                                    "Scenario": scenario_name,
                                    "Accuracy": f"{accuracy:.4f}",
                                    "Precision": f"{report['macro avg']['precision']:.4f}",
                                    "Recall": f"{report['macro avg']['recall']:.4f}",
                                    "F1-Score": f"{report['macro avg']['f1-score']:.4f}",
                                    "Delta %": f"{delta:.2f}",
                                    "Status": status
                                })
                                
                            except Exception as e:
                                st.warning(f"⚠️ Error in scenario '{scenario_name}': {str(e)}")
                        
                        # Extreme scenarios with dynamic parameters
                        extreme_scenarios = [
                            (f"Extreme Brightness (×{params['extreme_brightness']})", 
                             lambda imgs: np.clip(imgs * params['extreme_brightness'], 0, 1)),
                            (f"Heavy Blur (kernel={params['heavy_blur_kernel']})", 
                             lambda imgs: np.array([cv2.GaussianBlur((img*255).astype(np.uint8), 
                                                  (params['heavy_blur_kernel'], params['heavy_blur_kernel']), 0)/255.0 
                                                  for img in imgs])),
                            (f"Extreme Noise (σ={params['extreme_noise']})", 
                             lambda imgs: np.clip(imgs + np.random.normal(0, params['extreme_noise'], imgs.shape), 0, 1)),
                            (f"Combined Effects (intensity={params['combo_intensity']})", 
                             lambda imgs: np.clip(
                                 (imgs * (1 + params['combo_intensity'] * 0.3) + 
                                  np.random.normal(0, params['combo_intensity'] * 0.05, imgs.shape)), 0, 1))
                        ]
                        
                        for scenario_name, transform_fn in extreme_scenarios:
                            try:
                                transformed_images = transform_fn(test_images)
                                predictions = model.predict(transformed_images, verbose=0)
                                predicted_classes = np.argmax(predictions, axis=1)
                                accuracy = accuracy_score(test_labels, predicted_classes)
                                
                                report = classification_report(test_labels, predicted_classes, 
                                                             target_names=class_names, 
                                                             output_dict=True, 
                                                             zero_division=0)
                                
                                delta = ((baseline_accuracy - accuracy) / baseline_accuracy) * 100
                                status = "PASS" if delta <= threshold_pct else "FAIL"
                                
                                results.append({
                                    "Scenario": scenario_name,
                                    "Accuracy": f"{accuracy:.4f}",
                                    "Precision": f"{report['macro avg']['precision']:.4f}",
                                    "Recall": f"{report['macro avg']['recall']:.4f}",
                                    "F1-Score": f"{report['macro avg']['f1-score']:.4f}",
                                    "Delta %": f"{delta:.2f}",
                                    "Status": status
                                })
                                
                            except Exception as e:
                                st.warning(f"⚠️ Error in scenario '{scenario_name}': {str(e)}")
                        
                        display_cnn_results(results, baseline_accuracy, class_names, test_labels, model, test_images)
                        
                    # Cleanup
                    shutil.rmtree(temp_dir)
            else:
                st.error("❌ Test directory not found in dataset")
                if 'temp_dir' in locals():
                    shutil.rmtree(temp_dir)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir)

def display_results(results, baseline_value, metric_name):
    """Display results for time series and regression models"""
    st.markdown('<h3 class="sub-header">📊 Test Results</h3>', unsafe_allow_html=True)
    
    df_results = pd.DataFrame(results)
    
    # Color code the results
    def highlight_status(val):
        if val == "PASS":
            return 'background-color: #d4edda; color: #155724'
        elif val == "FAIL":
            return 'background-color: #f8d7da; color: #721c24'
        elif val == "BASELINE":
            return 'background-color: #cce5ff; color: #004085'
        return ''
    
    styled_df = df_results.style.applymap(highlight_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Summary metrics (excluding baseline)
    test_results = [r for r in results if r["Status"] != "BASELINE"]
    passed_tests = sum(1 for r in test_results if r["Status"] == "PASS")
    total_tests = len(test_results)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Passed Tests", passed_tests, f"{passed_tests}/{total_tests}")
    with col3:
        if total_tests > 0:
            pass_rate = (passed_tests / total_tests) * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col4:
        failed_tests = total_tests - passed_tests
        st.metric("Failed Tests", failed_tests, f"{failed_tests}/{total_tests}")
    
    # Detailed Analysis
    st.markdown("### 📈 Detailed Analysis")
    
    # Most vulnerable scenarios
    if test_results:
        worst_scenarios = sorted(test_results, key=lambda x: abs(float(x["Delta %"])), reverse=True)[:3]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most Vulnerable Scenarios:**")
            for i, scenario in enumerate(worst_scenarios, 1):
                status_emoji = "❌" if scenario["Status"] == "FAIL" else "⚠️"
                st.write(f"{i}. {status_emoji} {scenario['Scenario']} (Δ: {scenario['Delta %']}%)")
        
        with col2:
            st.markdown("**Robustness Score:**")
            robustness_score = (passed_tests / total_tests) * 100
            if robustness_score >= 80:
                st.success(f"🟢 Excellent: {robustness_score:.1f}%")
            elif robustness_score >= 60:
                st.warning(f"🟡 Good: {robustness_score:.1f}%")
            else:
                st.error(f"🔴 Needs Improvement: {robustness_score:.1f}%")
    
    # Visualization
    if total_tests > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Metric comparison
        test_names = [r["Scenario"] for r in test_results]
        if metric_name == "RMSE":
            metric_values = [float(r["RMSE"]) for r in test_results]
        else:
            metric_values = [float(r["MSE"]) for r in test_results]
        colors = ['green' if r["Status"] == "PASS" else 'red' for r in test_results]
        
        ax1.barh(test_names, metric_values, color=colors, alpha=0.7)
        ax1.axvline(baseline_value, color='blue', linestyle='--', label=f'Baseline {metric_name}')
        ax1.set_xlabel(metric_name)
        ax1.set_title(f'{metric_name} by Scenario')
        ax1.legend()
        plt.setp(ax1.get_yticklabels(), rotation=0, ha="right")
        
        # Pass/Fail pie chart
        status_counts = pd.Series([r["Status"] for r in test_results]).value_counts()
        if len(status_counts) > 0:
            ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                   colors=['lightgreen' if x == 'PASS' else 'lightcoral' for x in status_counts.index])
            ax2.set_title('Test Results Distribution')
        
        # Delta percentage
        deltas = [float(r["Delta %"]) for r in test_results]
        ax3.barh(test_names, deltas, color=colors, alpha=0.7)
        ax3.set_xlabel('Performance Change (%)')
        ax3.set_title('Performance Degradation by Scenario')
        ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
        plt.setp(ax3.get_yticklabels(), rotation=0, ha="right")
        
        # Robustness heatmap
        scenario_types = {
            'Noise': [r for r in test_results if 'noise' in r['Scenario'].lower()],
            'Feature': [r for r in test_results if any(word in r['Scenario'].lower() 
                       for word in ['feature', 'drift', 'missing', 'dropout'])],
            'Extreme': [r for r in test_results if 'extreme' in r['Scenario'].lower()],
            'Other': [r for r in test_results if not any(word in r['Scenario'].lower() 
                     for word in ['noise', 'feature', 'drift', 'missing', 'dropout', 'extreme'])]
        }
        
        category_scores = []
        category_names = []
        for category, scenarios in scenario_types.items():
            if scenarios:
                pass_rate = sum(1 for s in scenarios if s['Status'] == 'PASS') / len(scenarios)
                category_scores.append(pass_rate)
                category_names.append(f"{category} ({len(scenarios)})")
        
        if category_scores:
            ax4.bar(category_names, category_scores, color=['green' if s >= 0.8 else 'orange' if s >= 0.6 else 'red' 
                   for s in category_scores], alpha=0.7)
            ax4.set_ylabel('Pass Rate')
            ax4.set_title('Robustness by Category')
            ax4.set_ylim(0, 1)
            plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")
        
        plt.tight_layout()
        st.pyplot(fig)

def display_cnn_results(results, baseline_accuracy, class_names, test_labels, model, test_images):
    """Display results for CNN models with additional visualizations"""
    st.markdown('<h3 class="sub-header">📊 CNN Test Results</h3>', unsafe_allow_html=True)
    
    df_results = pd.DataFrame(results)
    
    # Color code the results
    def highlight_status(val):
        if val == "PASS":
            return 'background-color: #d4edda; color: #155724'
        elif val == "FAIL":
            return 'background-color: #f8d7da; color: #721c24'
        elif val == "BASELINE":
            return 'background-color: #cce5ff; color: #004085'
        return ''
    
    styled_df = df_results.style.applymap(highlight_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Summary metrics (excluding baseline)
    test_results = [r for r in results if r["Status"] != "BASELINE"]
    passed_tests = sum(1 for r in test_results if r["Status"] == "PASS")
    total_tests = len(test_results)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tests", total_tests)
    with col2:
        st.metric("Passed Tests", passed_tests, f"{passed_tests}/{total_tests}")
    with col3:
        if total_tests > 0:
            pass_rate = (passed_tests / total_tests) * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
    with col4:
        failed_tests = total_tests - passed_tests
        st.metric("Failed Tests", failed_tests)
    
    # CNN-specific analysis
    st.markdown("### 🖼️ CNN Robustness Analysis")
    
    if test_results:
        # Categorize CNN tests
        cnn_categories = {
            'Visual Quality': [r for r in test_results if any(word in r['Scenario'].lower() 
                              for word in ['noise', 'blur', 'jpeg'])],
            'Brightness/Contrast': [r for r in test_results if any(word in r['Scenario'].lower() 
                                   for word in ['brightness', 'contrast'])],
            'Color/Saturation': [r for r in test_results if any(word in r['Scenario'].lower() 
                                for word in ['saturation', 'color'])],
            'Geometric': [r for r in test_results if any(word in r['Scenario'].lower() 
                         for word in ['flip', 'rotation'])],
            'Extreme': [r for r in test_results if 'extreme' in r['Scenario'].lower() or 
                       'heavy' in r['Scenario'].lower() or 'combined' in r['Scenario'].lower()]
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Category Robustness:**")
            for category, scenarios in cnn_categories.items():
                if scenarios:
                    pass_rate = sum(1 for s in scenarios if s['Status'] == 'PASS') / len(scenarios) * 100
                    emoji = "🟢" if pass_rate >= 80 else "🟡" if pass_rate >= 60 else "🔴"
                    st.write(f"{emoji} {category}: {pass_rate:.1f}% ({len(scenarios)} tests)")
        
        with col2:
            st.markdown("**Most Challenging Tests:**")
            worst_cnn = sorted(test_results, key=lambda x: float(x["Delta %"]), reverse=True)[:3]
            for i, test in enumerate(worst_cnn, 1):
                status_emoji = "❌" if test["Status"] == "FAIL" else "⚠️"
                st.write(f"{i}. {status_emoji} {test['Scenario']}")
                st.write(f"   Accuracy drop: {test['Delta %']}%")
    
    # Visualization
    if total_tests > 0:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy comparison
        test_names = [r["Scenario"] for r in test_results]
        accuracies = [float(r["Accuracy"]) for r in test_results]
        colors = ['green' if r["Status"] == "PASS" else 'red' for r in test_results]
        
        ax1.barh(test_names, accuracies, color=colors, alpha=0.7)
        ax1.axvline(baseline_accuracy, color='blue', linestyle='--', label='Baseline Accuracy')
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Accuracy by Scenario')
        ax1.legend()
        plt.setp(ax1.get_yticklabels(), rotation=0, ha="right")
        
        # Pass/Fail pie chart
        status_counts = pd.Series([r["Status"] for r in test_results]).value_counts()
        if len(status_counts) > 0:
            ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                   colors=['lightgreen' if x == 'PASS' else 'lightcoral' for x in status_counts.index])
            ax2.set_title('Test Results Distribution')
        
        # F1-Score comparison
        f1_scores = [float(r["F1-Score"]) for r in test_results]
        ax3.barh(test_names, f1_scores, color=colors, alpha=0.7)
        baseline_f1 = float(results[0]["F1-Score"])  # Baseline is first result
        ax3.axvline(baseline_f1, color='blue', linestyle='--', label='Baseline F1-Score')
        ax3.set_xlabel('F1-Score')
        ax3.set_title('F1-Score by Scenario')
        ax3.legend()
        plt.setp(ax3.get_yticklabels(), rotation=0, ha="right")
        
        # Category performance
        category_data = []
        for category, scenarios in cnn_categories.items():
            if scenarios:
                avg_accuracy = np.mean([float(s["Accuracy"]) for s in scenarios])
                pass_rate = sum(1 for s in scenarios if s['Status'] == 'PASS') / len(scenarios)
                category_data.append((category, avg_accuracy, pass_rate))
        
        if category_data:
            categories, avg_accs, pass_rates = zip(*category_data)
            x_pos = np.arange(len(categories))
            
            ax4_twin = ax4.twinx()
            bars1 = ax4.bar(x_pos - 0.2, avg_accs, 0.4, label='Avg Accuracy', alpha=0.7, color='skyblue')
            bars2 = ax4_twin.bar(x_pos + 0.2, pass_rates, 0.4, label='Pass Rate', alpha=0.7, color='lightgreen')
            
            ax4.set_xlabel('Test Category')
            ax4.set_ylabel('Average Accuracy')
            ax4_twin.set_ylabel('Pass Rate')
            ax4.set_title('Performance by Test Category')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(categories, rotation=45, ha='right')
            
            # Add legends
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Confusion Matrix for baseline
        st.subheader("🔍 Baseline Confusion Matrix")
        baseline_predictions = model.predict(test_images, verbose=0)
        baseline_pred_classes = np.argmax(baseline_predictions, axis=1)
        
        cm = confusion_matrix(test_labels, baseline_pred_classes)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
        ax_cm.set_title('Baseline Confusion Matrix')
        ax_cm.set_xlabel('Predicted Label')
        ax_cm.set_ylabel('True Label')
        st.pyplot(fig_cm)

if __name__ == "__main__":
    main()