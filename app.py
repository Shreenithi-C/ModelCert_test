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
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ML Model Robustness Testing",
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
</style>
""", unsafe_allow_html=True)

# LSTM Model Class
class GlucoseLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GlucoseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# Helper Functions
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

# Main App
def main():
    st.markdown('<h1 class="main-header">🧪 ML Model Robustness Testing Suite</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Configuration")
    model_type = st.sidebar.selectbox("Select Model Type", ["Time Series (LSTM)", "Regression (MLR)"])
    
    if model_type == "Time Series (LSTM)":
        timeseries_interface()
    else:
        regression_interface()

def timeseries_interface():
    st.markdown('<h2 class="sub-header">📈 Time Series LSTM Model Testing</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📁 Upload Model & Dataset")
        model_file = st.file_uploader("Upload LSTM Model (.pth)", type=['pth'], key="lstm_model")
        data_file = st.file_uploader("Upload Dataset (.csv)", type=['csv'], key="lstm_data")
    
    with col2:
        st.subheader("⚙️ Test Configuration")
        threshold_pct = st.slider("Performance Degradation Threshold (%)", 5, 50, 20)
        
        # Scenario parameters
        st.markdown("**Scenario Parameters:**")
        noise_low = st.slider("Low Noise Level", 0.001, 0.1, 0.01, step=0.001)
        noise_high = st.slider("High Noise Level", 0.01, 0.2, 0.05, step=0.01)
        spike_value = st.slider("Glucose Spike Value", 0.05, 0.5, 0.1, step=0.05)
        heart_rate_increase = st.slider("Heart Rate Increase (%)", 10, 50, 20, step=5)
        adversarial_shift = st.slider("Adversarial Shift", 0.01, 0.2, 0.05, step=0.01)
        calorie_increase = st.slider("Calorie Increase", 0.1, 0.5, 0.2, step=0.1)
    
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
                    # Define scenarios with user parameters
                    scenarios = [
                        ("Baseline", None),
                        ("Gaussian noise (low)", lambda X: X + np.random.normal(0, noise_low, X.shape)),
                        ("Gaussian noise (high)", lambda X: X + np.random.normal(0, noise_high, X.shape)),
                        ("Zero-out steps feature", lambda X: np.where(
                            np.arange(X.shape[2]) == features.index('steps'), 0, X)),
                        ("Drop carb_input values", lambda X: np.where(
                            np.arange(X.shape[2]) == features.index('carb_input'), 0, X)),
                        ("Increase heart rate by {}%".format(heart_rate_increase), 
                         lambda X: X * (1 + (np.arange(X.shape[2]) == features.index('heart_rate')) * (heart_rate_increase/100))),
                        ("Sudden spike in glucose (+{})".format(spike_value), 
                         lambda X: np.where(np.arange(X.shape[2]) == features.index('glucose'), X + spike_value, X)),
                        ("Simulate missing basal_rate (set to mean)", 
                         lambda X: np.where(np.arange(X.shape[2]) == features.index('basal_rate'), 
                                           np.mean(X[:,:,features.index('basal_rate')]), X)),
                        ("Adversarial shift: all features +{}".format(adversarial_shift), 
                         lambda X: X + adversarial_shift),
                        ("Extreme calorie values (+{})".format(calorie_increase), 
                         lambda X: np.where(np.arange(X.shape[2]) == features.index('calories'), X + calorie_increase, X))
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
                    
                    # Display results
                    st.markdown('<h3 class="sub-header">📊 Test Results</h3>', unsafe_allow_html=True)
                    
                    df_results = pd.DataFrame(results)
                    
                    # Color code the results
                    def highlight_status(val):
                        if val == "PASS":
                            return 'background-color: #d4edda; color: #155724'
                        elif val == "FAIL":
                            return 'background-color: #f8d7da; color: #721c24'
                        return ''
                    
                    styled_df = df_results.style.applymap(highlight_status, subset=['Status'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Summary metrics
                    passed_tests = sum(1 for r in results if r["Status"] == "PASS")
                    total_tests = len(results)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tests", total_tests)
                    with col2:
                        st.metric("Passed Tests", passed_tests, f"{passed_tests}/{total_tests}")
                    with col3:
                        pass_rate = (passed_tests / total_tests) * 100
                        st.metric("Pass Rate", f"{pass_rate:.1f}%")
                    
                    # Visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # RMSE comparison
                    scenarios_names = [r["Scenario"] for r in results]
                    rmse_values = [float(r["RMSE"]) for r in results]
                    colors = ['green' if r["Status"] == "PASS" else 'red' for r in results]
                    
                    ax1.barh(scenarios_names, rmse_values, color=colors, alpha=0.7)
                    ax1.axvline(baseline_rmse, color='blue', linestyle='--', label='Baseline RMSE')
                    ax1.set_xlabel('RMSE')
                    ax1.set_title('RMSE by Scenario')
                    ax1.legend()
                    
                    # Pass/Fail pie chart
                    pass_fail_counts = df_results['Status'].value_counts()
                    ax2.pie(pass_fail_counts.values, labels=pass_fail_counts.index, autopct='%1.1f%%',
                           colors=['lightgreen', 'lightcoral'])
                    ax2.set_title('Test Results Distribution')
                    
                    st.pyplot(fig)
        
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
        threshold_pct = st.slider("Performance Degradation Threshold (%)", 5, 50, 10)
        
        # Scenario parameters
        st.markdown("**Scenario Parameters:**")
        noise_low = st.slider("Low Noise Standard Deviation", 0.01, 0.2, 0.05, step=0.01)
        noise_high = st.slider("High Noise Standard Deviation", 0.1, 1.0, 0.2, step=0.1)
        drift_up_pct = st.slider("Upward Feature Drift (%)", 5, 30, 10, step=5)
        drift_down_pct = st.slider("Downward Feature Drift (%)", 5, 30, 10, step=5)
        missing_low_pct = st.slider("Low Missing Values (%)", 5, 25, 10, step=5)
        missing_high_pct = st.slider("High Missing Values (%)", 20, 50, 30, step=5)
        outlier_pct = st.slider("Outlier Injection (%)", 1, 10, 5, step=1)
        outlier_multiplier = st.slider("Outlier Multiplier", 2, 20, 10, step=1)
        scaling_multiplier = st.slider("Feature Scaling Multiplier", 10, 1000, 100, step=10)
    
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
            
            # Auto-detect target column (assuming it's the last column or contains 'strength', 'target', etc.)
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
            
            # Show feature importance if available
            if hasattr(model, 'coef_'):
                st.subheader("📈 Feature Coefficients")
                coef_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Coefficient": model.coef_
                })
                coef_df['Abs_Coefficient'] = abs(coef_df['Coefficient'])
                coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(coef_df['Feature'], coef_df['Coefficient'])
                ax.set_xlabel('Coefficient Value')
                ax.set_title('Feature Coefficients')
                st.pyplot(fig)
            
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
                    
                    # Test scenarios with user parameters
                    test_scenarios = [
                        ("Gaussian noise (low)", lambda X: X + np.random.normal(0, noise_low, X.shape)),
                        ("Gaussian noise (high)", lambda X: X + np.random.normal(0, noise_high, X.shape)),
                        (f"Feature drift (+{drift_up_pct}%)", lambda X: X * (1 + drift_up_pct/100)),
                        (f"Feature drift (-{drift_down_pct}%)", lambda X: X * (1 - drift_down_pct/100)),
                    ]
                    
                    # Missing values scenarios
                    for pct, name in [(missing_low_pct, "low"), (missing_high_pct, "high")]:
                        def create_missing_scenario(percentage):
                            def scenario_fn(X):
                                X_missing = X.copy()
                                mask = np.random.rand(*X_missing.shape) < (percentage/100)
                                X_missing[mask] = np.nan
                                return X_missing.fillna(X.mean())
                            return scenario_fn
                        test_scenarios.append((f"{pct}% missing values ({name})", create_missing_scenario(pct)))
                    
                    # Outlier injection
                    def outlier_scenario(X):
                        X_outliers = X.copy()
                        n_outliers = int((outlier_pct/100) * len(X_outliers))
                        rows = np.random.choice(X_outliers.index, n_outliers, replace=False)
                        X_outliers.loc[rows] = X_outliers.loc[rows] * outlier_multiplier
                        return X_outliers
                    test_scenarios.append((f"{outlier_pct}% outliers (×{outlier_multiplier})", outlier_scenario))
                    
                    # Most important feature missing
                    if hasattr(model, 'coef_'):
                        important_feature = coef_df.sort_values('Abs_Coefficient', ascending=False).iloc[0]["Feature"]
                        def important_feature_scenario(X):
                            X_single_missing = X.copy()
                            X_single_missing[important_feature] = X_single_missing[important_feature].mean()
                            return X_single_missing
                        test_scenarios.append((f"Missing feature: {important_feature}", important_feature_scenario))
                    
                    # Feature scaling shift
                    def scaling_scenario(X):
                        X_scaled = X.copy()
                        X_scaled.iloc[:, 0] = X_scaled.iloc[:, 0] * scaling_multiplier
                        return X_scaled
                    test_scenarios.append((f"Feature scaling shift (×{scaling_multiplier})", scaling_scenario))
                    
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
                    
                    # Display results
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
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tests", total_tests)
                    with col2:
                        st.metric("Passed Tests", passed_tests, f"{passed_tests}/{total_tests}")
                    with col3:
                        if total_tests > 0:
                            pass_rate = (passed_tests / total_tests) * 100
                            st.metric("Pass Rate", f"{pass_rate:.1f}%")
                    
                    # Visualization
                    if total_tests > 0:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # RMSE comparison
                        test_names = [r["Scenario"] for r in test_results]
                        rmse_values = [float(r["RMSE"]) for r in test_results]
                        colors = ['green' if r["Status"] == "PASS" else 'red' for r in test_results]
                        
                        ax1.barh(test_names, rmse_values, color=colors, alpha=0.7)
                        ax1.axvline(baseline_rmse, color='blue', linestyle='--', label='Baseline RMSE')
                        ax1.set_xlabel('RMSE')
                        ax1.set_title('RMSE by Scenario')
                        ax1.legend()
                        plt.setp(ax1.get_yticklabels(), rotation=0, ha="right")
                        
                        # Pass/Fail pie chart
                        status_counts = pd.Series([r["Status"] for r in test_results]).value_counts()
                        if len(status_counts) > 0:
                            ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                                   colors=['lightgreen' if x == 'PASS' else 'lightcoral' for x in status_counts.index])
                            ax2.set_title('Test Results Distribution')
                        
                        st.pyplot(fig)
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.error("Please make sure your model file is compatible and the dataset format is correct.")

if __name__ == "__main__":
    main()