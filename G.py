import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
class_weight = None
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# ================= Plotting Configuration =================
try:
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# ================= Configuration Area =================

# 1. Path Configuration
INPUT_PATH = r"C:\Users\s5362998\Desktop\PPT\MBene_machinelearning\V27\pearson+spearman\Selected_Features_Shear_modulus_N_m.xlsx"
OUTPUT_DIR = r"C:\Users\s5362998\Desktop\PPT\MBene_machinelearning\V27\G\V29"

# 2. Field Configuration
FORMULA_COL = "Formula" # Chemical formula column name
TARGET = "Shear modulus N/m"

FEATURES = [
    "is bare", "has O", "has F", "number of X layers", "mean NpUnfilled",
    "avg_dev MendeleevNumber", "avg_dev NdValence", "mean MendeleevNumber",
    "avg_dev Row", "avg_dev NpUnfilled", "avg_dev NfValence",
    "minimum NValence", "avg_dev NUnfilled", "avg_dev Electronegativity", "mean NdUnfilled"
]

# 3. Interaction Analysis Pairs
INTERACTION_PAIRS = [
    ("number of X layers", "avg_dev NdValence"),
    ("number of X layers", "has O"),
    ("number of X layers", "mean MendeleevNumber"),
    ("number of X layers", "mean NpUnfilled"),
    ("avg_dev NdValence", "mean NpUnfilled") 
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Output directory created: {OUTPUT_DIR}")

# ================= Helper Functions =================

def plot_regression(y_train, y_train_pred, y_test, y_test_pred, model_name, r2_test, r2_train, save_dir):
    """Plots Predicted vs Actual values and exports regression data to Excel."""
    
    # --- Export Train Regression Data to Excel ---
    train_data = pd.DataFrame({
        'Actual_Value': y_train,
        'Predicted_Value': y_train_pred
    })
    train_data.to_excel(os.path.join(save_dir, f'{model_name}_Train_Regression_Data.xlsx'), index=False)

    # --- Export Test Regression Data to Excel ---
    test_data = pd.DataFrame({
        'Actual_Value': y_test,
        'Predicted_Value': y_test_pred
    })
    test_data.to_excel(os.path.join(save_dir, f'{model_name}_Test_Regression_Data.xlsx'), index=False)
    
    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_train_pred, color='lightgray', alpha=0.6, label=f'Train (R2={r2_train:.2f})')
    plt.scatter(y_test, y_test_pred, color='red', alpha=0.7, label=f'Test (R2={r2_test:.2f})')
    
    all_vals = np.concatenate([y_train, y_test])
    min_val, max_val = min(all_vals), max(all_vals)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    plt.xlabel(f'Actual {TARGET}')
    plt.ylabel(f'Predicted {TARGET}')
    plt.title(f'{model_name} Regression Plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_Regression_Plot.png'), dpi=300)
    plt.close()
    
    print(f"    >> [{model_name}] Exported Train and Test regression data to Excel")

def run_shap_analysis(model, X_df_scaled, X_raw, model_name, save_dir, full_df):
    """Performs SHAP analysis and exports summary/interaction data."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_df_scaled)
        
        if isinstance(shap_values, list): sv = shap_values[0]
        else: sv = shap_values

        # --- 1. Export SHAP Summary Plot Data ---
        shap_cols = [f"SHAP_{f}" for f in FEATURES]
        shap_df_vals = pd.DataFrame(sv, columns=shap_cols)
        raw_df_vals = X_raw.reset_index(drop=True)
        
        if FORMULA_COL in full_df.columns:
            formulas = full_df.loc[X_raw.index, FORMULA_COL].reset_index(drop=True)
            summary_final = pd.concat([formulas, raw_df_vals, shap_df_vals], axis=1)
        else:
            summary_final = pd.concat([raw_df_vals, shap_df_vals], axis=1)
        
        summary_final.to_excel(os.path.join(save_dir, f'{model_name}_SHAP_Summary_Plot_Data.xlsx'), index=False)

        # 2. Save Summary Plot Image
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_df_scaled, show=False, cmap='coolwarm')
        plt.title(f'SHAP Summary: {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{model_name}_SHAP_Summary.jpg'), dpi=300)
        plt.close()

        # 3. Interaction Plots and Data
        X_train_np = X_raw.values.astype(float)
        for feat_x, feat_color in INTERACTION_PAIRS:
            if feat_x in FEATURES and feat_color in FEATURES:
                safe_x = feat_x.replace(" ", "_")
                safe_color = feat_color.replace(" ", "_")

                try:
                    # Generate Dependence Plot
                    plt.figure(figsize=(8, 6))
                    shap.dependence_plot(
                        ind=feat_x, shap_values=sv, features=X_train_np,
                        feature_names=FEATURES, interaction_index=feat_color,
                        show=False, cmap='viridis'
                    )
                    plt.title(f'{model_name}: {feat_x} vs {feat_color}', fontsize=12)
                    plt.savefig(os.path.join(save_dir, f'{model_name}_Interact_{safe_x}_vs_{safe_color}.png'), dpi=300)
                    plt.close()

                    # Export Interaction Data to Excel
                    idx_x = FEATURES.index(feat_x)
                    interaction_df = pd.DataFrame({
                        'Formula': full_df.loc[X_raw.index, FORMULA_COL].values if FORMULA_COL in full_df.columns else "N/A",
                        f'{feat_x}_Raw': X_raw[feat_x].values,
                        f'{feat_x}_SHAP': sv[:, idx_x],
                        f'{feat_color}_Interact_Raw': X_raw[feat_color].values
                    })
                    interaction_df.to_excel(os.path.join(save_dir, f'{model_name}_Data_{safe_x}_vs_{safe_color}.xlsx'), index=False)
                except Exception as e:
                    print(f"      ⚠️ Interaction Plot failed ({feat_x} vs {feat_color}): {e}")

    except Exception as e:
        print(f"    ⚠️ SHAP analysis error: {e}")

# ================= Main Program =================

def main():
    print("="*60)
    print(f"      MBene {TARGET} Prediction System (Multi-Model Enhanced)")
    print("="*60)

    # --- 1. Data Loading ---
    df = pd.read_excel(INPUT_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    X = df[FEATURES]
    y = df[TARGET]

    # --- 2. Data Splitting & Preprocessing ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=FEATURES)

    # --- 3. Model Configuration ---
    models_config = {
        'XGBoost': {
            'model': XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [3000],
                'learning_rate': [0.01],
                'max_depth': [5],
                'min_child_weight': [5],
                'gamma': [0.5],
                'subsample': [0.7],
                'colsample_bytree': [0.6],
                'reg_alpha': [2],
                'reg_lambda': [1]
            },
            'type': 'tree'
        },
        'SVR': {
            'model': SVR(),
            'params': {'C': [100], 'gamma': [0.1], 'kernel': ['rbf']},
            'type': 'kernel'
        },
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 300], 
                'max_depth': [None, 10, 20], 
                'min_samples_split': [2, 5]
            },
            'type': 'tree'
        }
    }

    all_metrics = []

    # --- 4. Training Loop ---
    for model_name, config in models_config.items():
        print(f"\n⚡ Processing model: {model_name}")
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        
        best_model = grid.best_estimator_
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)
        
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        
        all_metrics.append({
            'Model': model_name, 'R2_Train': r2_train, 'R2_Test': r2_test,
            'Best_Params': str(grid.best_params_)
        })
        
        # Plotting and Exporting Regression Data
        plot_regression(y_train, y_train_pred, y_test, y_test_pred, model_name, r2_test, r2_train, OUTPUT_DIR)

        # --- 5. SHAP Analysis (Specifically for Tree models/XGBoost) ---
        if config['type'] == 'tree' and model_name == 'XGBoost':
            print(f"    >> Calculating SHAP values and exporting data for {model_name}...")
            run_shap_analysis(best_model, X_train_scaled_df, X_train, model_name, OUTPUT_DIR, df)

    # --- 6. Save Performance Metrics ---
    pd.DataFrame(all_metrics).to_csv(os.path.join(OUTPUT_DIR, 'model_performance.csv'), index=False)
    print(f"\n✅ Task completed! Results stored in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()