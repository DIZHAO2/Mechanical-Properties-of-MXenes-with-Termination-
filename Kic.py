import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# ================= Plotting Configuration =================
try:
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans'] 
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# ================= Configuration Area =================

# 1. Input file path
INPUT_PATH = r"C:\Users\s5362998\Desktop\PPT\MBene_machinelearning\V27\pearson+spearman\Selected_Features_KIC_MPa_m1_2.xlsx"

# 2. Output folder path
OUTPUT_DIR = r"C:\Users\s5362998\Desktop\PPT\MBene_machinelearning\V27\KIC\V29"

# 3. Target variable name
TARGET = "KIC MPa m1/2"

# 4. Chemical formula column name
FORMULA_COL = "Formula" 

# 5. Feature list
FEATURES = [
    "is bare", "has O", "has F", "number of X layers", "mean NpUnfilled",
    "avg_dev NdValence", "avg_dev MendeleevNumber", "mean MendeleevNumber",
    "mean NdUnfilled", "avg_dev NfValence", "minimum NValence",
    "Electroneg_diff_CN_TM", "avg_dev NUnfilled", "avg_dev Row", "range SpaceGroupNumber"
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= Helper Functions =================

def evaluate_model(name, model, param_grid, X_train, y_train, X_test, y_test):
    print(f"\n⚡ Training model: {name} ...")
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    # Predict on training and test sets
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    # --- [Feature] Export training set regression data to excel ---
    reg_data_train = pd.DataFrame({
        'Actual_Value': y_train,
        'Predicted_Value': y_train_pred
    })
    reg_data_train.to_excel(os.path.join(OUTPUT_DIR, f'{name}_Train_Regression_Data.xlsx'), index=False)
    
    # --- [Feature] Export test set regression data to excel ---
    reg_data_test = pd.DataFrame({
        'Actual_Value': y_test,
        'Predicted_Value': y_test_pred
    })
    reg_data_test.to_excel(os.path.join(OUTPUT_DIR, f'{name}_Test_Regression_Data.xlsx'), index=False)
    
    # --- [Feature] Plot regression graph including Train and Test ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_train_pred, color='lightgray', alpha=0.6, label=f'Train R2={r2_train:.3f}')
    plt.scatter(y_test, y_test_pred, color='blue', alpha=0.7, label=f'Test R2={r2_test:.3f}')
    
    combined_min = min(min(y_train), min(y_test))
    combined_max = max(max(y_train), max(y_test))
    plt.plot([combined_min, combined_max], [combined_min, combined_max], 'r--', lw=2)
    
    plt.xlabel(f'Actual {TARGET}')
    plt.ylabel(f'Predicted {TARGET}')
    plt.title(f'{name} Regression Analysis')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_Regression_Plot.png'), dpi=300)
    plt.close()
    
    print(f"    >> [{name}] Regression plot and Train/Test data saved. Train R2: {r2_train:.4f}, Test R2: {r2_test:.4f}")
    return best_model, r2_test

def run_shap_analysis(name, model, X_train_scaled, X_train_df, feature_names, original_df):
    print(f"    >> [{name}] Performing SHAP analysis (No Bare samples)...")
    mask = X_train_df['is bare'] == 0
    X_shap_input = X_train_scaled[mask]
    X_shap_df = X_train_df[mask]

    # SHAP is primarily performed for XGBoost here; SVR would require KernelExplainer
    if name == 'XGBoost':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_shap_input)
    else: 
        return None, None

    # Export Summary Data
    shap_cols = [f"SHAP_{f}" for f in feature_names]
    shap_df_vals = pd.DataFrame(shap_values, columns=shap_cols)
    raw_df_vals = X_shap_df.reset_index(drop=True)
    
    if FORMULA_COL in original_df.columns:
        formulas = original_df.loc[X_shap_df.index, FORMULA_COL].reset_index(drop=True)
        summary_final = pd.concat([formulas, raw_df_vals, shap_df_vals], axis=1)
        summary_final.to_excel(os.path.join(OUTPUT_DIR, f'{name}_SHAP_Summary_Data.xlsx'), index=False)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_shap_df, show=False, cmap='coolwarm')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_SHAP_Summary.jpg'), dpi=300)
    plt.close()
    
    return shap_values, X_shap_df

# ================= Main Program =================

def main():
    # Load data
    df = pd.read_excel(INPUT_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Configuration: XGBoost, SVR, RandomForest
    models_config = [
        {
            'name': 'XGBoost',
            'model': XGBRegressor(random_state=42),
            'params': {
                'n_estimators': [800], 'max_depth': [3], 'learning_rate': [0.02],
                'subsample': [0.7], 'colsample_bytree': [0.6], 'gamma': [0.1], 'min_child_weight': [5]
            }
        },
        {
            'name': 'SVR',
            'model': SVR(),
            'params': {
                'C': [1, 10, 100], 'gamma': [0.1, 0.01], 'kernel': ['rbf']
            }
        },
        {
            'name': 'RandomForest',
            'model': RandomForestRegressor(random_state=42),
            'params': {
                'n_estimators': [100, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]
            }
        }
    ]

    for config in models_config:
        model, r2 = evaluate_model(config['name'], config['model'], config['params'], 
                                   X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Deep SHAP interaction analysis for XGBoost specifically
        if config['name'] == 'XGBoost':
            xgb_shap, xgb_X_df = run_shap_analysis(config['name'], model, X_train_scaled, X_train, FEATURES, df)
            
            # Execute specified interaction analysis pairs
            target_interactions = [
                ("number of X layers", "avg_dev MendeleevNumber"),
                ("number of X layers", "avg_dev NdValence"),
                ("avg_dev NdValence", "mean NpUnfilled"),
                ("number of X layers", "mean NpUnfilled")
            ]
            
            for idx, (feat_x, feat_color) in enumerate(target_interactions, 1):
                safe_x, safe_color = feat_x.replace(" ", "_"), feat_color.replace(" ", "_")
                
                # Dependence Plot
                plt.figure(figsize=(8, 6))
                shap.dependence_plot(
                    ind=feat_x, shap_values=xgb_shap, features=xgb_X_df.values,
                    feature_names=FEATURES, interaction_index=feat_color, show=False
                )
                plt.savefig(os.path.join(OUTPUT_DIR, f'XGB_Interact_{idx}_{safe_x}_vs_{safe_color}.png'), dpi=300)
                plt.close()

                # Save interaction data to Excel
                inter_df = pd.DataFrame({
                    'Formula': df.loc[xgb_X_df.index, FORMULA_COL].values if FORMULA_COL in df.columns else "N/A",
                    f'{feat_x}_Raw': xgb_X_df[feat_x].values,
                    f'{feat_x}_SHAP': xgb_shap[:, FEATURES.index(feat_x)],
                    f'{feat_color}_Interact_Raw': xgb_X_df[feat_color].values
                })
                inter_df.to_excel(os.path.join(OUTPUT_DIR, f'XGB_Data_{idx}_{safe_x}_vs_{safe_color}.xlsx'), index=False)

    print(f"\n✅ Tasks completed! Regression analysis and interaction data saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()