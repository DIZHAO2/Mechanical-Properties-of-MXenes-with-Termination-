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
INPUT_PATH = r"C:\Users\s5362998\Desktop\PPT\MBene_machinelearning\V27\pearson+spearman\Selected_Features_bulk_modulus_N_m.xlsx"

# 2. Output directory path
OUTPUT_DIR = r"C:\Users\s5362998\Desktop\PPT\MBene_machinelearning\V27\Bulk\V29"

# 3. Target variable name
TARGET = "bulk modulus N/m"

# 4. Chemical formula column name
FORMULA_COL = "Formula" 

# 5. Feature list
FEATURES = [
    "is bare",
    "has O",
    "has F",
    "number of X layers",
    "mean NpUnfilled",
    "avg_dev NdValence",
    "range SpaceGroupNumber",
    "mean NdUnfilled",
    "mean MendeleevNumber",
    "mode MendeleevNumber",
    "avg_dev Row",
    "minimum Electronegativity",
    "Electroneg_diff_CN_TM",
    "avg_dev NfValence",
    "maximum SpaceGroupNumber"
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Output directory created: {OUTPUT_DIR}")

# ================= Helper Functions =================

def evaluate_model(name, model, param_grid, X_train, y_train, X_test, y_test, feature_names):
    print(f"\n⚡ Training model: {name} ...")
    grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    
    # Predict on training and testing sets
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    # --- [Feature] Export TRAIN regression data to excel ---
    reg_data_train = pd.DataFrame({
        'Actual_Value': y_train,
        'Predicted_Value': y_train_pred
    })
    reg_data_train.to_excel(os.path.join(OUTPUT_DIR, f'{name}_Train_Regression_Data.xlsx'), index=False)
    print(f"    >> Exported {name} training set data")

    # --- [Feature] Export TEST regression data to excel ---
    reg_data_test = pd.DataFrame({
        'Actual_Value': y_test,
        'Predicted_Value': y_test_pred
    })
    reg_data_test.to_excel(os.path.join(OUTPUT_DIR, f'{name}_Test_Regression_Data.xlsx'), index=False)
    print(f"    >> Exported {name} test set data")
    
    # Plot test set scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_test_pred, color='blue', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Value')
    plt.ylabel('Predicted Value')
    plt.title(f'{name}: Test R2 = {r2_test:.3f}')
    plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_Regression.png'), dpi=300)
    plt.close()
    
    metrics = {
        'Model': name, 'R2_Train': r2_train, 
        'R2_Test': r2_test, 'RMSE_Test': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'MAE_Test': mean_absolute_error(y_test, y_test_pred), 'Best_Params': str(grid.best_params_)
    }
    return best_model, metrics

def explain_model_shap(name, model, X_train_scaled, X_train_df, feature_names, original_df):
    print(f"    >> [{name}] Preparing SHAP analysis...")
    if 'is bare' in X_train_df.columns:
        mask_not_bare = X_train_df['is bare'] == 0
        X_shap_input = X_train_scaled[mask_not_bare]
        X_shap_df = X_train_df[mask_not_bare]
    else:
        X_shap_input = X_train_scaled
        X_shap_df = X_train_df

    try:
        if name in ['XGBoost', 'RandomForest']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap_input)
        elif name == 'SVR':
            X_summary = shap.kmeans(X_shap_input, min(50, len(X_shap_input))) 
            explainer = shap.KernelExplainer(model.predict, X_summary)
            shap_values = explainer.shap_values(X_shap_input)
        else: return None, None, None

        if isinstance(shap_values, list): shap_values = shap_values[0]

        # Export SHAP Summary Plot data to excel
        shap_df_export = pd.DataFrame(shap_values, columns=[f"SHAP_{f}" for f in feature_names])
        raw_values_export = X_shap_df.reset_index(drop=True)
        
        if FORMULA_COL in original_df.columns:
            formulas = original_df.loc[X_shap_df.index, FORMULA_COL].reset_index(drop=True)
            summary_data_final = pd.concat([formulas, raw_values_export, shap_df_export], axis=1)
        else:
            summary_data_final = pd.concat([raw_values_export, shap_df_export], axis=1)
        
        summary_data_final.to_excel(os.path.join(OUTPUT_DIR, f'{name}_SHAP_Summary_Plot_Data.xlsx'), index=False)

        # Plot SHAP Summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_shap_df, show=False, cmap='coolwarm')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_SHAP_Summary.jpg'), dpi=300)
        plt.close()

        # Export mean absolute SHAP values (Feature Importance)
        vals = np.abs(shap_values).mean(0)
        pd.DataFrame(list(zip(feature_names, vals)), columns=['Feature', 'Mean_SHAP']).to_csv(
            os.path.join(OUTPUT_DIR, f'{name}_Importance.csv'), index=False)
        
        return shap_values, explainer, X_shap_df
    except Exception as e:
        print(f"    ⚠️ SHAP analysis failed: {e}")
        return None, None, None

# ================= Main Program =================

def main():
    # Load data
    df = pd.read_excel(INPUT_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = X.columns.tolist()

    # Model configurations
    models_config = [
        {'name': 'XGBoost', 'model': XGBRegressor(random_state=42), 'params': {'max_depth': [3], 'n_estimators': [500]}},
        {'name': 'RandomForest', 'model': RandomForestRegressor(random_state=42), 'params': {'max_depth': [10, 20], 'n_estimators': [100, 200]}},
        {'name': 'SVR', 'model': SVR(), 'params': {'C': [100], 'gamma': [0.01], 'kernel': ['rbf']}}
    ]

    svr_data_bundle = None
    all_metrics = []

    for config in models_config:
        model, metrics = evaluate_model(config['name'], config['model'], config['params'], 
                                        X_train_scaled, y_train, X_test_scaled, y_test, feature_names)
        all_metrics.append(metrics)
        
        shap_values, _, X_shap_filtered = explain_model_shap(config['name'], model, X_train_scaled, X_train, feature_names, df)
        
        if config['name'] == 'SVR' and shap_values is not None:
            svr_data_bundle = {'shap_values': shap_values, 'X_shap_df': X_shap_filtered, 'feature_names': feature_names}

    # Save performance summary
    pd.DataFrame(all_metrics).to_excel(os.path.join(OUTPUT_DIR, 'Model_Performance_Comparison.xlsx'), index=False)

    # ================= SVR Specific Interaction Analysis =================
    if svr_data_bundle:
        svr_shap = svr_data_bundle['shap_values']
        svr_X_df = svr_data_bundle['X_shap_df']
        svr_names = svr_data_bundle['feature_names']
        svr_X_np = svr_X_df.values.astype(float)

        target_interactions = [
            ("number of X layers", "avg_dev NdValence"),
            ("avg_dev NdValence", "mean NpUnfilled"),
            ("number of X layers", "mean NpUnfilled")
        ]
        
        for feat_x, feat_color in target_interactions:
            if feat_x not in svr_names or feat_color not in svr_names: continue
            safe_x, safe_color = feat_x.replace(" ", "_"), feat_color.replace(" ", "_")
            
            # Generate Dependence Plot
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(ind=feat_x, shap_values=svr_shap, features=svr_X_np,
                                 feature_names=svr_names, interaction_index=feat_color, show=False)
            plt.savefig(os.path.join(OUTPUT_DIR, f'SVR_Interact_{safe_x}_vs_{safe_color}.png'), dpi=300)
            plt.close()

            # Export Interaction Data
            feat_idx = svr_names.index(feat_x)
            out_df = pd.DataFrame({
                'Formula': df.loc[svr_X_df.index, FORMULA_COL].values if FORMULA_COL in df.columns else "N/A",
                f'{feat_x}_Raw': svr_X_df[feat_x].values,
                f'{feat_x}_SHAP': svr_shap[:, feat_idx],
                f'{feat_color}_Interact_Raw': svr_X_df[feat_color].values
            })
            out_df.to_excel(os.path.join(OUTPUT_DIR, f'SVR_Data_{safe_x}_vs_{safe_color}.xlsx'), index=False)

    print(f"\nAll tasks completed! Data exported to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()