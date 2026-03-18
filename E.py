import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
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

# 1. Input file path
INPUT_PATH = r"C:\Users\s5362998\Desktop\PPT\MBene_machinelearning\V27\pearson+spearman\Selected_Features_E(y)_N_m.xlsx"

# 2. Output folder path
OUTPUT_DIR = r"C:\Users\s5362998\Desktop\PPT\MBene_machinelearning\V27\Ey\V29"

# 3. Feature list
FEATURES = [
    "is bare",
    "has O",
    "has F",
    "number of X layers",
    "mean NpUnfilled",
    "avg_dev MendeleevNumber",
    "avg_dev NdValence",
    "mean MendeleevNumber",
    "avg_dev Row",
    "minimum NValence",
    "mean NdUnfilled",
    "range SpaceGroupNumber",
    "avg_dev NfValence",
    "minimum Electronegativity",
    "Electroneg_diff_CN_TM"
]

# 4. Target variable name
TARGET = "E(y) N/m"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# ================= Helper Functions =================

def get_shap_explainer(model_name, model, X_train_scaled):
    """
    Returns the corresponding SHAP Explainer and SHAP Values based on model type.
    """
    try:
        if model_name in ['XGBoost', 'RandomForest']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train_scaled)
        elif model_name == 'SVR':
            # SVR requires KernelExplainer; clustering background data for speed
            # Use K-means to reduce background data (e.g., to 50 center points)
            X_summary = shap.kmeans(X_train_scaled, min(50, len(X_train_scaled)))
            explainer = shap.KernelExplainer(model.predict, X_summary)
            shap_values = explainer.shap_values(X_train_scaled)
        else:
            return None, None
        
        # Handle potential list outputs (specific versions of RF or multi-output)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        return explainer, shap_values
    except Exception as e:
        print(f"      ⚠️ Failed to create SHAP Explainer ({model_name}): {e}")
        return None, None

# ================= Main Program =================

def main():
    print("="*40)
    print("      MBene E(y) Multi-Model Prediction & SHAP Analysis")
    print("      (Train on All Data | SHAP excluding Bare)")
    print("="*40)

    # --- 1. Data Loading ---
    print(f"Reading data: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print(f"❌ Error: Cannot find file {INPUT_PATH}")
        return

    try:
        df = pd.read_excel(INPUT_PATH)
    except Exception as e:
        print(f"❌ Failed to read file: {e}")
        return

    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [col for col in FEATURES + [TARGET] if col not in df.columns]
    if missing_cols:
        print(f"❌ Error: Missing columns in data: {missing_cols}")
        return

    X = df[FEATURES]
    y = df[TARGET]
    print(f"✅ Data loaded successfully. Samples: {len(X)}, Features: {len(FEATURES)}")

    # --- 2. Data Splitting & Preprocessing ---
    # Training on full feature set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 3. Model Configuration ---
    models_config = [
        {
            'name': 'XGBoost',
            'model': XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [300, 500],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        },
        {
            'name': 'SVR',
            'model': SVR(),
            'params': {
                'C': [1, 10, 100],
                'gamma': ['scale', 0.1],
                'kernel': ['rbf']
            }
        },
        {
            'name': 'RandomForest',
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 300],
                'max_depth': [None, 10],
                'min_samples_leaf': [1, 2]
            }
        }
    ]

    all_metrics = []
    best_overall_r2 = -float('inf')
    best_overall_model_name = ""
    best_overall_shap_values = None
    best_overall_X_shap_raw = None # For final deep interaction analysis

    # --- 4. Training and Evaluation Loop ---
    for config in models_config:
        name = config['name']
        print(f"\n⚡ Processing model: {name} ...")

        # A. Training (Using full training data)
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        
        print(f"   [{name}] Best Parameters: {grid.best_params_}")

        # B. Prediction & Evaluation
        y_train_pred = best_model.predict(X_train_scaled)
        y_test_pred = best_model.predict(X_test_scaled)

        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae_test = mean_absolute_error(y_test, y_test_pred)

        print(f"   [{name}] Test R2: {r2_test:.4f}, RMSE: {rmse_test:.4f}")

        # Record Metrics
        metrics = {
            'Model': name,
            'R2_Train': r2_train,
            'R2_Test': r2_test,
            'RMSE_Test': rmse_test,
            'MAE_Test': mae_test,
            'Best_Params': str(grid.best_params_)
        }
        all_metrics.append(metrics)
        
        # --- Export Prediction Data to Excel ---
        
        # 1. Export Training Set Regression Data
        reg_data_train = pd.DataFrame({
            'Actual_Value': y_train,
            'Predicted_Value': y_train_pred
        })
        reg_data_train.to_excel(os.path.join(OUTPUT_DIR, f'{name}_Train_Regression_Data.xlsx'), index=False)
        print(f"   [{name}] Exported Training prediction Excel")

        # 2. Export Test Set Regression Data
        reg_data_test = pd.DataFrame({
            'Actual_Value': y_test,
            'Predicted_Value': y_test_pred
        })
        reg_data_test.to_excel(os.path.join(OUTPUT_DIR, f'{name}_Test_Regression_Data.xlsx'), index=False)
        print(f"   [{name}] Exported Test prediction Excel")
        
        # C. Plot Regression Graph
        plt.figure(figsize=(6, 6))
        plt.scatter(y_train, y_train_pred, color='gray', alpha=0.3, label='Train')
        plt.scatter(y_test, y_test_pred, color='blue', alpha=0.7, label='Test')
        
        combined_min = min(min(y_train), min(y_test))
        combined_max = max(max(y_train), max(y_test))
        plt.plot([combined_min, combined_max], [combined_min, combined_max], 'r--', lw=2)
        
        plt.xlabel(f'Actual {TARGET}')
        plt.ylabel(f'Predicted {TARGET}')
        plt.title(f'{name}: Test R2 = {r2_test:.3f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_Regression_Plot.png'), dpi=300)
        plt.close()

        # D. SHAP Analysis (Excluding Bare samples)
        print(f"   [{name}] Calculating SHAP (Excluding Bare samples)...")
        
        # 1. Filter Data
        if 'is bare' in X_train.columns:
            mask_not_bare = X_train['is bare'] == 0
            X_shap_input = X_train_scaled[mask_not_bare] # Scaled data for model input
            X_shap_raw = X_train[mask_not_bare]          # Raw data for labels and plotting
        else:
            X_shap_input = X_train_scaled
            X_shap_raw = X_train

        # 2. Compute SHAP Values
        explainer, shap_values = get_shap_explainer(name, best_model, X_shap_input)

        if shap_values is not None:
            # 3. Save Feature Importance CSV
            vals = np.abs(shap_values).mean(0)
            feat_imp = pd.DataFrame(list(zip(FEATURES, vals)), columns=['Feature', 'Mean_SHAP_Value'])
            feat_imp.sort_values(by='Mean_SHAP_Value', ascending=False, inplace=True)
            feat_imp.to_csv(os.path.join(OUTPUT_DIR, f'{name}_Feature_Importance_NoBare.csv'), index=False)
            print(f"      Saved importance table: {name}_Feature_Importance_NoBare.csv")

            # 4. Draw Summary Plot
            plt.figure(figsize=(10, 8))
            X_shap_input_df = pd.DataFrame(X_shap_input, columns=FEATURES) # Temp DF for column labels
            shap.summary_plot(shap_values, X_shap_input_df, show=False, cmap='coolwarm')
            plt.title(f'SHAP Summary (No Bare): {name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'{name}_SHAP_Summary_NoBare.jpg'), dpi=300)
            plt.close()

            # Record overall best model for deep analysis
            if r2_test > best_overall_r2:
                best_overall_r2 = r2_test
                best_overall_model_name = name
                best_overall_shap_values = shap_values
                best_overall_X_shap_raw = X_shap_raw

    # --- 5. Save Performance Summary ---
    metrics_df = pd.DataFrame(all_metrics)
    # Adjust column order
    cols = ['Model', 'R2_Train', 'R2_Test', 'RMSE_Test', 'MAE_Test', 'Best_Params']
    metrics_df = metrics_df[cols]
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'final_model_metrics.csv'), index=False)
    print(f"\n✅ All model metrics saved: final_model_metrics.csv")
    print(metrics_df)

    # =========================================================
    # Deep Interaction Analysis (Only for the best overall model)
    # =========================================================
    if best_overall_shap_values is not None:
        print("\n" + "="*50)
        print(f"★ Performing deep interaction analysis on best model [{best_overall_model_name}] (R2={best_overall_r2:.4f})")
        print("="*50)
        
        # Prepare Data
        X_shap_raw_np = best_overall_X_shap_raw.values.astype(float)
        feature_names_list = FEATURES
        sv = best_overall_shap_values

        # Define Interaction Pairs
        interaction_pairs = [
            ("number of X layers", "avg_dev NdValence"),
            ("number of X layers", "has O"),
            ("number of X layers", "mean MendeleevNumber"),
            ("number of X layers", "mean NpUnfilled")
        ]
        
        for feat_x, feat_color in interaction_pairs:
            if feat_x in FEATURES and feat_color in FEATURES:
                safe_x = feat_x.replace(" ", "_").replace("/", "_")
                safe_color = feat_color.replace(" ", "_").replace("/", "_")

                # A. Plotting
                try:
                    plt.figure(figsize=(8, 6))
                    shap.dependence_plot(
                        ind=feat_x,
                        shap_values=sv,
                        features=X_shap_raw_np,
                        feature_names=feature_names_list,
                        interaction_index=feat_color,
                        show=False,
                        alpha=0.8,
                        dot_size=40,
                        cmap='viridis'
                    )
                    plt.title(f'{best_overall_model_name} Interaction: {feat_x} vs {feat_color} (No Bare)', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(OUTPUT_DIR, f'BestModel_{best_overall_model_name}_Interact_{safe_x}_vs_{safe_color}.png'), dpi=300)
                    plt.close()
                except Exception as e:
                    print(f"      ⚠️ Plotting failed: {e}")

                # B. Data Export
                try:
                    x_data = best_overall_X_shap_raw[feat_x].values
                    c_data = best_overall_X_shap_raw[feat_color].values
                    feat_idx = FEATURES.index(feat_x)
                    y_shap_data = sv[:, feat_idx]
                    
                    interaction_df = pd.DataFrame({
                        f'{feat_x} (Raw Value)': x_data,
                        f'{feat_x} SHAP Value': y_shap_data,
                        f'{feat_color} (Interaction Value)': c_data
                    })
                    interaction_df.sort_values(by=f'{feat_x} (Raw Value)', inplace=True)
                    interaction_df.to_excel(os.path.join(OUTPUT_DIR, f'BestModel_{best_overall_model_name}_Data_{safe_x}_vs_{safe_color}.xlsx'), index=False)
                    print(f"      Exported interaction data: {feat_x} vs {feat_color}")
                except Exception as e:
                    print(f"      ⚠️ Data export failed: {e}")

    print("\n" + "="*50)
    print(f"Tasks completed successfully! Results saved in: {OUTPUT_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()