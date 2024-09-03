import pandas as pd
import shap
import warnings
import os 
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import load_data, load_model

warnings.filterwarnings('ignore')

# Correct the paths to the model and data
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, '../models/xgb_first_attempt.pkl')  # Corrected
data_path = os.path.join(script_dir, '../data/proc_test_first_attempt.csv')  # Corrected

# Load the model and data using utils
model = load_model(model_path)
test_df = load_data(data_path)
explainer = shap.Explainer(model)

def make_inference(lead_id, inference_copy=test_df.copy()):
    lead_df = inference_copy[inference_copy['id'] == lead_id]

    lead_df_x = inference_copy[inference_copy['id'] == lead_id]
    lead_df_x = lead_df_x.drop(['folds', 'id','born_project', 'deeper_target', 'createdAt_x', 'deeper_target_real', 'born_source', 'slice'], axis=1)
    lead_df_x = lead_df_x.astype(float)
    expected_feature_order = model.get_booster().feature_names
    lead_df_x = lead_df_x[expected_feature_order]
    lead_df['predicted_prob'] = model.predict_proba(lead_df_x)[0, 1]
    lead_df['predicted_class'] = (lead_df['predicted_prob'] > 0.055).astype(int)
    # lead_df.sort_values(by='predicted_prob', ascending=False, inplace=True)
    predicted_class = ['Interested' if lead_df['predicted_class'].iloc[0] == 1 else 'Not Interested']

    return {
        'lead_id': lead_id,
        'predicted_prob': float(model.predict_proba(lead_df_x)[0, 1]),
        'prediction_label': predicted_class[0]
    }
