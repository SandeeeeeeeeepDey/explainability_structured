from src.utils import load_data, load_model
import warnings

warnings.filterwarnings('ignore')

# Load the model and data using utils
model = load_model('./models/xgb_first_attempt.pkl')
test_df = load_data("./data/proc_test_first_attempt.csv")

test_df_x = test_df.drop(['folds', 'id','born_project', 'deeper_target', 'createdAt_x', 'deeper_target_real', 'born_source', 'slice'], axis=1)
test_df_x = test_df_x.astype(float)

def processing(lead_id, lead_df_x=test_df_x, inference_copy=test_df.copy()):

    expected_feature_order = model.get_booster().feature_names
    lead_df_x = lead_df_x[expected_feature_order]

    return 