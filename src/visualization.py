import shap
import matplotlib.pyplot as plt
import numpy as np


# def explain_result(lead_df_x, explainer):
#     shap_values = explainer.shap_values(lead_df_x)
#     plt.figure(figsize=(40, 6))
#     shap.force_plot(explainer.expected_value, shap_values[0], lead_df_x.iloc[0], matplotlib=True)
#     plt.show()
    
#     shap_row = explainer(lead_df_x)
#     shap.waterfall_plot(shap_row[0])
#     plt.show()




# def plot_recall_metrics(models, recall_interested, recall_not_interested):
#     print("plot_recall_metrics")
#     bar_width = 0.35
#     index = np.arange(len(models))

#     fig, ax = plt.subplots()
#     bar1 = ax.bar(index, recall_interested, bar_width, label='Interested', color='#2108')
#     bar2 = ax.bar(index + bar_width, recall_not_interested, bar_width, label='Not Interested', color='#168')

#     ax.set_xlabel('Models')
#     ax.set_ylabel('Recall')
#     ax.set_title('Recall Metrics for Interested and Not Interested Classes')
#     ax.set_xticks(index + bar_width / 2)
#     ax.set_xticklabels(models)
#     ax.legend()

#     def add_labels(bars):
#         for bar in bars:
#             height = bar.get_height()
#             ax.annotate(f'{height:.2f}',
#                         xy=(bar.get_x() + bar.get_width() / 2, height),
#                         xytext=(0, 3),  # 3 points vertical offset
#                         textcoords="offset points",
#                         ha='center', va='bottom')

#     add_labels(bar1)
#     add_labels(bar2)
#     ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=2)
#     plt.tight_layout()
#     plt.show()





################### CODE GENERATED ##############################
def explain_result(lead_df, explainer, lead_id):
    
    lead_df_x = lead_df[lead_df["id"] == lead_id]
    # print(lead_df_x)
    lead_df_x = lead_df_x.drop(['folds', 'id','born_project', 'deeper_target', 'createdAt_x', 'deeper_target_real', 'born_source', 'slice'], axis=1)
    lead_df_x = lead_df_x.astype(float)
    shap_values = explainer.shap_values(lead_df_x)
    expected_value = float(explainer.expected_value)  # Convert numpy.float32 to Python float

    # Convert SHAP values and feature values to lists of floats
    shap_values_converted = shap_values.astype(float).tolist()
    feature_values_converted = lead_df_x.iloc[0].astype(float).tolist()

    shap_table = {
        "expected_value": expected_value,
        "shap_values": shap_values_converted,
        "feature_names": lead_df_x.columns.tolist(),
        "feature_values": feature_values_converted
    }
    return shap_table


def plot_recall_metrics(models, recall_interested, recall_not_interested):
    print("plot_recall_metrics")

    # Returning the recall metrics as a dictionary
    recall_table = {
        "models": models,
        "recall_interested": recall_interested,
        "recall_not_interested": recall_not_interested
    }
    
    return recall_table
