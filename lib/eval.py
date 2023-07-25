import numpy as np
import pandas as pd
from sklearn.metrics import label_ranking_average_precision_score, multilabel_confusion_matrix, roc_auc_score


def calculate_binary_measures(mlcm, target_names):
    assert mlcm.shape[0] == len(target_names)
    mlcm = np.reshape(mlcm, (mlcm.shape[0], -1))
    df_tn_fp_fn_tp = pd.DataFrame(mlcm, columns=["TN", "FP", "FN", "TP"], index=target_names)

    micro_avg = np.sum(mlcm, axis=0, keepdims=True)  # Might not be meaningful since mistakes are counted twice
    mlcm = np.concatenate([mlcm, micro_avg], axis=0)
    mlcm = mlcm.T
    aggregates = np.stack([mlcm[3] / (mlcm[3] + mlcm[1]), mlcm[3] / (mlcm[3] + mlcm[2]),
                           2 * mlcm[3] / (2 * mlcm[3] + mlcm[1] + mlcm[2]),
                           mlcm[0] / (mlcm[0] + mlcm[1]), mlcm[2] + mlcm[3]], axis=-1)
    # Make sure P, R, F are nan for targets without examples
    aggregates[aggregates[:, -1] == 0, 0] = np.nan
    aggregates[aggregates[:, -1] == 0, 2] = np.nan
    macro_avg = np.nanmean(aggregates[:-1, ...], axis=0, keepdims=True)
    macro_avg[0, -1] = np.sum(aggregates[:-1, -1])
    df_p_r_f = pd.DataFrame(np.concatenate([aggregates, macro_avg], axis=0), columns=["P", "R", "F", "Specificity", "Support"], index=target_names + ["micro avg", "macro avg"])

    return df_tn_fp_fn_tp, df_p_r_f


def framewise_instrument_scores(output_path, test_labels, test_predictions, prediction_thresholds, class_names):
    test_predictions_thresholded = (test_predictions > prediction_thresholds)

    mlcm = multilabel_confusion_matrix(test_labels, test_predictions_thresholded)
    df_tn_fp_fn_tp, df_p_r_f = calculate_binary_measures(mlcm, class_names)
    df_tn_fp_fn_tp.to_csv(output_path + "/tn_fp_fn_tp")
    df_p_r_f.to_csv(output_path + "/p_r_f")

    lrap = label_ranking_average_precision_score(test_labels, test_predictions)
    df_lrap = pd.DataFrame([lrap], columns=["LRAP"])
    df_lrap.to_csv(output_path + "/lrap")

    targets_with_examples = np.where(np.logical_and(np.max(test_labels, axis=0) == 1, np.min(test_labels, axis=0) == 0))[0]
    # roc_auc not defined if one of the classes does not appear
    roc_auc = np.ones(len(class_names)) * np.nan
    if len(targets_with_examples) > 1:
        roc_auc[targets_with_examples] = roc_auc_score(test_labels[:, targets_with_examples], test_predictions[:, targets_with_examples], average=None)
        mean_roc_auc = np.nanmean(roc_auc)
    else:
        mean_roc_auc = np.nan
    df_roc_auc = pd.DataFrame(np.concatenate([roc_auc, [mean_roc_auc]]), columns=["ROC AUC"], index=class_names + ["Mean"])
    df_roc_auc.to_csv(output_path + "/roc_auc")

    with open(output_path + "/results", "w") as f:
        with pd.option_context('display.max_rows', 1000, 'display.max_columns', 1000, 'display.max_colwidth', 1000, 'display.width', None):
            print("Total number of frames:", len(test_labels), file=f)
            print("", file=f)
            print(df_lrap, file=f)
            print("", file=f)
            print(df_roc_auc, file=f)
            print("", file=f)
            print(df_tn_fp_fn_tp, file=f)
            print("", file=f)
            print(df_p_r_f, file=f)
