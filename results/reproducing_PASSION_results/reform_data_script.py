import csv
import itertools

import pandas as pd
import numpy as np
import re

from aif360.sklearn import metrics

from fairlearn.metrics import equalized_odds_ratio
from pandas import read_csv
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

is_test_5 = False
is_test_6 = False

passion_exp = 'experiment_standard_split_conditions_passion'
if is_test_5:
    code_state = 'bias_test_5-no_folds_labels_refactored_parallelDataLoader'
elif is_test_6:
    code_state = 'bias_test_6-reproduce_test_5'
else:
    #code_state = 'bias_test_7-labels_consistency'
    code_state = 'bias_test_x-PASSION_checkpoint_running_results'
#input_file = 'reformed_%s__%s.csv' % (passion_exp, code_state)
input_file = '%s__%s.csv' % (passion_exp, code_state)
output_file = 'analysis_%s__%s_img_lvl.csv' % (passion_exp, code_state)
create_data = False


if create_data:
    # Load the CSV with header
    df_results = pd.read_csv(input_file)  # assumes header is present
    df_labels = pd.read_csv('label.csv')
    df_split = pd.read_csv('PASSION_split.csv')

    def extract_subject_id(path: str):
        pattern = r"([A-Za-z]+[0-9]+)"
        match = re.search(pattern, path)
        if match:
            return str(match.group(1)).strip()
        else:
            return np.nan
    # Parse the image_paths and arrays
    def parse_image_paths(s):
        s = s.replace('\n', ' ').replace("'", '"')
        s = s.replace('[', '').replace(']', '')
        return s.split()

    def parse_numpy_array(s):
        return np.fromstring(s.strip('[]').replace('\n', ' '), sep=' ', dtype=int)

    df_results['FileNames'] = df_results['FileNames'].apply(parse_image_paths)
    df_results['Indices'] = df_results['Indices'].apply(parse_numpy_array)
    df_results['EvalTargets'] = df_results['EvalTargets'].apply(parse_numpy_array)
    df_results['EvalPredictions'] = df_results['EvalPredictions'].apply(parse_numpy_array)

    # Flatten the DataFrame into one row per image path
    rows = []
    unique_subject_ids = []
    for _, row in df_results.iterrows():
        for img_name, idx, lbl, pred in zip(row['FileNames'], row['Indices'], row['EvalTargets'], row['EvalPredictions']):
            subject_id = extract_subject_id(img_name)
            labels = df_labels[df_labels['subject_id'] == subject_id].iloc[0]
            split = df_split[df_split['subject_id'] == subject_id].iloc[0]
            rows.append({
                'correct': lbl == pred,
                'image_path': img_name,
                'index': idx,
                'targets': lbl,
                'predictions': pred,
                **labels.to_dict(),
                **split.drop('subject_id').to_dict()
            })
            if subject_id not in unique_subject_ids:
                unique_subject_ids.append(subject_id)

    print(len(unique_subject_ids))

    # Create new DataFrame and save
    output_df = pd.DataFrame(rows)
    output_df.to_csv(output_file, index=False)
else:
    output_df = pd.read_csv(output_file)

labels = [0, 1, 2, 3]
if is_test_5:
    labels = [2, 3, 0, 1]
elif is_test_6:
    labels = [1, 2, 3, 0]
target_names = ['Eczema', 'Fungal', 'Others', 'Scabies']
print_details = True
def print_detailed_bias_eval_scores(y_true: np.ndarray, y_pred: np.ndarray):
    rates = {
        'macro-tpr': 0,
        'macro-fpr': 0,
        'micro-tpr': 0,
        'micro-fpr': 0,
    }
    if print_details:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("Confusion Matrix:")
        print(cm)
        print(f'y_true: {set(y_true)}')
        print(f'y_pred: {set(y_pred)}')

        def get_values_from_cm(cm, class_index):
            total_classifications = cm.sum()
            tp = cm[class_index, class_index]
            fp = cm[:, class_index].sum() - tp
            fn = cm[class_index, :].sum() - tp
            tn = total_classifications - (tp + fp + fn)

            tp_n_fn = (tp + fn)
            tpr = round_2(tp / tp_n_fn) if tp_n_fn != 0 else 0

            fp_n_tn = (fp + tn)
            fpr = round_2(fp / fp_n_tn) if fp_n_tn != 0 else 0

            return tp, fp, fn, tn, tpr, fpr

        def round_2(val):
            return int(val * 100) / 100

        cumulated = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
            'tpr': [], 'fpr': [],
        }
        per_class = {}
        for i, name in enumerate(target_names):
            tp, fp, fn, tn, tpr, fpr = get_values_from_cm(cm, i)
            cumulated['tp'] += tp
            cumulated['fp'] += fp
            cumulated['fn'] += fn
            cumulated['tn'] += tn
            cumulated['tpr'].append(tpr)
            cumulated['fpr'].append(fpr)

            print(f"{name} — TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, TPR: {tpr}, FPR: {fpr}")
            per_class[name] = {
                'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'TPR': tpr, 'FPR': fpr
            }

        tp_n_fn = (cumulated['tp'] + cumulated['fn'])
        micro_tpr = round_2(cumulated['tp'] / tp_n_fn) if tp_n_fn != 0 else 0

        fp_n_tn = (cumulated['fp'] + cumulated['tn'])
        micro_fpr = round_2(cumulated['fp'] / fp_n_tn) if fp_n_tn != 0 else 0

        macro_tpr = round_2(np.average(cumulated['tpr']))
        macro_fpr = round_2(np.average(cumulated['fpr']))
        print(f"cumulated - TP: {cumulated['tp']}, FP: {cumulated['fp']}, FN: {cumulated['fn']}, TN: {cumulated['tn']}"
              f", macro-TPR (avg): {macro_tpr}, macro-FPR (avg): {macro_fpr}"
              f", micro-TPR: {micro_tpr}, micro-FPR: {micro_fpr}")
        rates = {
            'macro-tpr': macro_tpr,
            'macro-fpr': macro_fpr,
            'micro-tpr': micro_tpr,
            'micro-fpr': micro_fpr,
            'per_class': per_class,
            'cumulated': {
                'TP': cumulated['tp'],
                'FP': cumulated['fp'],
                'FN': cumulated['fn'],
                'TN': cumulated['tn'],
            }
        }


    print(
        classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        )
    )
    b_acc = balanced_accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
    )
    print(f"Balanced Acc: {b_acc}")
    return rates


def print_grouped_result(eval_df, group_by: str):
    group_types = sorted(eval_df[group_by].unique())
    group_name = group_by.capitalize()
    rates_eq_odds = {
        'macro-tpr': [],
        'macro-fpr': [],
        'micro-tpr': [],
        'micro-fpr': [],
    }
    for group_value in group_types:
        _df = eval_df[eval_df[group_by] == group_value]
        print(
            "~" * 20
            + f" {group_name}: {group_value}, Support: {_df.shape[0]} "
            + "~" * 20
        )
        rates = print_detailed_bias_eval_scores(
            y_true=eval_df["targets"][_df.index.values],
            y_pred=eval_df["predictions"][_df.index.values],
        )
        rates_eq_odds['macro-tpr'].append(rates['macro-tpr'])
        rates_eq_odds['macro-fpr'].append(rates['macro-fpr'])
        rates_eq_odds['micro-tpr'].append(rates['micro-tpr'])
        rates_eq_odds['micro-fpr'].append(rates['micro-fpr'])

    print(rates_eq_odds)

    def values_close(vals): #, rtol=1e-05, atol=1e-08):
        return np.allclose(vals, [vals[0]] * len(vals)) #, rtol=rtol, atol=atol)

    print('macro_eq_odds: ' + str(values_close(rates_eq_odds['macro-tpr']) and values_close(rates_eq_odds['macro-fpr'])))
    print('micro_eq_odds: ' + str(values_close(rates_eq_odds['micro-tpr']) and values_close(rates_eq_odds['micro-fpr'])))

    return group_types

def print_subgroup_results(eval_df, group_by: list[str]):
    def to_pascal_case(s: str) -> str:
        return "".join(word.capitalize() for word in s.split("_"))

    # Get all unique combinations of the group attributes
    grouped = sorted(eval_df.groupby(group_by))

    rates_eq_odds = {
        'macro-tpr': [],
        'macro-fpr': [],
        'micro-tpr': [],
        'micro-fpr': [],
    }
    for group_values, _df in grouped:
        if isinstance(group_values, str):
            group_values = (group_values,)  # Make it always a tuple

        if _df.shape[0] == 0:
            print(f'no support: group_by: {group_by}, group_values: {group_values}')
            continue  # Skip empty groups

        group_info = ", ".join(
            f"{to_pascal_case(attr)}: {val}"
            for attr, val in zip(group_by, group_values)
        )

        print("~" * 20 + f" {group_info}, Support: {_df.shape[0]} " + "~" * 20)
        y_true = eval_df["targets"][_df.index.values]
        y_pred = eval_df["predictions"][_df.index.values]
        rates = print_detailed_bias_eval_scores(
            y_true=y_true,
            y_pred=y_pred,
        )
        # todo: must filter out 0 probably, or not?
        rates_eq_odds['macro-tpr'].append(rates['macro-tpr'])
        rates_eq_odds['macro-fpr'].append(rates['macro-fpr'])
        rates_eq_odds['micro-tpr'].append(rates['micro-tpr'])
        rates_eq_odds['micro-fpr'].append(rates['micro-fpr'])

    print(rates_eq_odds)

    def values_close(vals): #, rtol=1e-05, atol=1e-08):
        return np.allclose(vals, [vals[0]] * len(vals)) #, rtol=rtol, atol=atol)

    print('macro_eq_odds: ' + str(values_close(rates_eq_odds['macro-tpr']) and values_close(rates_eq_odds['macro-fpr'])))
    print('micro_eq_odds: ' + str(values_close(rates_eq_odds['micro-tpr']) and values_close(rates_eq_odds['micro-fpr'])))

def print_aif360_results(y, y_true: np.ndarray, y_pred: np.ndarray):
    protected_attrs = ['sex', 'fitzpatrick']
    label_column = 'targets'

    # Binarize multiclass labels for one-vs-rest analysis
    classes = np.unique(y[label_column])
    y_true_bin = label_binarize(y[label_column], classes=classes)
    y_pred_bin = label_binarize(y_pred, classes=classes)  # assuming y_pred is available
    print(classes)
    print(y_true_bin)
    print(y_pred_bin)

    # Prepare index using protected attributes
    y_index = y.set_index(protected_attrs).index

    # Iterate through each class and compute metrics
    for i, class_label in enumerate(classes):
        print(f"\n--- Metrics for class '{class_label}' ---")

        y_true_class = y_true_bin[:, i]
        y_pred_class = y_pred_bin[:, i]

        # If you want fairness metrics per protected group, join protected info again
        y_series = pd.Series(y_true_class, index=y_index, name='target')

        print(f'num_samples: {metrics.num_samples(y_series, y_pred_class)}')
        print(f'num_pos_neg: {metrics.num_pos_neg(y_series, y_pred_class)}')
        print(f'specificity_score: {metrics.specificity_score(y_series, y_pred_class, zero_division=0)}')
        print(f'sensitivity_score: {metrics.sensitivity_score(y_series, y_pred_class)}')
        print(f'precision_score: {precision_score(y_true_class, y_pred_class)}')
        print(f'recall_score: {recall_score(y_true_class, y_pred_class)}')
        print(f'base_rate: {metrics.base_rate(y_series, y_pred_class)}')
        print(f'selection_rate: {metrics.selection_rate(y_series, y_pred_class)}')
        print(f'smoothed_base_rate: {metrics.smoothed_base_rate(y_series, y_pred_class)}')
        print(f'smoothed_selection_rate: {metrics.smoothed_selection_rate(y_series, y_pred_class)}')
        print(f'generalized_fpr: {metrics.generalized_fpr(y_series, y_pred_class)}')
        print(f'generalized_fnr: {metrics.generalized_fnr(y_series, y_pred_class)}')

        # Fairness metrics across groups
        print(f'equal_opportunity_difference: {metrics.equal_opportunity_difference(y_series, y_pred_class)}')
        print(f'average_odds_difference: {metrics.average_odds_difference(y_series, y_pred_class)}')
        #print(f'average_odds_error: {metrics.average_odds_error(y_series, y_pred_class)}')
        print(f'class_imbalance: {metrics.class_imbalance(y_series, y_pred_class)}')
        print(f'df_bias_amplification: {metrics.df_bias_amplification(y_series, y_pred_class)}')
        print(f'between_group_generalized_entropy_error: {metrics.between_group_generalized_entropy_error(y_series, y_pred_class)}')
        #print(f'consistency_score: {metrics.consistency_score(y_series, y_pred_class)}')

    # # Create the outcome vector with protected attributes as index
    # y = y.set_index(protected_attrs)[label_column]
    # print(y)
    #
    #
    # print(f'num_samples: {metrics.num_samples(y_true, y_pred)}')
    # print(f'num_pos_neg: {metrics.num_pos_neg(y_true, y_pred)}')
    # print(f'specifity_score: {metrics.specificity_score(y_true, y_pred, zero_division=0)}')
    # # - not working bc not binary - print(f'sensivity_score: {metrics.sensitivity_score(y_true, y_pred)}')
    # print(f'sklearn.precision_score: {precision_score(y_true, y_pred, average="macro")}')
    # print(f'sklearn.recall_score: {recall_score(y_true, y_pred, average="macro")}')
    # print(f'base_rate: {metrics.base_rate(y_true, y_pred)}')
    # print(f'selection_rate: {metrics.selection_rate(y_true, y_pred)}')
    # print(f'smoothed_base_rate: {metrics.smoothed_base_rate(y_true, y_pred)}')
    # print(f'smoothed_selection_rate: {metrics.smoothed_selection_rate(y_true, y_pred)}')
    # print(f'generalized_fpr: {metrics.generalized_fpr(y_true, y_pred)}')
    # print(f'generalized_fnr: {metrics.generalized_fnr(y_true, y_pred)}')
    #
    # print(f'equal_opportunity_difference: {metrics.one_vs_rest(metrics.equal_opportunity_difference, y, y_pred, return_groups=True)}')
    # print(f'average_odds_difference: {metrics.one_vs_rest(metrics.average_odds_difference, y, y_pred, return_groups=True)}')
    # print(f'average_odds_error: {metrics.one_vs_rest(metrics.average_odds_error, y, y_pred, return_groups=True)}')
    # print(f'class_imbalance: {metrics.one_vs_rest(metrics.class_imbalance, y, y_pred, return_groups=True)}')
    # print(f'df_bias_amplification: {metrics.one_vs_rest(metrics.df_bias_amplification, y, y_pred, return_groups=True)}')
    # print(f'between_group_generalized_entropy_error: {metrics.one_vs_rest(metrics.between_group_generalized_entropy_error, y, y_pred, return_groups=True)}')
    # print(f'consistency_score: {metrics.one_vs_rest(metrics.consistency_score, y, y_pred, return_groups=True)}')



def collect_subgroup_results(eval_df, group_by: list[str]):
    def to_pascal_case(s: str) -> str:
        return "".join(word.capitalize() for word in s.split("_"))

    grouped = sorted(eval_df.groupby(group_by))
    results = []
    result_keys = None

    for group_values, _df in grouped:
        if isinstance(group_values, str):
            group_values = (group_values,)  # Ensure tuple

        if _df.shape[0] == 0:
            print(f'no support: group_by: {group_by}, group_values: {group_values}')
            continue

        group_info = ", ".join(
            f"{to_pascal_case(attr)}: {val}"
            for attr, val in zip(group_by, group_values)
        )
        print("~" * 20 + f" {group_info}, Support: {_df.shape[0]} " + "~" * 20)

        y_true = eval_df["targets"][_df.index.values]
        y_pred = eval_df["predictions"][_df.index.values]
        rates = print_detailed_bias_eval_scores(y_true=y_true, y_pred=y_pred)

        result = {
            **{attr: val for attr, val in zip(group_by, group_values)},
            "Support": _df.shape[0],
            "Macro-TPR": rates['macro-tpr'],
            "Macro-FPR": rates['macro-fpr'],
            "Micro-TPR": rates['micro-tpr'],
            "Micro-FPR": rates['micro-fpr'],
            "TP": rates['cumulated']['TP'],
            "FP": rates['cumulated']['FP'],
            "FN": rates['cumulated']['FN'],
            "TN": rates['cumulated']['TN'],
        }
        if not result_keys:
            result_keys = list(result.keys())

        # Add per-class metrics
        for class_name, values in rates['per_class'].items():
            for metric_name, metric_val in values.items():
                result[f"{class_name}_{metric_name}"] = metric_val

        results.append(result)

    return pd.DataFrame(results), result_keys


def do_calculations(data):
    # Detailed evaluation
    print("*" * 20 + f" overall " + "*" * 20)
    print_detailed_bias_eval_scores(
        y_true=data["targets"],
        y_pred=data["predictions"],
    )

    print_aif360_results(
        y=data,
        y_true=data["targets"],
        y_pred=data["predictions"],
    )

    print('=' * 20 + ' now more dynamic (grouped) ' + '=' * 20)
    print_grouped_result(data, group_by="fitzpatrick")
    print_grouped_result(data, group_by="sex")

    print("=" * 20 + " grouped output per case using subgroup " + "=" * 20)
    print('dataset')
    bins = list(range(0, 100, 5))
    labels = [f'{i}-{i + 4}' for i in bins[:-1]]
    data['ageGroup'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)
    print(data)

    print_subgroup_results(data, group_by=["fitzpatrick"])
    print_subgroup_results(data, group_by=["sex"])
    print_subgroup_results(data, group_by=["ageGroup"])
    print_subgroup_results(data, group_by=["fitzpatrick", "sex"])
    print_subgroup_results( data, group_by=["fitzpatrick", "ageGroup"] )
    print_subgroup_results(data, group_by=["sex", "ageGroup"])
    print_subgroup_results( data, group_by=["fitzpatrick", "sex", "ageGroup"] )

    dfs = []
    group_by_key = "GroupBy"

    grouping_columns = ["fitzpatrick", "sex", "ageGroup"]
    # Create all possible combinations of the grouping columns dynamically
    group_combinations = []
    for r in range(1, len(grouping_columns) + 1):
        group_combinations.extend([list(comb) for comb in itertools.combinations(grouping_columns, r)])

    result_keys = None
    for group in group_combinations:
        subgroup_df, result_keys = collect_subgroup_results(data, group_by=group)
        subgroup_df[group_by_key] = ", ".join(group)  # Optional: keep track of grouping
        dfs.append(subgroup_df)

    final_df = pd.concat(dfs, ignore_index=True)
    all_other_cols = list(set(final_df.columns) - set(result_keys) - {group_by_key})
    all_other_cols.sort()
    ordered_cols = [group_by_key, *result_keys, *all_other_cols]
    final_df.to_csv("subgroup_bias_results.csv", columns=ordered_cols, index=False)

    # todo: introduce equalized odds


    # # TODO: also evaluate this
    # # Aggregate predictions per sample
    # data_aggregated = data.groupby("subject_id").agg(
    #     {"targets": list, "predictions": list}
    # )
    # case_targets = (
    #     data_aggregated["targets"].apply(lambda x: max(set(x), key=x.count)).values
    # )
    # case_predictions = (
    #     data_aggregated["predictions"].apply(lambda x: max(set(x), key=x.count)).values
    # )
    # print("*" * 20 + f" overall -> Case Agg. " + "*" * 20)
    # print_eval_scores(
    #     y_true=case_targets,
    #     y_pred=case_predictions,
    # )

#todo: reenable
do_calculations(output_df)




# def fun(preds, labels, sens_at):
#     uniq_g = np.unique(sens_at)
#     tpr, fpr = [], []
#
#     for g in uniq_g:
#         grp_idx = (sens_at == g)
#         grp_labels = labels[grp_idx]
#         grp_preds = preds[grp_idx]
#
#         tn, fp, fn, tp = confusion_matrix(grp_labels, grp_preds, labels=[0, 1]).ravel()
#         tpr.append(tp / (tp + fn))  # True Positive Rate
#         fpr.append(fp / (fp + tn))  # False Positive Rate
#     print(tpr)
#
#     print(fpr)
#     return np.allclose(tpr, tpr[0]) and np.allclose(fpr, fpr[0])
#
# labels = np.array([1, 0, 1, 0, 1, 0, 0])
# preds = np.array([1, 0, 1, 0, 1, 1, 0])
# sens_at = np.array(['male', 'male', 'female', 'female', 'female', 'male', 'female'])
#
# res = fun(preds, labels, sens_at)
# print(f"Does the model satisfy Equalized Odds? {res}")





# from aif360.datasets import StandardDataset
# from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
#
# dataset = StandardDataset(df,
#                           label_name='income',
#                           favorable_classes=[1],
#                           protected_attribute_names=['gender'],
#                           privileged_classes=[[1]])
#
#
# def fair_metrics(dataset, y_pred):
#     dataset_pred = dataset.copy()
#     dataset_pred.labels = y_pred
#
#     attr = dataset_pred.protected_attribute_names[0]
#
#     idx = dataset_pred.protected_attribute_names.index(attr)
#     privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
#     unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]
#
#     classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)
#
#     metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
#                                            privileged_groups=privileged_groups)
#
#     result = {'statistical_parity_difference': metric_pred.statistical_parity_difference(),
#               'disparate_impact': metric_pred.disparate_impact(),
#               'equal_opportunity_difference': classified_metric.equal_opportunity_difference()}
#
#     return result
#
#
# fair_metrics(dataset, y_pred)