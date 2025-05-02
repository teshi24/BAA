import csv
from dataclasses import dataclass

import pandas as pd
import numpy as np
import re

from pandas import read_csv
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

is_test_5 = True
is_test_6 = False

passion_exp = 'experiment_standard_split_conditions_passion'
if is_test_5:
    code_state = 'bias_test_5-no_folds_labels_refactored_parallelDataLoader'
if is_test_6:
    code_state = 'bias_test_6-reproduce_test_5'
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
if is_test_6:
    labels = [1, 2, 3, 0]
target_names = ['Eczema', 'Fungal', 'Others', 'Scabies']
print_details = True
def print_eval_scores(y_true: np.ndarray, y_pred: np.ndarray):
    if print_details:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        print("Confusion Matrix:")
        print(cm)

        def get_tp_fp_fn_tn(cm, class_index):
            total_classifications = cm.sum()
            tp = cm[class_index, class_index]
            fp = cm[:, class_index].sum() - tp
            fn = cm[class_index, :].sum() - tp
            tn = total_classifications - (tp + fp + fn)
            return tp, fp, fn, tn

        for i, name in enumerate(target_names):
            tp, fp, fn, tn = get_tp_fp_fn_tn(cm, i)
            print(f"{name} — TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    print(
        classification_report(
            y_true=y_true,
            y_pred=y_pred,
            labels=labels,
            target_names=target_names,
            #zero_division=0
        )
    )
    b_acc = balanced_accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
    )
    print(f"Balanced Acc: {b_acc}")

def print_grouped_result(eval_df, group_by: str):
    group_types = sorted(eval_df[group_by].unique())
    group_name = group_by.capitalize()
    for group_value in group_types:
        _df = eval_df[eval_df[group_by] == group_value]
        print(
            "~" * 20
            + f" {group_name}: {group_value}, Support: {_df.shape[0]} "
            + "~" * 20
        )
        print_eval_scores(
            y_true=eval_df["targets"][_df.index.values],
            y_pred=eval_df["predictions"][_df.index.values],
        )
    return group_types

def print_subgroup_results(eval_df, group_by: list[str]):
    def to_pascal_case(s: str) -> str:
        return "".join(word.capitalize() for word in s.split("_"))

    # Get all unique combinations of the group attributes
    grouped = sorted(eval_df.groupby(group_by))

    for group_values, _df in grouped:
        if isinstance(group_values, str):
            group_values = (group_values,)  # Make it always a tuple

        if _df.shape[0] == 0:
            continue  # Skip empty groups

        # Build header text
        group_info = ", ".join(
            f"{to_pascal_case(attr)}: {val}"
            for attr, val in zip(group_by, group_values)
        )

        print("~" * 20 + f" {group_info}, Support: {_df.shape[0]} " + "~" * 20)
        print_eval_scores(
            y_true=eval_df["targets"][_df.index.values],
            y_pred=eval_df["predictions"][_df.index.values],
        )


def do_calculations(data):
    # Detailed evaluation
    print("*" * 20 + f" overall " + "*" * 20)
    print_eval_scores(
        y_true=data["targets"],
        y_pred=data["predictions"],
    )
    # # Detailed evaluation per demographic
    # fst_types = sorted(data["fitzpatrick"].unique())
    # for fst in fst_types:
    #     _df = data[data["fitzpatrick"] == fst]
    #     print(
    #         "~" * 20
    #         + f" Fitzpatrick: {fst}, Support: {_df.shape[0]} "
    #         + "~" * 20
    #     )
    #     print_eval_scores(
    #         y_true=data["targets"][_df.index.values],
    #         y_pred=data["predictions"][_df.index.values],
    #     )
    # gender_types = sorted(data["sex"].unique())
    # for gender in gender_types:
    #     _df = data[data["sex"] == gender]
    #     print(
    #         "~" * 20 + f" Gender: {gender}, Support: {_df.shape[0]} " + "~" * 20
    #     )
    #     print_eval_scores(
    #         y_true=data["targets"][_df.index.values],
    #         y_pred=data["predictions"][_df.index.values],
    #     )

    print('=' * 20 + ' now more dynamic (grouped) ' + '=' * 20)
    print_grouped_result(data, group_by="fitzpatrick")
    print_grouped_result(data, group_by="sex")

    print("=" * 20 + " grouped output per case using subgroup " + "~=" * 20)
    print_subgroup_results(data, group_by=["fitzpatrick"])
    print_subgroup_results(data, group_by=["sex"])
    # todo: add bins for age
    # self.print_subgroup_results(data, group_by=["age"])
    print_subgroup_results(data, group_by=["fitzpatrick", "sex"])
    # print_subgroup_results( data, group_by=["fitzpatrick", "age"] )
    # print_subgroup_results(data, group_by=["sex", "age"])
    # print_subgroup_results( data, group_by=["fitzpatrick", "sex", "age"] )

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

do_calculations(output_df)
