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
)


passion_exp = 'experiment_standard_split_conditions_passion'
code_state = 'bias_test_5-no_folds_labels_refactored_parallelDataLoader'
#code_state = 'bias_test_6-reproduce_test_5'
#input_file = 'reformed_%s__%s.csv' % (passion_exp, code_state)
input_file = '%s__%s.csv' % (passion_exp, code_state)
output_file = 'analysis_%s__%s_img_lvl.csv' % (passion_exp, code_state)
create_data = True

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


def print_eval_scores(self, y_true: np.ndarray, y_pred: np.ndarray, target_names: np.ndarray):
    print(
        classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
        )
    )
    b_acc = balanced_accuracy_score(
        y_true=y_true,
        y_pred=y_pred,
    )
    print(f"Balanced Acc: {b_acc}")

def do_calculations(data):
    targets = np.array(data['targets'])
    names = np.array(data['conditions_PASSION'])

    unique_targets, indices = np.unique(targets, return_index=True)
    target_names = np.stack((unique_targets, names[indices]), axis=1)


    # Detailed evaluation
    print("*" * 20 + f" overall " + "*" * 20)
    print_eval_scores(
        y_true=score_dict["targets"],
        y_pred=score_dict["predictions"],
    )
    # Detailed evaluation per demographic
    eval_df = self.dataset.meta_data.iloc[eval_range].copy()
    eval_df.reset_index(drop=True, inplace=True)
    eval_df["targets"] = score_dict["targets"]
    eval_df["predictions"] = score_dict["predictions"]
    fst_types = eval_df["fitzpatrick"].unique()
    for fst in fst_types:
        _df = eval_df[eval_df["fitzpatrick"] == fst]
        print(
            "~" * 20
            + f" Fitzpatrick: {fst}, Support: {_df.shape[0]} "
            + "~" * 20
        )
        print_eval_scores(
            y_true=score_dict["targets"][_df.index.values],
            y_pred=score_dict["predictions"][_df.index.values],
        )
    gender_types = eval_df["sex"].unique()
    for gender in gender_types:
        _df = eval_df[eval_df["sex"] == gender]
        print(
            "~" * 20 + f" Gender: {gender}, Support: {_df.shape[0]} " + "~" * 20
        )
        print_eval_scores(
            y_true=score_dict["targets"][_df.index.values],
            y_pred=score_dict["predictions"][_df.index.values],
        )
    # Aggregate predictions per sample
    eval_df = eval_df.groupby("subject_id").agg(
        {"targets": list, "predictions": list}
    )
    case_targets = (
        eval_df["targets"].apply(lambda x: max(set(x), key=x.count)).values
    )
    case_predictions = (
        eval_df["predictions"].apply(lambda x: max(set(x), key=x.count)).values
    )
    print("*" * 20 + f" {e_type.name()} -> Case Agg. " + "*" * 20)
    print_eval_scores(
        y_true=case_targets,
        y_pred=case_predictions,
    )

do_calculations(output_df)
