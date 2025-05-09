# bias_evaluator.py
import pandas as pd
import numpy as np
import re

from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


class BiasEvaluator:
    def __init__(self, code_state: str, passion_exp: str = 'experiment_standard_split_conditions_passion'):
        self.code_state = code_state
        self.passion_exp = passion_exp
        self.input_file = f'{passion_exp}__{code_state}.csv'
        self.predictions_n_meta_data_file = f'predictions_with_metadata_{passion_exp}__{code_state}_img_lvl.csv'
        self.print_details = True
        # todo: provide them by param
        self.target_names = ['Eczema', 'Fungal', 'Others', 'Scabies']
        self.labels = [0, 1, 2, 3]

    @staticmethod
    def _parse_image_paths(s: str):
        return s.replace('\n', ' ').replace("'", '"').replace('[', '').replace(']', '').split()

    @staticmethod
    def _parse_numpy_array(s: str):
        return np.fromstring(s.strip('[]').replace('\n', ' '), sep=' ', dtype=int)

    @staticmethod
    def _extract_subject_id(path: str):
        match = re.search(r"([A-Za-z]+[0-9]+)", path)
        return str(match.group(1)).strip() if match else np.nan

    # todo: call this from the metadata result as csv
    def get_data_with_metadata_from_csv(self, create_data: bool = True):
        if create_data:
            df_results = pd.read_csv(self.input_file)
            data = self.aggregate_data_with_metadata(df_results)
            data.to_csv(self.predictions_n_meta_data_file, index=False)
            return data

        return pd.read_csv(self.predictions_n_meta_data_file)

    # todo: call this from the pipeline
    def aggregate_data_with_metadata(self, df_results):
        df_labels = pd.read_csv('label.csv')
        df_split = pd.read_csv('PASSION_split.csv')
        df_results['FileNames'] = df_results['FileNames'].apply(self._parse_image_paths)
        df_results['Indices'] = df_results['Indices'].apply(self._parse_numpy_array)
        df_results['EvalTargets'] = df_results['EvalTargets'].apply(self._parse_numpy_array)
        df_results['EvalPredictions'] = df_results['EvalPredictions'].apply(self._parse_numpy_array)
        output_rows = []
        seen_subject_ids = set()
        for _, row in df_results.iterrows():
            for img_name, idx, lbl, pred in zip(row['FileNames'], row['Indices'], row['EvalTargets'],
                                                row['EvalPredictions']):
                subject_id = self._extract_subject_id(img_name)
                labels = df_labels[df_labels['subject_id'] == subject_id].iloc[0]
                split = df_split[df_split['subject_id'] == subject_id].iloc[0]
                output_rows.append({
                    'correct': lbl == pred,
                    'image_path': img_name,
                    'index': idx,
                    'targets': lbl,
                    'predictions': pred,
                    **labels.to_dict(),
                    **split.drop('subject_id').to_dict()
                })
                seen_subject_ids.add(subject_id)
        print(f"{len(seen_subject_ids)} unique subjects processed.")
        data = pd.DataFrame(output_rows)
        return data

    def _print_classification_stats(self, y_true, y_pred, balanced_accuracy):
        print(classification_report(y_true, y_pred, labels=self.labels, target_names=self.target_names, zero_division=0))
        print(f"Balanced Accuracy: {balanced_accuracy}")

    def _get_confusion_metrics(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred, labels=self.labels)
        cumulated = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'tpr': [], 'fpr': []}
        per_class = {}

        for i, name in enumerate(self.target_names):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - (tp + fp + fn)

            tpr = round(tp / (tp + fn), 2) if (tp + fn) else 0
            fpr = round(fp / (fp + tn), 2) if (fp + tn) else 0

            cumulated['tp'] += tp
            cumulated['fp'] += fp
            cumulated['fn'] += fn
            cumulated['tn'] += tn
            cumulated['tpr'].append(tpr)
            cumulated['fpr'].append(fpr)

            per_class[name] = {'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn, 'TPR': tpr, 'FPR': fpr}
            print(f"{name} — TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}, TPR: {tpr}, FPR: {fpr}")

        tp_fn = cumulated['tp'] + cumulated['fn']
        micro_tpr = round(cumulated['tp'] / tp_fn, 2) if tp_fn else 0
        fp_tn = cumulated['fp'] + cumulated['tn']
        micro_fpr = round(cumulated['fp'] / fp_tn, 2) if fp_tn else 0
        macro_tpr = round(np.mean(cumulated['tpr']), 2)
        macro_fpr = round(np.mean(cumulated['fpr']), 2)

        return {
            'macro-tpr': macro_tpr,
            'macro-fpr': macro_fpr,
            'micro-tpr': micro_tpr,
            'micro-fpr': micro_fpr,
            'per_class': per_class,
            'cumulated': {'TP': cumulated['tp'], 'FP': cumulated['fp'], 'FN': cumulated['fn'], 'TN': cumulated['tn']},
            'balancedAcc': balanced_accuracy_score(y_true, y_pred)
        }

    # todo potentially add aif360_results

    def run_full_evaluation(self, create_data: bool = False):
        df = self.get_data_with_metadata_from_csv(create_data=create_data)
        bins = range(0, 100, 5)
        df['ageGroup'] = pd.cut(df['age'], bins=bins, labels=[f'{i:02}-{i + 4:02}' for i in bins[:-1]], right=False)

        y_true = df["targets"]
        y_pred = df["predictions"]
        print("**** Overall Evaluation ****")
        metrics = self._get_confusion_metrics(y_true, y_pred)
        self._print_classification_stats(y_true, y_pred, metrics['balancedAcc'])

        print("**** Detailed Evaluation ****")
        grouping_columns = ["fitzpatrick", "sex", "ageGroup"]
        self.detailed_evaluation(df, grouping_columns)

    def collect_subgroup_results(self, data, group_by: list[str]):
        def to_pascal_case(s: str) -> str:
            return "".join(word.capitalize() for word in s.split("_"))

        grouped = sorted(data.groupby(group_by))
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

            y_true = data["targets"][_df.index.values]
            y_pred = data["predictions"][_df.index.values]
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
                "balancedAcc": rates['balancedAcc']
            }
            if not result_keys:
                result_keys = list(result.keys())

            # Add per-class metrics
            for class_name, values in rates['per_class'].items():
                for metric_name, metric_val in values.items():
                    result[f"{class_name}_{metric_name}"] = metric_val

            results.append(result)

        return pd.DataFrame(results), result_keys

    def detailed_evaluation(self, data, grouping_columns, threshold=0.015):
        dfs = []
        group_by_key = "GroupBy"
        group_combinations = self._get_all_group_combinations(grouping_columns)
        report = {}
        result_keys = None
        for group in group_combinations:
            subgroup_df, result_keys = collect_subgroup_results(data, group_by=group)
            subgroup_df[group_by_key] = ", ".join(group)
            dfs.append(subgroup_df)

            macro_tpr_avg = subgroup_df["Macro-TPR"].mean()
            macro_fpr_avg = subgroup_df["Macro-FPR"].mean()

            privileged = []
            underprivileged = []
            avgprivileged = []


            for _, row in subgroup_df.iterrows():
                reasons = []
                macro_tpr = row["Macro-TPR"]
                macro_fpr = row['Macro-FPR']

                # Compare TPR
                tpr_diff = macro_tpr - macro_tpr_avg
                if abs(tpr_diff) <= threshold:
                    reasons.append(f"TPR ~ ({macro_tpr:.2f})")
                elif tpr_diff > 0:
                    reasons.append(f"TPR ↑ ({macro_tpr:.2f})")
                else:
                    reasons.append(f"TPR ↓ ({macro_tpr:.2f})")

                # Compare FPR
                fpr_diff = macro_fpr - macro_fpr_avg
                if abs(fpr_diff) <= threshold:
                    reasons.append(f"FPR ~ ({macro_fpr :.2f})")
                elif fpr_diff > 0:
                    reasons.append(f"FPR ↑ ({macro_fpr :.2f})")
                else:
                    reasons.append(f"FPR ↓ ({macro_fpr :.2f})")

                label = ", ".join(str(row[col]) for col in group)

                if ("TPR ↑" in " ".join(reasons) or "FPR ↓" in " ".join(reasons)):
                    privileged.append((label, reasons))
                elif ("TPR ↓" in " ".join(reasons) or "FPR ↑" in " ".join(reasons)):
                    underprivileged.append((label, reasons))
                else:
                    avgprivileged.append((label, reasons))

            report_key = ", ".join(group)
            report[report_key] = {
                "macro_tpr_avg": macro_tpr_avg,
                "macro_fpr_avg": macro_fpr_avg,
                "privileged": privileged,
                "underprivileged": underprivileged,
                "avgprivileged": avgprivileged
            }
        final_df = pd.concat(dfs, ignore_index=True)
        all_other_cols = list(set(final_df.columns) - set(result_keys) - {group_by_key})
        all_other_cols.sort()
        ordered_cols = [group_by_key, *result_keys, *all_other_cols]
        final_df.to_csv("subgroup_bias_results__" + code_state + ".csv", columns=ordered_cols, index=False)
        # Print privilege report
        for group_key, group_report in report.items():
            print(f"\n=== Grouping: {group_key} ===")
            print(f"macro-TPR avg: {group_report['macro_tpr_avg']}; macro-FPR avg: {group_report['macro_fpr_avg']}")

            if group_report["privileged"]:
                print("privileged:")
                for label, reasons in group_report["privileged"]:
                    print(f"  {label} - Reasons: {', '.join(reasons)}")
            if group_report["underprivileged"]:
                print("Underprivileged:")
                for label, reasons in group_report["underprivileged"]:
                    print(f"  {label} - Reasons: {', '.join(reasons)}")
            if group_report["avgprivileged"]:
                print("Average:")
                for label, reasons in group_report["avgprivileged"]:
                    print(f"  {label} - Reasons: {', '.join(reasons)}")

    def _get_all_group_combinations(self, grouping_columns):
        group_combinations = []
        for r in range(1, len(grouping_columns) + 1):
            group_combinations.extend([list(comb) for comb in itertools.combinations(grouping_columns, r)])
        return group_combinations


#from bias_evaluator import BiasEvaluator

evaluator = BiasEvaluator(code_state="big_model_test")
evaluator.run_full_evaluation(create_data=False)

# get_data_with_metadata_from_csv