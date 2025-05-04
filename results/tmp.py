#This code has already created a checkpoint, I want to reload the 'best model' ckp and only run the evaluation, no training needed

eval_type_dict = {
    # Models
    "fine_tuning": EvalFineTuning,
}

class EvaluationTrainer(ABC, object):
    def __init__(
        self,
        dataset_name: DatasetName,
        config: dict,
        SSL_model: str = "imagenet",
        output_path: Union[Path, str] = "assets/evaluation",
        cache_path: Union[Path, str] = "assets/evaluation/cache",
        n_layers: int = 1,
        append_to_df: bool = False,
    ):
        self.dataset_name = dataset_name
        self.config = config
        self.output_path = Path(output_path)
        self.cache_path = Path(cache_path)
        self.append_to_df = append_to_df
        self.seed = config["seed"]
        fix_random_seeds(self.seed)

        self.df_name = f"{self.experiment_name}_{self.dataset_name.value}.csv"
        self.df_path = self.output_path / self.df_name
        self.model_path = self.output_path / self.experiment_name

        # make sure the output and cache path exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.model_path.mkdir(parents=True, exist_ok=True)

        # parse the config to get the eval types
        self.eval_types = []
        for k, v in self.config.items():
            if k in eval_type_dict.keys():
                self.eval_types.append((eval_type_dict.get(k), v))

        # save the results to the dataframe
        self.df = pd.DataFrame(
            [],
            columns=[
                "Score",
                "FileNames",
                "Indices",
                "EvalTargets",
                "EvalPredictions",
                "EvalType",
                "AdditionalRunInfo",
                "SplitName",
            ],
        )
        if append_to_df:
            if not self.df_path.exists():
                print(f"Results for dataset: {self.dataset_name.value} not available.")
            else:
                print(f"Appending results to: {self.df_path}")
                self.df = pd.read_csv(self.df_path)

        # load the dataset to evaluate on
        self.input_size = config["input_size"]
        self.transform = transforms.Compose(
            [
                # transforms.Resize((144, 144)), # todo: activate for input_size 128
                transforms.Resize((256, 256)),  # activate for input_size 224
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        data_config = copy.deepcopy(config["dataset"])
        data_path = data_config[dataset_name.value].pop("path")
        self.dataset, self.torch_dataset = get_dataset(
            dataset_name=dataset_name,
            dataset_path=Path(data_path),
            batch_size=config.get("batch_size", 128),
            transform=self.transform,
            # num_workers=config.get("num_workers", 4),
            **data_config[dataset_name.value],
        )
        # load the correct model to use as initialization
        if SSL_model == "GoogleDermFound":
            self.dataset.return_embedding = True
            self.torch_dataset.dataset.return_embedding = True
        else:
            self.model, self.model_out_dim = self.load_model(SSL_model=SSL_model)
        #
        # check if the cache contains the embeddings already
        logger.debug("embed data")
        cache_file = (
            self.cache_path / f"{dataset_name.value}_{self.experiment_name}.pickle"
        )
        if cache_file.exists():
            print(f"Found cached file loading: {cache_file}")
            with open(cache_file, "rb") as file:
                cached_dict = pickle.load(file)
            self.emb_space = cached_dict["embedding_space"]
            self.labels = cached_dict["labels"]
            self.paths = cached_dict["paths"]
            self.indices = cached_dict["indices"]
            del cached_dict
        else:
            (
                self.emb_space,
                self.labels,
                self.images,
                self.paths,
                self.indices,
            ) = embed_dataset(
                torch_dataset=self.torch_dataset,
                model=self.model,
                n_layers=n_layers,
                # memmap=False,
                normalize=False,
            )
            # todo: what with those imgs
            # save the embeddings and issues to cache
            save_dict = {
                "embedding_space": self.emb_space,
                "labels": self.labels,
                "paths": self.paths,
                "indices": self.indices,
            }
            with open(cache_file, "wb") as handle:
                pickle.dump(save_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @property
    @abstractmethod
    def experiment_name(self) -> str:
        pass

    @abstractmethod
    def split_dataframe_iterator(self) -> Iterator[Tuple[np.ndarray, np.ndarray, str]]:
        pass

    def load_model(self, SSL_model: str):
        logger.debug("load model")
        model, info, _ = Embedder.load_pretrained(
            SSL_model,
            return_info=True,
            n_head_layers=0,
        )
        # set the model in evaluation mode
        model = model.eval()
        # move to correct device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"device: {device}")
        model = model.to(device)
        return model, info.out_dim

    def evaluate(self):
        if self.df_path.exists() and not self.append_to_df:
            raise ValueError(
                f"Dataframe already exists, remove to start: {self.df_path}"
            )

        for e_type, config in self.eval_types:
            for (
                train_valid_range,
                test_range,
                split_name,
            ) in self.split_dataframe_iterator():
                if config.get("n_folds", None) is not None:
                    k_fold = StratifiedGroupKFold(
                        n_splits=config["n_folds"],
                        random_state=self.seed,
                        shuffle=True,
                    )
                    labels = self.dataset.meta_data.loc[
                        train_valid_range, self.dataset.LBL_COL
                    ].values
                    groups = self.dataset.meta_data.loc[
                        train_valid_range, "subject_id"
                    ].values
                    fold_generator = k_fold.split(train_valid_range, labels, groups)
                    for i_fold, (train_range, valid_range) in tqdm(
                        enumerate(fold_generator),
                        total=config["n_folds"],
                        desc="K-Folds",
                    ):
                        self._run_evaluation_on_range(
                            e_type=e_type,
                            train_range=train_range,
                            eval_range=valid_range,
                            config=config,
                            add_run_info=f"Fold-{i_fold}",
                            split_name=split_name,
                            saved_model_path=None,
                        )
                if config["eval_test_performance"]:
                    self._run_evaluation_on_range(
                        e_type=e_type,
                        train_range=train_valid_range,
                        eval_range=test_range,
                        config=config,
                        add_run_info="Test",
                        split_name=split_name,
                        saved_model_path=self.model_path,
                        detailed_evaluation=True,
                    )

    def _run_evaluation_on_range(
        self,
        e_type: BaseEvalType,
        train_range: np.ndarray,
        eval_range: np.ndarray,
        config: dict,
        add_run_info: Optional[str] = None,
        split_name: Optional[str] = None,
        saved_model_path: Union[Path, str, None] = None,
        detailed_evaluation: bool = False,
    ):
        # get train / test set
        score_dict = e_type.evaluate(
            emb_space=self.emb_space,
            labels=self.labels,
            train_range=train_range,
            evaluation_range=eval_range,
            # only needed for fine-tuning
            dataset=self.dataset,
            model=self.model,
            model_out_dim=self.model_out_dim,
            saved_model_path=saved_model_path,
            # rest of the method specific parameters set with kwargs
            **config,
        )
        # save the results to the overall dataframe + save df
        self.df.loc[len(self.df)] = list(score_dict.values()) + [
            split_name,
            add_run_info,
            e_type.name(),
        ]
        self.df.to_csv(self.df_path, index=False)
        if detailed_evaluation:
            # Detailed evaluation
            print("*" * 20 + f" {e_type.name()} " + "*" * 20)
            self.print_eval_scores(
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
                self.print_eval_scores(
                    y_true=score_dict["targets"][_df.index.values],
                    y_pred=score_dict["predictions"][_df.index.values],
                )
            gender_types = eval_df["sex"].unique()
            for gender in gender_types:
                _df = eval_df[eval_df["sex"] == gender]
                print(
                    "~" * 20 + f" Gender: {gender}, Support: {_df.shape[0]} " + "~" * 20
                )
                self.print_eval_scores(
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
            self.print_eval_scores(
                y_true=case_targets,
                y_pred=case_predictions,
            )

            if e_type is EvalFineTuning:
                print("=" * 20 + f" {e_type.name()} = my analysis " + "=" * 20)
                self.print_eval_scores_bias(score_dict)

    def create_results(self, df_results):
        df_labels = pd.read_csv("data/PASSION/label.csv")
        df_split = pd.read_csv("data/PASSION/PASSION_split.csv")

        def extract_subject_id(path: str):
            pattern = r"([A-Za-z]+[0-9]+)"
            match = re.search(pattern, path)
            if match:
                return str(match.group(1)).strip()
            else:
                return np.nan

        # Flatten the DataFrame into one row per image path
        rows = []
        unique_subject_ids = []
        for img_name, idx, lbl, pred in zip(
            df_results["filenames"],
            df_results["indices"],
            df_results["targets"],
            df_results["predictions"],
        ):
            subject_id = extract_subject_id(img_name)
            labels = df_labels[df_labels["subject_id"] == subject_id].iloc[0]
            split = df_split[df_split["subject_id"] == subject_id].iloc[0]
            rows.append(
                {
                    "correct": lbl == pred,
                    "image_path": img_name,
                    "index": idx,
                    "targets": lbl,
                    "predictions": pred,
                    **labels.to_dict(),
                    **split.drop("subject_id").to_dict(),
                }
            )
            if subject_id not in unique_subject_ids:
                unique_subject_ids.append(subject_id)

        print(len(unique_subject_ids))

        return pd.DataFrame(rows)

    def print_eval_scores_bias(self, df_results):
        logger.debug(df_results)

        df_calc_in = self.create_results(df_results)

        def do_calculations(data):
            # Detailed evaluation
            print("*" * 20 + f" overall " + "*" * 20)
            print_detailed_bias_eval_scores(
                y_true=data["targets"],
                y_pred=data["predictions"],
            )

            print("=" * 20 + " now more dynamic (grouped) " + "=" * 20)
            print_grouped_result(data, group_by="fitzpatrick")
            print_grouped_result(data, group_by="sex")

        do_calculations(df_calc_in)

    def print_eval_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        if len(self.dataset.classes) == 2:
            f1 = f1_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=1,
                average="binary",
            )
            precision = precision_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=1,
                average="binary",
            )
            recall = recall_score(
                y_true=y_true,
                y_pred=y_pred,
                pos_label=1,
                average="binary",
            )
            print(f"Bin. F1: {f1:.2f}")
            print(f"Bin. Precision: {precision:.2f}")
            print(f"Bin. Recall: {recall:.2f}")
        else:
            try:
                print(
                    classification_report(
                        y_true=y_true,
                        y_pred=y_pred,
                        target_names=self.dataset.classes,
                    )
                )
            except Exception as e:
                print(f"Error generating classification report: {e}")
            b_acc = balanced_accuracy_score(
                y_true=y_true,
                y_pred=y_pred,
            )
            print(f"Balanced Acc: {b_acc}")




in the e_type.evaluation, there is this code which is relevant to run on the checkpoint model:
# create eval predictions for saving
        img_names, targets, predictions, indices = [], [], [], []
        classifier.eval()
        for img, img_name, target, index in eval_loader:
            logger.debug(f"img_name: {img_name}")
            logger.debug(f"index: {index}")
            img = img.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.no_grad():
                pred = classifier(img)
            targets.append(target.cpu())
            predictions.append(pred.cpu())

            img_names.append(img_name)
            indices.append(index)

        logger.debug(f"img_names: {img_names}")
        logger.debug(f"indices: {indices}")
        img_names = np.hstack(img_names)
        targets = torch.concat(targets).cpu().numpy()
        predictions = torch.concat(predictions).argmax(dim=-1).cpu().numpy()
        indices = torch.concat(indices).numpy()
        results = {
            "score": float(eval_scores_dict["f1"]["scores"][best_epoch] * 100),
            "filenames": img_names,
            "indices": indices,
            "targets": targets,
            "predictions": predictions,
        }
        logger.debug(f"evaluation results: {results}")
        return results