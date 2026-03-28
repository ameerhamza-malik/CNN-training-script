#!/usr/bin/env python3
"""Train a custom neural-network ATT&CK multi-label model.

This script mirrors the existing ATT&CK resolution + feature policy, but trains
an MLP-based multi-label model (OneVsRest + MLPClassifier).
Artifacts are written under CNN/artifacts by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, MultiLabelBinarizer, OneHotEncoder, StandardScaler


def _load_train_utils_module():
    project_root = Path(__file__).resolve().parents[1]
    train_script = project_root / "mitre_multilabel_pipeline" / "train_mitre_multilabel.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Could not locate training utilities: {train_script}")

    spec = importlib.util.spec_from_file_location("mitre_train_utils", train_script)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load training utilities module specification.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train custom NN ATT&CK multi-label classifier from malicious flow data."
    )
    parser.add_argument("--input", default="labeled_ja4_flows_malicious_only.csv")
    parser.add_argument("--label-column", default="Mitre_Techniques")
    parser.add_argument("--output-prefix", default="CNN/artifacts/mitre_multilabel")
    parser.add_argument("--model-tag", default="custom_mlp")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--cache-dir", default="mitre_multilabel_pipeline/.attack_cache")
    parser.add_argument("--refresh-attack-data", action="store_true")
    parser.add_argument(
        "--drop-columns",
        default="src_ip,dst_ip,src_port,dst_port,protocol,Mitre_Tactics,Mitre_Techniques,Label,ja4l_c,ja4l_s",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--min-label-frequency", type=int, default=5)
    parser.add_argument("--max-rows", type=int, default=None)

    parser.add_argument("--svd-components", type=int, default=256)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--no-save-resolved-csv", action="store_false", dest="save_resolved_csv")
    parser.set_defaults(save_resolved_csv=True)
    return parser.parse_args()


def ensure_non_empty(probabilities: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    empty_mask = predictions.sum(axis=1) == 0
    if not np.any(empty_mask):
        return predictions
    fallback_indices = np.argmax(probabilities[empty_mask], axis=1)
    predictions[empty_mask] = 0
    predictions[empty_mask, fallback_indices] = 1
    return predictions


def main() -> None:
    args = parse_args()

    if not 0.0 < args.test_size < 1.0:
        raise ValueError("--test-size must be between 0 and 1")
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.min_label_frequency < 1:
        raise ValueError("--min-label-frequency must be at least 1")
    if args.svd_components < 2:
        raise ValueError("--svd-components must be at least 2")

    utils = _load_train_utils_module()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    print(f"Loading dataset: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    original_row_count = len(df)

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
        print(f"Using first {len(df)} rows due to --max-rows")

    if args.label_column not in df.columns:
        raise ValueError(f"Label column not found: {args.label_column}")

    cache_dir = Path(args.cache_dir)
    print("Loading ATT&CK catalog...")
    name_to_parents, parent_to_subs, id_to_pattern = utils.load_attack_catalog(
        cache_dir=cache_dir,
        refresh=args.refresh_attack_data,
    )

    print("Resolving ATT&CK labels...")
    (
        resolved_id_rows,
        resolved_label_rows,
        unresolved_tokens,
        matrix_counts,
        resolution_type_counts,
    ) = utils.resolve_series_labels(
        labels_series=df[args.label_column],
        name_to_parents=name_to_parents,
        parent_to_subs=parent_to_subs,
        id_to_pattern=id_to_pattern,
    )

    processed_df = df.copy()
    processed_df["Resolved_Attack_IDs"] = [";".join(row) for row in resolved_id_rows]
    processed_df["Resolved_Attack_Labels"] = [";".join(row) for row in resolved_label_rows]
    processed_df["Resolved_Label_Count"] = [len(row) for row in resolved_id_rows]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_prefix = utils.resolve_unique_output_prefix(
        output_prefix=Path(args.output_prefix),
        model_tag=args.model_tag,
        run_id=run_id,
        allow_overwrite=args.allow_overwrite,
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    resolved_csv_path = Path(f"{output_prefix}_resolved_labels.csv")
    if args.save_resolved_csv:
        processed_df.to_csv(resolved_csv_path, index=False)
        print(f"Saved resolved dataset: {resolved_csv_path}")

    rows_with_labels_mask = np.array([len(row) > 0 for row in resolved_id_rows], dtype=bool)
    if rows_with_labels_mask.sum() == 0:
        raise RuntimeError("No rows contain resolvable ATT&CK labels; training cannot proceed")

    train_df = processed_df.loc[rows_with_labels_mask].copy()
    target_rows = [resolved_id_rows[idx] for idx in np.flatnonzero(rows_with_labels_mask)]

    mlb = MultiLabelBinarizer()
    y_matrix = mlb.fit_transform(target_rows)
    classes = np.asarray(mlb.classes_)
    class_counts = y_matrix.sum(axis=0).astype(int)

    min_freq_mask = class_counts >= args.min_label_frequency
    dropped_low_frequency_labels = classes[~min_freq_mask].tolist()
    if not np.any(min_freq_mask):
        raise RuntimeError("No labels remain after --min-label-frequency")

    y_matrix = y_matrix[:, min_freq_mask]
    classes = classes[min_freq_mask]

    non_empty_row_mask = y_matrix.sum(axis=1) > 0
    if not np.any(non_empty_row_mask):
        raise RuntimeError("All rows lost labels after low-frequency filtering")

    train_df = train_df.loc[non_empty_row_mask].copy()
    y_matrix = y_matrix[non_empty_row_mask]

    drop_columns = [col.strip() for col in args.drop_columns.split(",") if col.strip()]
    feature_df, applied_drop_columns, auto_identity_drop_columns, constant_columns = utils.build_feature_frame(
        train_df, drop_columns
    )
    if feature_df.empty:
        raise RuntimeError("No feature columns remain after drop policy")

    feature_indices = feature_df.index.to_numpy()
    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        feature_df,
        y_matrix,
        feature_indices,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    train_positive_counts = y_train.sum(axis=0)
    has_positive_in_train = train_positive_counts > 0
    dropped_no_train_positive_labels = classes[~has_positive_in_train].tolist()

    if not np.any(has_positive_in_train):
        raise RuntimeError("No labels have positive samples in training split")

    y_train = y_train[:, has_positive_in_train]
    y_test = y_test[:, has_positive_in_train]
    classes = classes[has_positive_in_train]

    numeric_features = x_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [column for column in x_train.columns if column not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    print("Fitting preprocessors...")
    x_train_sparse = preprocessor.fit_transform(x_train)
    x_test_sparse = preprocessor.transform(x_test)

    sparse_scaler = MaxAbsScaler()
    x_train_sparse = sparse_scaler.fit_transform(x_train_sparse)
    x_test_sparse = sparse_scaler.transform(x_test_sparse)

    max_possible_components = max(2, min(x_train_sparse.shape[0] - 1, x_train_sparse.shape[1] - 1))
    svd_components = min(args.svd_components, max_possible_components)
    if svd_components != args.svd_components:
        print(
            f"Adjusted svd-components from {args.svd_components} to {svd_components} based on feature matrix shape"
        )

    svd = TruncatedSVD(n_components=svd_components, random_state=args.random_state)
    x_train_dense = svd.fit_transform(x_train_sparse)
    x_test_dense = svd.transform(x_test_sparse)

    dense_scaler = StandardScaler()
    x_train_dense = dense_scaler.fit_transform(x_train_dense)
    x_test_dense = dense_scaler.transform(x_test_dense)

    hidden_layers: Sequence[int]
    if args.hidden2 > 0:
        hidden_layers = (args.hidden1, args.hidden2)
    else:
        hidden_layers = (args.hidden1,)

    classifier = OneVsRestClassifier(
        MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=args.alpha,
            batch_size=args.batch_size,
            learning_rate_init=args.learning_rate_init,
            max_iter=args.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=args.random_state,
            verbose=False,
        ),
        n_jobs=-1,
    )

    print("Training custom neural network model (OneVsRest + MLPClassifier)...")
    classifier.fit(x_train_dense, y_train)

    print("Evaluating on test split...")
    probabilities = classifier.predict_proba(x_test_dense)
    y_pred = (probabilities >= args.threshold).astype(int)
    y_pred = ensure_non_empty(probabilities, y_pred)

    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    samples_f1 = f1_score(y_test, y_pred, average="samples", zero_division=0)
    subset_accuracy = accuracy_score(y_test, y_pred)
    hamming = hamming_loss(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    class_frequency_map = {
        class_id: int(count)
        for class_id, count in zip(classes, y_matrix.sum(axis=0).astype(int))
    }

    attack_id_to_meta = {}
    for attack_id, pattern in id_to_pattern.items():
        parent_name = ""
        if pattern.parent_attack_id and pattern.parent_attack_id in id_to_pattern:
            parent_name = id_to_pattern[pattern.parent_attack_id].name
        attack_id_to_meta[attack_id] = {
            "attack_id": pattern.attack_id,
            "name": pattern.name,
            "matrix": pattern.matrix,
            "is_subtechnique": pattern.is_subtechnique,
            "parent_attack_id": pattern.parent_attack_id,
            "parent_name": parent_name,
        }

    metrics_payload = {
        "dataset": {
            "input_path": str(input_path),
            "original_rows": int(original_row_count),
            "rows_used_after_label_resolution": int(rows_with_labels_mask.sum()),
            "rows_used_for_training_after_label_filtering": int(len(train_df)),
            "features_used": int(x_train.shape[1]),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "numeric_feature_count": int(len(numeric_features)),
            "categorical_feature_count": int(len(categorical_features)),
            "auto_identity_drop_columns": auto_identity_drop_columns,
            "dropped_constant_columns": constant_columns,
            "applied_drop_columns": applied_drop_columns,
            "svd_components": int(svd_components),
        },
        "label_resolution": {
            "resolved_unique_labels": int(len(classes)),
            "matrix_counts": dict(matrix_counts),
            "resolution_type_counts": dict(resolution_type_counts),
            "unresolved_token_count": int(sum(unresolved_tokens.values())),
            "top_unresolved_tokens": unresolved_tokens.most_common(25),
            "dropped_low_frequency_labels": dropped_low_frequency_labels,
            "dropped_labels_without_train_positive": dropped_no_train_positive_labels,
        },
        "training": {
            "model": "OneVsRestClassifier(MLPClassifier)",
            "model_tag": args.model_tag,
            "run_id": run_id,
            "threshold": args.threshold,
            "min_label_frequency": args.min_label_frequency,
            "random_state": args.random_state,
            "allow_overwrite": bool(args.allow_overwrite),
            "output_prefix": str(output_prefix),
            "hidden_layers": list(hidden_layers),
            "max_iter": args.max_iter,
            "batch_size": args.batch_size,
            "learning_rate_init": args.learning_rate_init,
            "alpha": args.alpha,
        },
        "metrics": {
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "samples_f1": samples_f1,
            "subset_accuracy": subset_accuracy,
            "hamming_loss": hamming,
        },
        "class_frequencies": class_frequency_map,
        "classification_report": report,
    }

    bundle = {
        "preprocessor": preprocessor,
        "sparse_scaler": sparse_scaler,
        "svd": svd,
        "dense_scaler": dense_scaler,
        "classifier": classifier,
        "classes": classes.tolist(),
        "threshold": float(args.threshold),
        "model_tag": args.model_tag,
        "run_id": run_id,
        "feature_columns": x_train.columns.tolist(),
        "drop_columns": applied_drop_columns,
        "constant_columns": constant_columns,
        "attack_id_to_meta": attack_id_to_meta,
    }

    bundle_path = Path(f"{output_prefix}_bundle.pkl")
    metrics_path = Path(f"{output_prefix}_metrics.json")
    label_catalog_path = Path(f"{output_prefix}_label_catalog.json")
    test_predictions_path = Path(f"{output_prefix}_test_predictions.csv")

    joblib.dump(bundle, bundle_path)
    metrics_path.write_text(json.dumps(utils.to_builtin_types(metrics_payload), indent=2), encoding="utf-8")
    label_catalog_path.write_text(json.dumps(utils.to_builtin_types(attack_id_to_meta), indent=2), encoding="utf-8")

    resultsmetrics_path = output_prefix.parent / "Resultsmetrics.txt"
    utils.append_resultsmetrics_training(
        results_file=resultsmetrics_path,
        run_id=run_id,
        model_tag=args.model_tag,
        input_path=input_path,
        metrics_path=metrics_path,
        bundle_path=bundle_path,
        values={
            "micro_f1": float(micro_f1),
            "macro_f1": float(macro_f1),
            "samples_f1": float(samples_f1),
            "subset_accuracy": float(subset_accuracy),
            "hamming_loss": float(hamming),
        },
    )

    true_id_rows = utils.decode_binary_rows(y_test, classes)
    pred_id_rows = utils.decode_binary_rows(y_pred, classes)

    def id_list_to_pretty(ids: Sequence[str]) -> List[str]:
        labels = []
        for attack_id in ids:
            pattern = id_to_pattern.get(attack_id)
            if not pattern:
                labels.append(attack_id)
            else:
                labels.append(utils.format_attack_label(pattern, id_to_pattern))
        return labels

    test_view_columns = x_test.columns.tolist()[:10]
    test_predictions_df = processed_df.loc[idx_test, test_view_columns].copy()
    test_predictions_df["True_Attack_IDs"] = [";".join(ids) for ids in true_id_rows]
    test_predictions_df["Pred_Attack_IDs"] = [";".join(ids) for ids in pred_id_rows]
    test_predictions_df["True_Attack_Labels"] = [
        ";".join(id_list_to_pretty(ids)) for ids in true_id_rows
    ]
    test_predictions_df["Pred_Attack_Labels"] = [
        ";".join(id_list_to_pretty(ids)) for ids in pred_id_rows
    ]
    test_predictions_df.to_csv(test_predictions_path, index=False)

    print("\nTraining complete.")
    print(f"Run ID: {run_id}")
    print(f"Bundle: {bundle_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Label catalog: {label_catalog_path}")
    print(f"Test predictions: {test_predictions_path}")
    print(f"Results log: {resultsmetrics_path}")
    if args.save_resolved_csv:
        print(f"Resolved dataset: {resolved_csv_path}")

    print("\nEvaluation Summary")
    print(f"Micro F1: {micro_f1:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Samples F1: {samples_f1:.4f}")
    print(f"Subset Accuracy: {subset_accuracy:.4f}")
    print(f"Hamming Loss: {hamming:.4f}")


if __name__ == "__main__":
    main()
