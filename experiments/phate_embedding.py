from __future__ import annotations

import inspect
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomTreesEmbedding,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make local packages importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baselines import PageRankPHATE
from forestgeom import ForestProximity
from forestgeom.adapters import make_adapter
from experiments.runtime_utils import (
    kernel_percent_nnz,
    load_dataset_pair_with_raw_labels,
    log_progress,
    resolve_dataset_paths_from_base_names,
    timed_call,
)

try:
    from aeon.classification.sklearn import RotationForestClassifier
except Exception:
    RotationForestClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    import treeple as _treeple
except Exception:
    _treeple = None

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

DATASET_NAMES = [
    "sign_mnist",
]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"{RUN_ID}_gap_pagerank_phate_experiments"
RUN_DIR.mkdir(parents=True, exist_ok=True)

EMB_DIR = RUN_DIR / "embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = RUN_DIR / "gap_pagerank_phate_results.csv"
OUT_PARQUET = RUN_DIR / "gap_pagerank_phate_results.parquet"
PROGRESS_LOG = RUN_DIR / "gap_pagerank_phate_progress.log"

SEEDS = [44, 578, 9, 912, 345]

LABEL_COL_IDX = 0
DROP_MISSING_Y = True
VERBOSE_DATAPREP = False

IMAGE_DATASETS = {
    "pathmnist_28",
    "sign_mnist",
    "tissuemnist_28",
    "fashion_mnist",
}
DEFAULT_SCALE = "standardize"
DEFAULT_GLOBAL_TRANSFORM = False
IMAGE_SCALE = "normalize"
IMAGE_GLOBAL_TRANSFORM = True

SIGN_MNIST_ALLOWED_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K"]
SIGN_MNIST_ALLOWED_CODES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]

KNN_K_VALUES = [5, 10, 20]
LOGREG_MAX_ITER = 5000

FOREST_KWARGS = {
    "n_estimators": 100,
    "random_state": None,
    "n_jobs": -1,
}

PRECOMPUTED_PHATE_KWARGS = {
    "n_components": 2,
    "random_state": None,
    "knn": 50,
    "knn_dist": "precomputed_affinity",
    "verbose": 0,
}


# ---------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------
def flush_results(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQUET, index=False)


def append_and_flush(rows: list[dict], row: dict) -> None:
    rows.append(row)
    flush_results(rows)


def get_dataprep_kwargs(dataset_name: str) -> dict[str, object]:
    if dataset_name in IMAGE_DATASETS:
        return {
            "scale": IMAGE_SCALE,
            "global_transform": IMAGE_GLOBAL_TRANSFORM,
        }
    return {
        "scale": DEFAULT_SCALE,
        "global_transform": DEFAULT_GLOBAL_TRANSFORM,
    }


def crop_sign_mnist(
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    id_train_raw,
    id_test_raw,
):
    def _mask(y_raw):
        labels = np.asarray(y_raw)
        letter_mask = np.isin(
            np.char.upper(labels.astype(str)),
            SIGN_MNIST_ALLOWED_LETTERS,
        )
        numeric = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy()
        code_mask = np.isin(numeric, SIGN_MNIST_ALLOWED_CODES)
        return letter_mask | code_mask

    mask_train = _mask(y_train_raw)
    mask_test = _mask(y_test_raw)

    X_train = X_train[mask_train]
    X_test = X_test[mask_test]
    y_train = np.asarray(y_train)[mask_train]
    y_test = np.asarray(y_test)[mask_test]
    y_train_raw = np.asarray(y_train_raw)[mask_train]
    y_test_raw = np.asarray(y_test_raw)[mask_test]
    id_train_raw = np.asarray(id_train_raw)[mask_train]
    id_test_raw = np.asarray(id_test_raw)[mask_test]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_raw,
        y_test_raw,
        id_train_raw,
        id_test_raw,
    )


def get_knn_neighborhoods(n_train: int) -> list[int]:
    return [k for k in KNN_K_VALUES if k < n_train]


def knn_test_accuracy_multi(
    x_train_2d: np.ndarray,
    x_test_2d: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, list[int], dict[int, float]]:
    k_values = get_knn_neighborhoods(len(y_train))
    if not k_values:
        return np.nan, [], {}

    x_train_2d = np.asarray(x_train_2d, dtype=np.float32, order="C")
    x_test_2d = np.asarray(x_test_2d, dtype=np.float32, order="C")
    y_train = np.asarray(y_train)

    k_max = max(k_values)
    tree = KDTree(x_train_2d, leaf_size=64, metric="euclidean")
    nn_idx = tree.query(x_test_2d, k=k_max, return_distance=False)

    scores: dict[int, float] = {}
    for k in k_values:
        idx_k = nn_idx[:, :k]
        neigh_labels = y_train[idx_k]

        y_pred = np.empty(len(y_test), dtype=y_train.dtype)
        for i in range(len(y_test)):
            vals, counts = np.unique(neigh_labels[i], return_counts=True)
            y_pred[i] = vals[np.argmax(counts)]

        scores[k] = float(accuracy_score(y_test, y_pred))

    avg_score = float(np.mean(list(scores.values()))) if scores else np.nan
    return avg_score, k_values, scores


def logistic_test_accuracy(
    x_train_2d: np.ndarray,
    x_test_2d: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> float:
    clf = LogisticRegression(
        max_iter=LOGREG_MAX_ITER,
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(x_train_2d, y_train)
    y_pred = clf.predict(x_test_2d)
    return float(accuracy_score(y_test, y_pred))


def save_embedding(
    out_path: Path,
    row_ids: np.ndarray,
    y_raw: np.ndarray,
    emb_2d: np.ndarray,
) -> None:
    df = pd.DataFrame(
        {
            "row_id": np.asarray(row_ids),
            "label": np.asarray(y_raw),
            "x1": emb_2d[:, 0],
            "x2": emb_2d[:, 1],
        }
    )
    df.to_csv(out_path, index=False)


def save_embedding_plot(
    out_path: Path,
    y_raw: np.ndarray,
    emb_2d: np.ndarray,
    title: str,
) -> None:
    labels = pd.Index(np.asarray(y_raw))
    codes, uniques = pd.factorize(labels, sort=True)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab20", max(len(uniques), 1))
    scatter = ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=codes,
        cmap=cmap,
        s=10,
        alpha=0.8,
        linewidths=0,
    )
    ax.set_title(title)
    ax.set_xlabel("PHATE 1")
    ax.set_ylabel("PHATE 2")

    if len(uniques) <= 20:
        handles, _ = scatter.legend_elements(num=len(uniques))
        ax.legend(
            handles,
            [str(label) for label in uniques],
            title="label",
            loc="best",
            fontsize=8,
            title_fontsize=9,
            frameon=False,
        )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_dataset_seed_dir(dataset_name: str, seed: int) -> Path:
    out_dir = EMB_DIR / dataset_name / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def base_result_dict() -> dict:
    return {
        "adapter_name": "",
        "forest_fit_time_s": np.nan,
        "cache_build_time_s": np.nan,
        "train_kernel_time_s": np.nan,
        "phate_fit_transform_time_s": np.nan,
        "train_total_time_s": np.nan,
        "train_total_peak_mb": np.nan,
        "kernel_percent_nnz": np.nan,
        "knn_test_acc_avg": np.nan,
        "knn_k_values": "",
        "knn_test_acc_by_k": "{}",
        "linear_test_acc": np.nan,
        "train_embedding_file": "",
        "test_embedding_file": "",
        "train_plot_file": "",
        "test_plot_file": "",
        "all_plot_file": "",
        "status": "ok",
        "error": "",
        "skip_reason": "",
    }


def treeple_class(name: str):
    if _treeple is None:
        return None
    return getattr(_treeple, name, None)


MODEL_SPECS = [
    {
        "model_name": "rf_classifier",
        "builder": lambda seed: RandomForestClassifier(
            n_estimators=FOREST_KWARGS["n_estimators"],
            bootstrap=True,
            random_state=seed,
            n_jobs=FOREST_KWARGS["n_jobs"],
        ),
        "fit_with_y": True,
    },
    {
        "model_name": "et_classifier",
        "builder": lambda seed: ExtraTreesClassifier(
            n_estimators=FOREST_KWARGS["n_estimators"],
            bootstrap=True,
            random_state=seed,
            n_jobs=FOREST_KWARGS["n_jobs"],
        ),
        "fit_with_y": True,
    },
    {
        "model_name": "random_trees_embedding",
        "builder": lambda seed: RandomTreesEmbedding(
            n_estimators=FOREST_KWARGS["n_estimators"],
            bootstrap=True,
            random_state=seed,
            n_jobs=FOREST_KWARGS["n_jobs"],
        ),
        "fit_with_y": False,
    },
    {
        "model_name": "gbt_classifier",
        "builder": lambda seed: GradientBoostingClassifier(
            n_estimators=FOREST_KWARGS["n_estimators"],
            random_state=seed,
        ),
        "fit_with_y": True,
    },
    {
        "model_name": "lgbm_classifier",
        "builder": (
            None
            if LGBMClassifier is None
            else lambda seed: LGBMClassifier(
                n_estimators=FOREST_KWARGS["n_estimators"],
                learning_rate=0.1,
                random_state=seed,
                verbose=-1,
                n_jobs=FOREST_KWARGS["n_jobs"],
            )
        ),
        "fit_with_y": True,
    },
    {
        "model_name": "xgb_classifier",
        "builder": (
            None
            if XGBClassifier is None
            else lambda seed: XGBClassifier(
                n_estimators=FOREST_KWARGS["n_estimators"],
                learning_rate=0.1,
                random_state=seed,
                verbosity=0,
                n_jobs=FOREST_KWARGS["n_jobs"],
                eval_metric="logloss",
            )
        ),
        "fit_with_y": True,
    },
    {
        "model_name": "rotation_forest_classifier",
        "builder": (
            None
            if RotationForestClassifier is None
            else lambda seed: RotationForestClassifier(
                n_estimators=FOREST_KWARGS["n_estimators"],
                random_state=seed,
            )
        ),
        "fit_with_y": True,
    },
    {
        "model_name": "treeple_oblique_classifier",
        "builder": (
            None
            if treeple_class("ObliqueRandomForestClassifier") is None
            else lambda seed: treeple_class("ObliqueRandomForestClassifier")(
                n_estimators=FOREST_KWARGS["n_estimators"],
                bootstrap=True,
                random_state=seed,
                n_jobs=FOREST_KWARGS["n_jobs"],
            )
        ),
        "fit_with_y": True,
    },
    {
        "model_name": "treeple_patch_oblique_classifier",
        "builder": (
            None
            if treeple_class("PatchObliqueRandomForestClassifier") is None
            else lambda seed: treeple_class("PatchObliqueRandomForestClassifier")(
                n_estimators=FOREST_KWARGS["n_estimators"],
                bootstrap=True,
                random_state=seed,
                n_jobs=FOREST_KWARGS["n_jobs"],
            )
        ),
        "fit_with_y": True,
    },
    {
        "model_name": "treeple_extra_oblique_classifier",
        "builder": (
            None
            if treeple_class("ExtraObliqueRandomForestClassifier") is None
            else lambda seed: treeple_class("ExtraObliqueRandomForestClassifier")(
                n_estimators=FOREST_KWARGS["n_estimators"],
                bootstrap=True,
                random_state=seed,
                n_jobs=FOREST_KWARGS["n_jobs"],
            )
        ),
        "fit_with_y": True,
    },
    {
        "model_name": "treeple_honest_classifier",
        "builder": (
            None
            if treeple_class("HonestForestClassifier") is None
            else lambda seed: treeple_class("HonestForestClassifier")(
                n_estimators=FOREST_KWARGS["n_estimators"],
                bootstrap=True,
                random_state=seed,
                n_jobs=FOREST_KWARGS["n_jobs"],
            )
        ),
        "fit_with_y": True,
    },
    {
        "model_name": "treeple_unsupervised_random_forest",
        "builder": (
            None
            if treeple_class("UnsupervisedRandomForest") is None
            else lambda seed: treeple_class("UnsupervisedRandomForest")(
                n_estimators=FOREST_KWARGS["n_estimators"],
                bootstrap=True,
                random_state=seed,
                n_jobs=FOREST_KWARGS["n_jobs"],
            )
        ),
        "fit_with_y": False,
    },
    {
        "model_name": "treeple_extended_isolation_forest",
        "builder": (
            None
            if treeple_class("ExtendedIsolationForest") is None
            else lambda seed: treeple_class("ExtendedIsolationForest")(
                n_estimators=FOREST_KWARGS["n_estimators"],
                random_state=seed,
                n_jobs=FOREST_KWARGS["n_jobs"],
            )
        ),
        "fit_with_y": False,
    },
]


def resolve_model_entry(model_spec: dict, seed: int) -> dict:
    result = {
        "model_name": model_spec["model_name"],
        "fit_with_y": model_spec["fit_with_y"],
        "status": "ready",
        "skip_reason": "",
        "forest": None,
        "adapter_name": "",
    }

    builder = model_spec["builder"]
    if builder is None:
        result["status"] = "skipped"
        result["skip_reason"] = "model dependency is not installed"
        return result

    try:
        forest = builder(seed)
    except Exception as exc:
        result["status"] = "skipped"
        result["skip_reason"] = f"model construction failed: {exc}"
        return result

    try:
        adapter = make_adapter(forest)
        result["adapter_name"] = type(adapter).__name__
        adapter.validate_weight_scheme("gap")
    except Exception as exc:
        result["status"] = "skipped"
        result["skip_reason"] = f"gap unsupported: {exc}"
        return result

    result["forest"] = forest
    return result


def run_gap_pagerank_phate(
    forest,
    adapter_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    y_train_raw,
    y_test_raw,
    id_train_raw,
    id_test_raw,
    seed: int,
    out_dir: Path,
    fit_with_y: bool,
    model_name: str,
) -> dict:
    X_all = np.vstack([X_train, X_test])
    y_all = np.concatenate([y_train, y_test])
    y_fit = y_all if fit_with_y else None

    # Precomputed PHATE is transductive, so we embed the full symmetric GAP
    # affinity on the concatenated train/test split and then split the 2D
    # coordinates back into train/test blocks for downstream evaluation.
    def train_pipeline():
        fk = ForestProximity(forest=forest, weight_scheme="gap")

        t0 = time.perf_counter()
        fk._fit_forest(X_all, y_fit)
        forest_fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        fk._build_cache()
        cache_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        K_all = fk.training_proximity(
            force_symmetric=True,
            return_dense=False,
        )
        kernel_time = time.perf_counter() - t0

        phate_op = PageRankPHATE(**PRECOMPUTED_PHATE_KWARGS)

        t0 = time.perf_counter()
        x_all_2d = phate_op.fit_transform(K_all)
        phate_time = time.perf_counter() - t0

        return fk, K_all, x_all_2d, forest_fit_time, cache_time, kernel_time, phate_time

    (
        fk,
        K_all,
        x_all_2d,
        forest_fit_time,
        cache_time,
        kernel_time,
        phate_time,
    ), train_total_time, train_total_peak = timed_call(train_pipeline)

    n_train = X_train.shape[0]
    x_train_2d = np.asarray(x_all_2d[:n_train])
    x_test_2d = np.asarray(x_all_2d[n_train:])

    knn_acc, knn_k_values, knn_scores = knn_test_accuracy_multi(
        x_train_2d,
        x_test_2d,
        y_train,
        y_test,
    )
    lin_acc = logistic_test_accuracy(
        x_train_2d,
        x_test_2d,
        y_train,
        y_test,
        seed,
    )

    train_file = out_dir / f"{model_name}_gap_pagerank_phate_train.csv"
    test_file = out_dir / f"{model_name}_gap_pagerank_phate_test.csv"
    train_plot_file = out_dir / f"{model_name}_gap_pagerank_phate_train.png"
    test_plot_file = out_dir / f"{model_name}_gap_pagerank_phate_test.png"
    all_plot_file = out_dir / f"{model_name}_gap_pagerank_phate_all.png"
    save_embedding(train_file, id_train_raw, y_train_raw, x_train_2d)
    save_embedding(test_file, id_test_raw, y_test_raw, x_test_2d)
    save_embedding_plot(
        train_plot_file,
        y_train_raw,
        x_train_2d,
        title=f"{model_name} train GAP PageRankPHATE",
    )
    save_embedding_plot(
        test_plot_file,
        y_test_raw,
        x_test_2d,
        title=f"{model_name} test GAP PageRankPHATE",
    )
    save_embedding_plot(
        all_plot_file,
        np.concatenate([np.asarray(y_train_raw), np.asarray(y_test_raw)]),
        x_all_2d,
        title=f"{model_name} all GAP PageRankPHATE",
    )

    result = base_result_dict()
    result.update(
        {
            "method_name": "gap_pagerank_phate_precomputed",
            "adapter_name": adapter_name,
            "forest_fit_time_s": forest_fit_time,
            "cache_build_time_s": cache_time,
            "train_kernel_time_s": kernel_time,
            "phate_fit_transform_time_s": phate_time,
            "train_total_time_s": train_total_time,
            "train_total_peak_mb": train_total_peak,
            "kernel_percent_nnz": kernel_percent_nnz(K_all),
            "knn_test_acc_avg": knn_acc,
            "knn_k_values": str(knn_k_values),
            "knn_test_acc_by_k": str(knn_scores),
            "linear_test_acc": lin_acc,
            "train_embedding_file": str(train_file),
            "test_embedding_file": str(test_file),
            "train_plot_file": str(train_plot_file),
            "test_plot_file": str(test_plot_file),
            "all_plot_file": str(all_plot_file),
            "transductive_precomputed": True,
        }
    )
    del fk
    return result


def main() -> None:
    dataset_groups = resolve_dataset_paths_from_base_names(DATA_DIR, DATASET_NAMES)
    rows: list[dict] = []

    log_progress(f"Run ID: {RUN_ID}", PROGRESS_LOG)
    log_progress(f"Run directory: {RUN_DIR}", PROGRESS_LOG)
    log_progress(f"Embedding directory: {EMB_DIR}", PROGRESS_LOG)
    log_progress(f"Datasets: {sorted(dataset_groups.keys())}", PROGRESS_LOG)
    log_progress(f"Seeds: {SEEDS}", PROGRESS_LOG)
    log_progress("Method: precomputed PageRankPHATE on symmetric GAP affinity", PROGRESS_LOG)
    log_progress(f"PHATE kwargs: {PRECOMPUTED_PHATE_KWARGS}", PROGRESS_LOG)
    log_progress(
        "Embedding mode: transductive full-dataset fit because precomputed PHATE "
        "does not support inductive transform.",
        PROGRESS_LOG,
    )
    log_progress(
        f"Candidate models: {[spec['model_name'] for spec in MODEL_SPECS]}",
        PROGRESS_LOG,
    )

    def save_result_row(
        result: dict,
        dataset_name: str,
        seed: int,
        meta: dict,
        scale,
        global_transform,
        n_train: int,
        n_test: int,
        model_name: str,
    ) -> None:
        row = {
            "run_id": RUN_ID,
            "dataset": dataset_name,
            "seed": seed,
            "model_name": model_name,
            "predefined_split": meta["predefined_split"],
            "scale": scale,
            "global_transform": global_transform,
            "n_train": n_train,
            "n_test": n_test,
            **result,
        }
        append_and_flush(rows, row)

    for dataset_name, dataset_paths in dataset_groups.items():
        log_progress(f"=== DATASET: {dataset_name} ===", PROGRESS_LOG)

        dataprep_kwargs = get_dataprep_kwargs(dataset_name)
        scale = dataprep_kwargs["scale"]
        global_transform = dataprep_kwargs["global_transform"]

        for seed in SEEDS:
            log_progress(f">>> SEED: {seed}", PROGRESS_LOG)

            (
                X_train,
                X_test,
                y_train,
                y_test,
                y_train_raw,
                y_test_raw,
                id_train_raw,
                id_test_raw,
                meta,
            ) = load_dataset_pair_with_raw_labels(
                dataset_name=dataset_name,
                paths=dataset_paths,
                seed=seed,
                label_col_idx=LABEL_COL_IDX,
                scale=scale,
                global_transform=global_transform,
                drop_missing_y=DROP_MISSING_Y,
                verbose_dataprep=VERBOSE_DATAPREP,
            )

            if dataset_name == "sign_mnist":
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    y_train_raw,
                    y_test_raw,
                    id_train_raw,
                    id_test_raw,
                ) = crop_sign_mnist(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    y_train_raw,
                    y_test_raw,
                    id_train_raw,
                    id_test_raw,
                )

            out_dir = make_dataset_seed_dir(dataset_name, seed)

            for model_spec in MODEL_SPECS:
                model_name = model_spec["model_name"]
                log_progress(
                    f"Model: {model_name} | resolving adapter availability",
                    PROGRESS_LOG,
                )
                model_entry = resolve_model_entry(model_spec, seed)

                if model_entry["status"] != "ready":
                    result = base_result_dict()
                    result.update(
                        {
                            "method_name": "gap_pagerank_phate_precomputed",
                            "adapter_name": model_entry["adapter_name"],
                            "status": "skipped",
                            "skip_reason": model_entry["skip_reason"],
                            "transductive_precomputed": True,
                        }
                    )
                    save_result_row(
                        result,
                        dataset_name,
                        seed,
                        meta,
                        scale,
                        global_transform,
                        len(y_train),
                        len(y_test),
                        model_name,
                    )
                    log_progress(
                        f"Skipped {model_name}: {model_entry['skip_reason']}",
                        PROGRESS_LOG,
                    )
                    continue

                model_out_dir = out_dir / model_name
                model_out_dir.mkdir(parents=True, exist_ok=True)

                try:
                    result = run_gap_pagerank_phate(
                        forest=model_entry["forest"],
                        adapter_name=model_entry["adapter_name"],
                        X_train=X_train,
                        X_test=X_test,
                        y_train=y_train,
                        y_test=y_test,
                        y_train_raw=y_train_raw,
                        y_test_raw=y_test_raw,
                        id_train_raw=id_train_raw,
                        id_test_raw=id_test_raw,
                        seed=seed,
                        out_dir=model_out_dir,
                        fit_with_y=model_entry["fit_with_y"],
                        model_name=model_name,
                    )
                except Exception as exc:
                    result = base_result_dict()
                    result.update(
                        {
                            "method_name": "gap_pagerank_phate_precomputed",
                            "adapter_name": model_entry["adapter_name"],
                            "status": "failed",
                            "error": str(exc),
                            "transductive_precomputed": True,
                        }
                    )
                    save_result_row(
                        result,
                        dataset_name,
                        seed,
                        meta,
                        scale,
                        global_transform,
                        len(y_train),
                        len(y_test),
                        model_name,
                    )
                    log_progress(
                        f"Failed {model_name}: {exc}",
                        PROGRESS_LOG,
                    )
                    continue

                save_result_row(
                    result,
                    dataset_name,
                    seed,
                    meta,
                    scale,
                    global_transform,
                    len(y_train),
                    len(y_test),
                    model_name,
                )
                log_progress(
                    f"Done {model_name} | adapter={result['adapter_name']} | "
                    f"knn_acc={result['knn_test_acc_avg']:.4f} | "
                    f"lin_acc={result['linear_test_acc']:.4f}",
                    PROGRESS_LOG,
                )

    flush_results(rows)
    log_progress(f"Saved results to: {OUT_CSV}", PROGRESS_LOG)
    log_progress(f"Saved results to: {OUT_PARQUET}", PROGRESS_LOG)
    log_progress("Done.", PROGRESS_LOG)


if __name__ == "__main__":
    main()
