from __future__ import annotations

import json
import sys
import time
import tempfile
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

# ---------------------------------------------------------------------
# Make local packages importable
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forestgeom import ForestProximity
from experiments.runtime_utils import (
    kernel_percent_nnz,
    log_progress,
    predict_classifier_from_proximity,
    resolve_dataset_paths_from_base_names,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"

# Used only for dataset ablation
DATASET_ABLATION_DATASET_NAMES = [

    "epsilon",
    "airlines",
    "celegans",
    "covertype",
    # "nsl_kdd+",
    "pathmnist_28",
    "pbmc",
    "sign_mnist",
    "tissuemnist_28",
    "tv_news_combined",
    "zilionis",

    "higgs",
    "susy",
]

# Used for all non-dataset ablations
FIXED_ABLATION_DATASET_NAMES = [
    "airlines",
    # "pathmnist_28",
    # "tv_news_combined",
    "covertype",
]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
RUN_DIR = RESULTS_DIR / f"{RUN_ID}_ablation"
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [44, 578, 9, 912, 345]

LABEL_COL_IDX = 0
DROP_MISSING_Y = True
VERBOSE_DATAPREP = False

# ---------------------------------------------------------
# Train subset sizes used for scaling curves.
# Sizes are generated per dataset from a global minimum power
# of two, then doubled until the largest power of two below
# full size. The exact full size is appended if needed.
# ---------------------------------------------------------
MIN_POW = 14   # 2**14 = 16384

RUN_DATASET_ABLATION = True
RUN_KERNEL_METHOD_ABLATION = True
RUN_MODEL_TYPE_ABLATION = True
RUN_MAX_DEPTH_ABLATION = True
RUN_MIN_SAMPLES_LEAF_ABLATION = True

RUN_FULL_KERNEL = True


# ---------------------------------------------------------------------
# ABLATION SETTINGS
# ---------------------------------------------------------------------

# 1) dataset ablation
DATASET_ABLATION_SETTINGS = [
    {
        "model_type": "rf",
        "kernel_method": "gap",
        "ablation_name": "dataset_ablation",
        "ablation_cfg": {"bootstrap": True},
    }
]

# 2) kernel method ablation
KERNEL_METHOD_SETTINGS = [
    {
        "model_type": "rf",
        "kernel_method": "original",
        "ablation_name": "kernel_method=original",
        "ablation_cfg": {"bootstrap": True},
    },
    {
        "model_type": "rf",
        "kernel_method": "kerf",
        "ablation_name": "kernel_method=kerf",
        "ablation_cfg": {"bootstrap": True},
    },
    {
        "model_type": "rf",
        "kernel_method": "oob",
        "ablation_name": "kernel_method=oob",
        "ablation_cfg": {"bootstrap": True},
    },
    {
        "model_type": "rf",
        "kernel_method": "gap",
        "ablation_name": "kernel_method=gap",
        "ablation_cfg": {"bootstrap": True},
    },
]

# 3) model type ablation
MODEL_TYPE_SETTINGS = [
    {
        "model_type": "rf",
        "kernel_method": "gap",
        "ablation_name": "model_type=rf",
        "ablation_cfg": {"bootstrap": True},
    },
    {
        "model_type": "et",
        "kernel_method": "gap",
        "ablation_name": "model_type=et",
        "ablation_cfg": {"bootstrap": True},
    },
]

# 4) max depth ablation
MAX_DEPTH_VALUES = [10, 20, 30, None]
MAX_DEPTH_FIXED_CFG = {
    "bootstrap": True,
    "min_samples_leaf": 1,
}
MAX_DEPTH_SETTINGS = [
    {
        "model_type": "rf",
        "kernel_method": "gap",
        "ablation_name": f"max_depth={value}",
        "ablation_cfg": {"max_depth": value, **MAX_DEPTH_FIXED_CFG},
    }
    for value in MAX_DEPTH_VALUES
]

# 5) min samples leaf ablation
MIN_SAMPLES_LEAF_VALUES = [1, 5, 10, 20]
MIN_SAMPLES_LEAF_FIXED_CFG = {
    "bootstrap": True,
    "max_depth": None,
}
MIN_SAMPLES_LEAF_SETTINGS = [
    {
        "model_type": "rf",
        "kernel_method": "gap",
        "ablation_name": f"min_samples_leaf={value}",
        "ablation_cfg": {"min_samples_leaf": value, **MIN_SAMPLES_LEAF_FIXED_CFG},
    }
    for value in MIN_SAMPLES_LEAF_VALUES
]


# ---------------------------------------------------------------------
# Script-local utilities
# ---------------------------------------------------------------------
def make_output_paths(mode_name: str) -> dict[str, Path]:
    mode_dir = RUN_DIR / mode_name
    mode_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": mode_dir,
        "csv": mode_dir / f"{mode_name}_results.csv",
        "parquet": mode_dir / f"{mode_name}_results.parquet",
        "log": mode_dir / f"{mode_name}_progress.log",
    }


def flush_results(rows: list[dict], out_csv: Path, out_parquet: Path) -> None:
    df_results = pd.DataFrame(rows)
    df_results.to_csv(out_csv, index=False)
    df_results.to_parquet(out_parquet, index=False)


def append_and_flush(rows: list[dict], row: dict, out_csv: Path, out_parquet: Path) -> None:
    rows.append(row)
    flush_results(rows, out_csv, out_parquet)


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def floor_log2_int(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive.")
    return int(np.floor(np.log2(n)))


def make_train_size_grid(
    n_max: int,
    min_pow: int = MIN_POW,
) -> tuple[list[int], int | None, int | None]:
    if n_max <= 0:
        raise ValueError("n_max must be positive.")

    k_max = floor_log2_int(n_max)

    if min_pow > k_max:
        return [n_max], None, None

    sizes = [2 ** k for k in range(min_pow, k_max + 1)]

    if sizes[-1] != n_max:
        sizes.append(n_max)

    sizes = sorted(set(sizes))
    return sizes, min_pow, k_max


def parquet_num_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def parquet_column_name(path: Path, col_idx: int) -> str:
    schema = pq.ParquetFile(path).schema_arrow
    return schema.names[col_idx]


def count_effective_rows(path: Path, label_col_idx: int, drop_missing_y: bool) -> int:
    if label_col_idx is None or not drop_missing_y:
        return parquet_num_rows(path)

    label_col = parquet_column_name(path, label_col_idx)
    y = pd.read_parquet(path, columns=[label_col]).iloc[:, 0]
    return int(y.notna().sum())


def train_test_sizes_from_metadata(
    dataset_name: str,
    dataset_paths: dict[str, Path | None],
) -> tuple[int, int, dict[str, object]]:
    meta = {
        "dataset": dataset_name,
        "predefined_split": False,
        "train_path": None,
        "test_path": None,
        "single_path": None,
    }

    if dataset_paths["train"] is not None and dataset_paths["test"] is not None:
        meta["predefined_split"] = True
        meta["train_path"] = str(dataset_paths["train"])
        meta["test_path"] = str(dataset_paths["test"])

        n_train = count_effective_rows(
            dataset_paths["train"],
            LABEL_COL_IDX,
            DROP_MISSING_Y,
        )
        n_test = count_effective_rows(
            dataset_paths["test"],
            LABEL_COL_IDX,
            DROP_MISSING_Y,
        )
        return n_train, n_test, meta

    if dataset_paths["single"] is not None:
        meta["single_path"] = str(dataset_paths["single"])
        n_rows = count_effective_rows(
            dataset_paths["single"],
            LABEL_COL_IDX,
            DROP_MISSING_Y,
        )
        n_test = int(np.ceil(0.1 * n_rows))
        n_train = n_rows - n_test
        return n_train, n_test, meta

    raise ValueError(f"Dataset '{dataset_name}' has no usable parquet file.")


def instantiate_fk(
    model_type: str,
    kernel_method: str,
    seed: int,
    model_kwargs: dict[str, object],
) -> ForestProximity:
    kwargs = dict(model_kwargs)

    if model_type in {"rf", "et"}:
        kwargs.setdefault("n_jobs", -1)
    elif model_type == "xgb":
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("device", "cuda")

    kwargs["random_state"] = seed

    if model_type == "rf":
        forest = RandomForestClassifier(**kwargs)
    elif model_type == "et":
        forest = ExtraTreesClassifier(**kwargs)
    elif model_type == "xgb":
        if XGBClassifier is None:
            raise ImportError("model_type='xgb' requires xgboost to be installed.")
        forest = XGBClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported model_type for this script: {model_type!r}")

    weight_scheme = "uniform" if kernel_method == "original" else kernel_method
    return ForestProximity(forest=forest, weight_scheme=weight_scheme)


def run_fk_full_pipeline(
    fk: ForestProximity,
    X_sub,
    y_sub,
    X_test,
    y_test,
    kernel_method: str,
):
    t0 = time.perf_counter()
    fk._fit_forest(X_sub, y_sub)
    forest_fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred_forest = fk.forest_.estimator.predict(X_test)
    forest_pred_time = time.perf_counter() - t0
    forest_acc = accuracy_score(y_test, y_pred_forest)

    t0 = time.perf_counter()
    fk._build_cache()
    cache_time = time.perf_counter() - t0

    if kernel_method == "oob":
        q_time = np.nan
    else:
        t0 = time.perf_counter()
        fk.query_map()
        q_time = time.perf_counter() - t0

    if RUN_FULL_KERNEL:
        t0 = time.perf_counter()
        K_fk = fk.training_proximity(return_dense=False)
        k_time = time.perf_counter() - t0
        k_percent_nnz = kernel_percent_nnz(K_fk)
    else:
        k_time = np.nan
        k_percent_nnz = np.nan

    t0 = time.perf_counter()
    K_test = fk.transform(X_test, return_dense=False)
    y_pred_kp, _ = predict_classifier_from_proximity(K_test, y_sub, fk.classes_)
    kp_time = time.perf_counter() - t0
    kp_acc = accuracy_score(y_test, y_pred_kp)

    return {
        "forest_fit_time_s": forest_fit_time,
        "forest_test_predict_time_s": forest_pred_time,
        "forest_test_acc": forest_acc,
        "cache_build_time_s": cache_time,
        "q_build_time_s": q_time,
        "full_kernel_time_s": k_time,
        "kernel_percent_nnz": k_percent_nnz,
        "kernel_predict_time_s": kp_time,
        "kernel_predict_test_acc": kp_acc,
    }


def _write_subprocess_payload(
    payload_path: Path,
    *,
    dataset_name: str,
    dataset_paths: dict[str, str | None],
    seed: int,
    train_size: int,
    subset_seed: int,
    label_col_idx: int,
    drop_missing_y: bool,
    verbose_dataprep: bool,
    model_type: str,
    kernel_method: str,
    ablation_cfg: dict[str, object],
    run_full_kernel: bool,
):
    payload = {
        "dataset_name": dataset_name,
        "dataset_paths": dataset_paths,
        "seed": seed,
        "train_size": train_size,
        "subset_seed": subset_seed,
        "label_col_idx": label_col_idx,
        "drop_missing_y": drop_missing_y,
        "verbose_dataprep": verbose_dataprep,
        "model_type": model_type,
        "kernel_method": kernel_method,
        "ablation_cfg": ablation_cfg,
        "run_full_kernel": run_full_kernel,
        "project_root": str(PROJECT_ROOT),
    }
    payload_path.write_text(json.dumps(payload), encoding="utf-8")


def _subprocess_worker_code() -> str:
    return r"""
import json
import sys
import time
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

project_root = Path(sys.argv[1])
src_root = project_root / "src"
payload_path = Path(sys.argv[2])

if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from forestgeom import ForestProximity
from dataset import dataprep
from experiments.runtime_utils import (
    MemoryMonitor,
    kernel_percent_nnz,
    predict_classifier_from_proximity,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

payload = json.loads(payload_path.read_text(encoding="utf-8"))

def sample_train_subset_positions(y, train_size, seed):
    if train_size >= len(y):
        return np.arange(len(y))
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=seed,
    )
    idx, _ = next(splitter.split(np.zeros(len(y)), y))
    return idx

def valid_label_positions(path, label_col_idx, drop_missing_y):
    label_name = pq.ParquetFile(path).schema_arrow.names[label_col_idx]
    y = pd.read_parquet(path, columns=[label_name]).iloc[:, 0]
    if drop_missing_y:
        positions = np.flatnonzero(y.notna().to_numpy())
        y = y.iloc[positions]
        return positions, y.to_numpy()
    return np.arange(len(y)), y.to_numpy()

def read_selected_rows(path, positions, batch_size=65536):
    positions = np.asarray(positions, dtype=np.int64)
    if positions.size == 0:
        return pq.read_table(path).slice(0, 0).to_pandas()

    order = np.argsort(positions)
    sorted_positions = positions[order]
    tables = []
    offset = 0

    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        batch_len = batch.num_rows
        start = np.searchsorted(sorted_positions, offset, side="left")
        end = np.searchsorted(sorted_positions, offset + batch_len, side="left")

        if start != end:
            local_positions = sorted_positions[start:end] - offset
            tables.append(batch.take(pa.array(local_positions)))

        offset += batch_len

    if not tables:
        return pq.read_table(path).slice(0, 0).to_pandas()

    table = pa.Table.from_batches(tables)
    df = table.to_pandas().reset_index(drop=True)

    if len(order) == len(df):
        inverse_order = np.argsort(order)
        df = df.iloc[inverse_order].reset_index(drop=True)

    return df

def load_subset_dataset_pair(paths, seed, train_size, subset_seed, label_col_idx, drop_missing_y, verbose_dataprep):
    if paths["train"] is not None and paths["test"] is not None:
        train_positions, y_train_labels = valid_label_positions(
            paths["train"],
            label_col_idx,
            drop_missing_y,
        )
        subset_local = sample_train_subset_positions(
            y_train_labels,
            train_size=train_size,
            seed=subset_seed,
        )
        train_subset_positions = train_positions[subset_local]

        test_positions, _ = valid_label_positions(
            paths["test"],
            label_col_idx,
            drop_missing_y,
        )

        df_train = read_selected_rows(paths["train"], train_subset_positions)
        del y_train_labels, train_positions, subset_local, train_subset_positions
        gc.collect()

        df_test = read_selected_rows(paths["test"], test_positions)
        del test_positions
        gc.collect()

    elif paths["single"] is not None:
        positions, labels = valid_label_positions(
            paths["single"],
            label_col_idx,
            drop_missing_y,
        )
        train_local, test_local = train_test_split(
            np.arange(len(labels)),
            test_size=0.1,
            random_state=seed,
            stratify=labels,
        )
        subset_local = sample_train_subset_positions(
            labels[train_local],
            train_size=train_size,
            seed=subset_seed,
        )
        train_positions = positions[train_local[subset_local]]
        test_positions = positions[test_local]

        selected_positions = np.concatenate([train_positions, test_positions])
        df_selected = read_selected_rows(paths["single"], selected_positions)
        df_train = df_selected.iloc[:len(train_positions)].reset_index(drop=True)
        df_test = df_selected.iloc[len(train_positions):].reset_index(drop=True)
        del df_selected, labels, positions, train_local, test_local, subset_local
        del train_positions, test_positions
        gc.collect()

    else:
        raise ValueError("No usable parquet file was provided.")

    n_train = len(df_train)
    n_test = len(df_test)
    df_full = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    del df_train, df_test
    gc.collect()

    X_full, y_full = dataprep(
        df_full,
        label_col_idx=label_col_idx,
        scale=None,
        global_transform=False,
        drop_missing_y=drop_missing_y,
        verbose=verbose_dataprep,
    )
    del df_full
    gc.collect()

    X_full = np.asarray(X_full)
    y_full = np.asarray(y_full).reshape(-1)

    X_sub = X_full[:n_train]
    X_test = X_full[n_train:n_train + n_test]
    y_sub = y_full[:n_train]
    y_test = y_full[n_train:n_train + n_test]

    return X_sub, X_test, y_sub, y_test

def instantiate_fk(model_type, kernel_method, seed, model_kwargs):
    kwargs = dict(model_kwargs)
    if model_type in {"rf", "et"}:
        kwargs.setdefault("n_jobs", -1)
    elif model_type == "xgb":
        kwargs.setdefault("n_jobs", -1)
        kwargs.setdefault("device", "cuda")
    kwargs["random_state"] = seed
    if model_type == "rf":
        forest = RandomForestClassifier(**kwargs)
    elif model_type == "et":
        forest = ExtraTreesClassifier(**kwargs)
    elif model_type == "xgb":
        if XGBClassifier is None:
            raise ImportError("model_type='xgb' requires xgboost to be installed.")
        forest = XGBClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported model_type for this script: {model_type!r}")
    weight_scheme = "uniform" if kernel_method == "original" else kernel_method
    return ForestProximity(forest=forest, weight_scheme=weight_scheme)

paths = payload["dataset_paths"]
X_sub, X_test, y_sub, y_test = load_subset_dataset_pair(
    paths=paths,
    seed=payload["seed"],
    train_size=payload["train_size"],
    subset_seed=payload["subset_seed"],
    label_col_idx=payload["label_col_idx"],
    drop_missing_y=payload["drop_missing_y"],
    verbose_dataprep=payload["verbose_dataprep"],
)
gc.collect()

fk = instantiate_fk(
    model_type=payload["model_type"],
    kernel_method=payload["kernel_method"],
    seed=payload["seed"],
    model_kwargs=payload["ablation_cfg"],
)

run_full_kernel = payload["run_full_kernel"]
kernel_method = payload["kernel_method"]

# -------------------------------------------------
# Unmeasured: forest fit
# -------------------------------------------------
t0 = time.perf_counter()
fk._fit_forest(X_sub, y_sub)
forest_fit_time = time.perf_counter() - t0

# Optional forest prediction time/acc, also unmeasured for memory
t0 = time.perf_counter()
y_pred_forest = fk.forest_.estimator.predict(X_test)
forest_pred_time = time.perf_counter() - t0
forest_acc = accuracy_score(y_test, y_pred_forest)

gc.collect()

# -------------------------------------------------
# Measured: cache + query map + full kernel only
# -------------------------------------------------
with MemoryMonitor(poll_seconds=0.005) as mm:
    t0 = time.perf_counter()
    fk._build_cache()
    cache_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    if run_full_kernel and kernel_method != "oob":
        fk.query_map()
    q_time = time.perf_counter() - t0

    if run_full_kernel:
        t0 = time.perf_counter()
        K_fk = fk.training_proximity(return_dense=False)
        k_time = time.perf_counter() - t0
        k_percent_nnz = kernel_percent_nnz(K_fk)
    else:
        k_time = float("nan")
        k_percent_nnz = float("nan")

kernel_build_peak_mb = mm.peak_delta_mb

# -------------------------------------------------
# Unmeasured: proximity-weighted prediction
# -------------------------------------------------
t0 = time.perf_counter()
K_test = fk.transform(X_test, return_dense=False)
y_pred_kp, _ = predict_classifier_from_proximity(K_test, y_sub, fk.classes_)
kp_time = time.perf_counter() - t0
kp_acc = accuracy_score(y_test, y_pred_kp)

result = {
    "forest_fit_time_s": forest_fit_time,
    "forest_test_predict_time_s": forest_pred_time,
    "forest_test_acc": forest_acc,
    "cache_build_time_s": cache_time,
    "q_build_time_s": q_time,
    "full_kernel_time_s": k_time,
    "kernel_percent_nnz": k_percent_nnz,
    "kernel_predict_time_s": kp_time,
    "kernel_predict_test_acc": kp_acc,
    "kernel_build_peak_mb": kernel_build_peak_mb,
    "status": "ok",
    "error": "",
}
print(json.dumps(result))
"""


def run_fk_full_pipeline_subprocess(
    *,
    dataset_name: str,
    dataset_paths: dict[str, Path | None],
    seed: int,
    train_size: int,
    subset_seed: int,
    model_type: str,
    kernel_method: str,
    ablation_cfg: dict[str, object],
):
    serializable_paths = {
        "train": str(dataset_paths["train"]) if dataset_paths["train"] is not None else None,
        "test": str(dataset_paths["test"]) if dataset_paths["test"] is not None else None,
        "single": str(dataset_paths["single"]) if dataset_paths["single"] is not None else None,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        payload_path = tmpdir / "payload.json"
        worker_path = tmpdir / "worker.py"

        _write_subprocess_payload(
            payload_path,
            dataset_name=dataset_name,
            dataset_paths=serializable_paths,
            seed=seed,
            train_size=train_size,
            subset_seed=subset_seed,
            label_col_idx=LABEL_COL_IDX,
            drop_missing_y=DROP_MISSING_Y,
            verbose_dataprep=VERBOSE_DATAPREP,
            model_type=model_type,
            kernel_method=kernel_method,
            ablation_cfg=ablation_cfg,
            run_full_kernel=RUN_FULL_KERNEL,
        )
        worker_path.write_text(_subprocess_worker_code(), encoding="utf-8")

        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(worker_path), str(PROJECT_ROOT), str(payload_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        wall_time = time.perf_counter() - t0

        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip() or f"Subprocess failed with code {proc.returncode}"
            return None, wall_time, np.nan, "failed", err

        try:
            result = json.loads(proc.stdout.strip())
        except Exception as e:
            return None, wall_time, np.nan, "failed", f"Failed to parse subprocess output: {e}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

        return result, wall_time, result.get("kernel_build_peak_mb", np.nan), result.get("status", "ok"), result.get("error", "")


# ---------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------
def run_one_ablation_mode(
    mode_name: str,
    settings: list[dict[str, object]],
    dataset_groups: dict[str, dict[str, Path | None]],
) -> None:
    paths = make_output_paths(mode_name)
    rows: list[dict] = []

    log_progress(f"Run ID: {RUN_ID}", paths["log"])
    log_progress(f"Mode: {mode_name}", paths["log"])
    log_progress(f"Run directory: {paths['dir']}", paths["log"])
    log_progress(f"Resolved datasets: {sorted(dataset_groups.keys())}", paths["log"])
    log_progress(f"MIN_POW: {MIN_POW}", paths["log"])
    log_progress("Grid type: powers_of_two_plus_full_size", paths["log"])
    log_progress(f"Seeds: {SEEDS}", paths["log"])
    log_progress("Scale: None", paths["log"])
    log_progress("Global transform: False", paths["log"])
    log_progress(f"Number of settings: {len(settings)}", paths["log"])
    log_progress(f"CSV output: {paths['csv']}", paths["log"])
    log_progress(f"Parquet output: {paths['parquet']}", paths["log"])
    log_progress(f"Progress log: {paths['log']}", paths["log"])
    log_progress("Memory measurement mode: fresh subprocess peak RSS per run", paths["log"])

    for setting in settings:
        log_progress(
            f"Setting prepared | model_type={setting['model_type']} | "
            f"kernel_method={setting['kernel_method']} | "
            f"name={setting['ablation_name']} | cfg={setting['ablation_cfg']}",
            paths["log"],
        )

    for dataset_name, dataset_paths in dataset_groups.items():
        log_progress(f"=== DATASET: {dataset_name} ===", paths["log"])

        try:
            available_train_size, test_size, meta = train_test_sizes_from_metadata(
                dataset_name,
                dataset_paths,
            )
        except Exception as e:
            log_progress(
                f"Failed to inspect dataset {dataset_name} for grid construction: {e}",
                paths["log"],
            )
            continue

        train_sizes, k_min, k_max = make_train_size_grid(
            n_max=available_train_size,
            min_pow=MIN_POW,
        )

        log_progress(
            f"Inspected {dataset_name}: train_pool_rows={available_train_size}, "
            f"test_rows={test_size}, predefined_split={meta['predefined_split']}, "
            f"available_train_size={available_train_size}",
            paths["log"],
        )
        log_progress(f"Dataset-specific k_min: {k_min}", paths["log"])
        log_progress(f"Dataset-specific k_max: {k_max}", paths["log"])
        log_progress(f"Train sizes: {train_sizes}", paths["log"])

        for seed in SEEDS:
            log_progress(f">>> SEED: {seed}", paths["log"])

            for size_id, train_size in enumerate(train_sizes, start=1):
                log_progress(
                    f"--- size {size_id}/{len(train_sizes)} | "
                    f"train_size={train_size} | seed={seed} ---",
                    paths["log"],
                )

                subset_seed = seed + size_id

                for ablation_id, setting in enumerate(settings, start=1):
                    model_type = setting["model_type"]
                    kernel_method = setting["kernel_method"]
                    ablation_name = setting["ablation_name"]
                    ablation_cfg = setting["ablation_cfg"]

                    log_progress(
                        f"Ablation {ablation_id}/{len(settings)} | "
                        f"model_type={model_type} | kernel_method={kernel_method} | "
                        f"{ablation_name}",
                        paths["log"],
                    )

                    pipeline_out, pipeline_wall_time, kernel_build_peak_mb, pipeline_status, pipeline_error = run_fk_full_pipeline_subprocess(
                        dataset_name=dataset_name,
                        dataset_paths=dataset_paths,
                        seed=seed,
                        train_size=train_size,
                        subset_seed=subset_seed,
                        model_type=model_type,
                        kernel_method=kernel_method,
                        ablation_cfg=ablation_cfg,
                    )

                    if pipeline_status == "ok":
                        forest_fit_time = pipeline_out["forest_fit_time_s"]
                        forest_pred_time = pipeline_out["forest_test_predict_time_s"]
                        forest_acc = pipeline_out["forest_test_acc"]
                        cache_time = pipeline_out["cache_build_time_s"]
                        q_time = pipeline_out["q_build_time_s"]
                        k_time = pipeline_out["full_kernel_time_s"]
                        k_percent_nnz = pipeline_out["kernel_percent_nnz"]
                        kp_time = pipeline_out["kernel_predict_time_s"]
                        kp_acc = pipeline_out["kernel_predict_test_acc"]
                    else:
                        forest_fit_time = np.nan
                        forest_pred_time = np.nan
                        forest_acc = np.nan
                        cache_time = np.nan
                        q_time = np.nan
                        k_time = np.nan
                        k_percent_nnz = np.nan
                        kp_time = np.nan
                        kp_acc = np.nan

                    row = {
                        "run_id": RUN_ID,
                        "ablation_mode": mode_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "predefined_split": meta["predefined_split"],
                        "scale": None,
                        "global_transform": False,
                        "model_type": model_type,
                        "kernel_method": kernel_method,
                        "ablation_id": ablation_id,
                        "ablation_name": ablation_name,
                        "ablation_cfg": str(ablation_cfg),
                        "available_train_size": available_train_size,
                        "min_pow": MIN_POW,
                        "dataset_k_min": k_min,
                        "dataset_k_max": k_max,
                        "size_id": size_id,
                        "requested_train_size": train_size,
                        "is_power_of_two_size": is_power_of_two(train_size),
                        "log2_requested_train_size": np.log2(train_size),
                        "n_train_subset": train_size,
                        "n_test": test_size,
                        "forest_fit_time_s": forest_fit_time,
                        "forest_test_predict_time_s": forest_pred_time,
                        "forest_test_acc": forest_acc,
                        "cache_build_time_s": cache_time,
                        "q_build_time_s": q_time,
                        "full_kernel_time_s": k_time,
                        "kernel_percent_nnz": k_percent_nnz,
                        "kernel_predict_time_s": kp_time,
                        "kernel_predict_test_acc": kp_acc,
                        "kernel_build_peak_mb": kernel_build_peak_mb,
                        "pipeline_wall_time_s": pipeline_wall_time,
                        "status": pipeline_status,
                        "error": pipeline_error,
                    }
                    append_and_flush(rows, row, paths["csv"], paths["parquet"])

                    log_progress(
                        f"Done | dataset={dataset_name} | seed={seed} | "
                        f"train_size={train_size} | n_train={train_size} | "
                        f"model_type={model_type} | kernel_method={kernel_method} | "
                        f"ablation={ablation_name} | "
                        f"fit={forest_fit_time if not np.isnan(forest_fit_time) else 'nan'}s | "
                        f"cache={cache_time if not np.isnan(cache_time) else 'nan'}s | "
                        f"q={q_time if not np.isnan(q_time) else 'nan'}s | "
                        f"kernel={k_time if not np.isnan(k_time) else 'nan'}s | "
                        f"kp={kp_time if not np.isnan(kp_time) else 'nan'}s | "
                        f"kernel_build_peak_mb={kernel_build_peak_mb if not np.isnan(kernel_build_peak_mb) else 'nan'} | "
                        f"%nnz={k_percent_nnz if not np.isnan(k_percent_nnz) else 'nan'} | "
                        f"forest_acc={forest_acc if not np.isnan(forest_acc) else 'nan'} | "
                        f"kp_acc={kp_acc if not np.isnan(kp_acc) else 'nan'} | "
                        f"status={pipeline_status}",
                        paths["log"],
                    )

    flush_results(rows, paths["csv"], paths["parquet"])
    log_progress(f"Saved results to: {paths['csv']}", paths["log"])
    log_progress(f"Saved results to: {paths['parquet']}", paths["log"])
    log_progress("Done.", paths["log"])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    dataset_ablation_groups = resolve_dataset_paths_from_base_names(
        DATA_DIR,
        DATASET_ABLATION_DATASET_NAMES,
    )
    fixed_ablation_groups = resolve_dataset_paths_from_base_names(
        DATA_DIR,
        FIXED_ABLATION_DATASET_NAMES,
    )

    if RUN_DATASET_ABLATION:
        run_one_ablation_mode(
            mode_name="dataset",
            settings=DATASET_ABLATION_SETTINGS,
            dataset_groups=dataset_ablation_groups,
        )

    if RUN_KERNEL_METHOD_ABLATION:
        run_one_ablation_mode(
            mode_name="kernel_method",
            settings=KERNEL_METHOD_SETTINGS,
            dataset_groups=fixed_ablation_groups,
        )

    if RUN_MODEL_TYPE_ABLATION:
        run_one_ablation_mode(
            mode_name="model_type",
            settings=MODEL_TYPE_SETTINGS,
            dataset_groups=fixed_ablation_groups,
        )

    if RUN_MAX_DEPTH_ABLATION:
        run_one_ablation_mode(
            mode_name="max_depth",
            settings=MAX_DEPTH_SETTINGS,
            dataset_groups=fixed_ablation_groups,
        )

    if RUN_MIN_SAMPLES_LEAF_ABLATION:
        run_one_ablation_mode(
            mode_name="min_samples_leaf",
            settings=MIN_SAMPLES_LEAF_SETTINGS,
            dataset_groups=fixed_ablation_groups,
        )


if __name__ == "__main__":
    main()
