"""Benchmark uniform, GAP, and OOB proximity construction on synthetic data.

The parent process launches a fresh child process for each scheme/split
combination, so memory peak measurements are comparable across runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from forestgeom import ForestProximity
from forestgeom.adapters import make_adapter


@dataclass
class BenchmarkResult:
    scheme: str
    split: str
    elapsed_s: float
    peak_rss_mb: float
    peak_delta_mb: float
    shape: tuple[int, int]
    nnz: int


class PeakRssSampler:
    def __init__(self, interval_s: float = 0.02):
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.start_rss = 0
        self.peak_rss = 0

    @staticmethod
    def _current_rss_bytes() -> int:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(rss)
        return int(rss) * 1024

    def __enter__(self):
        self.start_rss = self._current_rss_bytes()
        self.peak_rss = self.start_rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            rss = self._current_rss_bytes()
            if rss > self.peak_rss:
                self.peak_rss = rss
            time.sleep(self.interval_s)

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_mb(self) -> float:
        return self.peak_rss / (1024**2)

    @property
    def delta_mb(self) -> float:
        return (self.peak_rss - self.start_rss) / (1024**2)


def build_synthetic_split(seed: int, train_size: int, test_size: int, n_features: int):
    n_samples = train_size + test_size
    n_informative = max(2, n_features // 2)
    n_redundant = max(0, n_features // 4)

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_repeated=0,
        n_classes=10,
        n_clusters_per_class=2,
        weights=None,
        flip_y=0.01,
        class_sep=1.0,
        random_state=seed,
    )

    return train_test_split(
        X,
        y,
        train_size=train_size,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )


def build_forest(seed: int, n_estimators: int, max_depth: int | None, n_jobs: int):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        bootstrap=True,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=seed,
    )


def build_proximity_model(forest, scheme: str, X_train, y_train):
    fp = ForestProximity(forest=forest, weight_scheme=scheme)
    fp.forest_ = make_adapter(forest, weight_scheme=scheme)
    fp.X_fit_ = X_train
    fp.y_ = y_train
    fp.classes_ = None
    fp.cache_ = None
    fp._build_cache()
    return fp


def benchmark_once(
    scheme: str,
    split: str,
    seed: int,
    train_size: int,
    test_size: int,
    n_features: int,
    n_estimators: int,
    max_depth: int | None,
    n_jobs: int,
):
    X_train, X_test, y_train, y_test = build_synthetic_split(
        seed=seed,
        train_size=train_size,
        test_size=test_size,
        n_features=n_features,
    )

    forest = build_forest(
        seed=seed,
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=n_jobs,
    )
    forest.fit(X_train, y_train)

    fp = build_proximity_model(forest, scheme, X_train, y_train)

    if split == "train":
        fn = lambda: fp.training_proximity(return_dense=False)
    elif split == "test":
        fn = lambda: fp.transform(X_test, return_dense=False)
    else:
        raise ValueError("split must be 'train' or 'test'.")

    with PeakRssSampler() as sampler:
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start

    out = BenchmarkResult(
        scheme=scheme,
        split=split,
        elapsed_s=elapsed,
        peak_rss_mb=sampler.peak_mb,
        peak_delta_mb=sampler.delta_mb,
        shape=result.shape,
        nnz=result.nnz,
    )

    del fp, result, forest, X_train, X_test, y_train, y_test
    gc.collect()
    return out


def format_rows(rows):
    headers = ["scheme", "split", "elapsed_s", "peak_rss_mb", "peak_delta_mb", "shape", "nnz"]
    values = [
        [
            row.scheme,
            row.split,
            f"{row.elapsed_s:.3f}",
            f"{row.peak_rss_mb:.1f}",
            f"{row.peak_delta_mb:.1f}",
            f"{row.shape[0]}x{row.shape[1]}",
            str(row.nnz),
        ]
        for row in rows
    ]

    widths = [max(len(h), max(len(v[i]) for v in values)) for i, h in enumerate(headers)]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = ["  ".join(v[i].ljust(widths[i]) for i in range(len(headers))) for v in values]
    return "\n".join([line, sep, *body])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=50_000)
    parser.add_argument("--test-size", type=int, default=20_000)
    parser.add_argument("--n-features", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--scheme", choices=("uniform", "gap", "oob"))
    parser.add_argument("--split", choices=("train", "test"))
    return parser.parse_args()


def main():
    args = parse_args()

    if args.scheme and args.split:
        result = benchmark_once(
            scheme=args.scheme,
            split=args.split,
            seed=args.seed,
            train_size=args.train_size,
            test_size=args.test_size,
            n_features=args.n_features,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            n_jobs=args.n_jobs,
        )
        print(json.dumps(result.__dict__))
        return

    rows = []
    for scheme in ("uniform", "gap", "oob"):
        for split in ("train", "test"):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--scheme",
                scheme,
                "--split",
                split,
                "--seed",
                str(args.seed),
                "--train-size",
                str(args.train_size),
                "--test-size",
                str(args.test_size),
                "--n-features",
                str(args.n_features),
                "--n-estimators",
                str(args.n_estimators),
                "--max-depth",
                str(args.max_depth),
                "--n-jobs",
                str(args.n_jobs),
            ]

            completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
            payload = json.loads(completed.stdout.strip())
            rows.append(
                BenchmarkResult(
                    scheme=payload["scheme"],
                    split=payload["split"],
                    elapsed_s=float(payload["elapsed_s"]),
                    peak_rss_mb=float(payload["peak_rss_mb"]),
                    peak_delta_mb=float(payload["peak_delta_mb"]),
                    shape=tuple(payload["shape"]),
                    nnz=int(payload["nnz"]),
                )
            )

    print(format_rows(rows))


if __name__ == "__main__":
    main()