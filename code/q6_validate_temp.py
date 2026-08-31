from __future__ import annotations

import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit


def benchmark(cases: dict[str, callable], *, repeat: int = 5, number: int = 1, warmup: int = 1) -> pd.DataFrame:
    records = []
    for name, func in cases.items():
        for _ in range(warmup):
            func()
        times = np.asarray(timeit.repeat(stmt=func, repeat=repeat, number=number), dtype=np.float64) / number
        records.append(
            {
                "name": name,
                "median_ms": np.median(times) * 1000,
                "min_ms": times.min() * 1000,
                "max_ms": times.max() * 1000,
            }
        )
    return pd.DataFrame(records).sort_values("median_ms").reset_index(drop=True)


def dataframe_memory_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(index=True, deep=True).sum() / 1024**2


DATA_DIR = Path("data")
TAXI_PARQUET_PATH = DATA_DIR / "yellow_tripdata_2025-12.parquet"
BENCH_ROWS = 100000

bench_cols = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
]

bench_df = next(
    pq.ParquetFile(TAXI_PARQUET_PATH).iter_batches(
        batch_size=BENCH_ROWS,
        columns=bench_cols,
    )
).to_pandas()

bench_df = (
    bench_df
    .assign(
        trip_minutes=lambda x: (
            x["tpep_dropoff_datetime"] - x["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60,
        passenger_count_filled=lambda x: x["passenger_count"].fillna(1.0).clip(lower=1),
    )
    .query(
        "trip_minutes > 0 and trip_distance > 0 "
        "and total_amount > 0 and fare_amount > 0"
    )
    .copy()
)


def python_loop_efficiency_score() -> list[float]:
    out = []
    total_amounts = bench_df["total_amount"].to_list()
    trip_minutes = bench_df["trip_minutes"].to_list()
    trip_distance = bench_df["trip_distance"].to_list()
    fare_amounts = bench_df["fare_amount"].to_list()
    tip_amounts = bench_df["tip_amount"].to_list()
    payment_types = bench_df["payment_type"].to_list()
    passenger_counts = bench_df["passenger_count_filled"].to_list()

    for total_amount, minutes, distance, fare, tip, payment_type, passenger_count in zip(
        total_amounts,
        trip_minutes,
        trip_distance,
        fare_amounts,
        tip_amounts,
        payment_types,
        passenger_counts,
    ):
        speed = distance / minutes
        spend_rate = total_amount / minutes
        score = (
            0.20 * speed
            + 0.04 * spend_rate
            + 0.03 * fare
            + 0.02 * passenger_count
            - 0.015 * minutes
        )

        for k in range(5):
            mix = (
                speed * (1.03 + 0.02 * k)
                + spend_rate * (0.70 + 0.03 * k)
                + passenger_count * (0.24 + 0.04 * k)
            )

            if payment_type == 1:
                score += 0.10 * fare + 0.05 * mix + 0.01 * tip
            elif payment_type == 2:
                score += 0.08 * total_amount + 0.04 * mix - 0.12
            else:
                score += 0.06 * fare + 0.05 * mix - 0.02 * payment_type

            if score > 11.0 + 0.9 * k:
                score = score * 0.85 + total_amount * (0.025 + 0.003 * k)
            else:
                score = score * 1.04 - minutes * (0.010 + 0.002 * k)

            if tip > fare * (0.18 + 0.012 * k):
                score += 0.16 * (k + 1)
            else:
                score -= 0.06 * (k + 1)

            if mix > total_amount * (0.30 + 0.015 * k):
                score += 0.020 * mix - 0.010 * passenger_count
            else:
                score = score - 0.015 * mix + 0.008 * speed

        out.append(score)

    return out


def pandas_apply_efficiency_score() -> pd.Series:
    def score_row(row: pd.Series) -> float:
        speed = row["trip_distance"] / row["trip_minutes"]
        spend_rate = row["total_amount"] / row["trip_minutes"]
        score = (
            0.20 * speed
            + 0.04 * spend_rate
            + 0.03 * row["fare_amount"]
            + 0.02 * row["passenger_count_filled"]
            - 0.015 * row["trip_minutes"]
        )

        for k in range(5):
            mix = (
                speed * (1.03 + 0.02 * k)
                + spend_rate * (0.70 + 0.03 * k)
                + row["passenger_count_filled"] * (0.24 + 0.04 * k)
            )

            if row["payment_type"] == 1:
                score += 0.10 * row["fare_amount"] + 0.05 * mix + 0.01 * row["tip_amount"]
            elif row["payment_type"] == 2:
                score += 0.08 * row["total_amount"] + 0.04 * mix - 0.12
            else:
                score += 0.06 * row["fare_amount"] + 0.05 * mix - 0.02 * row["payment_type"]

            if score > 11.0 + 0.9 * k:
                score = score * 0.85 + row["total_amount"] * (0.025 + 0.003 * k)
            else:
                score = score * 1.04 - row["trip_minutes"] * (0.010 + 0.002 * k)

            if row["tip_amount"] > row["fare_amount"] * (0.18 + 0.012 * k):
                score += 0.16 * (k + 1)
            else:
                score -= 0.06 * (k + 1)

            if mix > row["total_amount"] * (0.30 + 0.015 * k):
                score += 0.020 * mix - 0.010 * row["passenger_count_filled"]
            else:
                score = score - 0.015 * mix + 0.008 * speed

        return score

    return bench_df.apply(score_row, axis=1)


def vectorized_efficiency_score() -> np.ndarray:
    speed = bench_df["trip_distance"] / bench_df["trip_minutes"]
    spend_rate = bench_df["total_amount"] / bench_df["trip_minutes"]
    fare = bench_df["fare_amount"]
    tip = bench_df["tip_amount"]
    total = bench_df["total_amount"]
    minutes = bench_df["trip_minutes"]
    payment = bench_df["payment_type"]
    passenger = bench_df["passenger_count_filled"]

    score = (
        0.20 * speed
        + 0.04 * spend_rate
        + 0.03 * fare
        + 0.02 * passenger
        - 0.015 * minutes
    )

    for k in range(5):
        mix = (
            speed * (1.03 + 0.02 * k)
            + spend_rate * (0.70 + 0.03 * k)
            + passenger * (0.24 + 0.04 * k)
        )

        score = score + np.where(
            payment == 1,
            0.10 * fare + 0.05 * mix + 0.01 * tip,
            np.where(
                payment == 2,
                0.08 * total + 0.04 * mix - 0.12,
                0.06 * fare + 0.05 * mix - 0.02 * payment,
            ),
        )

        score = np.where(
            score > 11.0 + 0.9 * k,
            score * 0.85 + total * (0.025 + 0.003 * k),
            score * 1.04 - minutes * (0.010 + 0.002 * k),
        )

        score = np.where(
            tip > fare * (0.18 + 0.012 * k),
            score + 0.16 * (k + 1),
            score - 0.06 * (k + 1),
        )

        score = np.where(
            mix > total * (0.30 + 0.015 * k),
            score + 0.020 * mix - 0.010 * passenger,
            score - 0.015 * mix + 0.008 * speed,
        )

    return np.asarray(score, dtype=np.float64)


@njit
def numba_efficiency_score(
    total_amounts: np.ndarray,
    trip_minutes: np.ndarray,
    trip_distance: np.ndarray,
    fare_amounts: np.ndarray,
    tip_amounts: np.ndarray,
    payment_types: np.ndarray,
    passenger_counts: np.ndarray,
) -> np.ndarray:
    out = np.empty(total_amounts.shape[0], dtype=np.float64)

    for i in range(total_amounts.shape[0]):
        total_amount = total_amounts[i]
        minutes = trip_minutes[i]
        distance = trip_distance[i]
        fare = fare_amounts[i]
        tip = tip_amounts[i]
        payment_type = payment_types[i]
        passenger_count = passenger_counts[i]

        speed = distance / minutes
        spend_rate = total_amount / minutes
        score = (
            0.20 * speed
            + 0.04 * spend_rate
            + 0.03 * fare
            + 0.02 * passenger_count
            - 0.015 * minutes
        )

        for k in range(5):
            mix = (
                speed * (1.03 + 0.02 * k)
                + spend_rate * (0.70 + 0.03 * k)
                + passenger_count * (0.24 + 0.04 * k)
            )

            if payment_type == 1:
                score += 0.10 * fare + 0.05 * mix + 0.01 * tip
            elif payment_type == 2:
                score += 0.08 * total_amount + 0.04 * mix - 0.12
            else:
                score += 0.06 * fare + 0.05 * mix - 0.02 * payment_type

            if score > 11.0 + 0.9 * k:
                score = score * 0.85 + total_amount * (0.025 + 0.003 * k)
            else:
                score = score * 1.04 - minutes * (0.010 + 0.002 * k)

            if tip > fare * (0.18 + 0.012 * k):
                score += 0.16 * (k + 1)
            else:
                score -= 0.06 * (k + 1)

            if mix > total_amount * (0.30 + 0.015 * k):
                score += 0.020 * mix - 0.010 * passenger_count
            else:
                score = score - 0.015 * mix + 0.008 * speed

        out[i] = score

    return out


def numba_case() -> np.ndarray:
    return numba_efficiency_score(
        bench_df["total_amount"].to_numpy(dtype=np.float64, copy=False),
        bench_df["trip_minutes"].to_numpy(dtype=np.float64, copy=False),
        bench_df["trip_distance"].to_numpy(dtype=np.float64, copy=False),
        bench_df["fare_amount"].to_numpy(dtype=np.float64, copy=False),
        bench_df["tip_amount"].to_numpy(dtype=np.float64, copy=False),
        bench_df["payment_type"].to_numpy(dtype=np.int64, copy=False),
        bench_df["passenger_count_filled"].to_numpy(dtype=np.float64, copy=False),
    )


baseline = np.asarray(python_loop_efficiency_score(), dtype=np.float64)
apply_result = np.asarray(pandas_apply_efficiency_score(), dtype=np.float64)
vectorized_result = np.asarray(vectorized_efficiency_score(), dtype=np.float64)
numba_result = numba_case()

print("rows:", len(bench_df))
print("shapes:", baseline.shape, apply_result.shape, vectorized_result.shape, numba_result.shape)
print("equal apply:", np.allclose(baseline, apply_result, equal_nan=True))
print("equal vectorized:", np.allclose(baseline, vectorized_result, equal_nan=True))
print("equal numba:", np.allclose(baseline, numba_result, equal_nan=True))

optimized_df = bench_df.copy()
optimized_df["passenger_count_filled"] = optimized_df["passenger_count_filled"].astype(np.float32)
optimized_df["trip_distance"] = optimized_df["trip_distance"].astype(np.float32)
optimized_df["fare_amount"] = optimized_df["fare_amount"].astype(np.float32)
optimized_df["tip_amount"] = optimized_df["tip_amount"].astype(np.float32)
optimized_df["total_amount"] = optimized_df["total_amount"].astype(np.float32)
optimized_df["trip_minutes"] = optimized_df["trip_minutes"].astype(np.float32)
optimized_df["payment_type"] = optimized_df["payment_type"].astype(np.int8)

print("memory before MB:", dataframe_memory_mb(bench_df))
print("memory after MB:", dataframe_memory_mb(optimized_df))

perf_result = benchmark(
    {
        "python_loop": python_loop_efficiency_score,
        "pandas_apply": pandas_apply_efficiency_score,
        "vectorized": vectorized_efficiency_score,
        "numba": numba_case,
    },
    repeat=5,
    number=1,
    warmup=1,
)
print(perf_result.to_string(index=False))
