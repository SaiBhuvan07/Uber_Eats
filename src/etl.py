
import pandas as pd
from pathlib import Path

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# 1. EXTRACT
customers = pd.read_csv(RAW_DIR / "customers.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")
deliveries = pd.read_csv(RAW_DIR / "deliveries.csv")
reviews = pd.read_csv(RAW_DIR / "reviews.csv")

datasets = {
    "customers": customers,
    "orders": orders,
    "deliveries": deliveries,
    "reviews": reviews
}

# 2. INSPECT
for name, df in datasets.items():
    print(f"\n{'=' * 40}")
    print(f"DATASET: {name.upper()}")
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

# 3. TRANSFORM: correct data types
customers["signup_date"] = pd.to_datetime(
    customers["signup_date"], errors="coerce"
)

orders["order_time"] = pd.to_datetime(
    orders["order_time"], errors="coerce"
)

deliveries["pickup_time"] = pd.to_datetime(
    deliveries["pickup_time"], errors="coerce"
)

deliveries["delivery_time"] = pd.to_datetime(
    deliveries["delivery_time"], errors="coerce"
)

# Recalculate delivery duration from timestamps
calculated_duration = (
    deliveries["delivery_time"] - deliveries["pickup_time"]
).dt.total_seconds() / 60

# Flag invalid durations before correcting anything
invalid_duration = (
    calculated_duration.isna()
    | (calculated_duration <= 0)
)

print("\nInvalid delivery durations:", invalid_duration.sum())

# Use the timestamp-derived duration as the authoritative value
deliveries["delivery_duration"] = calculated_duration

# Validate rating values
print(
    "Invalid ratings:",
    (~reviews["rating"].between(1, 5)).sum()
)

# 4. LOAD: save processed datasets
datasets = {
    "customers": customers,
    "orders": orders,
    "deliveries": deliveries,
    "reviews": reviews
}

for name, df in datasets.items():
    df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

print("\nETL completed. Processed files saved.")
