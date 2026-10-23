
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Make the results reproducible
rng = np.random.default_rng(42)

# 2. Define the project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# 3. Generate synthetic customers
n_customers = 1000

customers = pd.DataFrame({
    "customer_id": range(1, n_customers + 1),
    "signup_date": rng.choice(
        pd.date_range("2024-01-01", "2026-09-30"),
        size=n_customers
    ),
    "city": rng.choice(
        ["Chennai", "Bengaluru", "Hyderabad", "Mumbai", "Pune"],
        size=n_customers
    ),
    "customer_segment": rng.choice(
        ["New", "Regular", "Loyal"],
        size=n_customers,
        p=[0.25, 0.50, 0.25]
    )
})

# 4. Save the dataset
customers.to_csv(RAW_DIR / "customers.csv", index=False)

# 5. Inspect the result
print(customers.head())
print("\nDataset shape:", customers.shape)
print("\nMissing values:\n", customers.isnull().sum())
print("\nSegment counts:\n", customers["customer_segment"].value_counts())


# 6. Generate synthetic orders
n_orders = 5000

orders = pd.DataFrame({
    "order_id": range(10001, 10001 + n_orders),
    "customer_id": rng.integers(1, n_customers + 1, size=n_orders),
    "restaurant_id": rng.integers(101, 151, size=n_orders),
    "order_time": (
        pd.Timestamp("2025-01-01")
        + pd.to_timedelta(
            rng.integers(0, 365 * 24 * 60, size=n_orders),
            unit="min"
        )
    ),
    "order_value": np.round(
        rng.uniform(100, 1500, size=n_orders), 2
    ),
    "status": rng.choice(
        ["Completed", "Cancelled", "In Progress"],
        size=n_orders,
        p=[0.82, 0.12, 0.06]
    )
})

orders.to_csv(RAW_DIR / "orders.csv", index=False)


# 7. Generate delivery records for completed orders
completed_orders = orders.loc[
    orders["status"] == "Completed"
].copy()

n_deliveries = len(completed_orders)

delivery_duration = rng.integers(15, 91, size=n_deliveries)

deliveries = pd.DataFrame({
    "delivery_id": range(20001, 20001 + n_deliveries),
    "order_id": completed_orders["order_id"].to_numpy(),
    "courier_id": rng.integers(501, 701, size=n_deliveries),
    "pickup_time": (
        completed_orders["order_time"]
        + pd.to_timedelta(
            rng.integers(5, 26, size=n_deliveries),
            unit="min"
        )
    )
})

deliveries["delivery_duration"] = delivery_duration

deliveries["delivery_time"] = (
    deliveries["pickup_time"]
    + pd.to_timedelta(deliveries["delivery_duration"], unit="min")
)

deliveries.to_csv(RAW_DIR / "deliveries.csv", index=False)


# 8. Generate customer reviews for a subset of completed orders
n_reviews = int(n_deliveries * 0.60)

reviewed_orders = completed_orders.sample(
    n=n_reviews,
    random_state=42
).reset_index(drop=True)

review_templates = {
    1: [
        "Very late delivery and cold food",
        "Terrible experience, order arrived too late"
    ],
    2: [
        "Delivery was slow and food was not great",
        "Not satisfied with the service"
    ],
    3: [
        "Average experience, nothing special",
        "Food was okay but delivery could improve"
    ],
    4: [
        "Good food and quick delivery",
        "Nice experience, would order again"
    ],
    5: [
        "Excellent service and delicious food",
        "Amazing experience, very fast delivery"
    ]
}

ratings = rng.choice([1, 2, 3, 4, 5], size=n_reviews,
                     p=[0.08, 0.10, 0.17, 0.35, 0.30])

review_text = [
    rng.choice(review_templates[int(rating)])
    for rating in ratings
]

reviews = pd.DataFrame({
    "review_id": range(30001, 30001 + n_reviews),
    "order_id": reviewed_orders["order_id"].to_numpy(),
    "rating": ratings,
    "review_text": review_text
})

reviews.to_csv(RAW_DIR / "reviews.csv", index=False)


# 9. Print dataset summaries
for name, df in {
    "customers": customers,
    "orders": orders,
    "deliveries": deliveries,
    "reviews": reviews
}.items():
    print(f"\n{name.upper()}")
    print("Shape:", df.shape)
    print(df.head(3))


print("\n--- DATA VALIDATION ---")

# Every order should reference an existing customer
print(
    "Invalid customer references:",
    (~orders["customer_id"].isin(customers["customer_id"])).sum()
)

# Every delivery should reference a completed order
print(
    "Deliveries linked to non-completed orders:",
    (~deliveries["order_id"].isin(completed_orders["order_id"])).sum()
)

# Every review should reference an existing order
print(
    "Invalid review references:",
    (~reviews["order_id"].isin(orders["order_id"])).sum()
)

# Check duplicate primary keys
print("Duplicate customer IDs:", customers["customer_id"].duplicated().sum())
print("Duplicate order IDs:", orders["order_id"].duplicated().sum())
print("Duplicate delivery IDs:", deliveries["delivery_id"].duplicated().sum())
print("Duplicate review IDs:", reviews["review_id"].duplicated().sum())
