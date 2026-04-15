# =========================================================
# PRT564 - Assessment 2
# Group 9
# RQ2 + RQ3 Regression Workflow
#
# Datasets used:
# 1) nt-government-regions_1986-to-2025.xlsx
# 2) nt_crime_statistics_jan_2020.csv
# 3) nt_crime_statistics_dec_2025.csv
#
# This script covers:
# - data loading
# - preprocessing / cleaning
# - missing value handling
# - descriptive summaries
# - EDA charts
# - feature engineering
# - RQ2 explanatory regression
# - RQ3 predictive regression
# - model evaluation
# - paired t-tests
# =========================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. FILE PATHS
# =========================================================

POP_FILE = r"C:\Users\khuep\Downloads\YEN\UNITS\PRT564\nt-government-regions_1986-to-2025.xlsx"
CRIME_2020_FILE = r"C:\Users\khuep\Downloads\YEN\UNITS\PRT564\nt_crime_statistics_jan_2020.csv"
CRIME_2025_FILE = r"C:\Users\khuep\Downloads\YEN\UNITS\PRT564\nt_crime_statistics_dec_2025.csv"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names."""
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[ /-]+", "_", regex=True)
        .str.replace(r"__+", "_", regex=True)
    )
    return df


def safe_divide(a, b):
    """Safely divide arrays/series and return NaN if denominator is 0."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    out = np.where((b == 0) | np.isnan(b), np.nan, a / b)
    return out


def evaluate_regression(y_true, y_pred, model_name="Model"):
    """Return regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def standardise_yes_no(series: pd.Series) -> pd.Series:
    """
    Convert yes/no style values to 1/0.
    '-' is treated as 0 for simplicity in this project.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "yes": 1, "y": 1, "true": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "0": 0,
            "-": 0,
            "nan": np.nan,
            "none": np.nan,
            "": np.nan
        })
    )


def parse_age_bounds(age_text: str):
    """
    Extract age bounds from labels like:
    - '15-19'
    - '20-24 years'
    - '65 years and over'
    - '85+'
    """
    if pd.isna(age_text):
        return (np.nan, np.nan)

    text = str(age_text).strip().lower()

    # Example: 15-19
    match_range = re.search(r"(\d+)\s*-\s*(\d+)", text)
    if match_range:
        return int(match_range.group(1)), int(match_range.group(2))

    # Example: 65 years and over / 85+
    match_over = re.search(r"(\d+)", text)
    if ("over" in text or "+" in text) and match_over:
        return int(match_over.group(1)), 120

    return (np.nan, np.nan)


def is_young_group(age_text: str) -> int:
    """
    Return 1 if age group overlaps with 15-34.
    """
    low, high = parse_age_bounds(age_text)
    if pd.isna(low) or pd.isna(high):
        return 0
    return int(high >= 15 and low <= 34)


def pick_population_total_rows(pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to select rows representing 'total population' per region-year.
    This dataset may contain cross-classified breakdowns.
    We prefer rows where:
    - sex indicates total/persons/all
    - age_group indicates total/all ages
    - aboriginal_status indicates total/all
    - status indicates estimate/total if available

    If no such rows are found, fall back carefully.
    """
    df = pop_df.copy()

    def text_contains_any(series, patterns):
        s = series.astype(str).str.lower()
        mask = pd.Series(False, index=series.index)
        for p in patterns:
            mask = mask | s.str.contains(p, na=False)
        return mask

    sex_ok = pd.Series(True, index=df.index)
    age_ok = pd.Series(True, index=df.index)
    ab_ok = pd.Series(True, index=df.index)
    status_ok = pd.Series(True, index=df.index)

    if "sex" in df.columns:
        sex_ok = text_contains_any(df["sex"], ["persons", "persons_total", "all", "total", "both"])

    if "age_group" in df.columns:
        age_ok = text_contains_any(df["age_group"], ["all ages", "total", "all"])

    if "aboriginal_status" in df.columns:
        ab_ok = text_contains_any(df["aboriginal_status"], ["all", "total"])

    if "status" in df.columns:
        # keep broad because labels vary
        status_ok = text_contains_any(df["status"], ["estimate", "erp", "total", "final", "preliminary"])

    subset = df.loc[sex_ok & age_ok & ab_ok & status_ok].copy()

    # If too strict and empty, relax step by step
    if subset.empty and "sex" in df.columns and "age_group" in df.columns and "aboriginal_status" in df.columns:
        subset = df.loc[sex_ok & age_ok & ab_ok].copy()

    if subset.empty and "sex" in df.columns and "age_group" in df.columns:
        subset = df.loc[sex_ok & age_ok].copy()

    if subset.empty:
        subset = df.copy()

    return subset


def build_population_features(pop_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build population-based features for modelling:
    - population_total
    - young_ratio
    - male_ratio
    - aboriginal_ratio
    - population_growth_rate
    """

    df = pop_df.copy()

    # Standardise text
    for col in ["status", "sex", "age_group", "aboriginal_status", "region"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Region key
    df["region_key"] = df["region"].astype(str).str.strip().str.lower()

    # Use selected rows for total population
    pop_total_rows = pick_population_total_rows(df)

    pop_total = (
        pop_total_rows.groupby(["region_key", "year"], as_index=False)["population"]
        .sum()
        .rename(columns={"population": "population_total"})
    )

    # Young population
    df["is_young_group"] = df["age_group"].apply(is_young_group) if "age_group" in df.columns else 0
    pop_young = (
        df.loc[df["is_young_group"] == 1]
        .groupby(["region_key", "year"], as_index=False)["population"]
        .sum()
        .rename(columns={"population": "young_population"})
    )

    # Male population
    if "sex" in df.columns:
        male_mask = df["sex"].astype(str).str.lower().str.contains("male", na=False)
        pop_male = (
            df.loc[male_mask]
            .groupby(["region_key", "year"], as_index=False)["population"]
            .sum()
            .rename(columns={"population": "male_population"})
        )
    else:
        pop_male = pop_total[["region_key", "year"]].copy()
        pop_male["male_population"] = np.nan

    # Aboriginal population
    if "aboriginal_status" in df.columns:
        ab_mask = df["aboriginal_status"].astype(str).str.lower().str.contains("aboriginal", na=False)
        pop_aboriginal = (
            df.loc[ab_mask]
            .groupby(["region_key", "year"], as_index=False)["population"]
            .sum()
            .rename(columns={"population": "aboriginal_population"})
        )
    else:
        pop_aboriginal = pop_total[["region_key", "year"]].copy()
        pop_aboriginal["aboriginal_population"] = np.nan

    # Merge
    pop_features = pop_total.merge(pop_young, on=["region_key", "year"], how="left")
    pop_features = pop_features.merge(pop_male, on=["region_key", "year"], how="left")
    pop_features = pop_features.merge(pop_aboriginal, on=["region_key", "year"], how="left")

    for col in ["young_population", "male_population", "aboriginal_population"]:
        pop_features[col] = pop_features[col].fillna(0)

    pop_features["young_ratio"] = safe_divide(pop_features["young_population"], pop_features["population_total"])
    pop_features["male_ratio"] = safe_divide(pop_features["male_population"], pop_features["population_total"])
    pop_features["aboriginal_ratio"] = safe_divide(pop_features["aboriginal_population"], pop_features["population_total"])

    # Growth rate
    pop_features = pop_features.sort_values(["region_key", "year"])
    pop_features["population_growth_rate"] = (
        pop_features.groupby("region_key")["population_total"].pct_change()
    )

    return pop_features


# =========================================================
# 3. LOAD DATA
# =========================================================

pop_df = pd.read_excel(POP_FILE)
crime_2020_df = pd.read_csv(CRIME_2020_FILE)
crime_2025_df = pd.read_csv(CRIME_2025_FILE)

pop_df = clean_column_names(pop_df)
crime_2020_df = clean_column_names(crime_2020_df)
crime_2025_df = clean_column_names(crime_2025_df)

print("Population columns:", pop_df.columns.tolist())
print("Crime 2020 columns:", crime_2020_df.columns.tolist())
print("Crime 2025 columns:", crime_2025_df.columns.tolist())


# =========================================================
# 4. BASIC CLEANING
# =========================================================

# Convert numeric
for col in ["year", "population"]:
    if col in pop_df.columns:
        pop_df[col] = pd.to_numeric(pop_df[col], errors="coerce")

for col in ["year", "month_number", "number_of_offences"]:
    if col in crime_2020_df.columns:
        crime_2020_df[col] = pd.to_numeric(crime_2020_df[col], errors="coerce")
    if col in crime_2025_df.columns:
        crime_2025_df[col] = pd.to_numeric(crime_2025_df[col], errors="coerce")

# Combine crime datasets
crime_df = pd.concat([crime_2020_df, crime_2025_df], ignore_index=True)

# Standardise text columns
for col in ["offence_category", "offence_type", "reporting_region", "statistical_area_2",
            "alcohol_involvement", "dv_involvement"]:
    if col in crime_df.columns:
        crime_df[col] = crime_df[col].astype(str).str.strip()

for col in ["status", "sex", "age_group", "aboriginal_status", "region"]:
    if col in pop_df.columns:
        pop_df[col] = pop_df[col].astype(str).str.strip()

# Build region keys
pop_df["region_key"] = pop_df["region"].astype(str).str.strip().str.lower()
crime_df["region_key"] = crime_df["reporting_region"].astype(str).str.strip().str.lower()

# Harmonise crime regions to population regions
REGION_MAP = {
    "alice springs": "central australia",
    "tennant creek": "barkly",
    "darwin": "greater darwin",
    "palmerston": "greater darwin",
    "nhulunbuy": "east arnhem",
    "katherine": "big rivers",
    "nt balance": "top end",
    "unknown": np.nan
}

crime_df["region_key"] = crime_df["region_key"].replace(REGION_MAP)

# Standardise alcohol / DV
crime_df["alcohol_flag"] = standardise_yes_no(crime_df["alcohol_involvement"]) \
    if "alcohol_involvement" in crime_df.columns else np.nan

crime_df["dv_flag"] = standardise_yes_no(crime_df["dv_involvement"]) \
    if "dv_involvement" in crime_df.columns else np.nan

# Remove unusable rows
pop_df = pop_df.dropna(subset=["year", "region_key", "population"])
crime_df = crime_df.dropna(subset=["year", "month_number", "region_key", "number_of_offences"])

# Keep only years common to available crime data
# Crime starts much later than population
pop_df = pop_df.loc[pop_df["year"].between(crime_df["year"].min(), crime_df["year"].max())].copy()


# =========================================================
# 5. POPULATION FEATURES
# =========================================================

pop_features = build_population_features(pop_df)
pop_features.to_csv(OUTPUT_DIR / "population_features.csv", index=False)


# =========================================================
# 6. CRIME FEATURES
# =========================================================

# Aggregate to region-year-month
crime_monthly = (
    crime_df.groupby(["region_key", "year", "month_number"], as_index=False)
    .agg(
        number_of_offences=("number_of_offences", "sum"),
        alcohol_related_offences=("alcohol_flag", lambda x: np.nansum(pd.to_numeric(x, errors="coerce"))),
        dv_related_offences=("dv_flag", lambda x: np.nansum(pd.to_numeric(x, errors="coerce")))
    )
)

crime_monthly["alcohol_ratio"] = safe_divide(
    crime_monthly["alcohol_related_offences"],
    crime_monthly["number_of_offences"]
)

crime_monthly["dv_ratio"] = safe_divide(
    crime_monthly["dv_related_offences"],
    crime_monthly["number_of_offences"]
)

# Offence composition ratios
if "offence_category" in crime_df.columns:
    cat_monthly = (
        crime_df.groupby(["region_key", "year", "month_number", "offence_category"], as_index=False)["number_of_offences"]
        .sum()
    )

    cat_pivot = cat_monthly.pivot_table(
        index=["region_key", "year", "month_number"],
        columns="offence_category",
        values="number_of_offences",
        aggfunc="sum",
        fill_value=0
    ).reset_index()

    # Clean column names
    cat_pivot.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_") if isinstance(c, str) else c
        for c in cat_pivot.columns
    ]

    offence_cols = [c for c in cat_pivot.columns if c not in ["region_key", "year", "month_number"]]

    total_offence_by_row = cat_pivot[offence_cols].sum(axis=1)
    for col in offence_cols:
        cat_pivot[f"{col}_ratio"] = safe_divide(cat_pivot[col], total_offence_by_row)

    ratio_cols = ["region_key", "year", "month_number"] + [f"{c}_ratio" for c in offence_cols]
    offence_comp = cat_pivot[ratio_cols].copy()
else:
    offence_comp = crime_monthly[["region_key", "year", "month_number"]].copy()

crime_monthly = crime_monthly.merge(
    offence_comp,
    on=["region_key", "year", "month_number"],
    how="left"
)

# Lag features
crime_monthly = crime_monthly.sort_values(["region_key", "year", "month_number"])
crime_monthly["lag_offence_1"] = crime_monthly.groupby("region_key")["number_of_offences"].shift(1)
crime_monthly["lag_offence_2"] = crime_monthly.groupby("region_key")["number_of_offences"].shift(2)

crime_monthly.to_csv(OUTPUT_DIR / "crime_monthly_features.csv", index=False)


# =========================================================
# 7. MERGE DATASETS
# =========================================================

model_df = crime_monthly.merge(
    pop_features,
    on=["region_key", "year"],
    how="left"
)

# Targets and time vars
model_df["crime_rate"] = safe_divide(model_df["number_of_offences"], model_df["population_total"])
model_df["month"] = model_df["month_number"]
model_df["year_num"] = model_df["year"]

# Save pre-missing-value version
model_df.to_csv(OUTPUT_DIR / "merged_before_missing_handling.csv", index=False)


# =========================================================
# 8. MISSING VALUES
# =========================================================

# Fill region-based medians first, then global medians
for col in ["population_growth_rate", "alcohol_ratio", "dv_ratio", "lag_offence_1", "lag_offence_2"]:
    if col in model_df.columns:
        model_df[col] = model_df.groupby("region_key")[col].transform(lambda x: x.fillna(x.median()))
        model_df[col] = model_df[col].fillna(model_df[col].median())

# Fill ratio columns with 0 if absent for a specific region-month
ratio_cols_all = [c for c in model_df.columns if c.endswith("_ratio")]
for col in ratio_cols_all:
    model_df[col] = model_df[col].fillna(0)

# Drop rows missing essential targets/population
model_df = model_df.dropna(subset=["population_total", "number_of_offences", "crime_rate"])

# Final dataset
model_df.to_csv(OUTPUT_DIR / "merged_model_dataset.csv", index=False)
print("Final modelling dataset saved.")


# =========================================================
# 9. DESCRIPTIVE SUMMARIES
# =========================================================

summary_stats = model_df[[
    "number_of_offences", "crime_rate", "population_total",
    "young_ratio", "male_ratio", "aboriginal_ratio",
    "population_growth_rate", "alcohol_ratio", "dv_ratio"
]].describe().T

summary_stats.to_csv(OUTPUT_DIR / "descriptive_summary.csv")

region_summary = (
    model_df.groupby("region_key", as_index=False)
    .agg(
        total_offences=("number_of_offences", "sum"),
        avg_crime_rate=("crime_rate", "mean"),
        avg_population=("population_total", "mean")
    )
    .sort_values("total_offences", ascending=False)
)

region_summary.to_csv(OUTPUT_DIR / "region_summary.csv", index=False)

print("\nDescriptive summaries saved.")


# =========================================================
# 10. EDA VISUALISATIONS
# =========================================================

# 10.1 Line chart: crime over time
trend_df = (
    model_df.groupby(["year", "month"], as_index=False)["number_of_offences"]
    .sum()
    .sort_values(["year", "month"])
)
trend_df["date_index"] = np.arange(len(trend_df))

plt.figure(figsize=(10, 5))
plt.plot(trend_df["date_index"], trend_df["number_of_offences"])
plt.title("Total Monthly Offences Over Time")
plt.xlabel("Time Index")
plt.ylabel("Number of Offences")
plt.tight_layout()
plt.show()

# 10.2 Bar chart: crime by region
region_plot_df = (
    model_df.groupby("region_key", as_index=False)["number_of_offences"]
    .sum()
    .sort_values("number_of_offences", ascending=False)
)

plt.figure(figsize=(10, 5))
plt.bar(region_plot_df["region_key"], region_plot_df["number_of_offences"])
plt.title("Total Offences by Region")
plt.xlabel("Region")
plt.ylabel("Number of Offences")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 10.3 Scatter plot: population vs crime
scatter_df = (
    model_df.groupby(["region_key", "year"], as_index=False)
    .agg(
        population_total=("population_total", "first"),
        avg_crime_rate=("crime_rate", "mean")
    )
)

plt.figure(figsize=(8, 5))
plt.scatter(scatter_df["population_total"], scatter_df["avg_crime_rate"])
plt.title("Population vs Average Crime Rate")
plt.xlabel("Population")
plt.ylabel("Average Crime Rate")
plt.tight_layout()
plt.show()

print("EDA charts saved.")

# 10.4 Alcohol ratio over time
alcohol_trend = (
    model_df.groupby(["year", "month"], as_index=False)["alcohol_ratio"]
    .mean()
    .sort_values(["year", "month"])
)
alcohol_trend["date_index"] = np.arange(len(alcohol_trend))

plt.figure(figsize=(10, 5))
plt.plot(alcohol_trend["date_index"], alcohol_trend["alcohol_ratio"])
plt.title("Average Alcohol-Related Offence Ratio Over Time")
plt.xlabel("Time Index")
plt.ylabel("Alcohol Ratio")
plt.tight_layout()
plt.show()


# 10.5 Domestic violence ratio over time
dv_trend = (
    model_df.groupby(["year", "month"], as_index=False)["dv_ratio"]
    .mean()
    .sort_values(["year", "month"])
)
dv_trend["date_index"] = np.arange(len(dv_trend))

plt.figure(figsize=(10, 5))
plt.plot(dv_trend["date_index"], dv_trend["dv_ratio"])
plt.title("Average Domestic Violence Ratio Over Time")
plt.xlabel("Time Index")
plt.ylabel("DV Ratio")
plt.tight_layout()
plt.show()


# 10.6 Monthly seasonality
seasonality_df = (
    model_df.groupby("month", as_index=False)["number_of_offences"]
    .mean()
    .sort_values("month")
)

plt.figure(figsize=(8, 5))
plt.plot(seasonality_df["month"], seasonality_df["number_of_offences"], marker="o")
plt.title("Average Monthly Offences by Month")
plt.xlabel("Month")
plt.ylabel("Average Number of Offences")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.show()


# 10.7 Top offence categories
if "offence_category" in crime_df.columns:
    offence_cat_df = (
        crime_df.groupby("offence_category", as_index=False)["number_of_offences"]
        .sum()
        .sort_values("number_of_offences", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 5))
    plt.bar(offence_cat_df["offence_category"], offence_cat_df["number_of_offences"])
    plt.title("Top 10 Offence Categories by Number of Offences")
    plt.xlabel("Offence Category")
    plt.ylabel("Number of Offences")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


# 10.8 Correlation heatmap
corr_cols = [
    "number_of_offences",
    "crime_rate",
    "population_total",
    "young_ratio",
    "male_ratio",
    "aboriginal_ratio",
    "population_growth_rate",
    "alcohol_ratio",
    "dv_ratio",
    "lag_offence_1",
    "lag_offence_2"
]
corr_cols = [c for c in corr_cols if c in model_df.columns]

corr_matrix = model_df[corr_cols].corr()

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar()
plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
plt.yticks(range(len(corr_cols)), corr_cols)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# 10.9 Boxplot of crime rate by region
region_groups = []
region_labels = []

for region, grp in model_df.groupby("region_key"):
    region_groups.append(grp["crime_rate"].dropna())
    region_labels.append(region)

plt.figure(figsize=(10, 5))
plt.boxplot(region_groups, labels=region_labels)
plt.title("Crime Rate Distribution by Region")
plt.xlabel("Region")
plt.ylabel("Crime Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# 10.10 Lagged offence count vs current offence count
if "lag_offence_1" in model_df.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(model_df["lag_offence_1"], model_df["number_of_offences"])
    plt.title("Lagged Offence Count vs Current Offence Count")
    plt.xlabel("Lagged Offence Count (t-1)")
    plt.ylabel("Current Offence Count")
    plt.tight_layout()
    plt.show()


# =========================================================
# 11. RQ2 - EXPLANATORY REGRESSION
# =========================================================

# RQ2 target: crime_rate
# Focus: explaining regional crime-rate variation

rq2_base_features = [
    "population_total",
    "young_ratio",
    "male_ratio",
    "aboriginal_ratio",
    "population_growth_rate",
    "alcohol_ratio",
    "dv_ratio",
    "month",
    "year_num"
]

# Add a few offence composition ratios if available
offence_ratio_cols = [
    c for c in model_df.columns
    if c.endswith("_ratio") and c not in [
        "young_ratio", "male_ratio", "aboriginal_ratio", "alcohol_ratio", "dv_ratio"
    ]
]

rq2_features = rq2_base_features + offence_ratio_cols[:5]
rq2_features = [c for c in rq2_features if c in model_df.columns]

rq2_df = model_df.dropna(subset=rq2_features + ["crime_rate"]).copy()

X_rq2 = rq2_df[rq2_features]
y_rq2 = rq2_df["crime_rate"]

X_train_rq2, X_test_rq2, y_train_rq2, y_test_rq2 = train_test_split(
    X_rq2, y_rq2, test_size=0.2, random_state=42
)

rq2_model = LinearRegression()
rq2_model.fit(X_train_rq2, y_train_rq2)
rq2_pred = rq2_model.predict(X_test_rq2)

rq2_metrics = evaluate_regression(y_test_rq2, rq2_pred, "RQ2_MultipleLinearRegression")
print("\nRQ2 metrics:", rq2_metrics)

rq2_coef_df = pd.DataFrame({
    "feature": X_rq2.columns,
    "coefficient": rq2_model.coef_
}).sort_values("coefficient", key=np.abs, ascending=False)

rq2_coef_df.to_csv(OUTPUT_DIR / "rq2_coefficients.csv", index=False)



# =========================================================
# 12. RQ3 - PREDICTIVE REGRESSION
# =========================================================

# RQ3 target: monthly offence count
# Compare models

rq3_base_features = [
    "population_total",
    "young_ratio",
    "male_ratio",
    "aboriginal_ratio",
    "population_growth_rate",
    "alcohol_ratio",
    "dv_ratio",
    "lag_offence_1",
    "lag_offence_2",
    "month",
    "year_num"
]

rq3_features = rq3_base_features + offence_ratio_cols[:5]
rq3_features = [c for c in rq3_features if c in model_df.columns]

rq3_df = model_df.dropna(subset=rq3_features + ["number_of_offences"]).copy()

X_rq3 = rq3_df[rq3_features]
y_rq3 = rq3_df["number_of_offences"]

X_train_rq3, X_test_rq3, y_train_rq3, y_test_rq3 = train_test_split(
    X_rq3, y_rq3, test_size=0.2, random_state=42
)

# Model 1: Linear Regression
rq3_lr = LinearRegression()
rq3_lr.fit(X_train_rq3, y_train_rq3)
rq3_lr_pred = rq3_lr.predict(X_test_rq3)
rq3_lr_metrics = evaluate_regression(y_test_rq3, rq3_lr_pred, "RQ3_LinearRegression")

# Model 2: Ridge Regression
rq3_ridge = Ridge(alpha=1.0)
rq3_ridge.fit(X_train_rq3, y_train_rq3)
rq3_ridge_pred = rq3_ridge.predict(X_test_rq3)
rq3_ridge_metrics = evaluate_regression(y_test_rq3, rq3_ridge_pred, "RQ3_Ridge")

# Model 3: Random Forest Regression
rq3_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rq3_rf.fit(X_train_rq3, y_train_rq3)
rq3_rf_pred = rq3_rf.predict(X_test_rq3)
rq3_rf_metrics = evaluate_regression(y_test_rq3, rq3_rf_pred, "RQ3_RandomForest")

print("\nRQ3 metrics:")
print(rq3_lr_metrics)
print(rq3_ridge_metrics)
print(rq3_rf_metrics)

# =========================================================
# 12A. REGRESSION VISUALISATIONS
# =========================================================

# 12A.1 RQ2 coefficients plot
coef_df = pd.DataFrame({
    "feature": X_rq2.columns,
    "coef": rq2_model.coef_
}).sort_values("coef", key=abs, ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(coef_df["feature"], coef_df["coef"])
plt.xticks(rotation=45, ha="right")
plt.title("Regression Coefficients (RQ2)")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.tight_layout()
plt.show()


# 12A.2 RQ3 Actual vs Predicted - Linear Regression
plt.figure(figsize=(8, 5))
plt.scatter(y_test_rq3, rq3_lr_pred)
plt.title("Actual vs Predicted (Linear Regression)")
plt.xlabel("Actual Offence Count")
plt.ylabel("Predicted Offence Count")
plt.tight_layout()
plt.show()


# 12A.3 RQ3 Actual vs Predicted - Random Forest
plt.figure(figsize=(8, 5))
plt.scatter(y_test_rq3, rq3_rf_pred)
plt.title("Actual vs Predicted (Random Forest)")
plt.xlabel("Actual Offence Count")
plt.ylabel("Predicted Offence Count")
plt.tight_layout()
plt.show()


# 12A.4 RQ3 Residual Plot - Linear Regression
residuals = y_test_rq3 - rq3_lr_pred

plt.figure(figsize=(8, 5))
plt.scatter(rq3_lr_pred, residuals)
plt.axhline(y=0)
plt.title("Residual Plot (Linear Regression)")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()


# 12A.5 RQ3 Feature Importance - Random Forest
importance_df = pd.DataFrame({
    "feature": X_rq3.columns,
    "importance": rq3_rf.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 5))
plt.bar(importance_df["feature"], importance_df["importance"])
plt.xticks(rotation=45, ha="right")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# 12A.6 Model comparison by MAE
models = ["Linear", "Ridge", "Random Forest"]
mae_values = [
    rq3_lr_metrics["MAE"],
    rq3_ridge_metrics["MAE"],
    rq3_rf_metrics["MAE"]
]

plt.figure(figsize=(7, 5))
plt.bar(models, mae_values)
plt.title("Model Comparison by MAE")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.tight_layout()
plt.show()

rf_importance_df = pd.DataFrame({
    "feature": X_rq3.columns,
    "importance": rq3_rf.feature_importances_
}).sort_values("importance", ascending=False)

rf_importance_df.to_csv(OUTPUT_DIR / "rq3_rf_feature_importance.csv", index=False)

# Actual vs predicted plots
plt.figure(figsize=(8, 5))
plt.scatter(y_test_rq3, rq3_rf_pred)

plt.plot([y_test_rq3.min(), y_test_rq3.max()],
         [y_test_rq3.min(), y_test_rq3.max()])

plt.title("RQ3 Random Forest: Actual vs Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(y_test_rq3, rq3_rf_pred)
plt.title("RQ3 Random Forest: Actual vs Predicted")
plt.xlabel("Actual Offence Count")
plt.ylabel("Predicted Offence Count")
plt.tight_layout()
plt.show()


# =========================================================
# 13. MODEL METRICS TABLE
# =========================================================

metrics_df = pd.DataFrame([
    rq2_metrics,
    rq3_lr_metrics,
    rq3_ridge_metrics,
    rq3_rf_metrics
])

metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
print("\n================ MODEL METRICS ================")
print(metrics_df.round(4).to_string(index=False))

# =========================================================
# MODEL COMPARISON VISUALISATION
# =========================================================

models = ["Linear", "Ridge", "Random Forest"]
mae_values = [
    rq3_lr_metrics["MAE"],
    rq3_ridge_metrics["MAE"],
    rq3_rf_metrics["MAE"]
]

plt.figure(figsize=(6, 4))
plt.bar(models, mae_values)
plt.title("Model Comparison (MAE)")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.tight_layout()
plt.show()

# =========================================================
# 14. STATISTICAL TESTING
# =========================================================

# Paired t-tests on absolute errors for RQ3 model comparison
lr_abs_errors = np.abs(y_test_rq3 - rq3_lr_pred)
ridge_abs_errors = np.abs(y_test_rq3 - rq3_ridge_pred)
rf_abs_errors = np.abs(y_test_rq3 - rq3_rf_pred)

t_stat_lr_ridge, p_lr_ridge = ttest_rel(lr_abs_errors, ridge_abs_errors, nan_policy="omit")
t_stat_lr_rf, p_lr_rf = ttest_rel(lr_abs_errors, rf_abs_errors, nan_policy="omit")
t_stat_ridge_rf, p_ridge_rf = ttest_rel(ridge_abs_errors, rf_abs_errors, nan_policy="omit")

ttest_df = pd.DataFrame({
    "comparison": ["LR vs Ridge", "LR vs RF", "Ridge vs RF"],
    "t_statistic": [t_stat_lr_ridge, t_stat_lr_rf, t_stat_ridge_rf],
    "p_value": [p_lr_ridge, p_lr_rf, p_ridge_rf]
})

ttest_df.to_csv(OUTPUT_DIR / "paired_ttest_results.csv", index=False)
print("\n================ PAIRED T-TEST RESULTS ================")
print(ttest_df.round(4).to_string(index=False))


# =========================================================
# 15. QUICK OUTPUTS FOR REPORT / SLIDES
# =========================================================

# Save sample rows
model_df.head(50).to_csv(OUTPUT_DIR / "model_df_sample.csv", index=False)

# Save top RQ2 coefficients
rq2_coef_df.head(10).to_csv(OUTPUT_DIR / "rq2_top_coefficients.csv", index=False)

# Save top RF importance
rf_importance_df.head(10).to_csv(OUTPUT_DIR / "rq3_top_rf_importance.csv", index=False)


# =========================================================
# 16. CONSOLE SUMMARY
# =========================================================

print("\n================ FINAL SUMMARY ================")
print("Files created in:", OUTPUT_DIR.resolve())
print("\nMain outputs:")
print("- population_features.csv")
print("- crime_monthly_features.csv")
print("- merged_model_dataset.csv")
print("- descriptive_summary.csv")
print("- region_summary.csv")
print("- eda_line_crime_over_time.png")
print("- eda_bar_crime_by_region.png")
print("- eda_scatter_population_vs_crime.png")
print("- rq2_coefficients.csv")
print("- rq2_coefficients_plot.png")
print("- rq3_rf_feature_importance.csv")
print("- rq3_lr_actual_vs_predicted.png")
print("- rq3_rf_actual_vs_predicted.png")
print("- model_metrics.csv")
print("- paired_ttest_results.csv")

print("\nInterpretation notes:")
print("RQ2:")
print("- Use rq2_coefficients.csv to explain which variables are most strongly associated with crime_rate.")
print("- Positive coefficient = higher value associated with higher crime_rate.")
print("- Negative coefficient = higher value associated with lower crime_rate.")
print("- Emphasise association, not causation.")

print("\nRQ3:")
print("- Compare MAE, RMSE, and R2 in model_metrics.csv.")
print("- Lower MAE/RMSE is better; higher R2 is better.")
print("- Use paired_ttest_results.csv to discuss whether model differences are statistically significant.")
print("- If p < 0.05, the difference between model errors is statistically significant.")
print("================================================")
