# ============================================
# PRT564 - Assessment 2
# Regression analysis for RQ2 and RQ3
# ============================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import ttest_rel

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

# ============================================
# 1. FILE PATHS
# ============================================

# Update these file names/paths as needed
POP_FILE = "population_estimates_ntg_service_regions.xlsx"
CRIME_2023_FILE = "nt_crime_statistics_november_2023.csv"
CRIME_2025_FILE = "nt_crime_statistics_dec_2025.csv"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================
# 2. HELPER FUNCTIONS
# ============================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
    )
    return df


def standardise_yes_no(series: pd.Series) -> pd.Series:
    """Convert yes/no style text to binary 1/0."""
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "yes": 1, "y": 1, "true": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "0": 0,
            "nan": np.nan
        })
    )


def parse_age_group(age_text: str):
    """
    Extract lower and upper age bounds from age group text.
    Returns (lower, upper) where possible.
    """
    if pd.isna(age_text):
        return (np.nan, np.nan)

    text = str(age_text).strip().lower()

    # Common patterns like '15-19', '20-24'
    if "-" in text:
        parts = text.replace("years", "").replace("year", "").strip().split("-")
        try:
            lower = int(''.join(filter(str.isdigit, parts[0])))
            upper = int(''.join(filter(str.isdigit, parts[1])))
            return (lower, upper)
        except:
            return (np.nan, np.nan)

    # Common patterns like '85 years and over', '65+'
    if "over" in text or "+" in text:
        try:
            lower = int(''.join(filter(str.isdigit, text)))
            return (lower, 120)
        except:
            return (np.nan, np.nan)

    return (np.nan, np.nan)


def classify_young_age(age_text: str) -> int:
    """
    Returns 1 if age group is broadly 15-34, else 0.
    Adjust if your lecturer/group wants a different definition.
    """
    lower, upper = parse_age_group(age_text)
    if pd.isna(lower) or pd.isna(upper):
        return 0

    # overlaps with 15-34
    if upper >= 15 and lower <= 34:
        return 1
    return 0


def evaluate_regression(y_true, y_pred, model_name="Model") -> dict:
    """Return common regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def safe_divide(a, b):
    """Avoid division by zero."""
    return np.where((b == 0) | pd.isna(b), np.nan, a / b)


# ============================================
# 3. LOAD DATA
# ============================================

# Population data
pop_df = pd.read_excel(POP_FILE)
pop_df = clean_column_names(pop_df)

# Crime data
crime_2023_df = pd.read_csv(CRIME_2023_FILE)
crime_2023_df = clean_column_names(crime_2023_df)

crime_2025_df = pd.read_csv(CRIME_2025_FILE)
crime_2025_df = clean_column_names(crime_2025_df)

print("Population columns:", pop_df.columns.tolist())
print("Crime 2023 columns:", crime_2023_df.columns.tolist())
print("Crime 2025 columns:", crime_2025_df.columns.tolist())


# ============================================
# 4. BASIC CLEANING
# ============================================

# Standardise likely key columns in population data
# Expected columns based on your description:
# year, status, sex, age_group, aboriginal_status, region, population

required_pop_cols = ["year", "sex", "age_group", "aboriginal_status", "region", "population"]
for col in required_pop_cols:
    if col not in pop_df.columns:
        print(f"WARNING: population column '{col}' not found.")

# Standardise likely key columns in crime data
# Expected columns:
# as_at, year, month_number, offence_category, offence_type,
# alcohol_involvement, dv_involvement, reporting_region,
# statistical_area_2, number_of_offences

required_crime_cols = [
    "year", "month_number", "offence_category", "offence_type",
    "alcohol_involvement", "dv_involvement", "reporting_region",
    "number_of_offences"
]
for col in required_crime_cols:
    if col not in crime_2023_df.columns:
        print(f"WARNING: crime column '{col}' not found in 2023 data.")
    if col not in crime_2025_df.columns:
        print(f"WARNING: crime column '{col}' not found in 2025 data.")


# Combine crime datasets
crime_df = pd.concat([crime_2023_df, crime_2025_df], ignore_index=True)

# Convert numeric columns
for col in ["year", "month_number", "number_of_offences"]:
    if col in crime_df.columns:
        crime_df[col] = pd.to_numeric(crime_df[col], errors="coerce")

if "population" in pop_df.columns:
    pop_df["population"] = pd.to_numeric(pop_df["population"], errors="coerce")

if "year" in pop_df.columns:
    pop_df["year"] = pd.to_numeric(pop_df["year"], errors="coerce")

# Standardise text columns
text_cols_pop = ["sex", "age_group", "aboriginal_status", "region", "status"]
for col in text_cols_pop:
    if col in pop_df.columns:
        pop_df[col] = pop_df[col].astype(str).str.strip()

text_cols_crime = [
    "offence_category", "offence_type", "reporting_region",
    "statistical_area_2", "alcohol_involvement", "dv_involvement"
]
for col in text_cols_crime:
    if col in crime_df.columns:
        crime_df[col] = crime_df[col].astype(str).str.strip()

# Standardise region keys
if "region" in pop_df.columns:
    pop_df["region_key"] = pop_df["region"].astype(str).str.strip().str.lower()

if "reporting_region" in crime_df.columns:
    crime_df["region_key"] = crime_df["reporting_region"].astype(str).str.strip().str.lower()

# Optional: region harmonisation mapping
# Update as needed if labels differ between datasets
REGION_MAP = {
    # "alice springs region": "alice springs",
    # "darwin urban": "darwin",
}

pop_df["region_key"] = pop_df["region_key"].replace(REGION_MAP)
crime_df["region_key"] = crime_df["region_key"].replace(REGION_MAP)

# Binary conversions for alcohol and DV
if "alcohol_involvement" in crime_df.columns:
    crime_df["alcohol_flag"] = standardise_yes_no(crime_df["alcohol_involvement"])
else:
    crime_df["alcohol_flag"] = np.nan

if "dv_involvement" in crime_df.columns:
    crime_df["dv_flag"] = standardise_yes_no(crime_df["dv_involvement"])
else:
    crime_df["dv_flag"] = np.nan

# Drop obviously unusable rows
crime_df = crime_df.dropna(subset=["year", "month_number", "region_key", "number_of_offences"])
pop_df = pop_df.dropna(subset=["year", "region_key", "population"])

# ============================================
# 5. BUILD POPULATION FEATURES
# ============================================

# 5.1 Total population by region-year
pop_total = (
    pop_df.groupby(["region_key", "year"], as_index=False)["population"]
    .sum()
    .rename(columns={"population": "population_total"})
)

# 5.2 Young population ratio (15-34)
if "age_group" in pop_df.columns:
    pop_df["is_young_group"] = pop_df["age_group"].apply(classify_young_age)

    pop_young = (
        pop_df.loc[pop_df["is_young_group"] == 1]
        .groupby(["region_key", "year"], as_index=False)["population"]
        .sum()
        .rename(columns={"population": "young_population"})
    )
else:
    pop_young = pop_total[["region_key", "year"]].copy()
    pop_young["young_population"] = np.nan

# 5.3 Male population ratio
if "sex" in pop_df.columns:
    male_mask = pop_df["sex"].astype(str).str.lower().str.contains("male", na=False)
    pop_male = (
        pop_df.loc[male_mask]
        .groupby(["region_key", "year"], as_index=False)["population"]
        .sum()
        .rename(columns={"population": "male_population"})
    )
else:
    pop_male = pop_total[["region_key", "year"]].copy()
    pop_male["male_population"] = np.nan

# 5.4 Aboriginal population ratio
if "aboriginal_status" in pop_df.columns:
    aboriginal_mask = pop_df["aboriginal_status"].astype(str).str.lower().str.contains("aboriginal", na=False)
    pop_aboriginal = (
        pop_df.loc[aboriginal_mask]
        .groupby(["region_key", "year"], as_index=False)["population"]
        .sum()
        .rename(columns={"population": "aboriginal_population"})
    )
else:
    pop_aboriginal = pop_total[["region_key", "year"]].copy()
    pop_aboriginal["aboriginal_population"] = np.nan

# 5.5 Merge population features
pop_features = pop_total.merge(pop_young, on=["region_key", "year"], how="left")
pop_features = pop_features.merge(pop_male, on=["region_key", "year"], how="left")
pop_features = pop_features.merge(pop_aboriginal, on=["region_key", "year"], how="left")

# Fill missing component populations with 0 if appropriate
for col in ["young_population", "male_population", "aboriginal_population"]:
    if col in pop_features.columns:
        pop_features[col] = pop_features[col].fillna(0)

# Ratios
pop_features["young_ratio"] = safe_divide(pop_features["young_population"], pop_features["population_total"])
pop_features["male_ratio"] = safe_divide(pop_features["male_population"], pop_features["population_total"])
pop_features["aboriginal_ratio"] = safe_divide(pop_features["aboriginal_population"], pop_features["population_total"])

# Population growth rate by region-year
pop_features = pop_features.sort_values(["region_key", "year"])
pop_features["population_growth_rate"] = (
    pop_features.groupby("region_key")["population_total"].pct_change()
)

# ============================================
# 6. BUILD CRIME FEATURES
# ============================================

# 6.1 Aggregate offence totals by region-year-month
crime_monthly = (
    crime_df.groupby(["region_key", "year", "month_number"], as_index=False)
    .agg(
        number_of_offences=("number_of_offences", "sum"),
        alcohol_related_offences=("alcohol_flag", lambda x: np.nansum(x)),
        dv_related_offences=("dv_flag", lambda x: np.nansum(x))
    )
)

# Ratios
crime_monthly["alcohol_ratio"] = safe_divide(
    crime_monthly["alcohol_related_offences"],
    crime_monthly["number_of_offences"]
)

crime_monthly["dv_ratio"] = safe_divide(
    crime_monthly["dv_related_offences"],
    crime_monthly["number_of_offences"]
)

# 6.2 Offence composition ratios
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

    # Clean category column names
    cat_pivot.columns = [
        str(c).strip().lower().replace(" ", "_").replace("-", "_") if isinstance(c, str) else c
        for c in cat_pivot.columns
    ]

    # Convert counts to ratios
    offence_cols = [c for c in cat_pivot.columns if c not in ["region_key", "year", "month_number"]]
    for c in offence_cols:
        cat_pivot[f"{c}_ratio"] = safe_divide(cat_pivot[c], cat_pivot[offence_cols].sum(axis=1))

    # Keep only ratios
    ratio_cols = ["region_key", "year", "month_number"] + [f"{c}_ratio" for c in offence_cols]
    offence_comp = cat_pivot[ratio_cols]
else:
    offence_comp = crime_monthly[["region_key", "year", "month_number"]].copy()

# Merge offence composition
crime_monthly = crime_monthly.merge(
    offence_comp, on=["region_key", "year", "month_number"], how="left"
)

# 6.3 Lagged offence counts
crime_monthly = crime_monthly.sort_values(["region_key", "year", "month_number"])
crime_monthly["lag_offence_1"] = crime_monthly.groupby("region_key")["number_of_offences"].shift(1)
crime_monthly["lag_offence_2"] = crime_monthly.groupby("region_key")["number_of_offences"].shift(2)

# ============================================
# 7. MERGE POPULATION + CRIME
# ============================================

model_df = crime_monthly.merge(
    pop_features,
    on=["region_key", "year"],
    how="left"
)

# Create crime rate for RQ2
model_df["crime_rate"] = safe_divide(
    model_df["number_of_offences"],
    model_df["population_total"]
)

# Keep month/year as temporal variables
model_df["month"] = model_df["month_number"]
model_df["year_num"] = model_df["year"]

# ============================================
# 8. MISSING VALUES
# ============================================

# Fill lag values with regional median or overall median
for col in ["lag_offence_1", "lag_offence_2", "population_growth_rate",
            "alcohol_ratio", "dv_ratio"]:
    if col in model_df.columns:
        model_df[col] = model_df.groupby("region_key")[col].transform(
            lambda x: x.fillna(x.median())
        )
        model_df[col] = model_df[col].fillna(model_df[col].median())

# Fill ratio columns if needed
ratio_feature_cols = [c for c in model_df.columns if c.endswith("_ratio")]
for col in ratio_feature_cols:
    model_df[col] = model_df[col].fillna(0)

# Drop rows with missing core targets/features
model_df = model_df.dropna(subset=["crime_rate", "number_of_offences", "population_total"])

# Save merged dataset
model_df.to_csv(OUTPUT_DIR / "merged_model_dataset.csv", index=False)
print("Merged dataset saved to outputs/merged_model_dataset.csv")

# ============================================
# 9. EDA VISUALISATIONS
# ============================================

# 9.1 Total offences over time
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
plt.savefig(OUTPUT_DIR / "eda_total_monthly_offences.png", dpi=300)
plt.close()

# 9.2 Crime by region
region_df = (
    model_df.groupby("region_key", as_index=False)["number_of_offences"]
    .sum()
    .sort_values("number_of_offences", ascending=False)
)

plt.figure(figsize=(10, 5))
plt.bar(region_df["region_key"], region_df["number_of_offences"])
plt.title("Total Offences by Region")
plt.xlabel("Region")
plt.ylabel("Number of Offences")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "eda_offences_by_region.png", dpi=300)
plt.close()

# 9.3 Population vs crime rate
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
plt.savefig(OUTPUT_DIR / "eda_population_vs_crime_rate.png", dpi=300)
plt.close()

print("EDA charts saved to outputs/")

# ============================================
# 10. RQ2 - EXPLANATORY REGRESSION
# ============================================

# Features for RQ2
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

# Add offence composition ratio features if present
offence_ratio_cols = [c for c in model_df.columns if c.endswith("_ratio") and c not in [
    "young_ratio", "male_ratio", "aboriginal_ratio", "alcohol_ratio", "dv_ratio"
]]

rq2_features = rq2_base_features + offence_ratio_cols[:5]  # limit to first few if many
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

rq2_metrics = evaluate_regression(y_test_rq2, rq2_pred, "RQ2_LinearRegression")
print("\nRQ2 Metrics")
print(rq2_metrics)

# Coefficients table for interpretation
rq2_coef_df = pd.DataFrame({
    "feature": X_rq2.columns,
    "coefficient": rq2_model.coef_
}).sort_values("coefficient", key=np.abs, ascending=False)

rq2_coef_df.to_csv(OUTPUT_DIR / "rq2_coefficients.csv", index=False)

# ============================================
# 11. RQ3 - PREDICTIVE REGRESSION
# ============================================

# Features for RQ3
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

# Model 2: Random Forest Regression
rq3_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rq3_rf.fit(X_train_rq3, y_train_rq3)
rq3_rf_pred = rq3_rf.predict(X_test_rq3)

rq3_rf_metrics = evaluate_regression(y_test_rq3, rq3_rf_pred, "RQ3_RandomForest")

# Optional Model 3: Ridge Regression
rq3_ridge = Ridge(alpha=1.0)
rq3_ridge.fit(X_train_rq3, y_train_rq3)
rq3_ridge_pred = rq3_ridge.predict(X_test_rq3)

rq3_ridge_metrics = evaluate_regression(y_test_rq3, rq3_ridge_pred, "RQ3_Ridge")

print("\nRQ3 Metrics")
print(rq3_lr_metrics)
print(rq3_rf_metrics)
print(rq3_ridge_metrics)

# Save feature importance for RF
rf_importance_df = pd.DataFrame({
    "feature": X_rq3.columns,
    "importance": rq3_rf.feature_importances_
}).sort_values("importance", ascending=False)

rf_importance_df.to_csv(OUTPUT_DIR / "rq3_rf_feature_importance.csv", index=False)

# ============================================
# 12. MODEL COMPARISON TABLE
# ============================================

metrics_df = pd.DataFrame([
    rq2_metrics,
    rq3_lr_metrics,
    rq3_rf_metrics,
    rq3_ridge_metrics
])
metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

print("\nAll metrics:")
print(metrics_df)

# ============================================
# 13. PAIRED T-TEST ON ABSOLUTE ERRORS (RQ3)
# ============================================

# Compare Linear Regression vs Random Forest on same test set
lr_abs_errors = np.abs(y_test_rq3 - rq3_lr_pred)
rf_abs_errors = np.abs(y_test_rq3 - rq3_rf_pred)
ridge_abs_errors = np.abs(y_test_rq3 - rq3_ridge_pred)

# t-test: LR vs RF
t_stat_lr_rf, p_value_lr_rf = ttest_rel(lr_abs_errors, rf_abs_errors, nan_policy="omit")

# t-test: LR vs Ridge
t_stat_lr_ridge, p_value_lr_ridge = ttest_rel(lr_abs_errors, ridge_abs_errors, nan_policy="omit")

# t-test: RF vs Ridge
t_stat_rf_ridge, p_value_rf_ridge = ttest_rel(rf_abs_errors, ridge_abs_errors, nan_policy="omit")

ttest_df = pd.DataFrame({
    "comparison": ["LR vs RF", "LR vs Ridge", "RF vs Ridge"],
    "t_statistic": [t_stat_lr_rf, t_stat_lr_ridge, t_stat_rf_ridge],
    "p_value": [p_value_lr_rf, p_value_lr_ridge, p_value_rf_ridge]
})

ttest_df.to_csv(OUTPUT_DIR / "paired_ttest_results.csv", index=False)

print("\nPaired t-test results:")
print(ttest_df)

# ============================================
# 14. PREDICTION PLOTS
# ============================================

# 14.1 Actual vs Predicted for RQ3 Linear Regression
plt.figure(figsize=(8, 5))
plt.scatter(y_test_rq3, rq3_lr_pred)
plt.title("RQ3 Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Offence Count")
plt.ylabel("Predicted Offence Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rq3_lr_actual_vs_predicted.png", dpi=300)
plt.close()

# 14.2 Actual vs Predicted for RQ3 Random Forest
plt.figure(figsize=(8, 5))
plt.scatter(y_test_rq3, rq3_rf_pred)
plt.title("RQ3 Random Forest: Actual vs Predicted")
plt.xlabel("Actual Offence Count")
plt.ylabel("Predicted Offence Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rq3_rf_actual_vs_predicted.png", dpi=300)
plt.close()

# ============================================
# 15. SAVE SAMPLE OUTPUTS
# ============================================

# Save a sample of the modelling dataframe
model_df.head(50).to_csv(OUTPUT_DIR / "model_df_sample.csv", index=False)

print("\nDone. Output files generated in the 'outputs' folder.")

# ============================================
# 16. QUICK INTERPRETATION NOTES
# ============================================

print("\n--- QUICK INTERPRETATION GUIDE ---")
print("RQ2:")
print("- Use rq2_coefficients.csv to explain which variables are most strongly associated with crime_rate.")
print("- Positive coefficient = higher value associated with higher crime_rate.")
print("- Negative coefficient = higher value associated with lower crime_rate.")
print("\nRQ3:")
print("- Compare MAE, RMSE, and R2 in model_metrics.csv.")
print("- Lower MAE/RMSE is better; higher R2 is better.")
print("- Use paired_ttest_results.csv to discuss whether model differences are statistically significant.")