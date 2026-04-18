# =========================================================
# PRT564 - Assessment 2
# FINAL BEST VERSION (Merged from both approaches)
# Focus: RQ2 + RQ3
# =========================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================================
# 1. FILE PATHS
# =========================================================

POP_FILE = r"C:\Users\khuep\Downloads\YEN\UNITS\PRT564\nt-government-regions_1986-to-2025.xlsx"
CRIME_NOV_2023_FILE = r"C:\Users\khuep\Downloads\YEN\UNITS\PRT564\nt_crime_statistics_nov_2023.csv"
CRIME_DEC_2025_FILE = r"C:\Users\khuep\Downloads\YEN\UNITS\PRT564\nt_crime_statistics_dec_2025.csv"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# 2. HELPER FUNCTIONS
# =========================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
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
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return np.where((b == 0) | np.isnan(b), np.nan, a / b)


def evaluate_regression(y_true, y_pred, model_name="Model"):
    return {
        "model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }


def standardise_yes_no(series):
    """
    Convert yes/no values to 1/0.
    '-' and blanks are treated as missing.
    """
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "yes": 1, "y": 1, "true": 1, "1": 1,
            "no": 0, "n": 0, "false": 0, "0": 0,
            "-": np.nan,
            "nan": np.nan,
            "none": np.nan,
            "": np.nan
        })
    )


def parse_age_bounds(age_text: str):
    if pd.isna(age_text):
        return (np.nan, np.nan)

    text = str(age_text).strip().lower()

    if "+" in text:
        digits = re.findall(r"\d+", text)
        if digits:
            low = int(digits[0])
            return low, 120
        return (np.nan, np.nan)

    if "-" in text:
        digits = re.findall(r"\d+", text)
        if digits:
            low = int(digits[0])
            return low, low + 4
        return (np.nan, np.nan)

    return (np.nan, np.nan)


def is_young_group(age_text: str) -> int:
    low, high = parse_age_bounds(age_text)
    if pd.isna(low) or pd.isna(high):
        return 0
    return int(high >= 15 and low <= 34)


def harmonise_offence_category(series: pd.Series) -> pd.Series:
    mapping = {
        "01 Homicide": "Homicide and related offences",
        "02 Assault": "Assault and related offences",
        "03 Sexual assault and related offences": "Sexual assault and related offences",
        "04 Dangerous or negligent acts endangering persons": "Dangerous or negligent acts endangering persons",
        "05 Abduction, harassment and other offences against the person": "Abduction - harassment and other offences against the person",
        "06 Robbery, extortion and related offences": "Robbery, extortion and related offences",
        "07 Unlawful entry with intent/burglary, break and enter": "Unlawful entry with intent/burglary, break and enter",
        "08 Theft and related offences": "Theft and related offences",
        "09 Motor vehicle theft and related offences": "Motor vehicle theft and related offences",
        "10 Fraud, deception and related offences": "Fraud, deception and related offences",
        "11 Illicit drug offences": "Illicit drug offences",
        "12 Property damage and environmental pollution": "Property damage and environmental pollution",
        "13 Public order offences": "Public order offences",
        "14 Traffic and vehicle regulatory offences": "Traffic and vehicle regulatory offences",
        "15 Offences against justice procedures, govt security and govt operations": "Offences against justice procedures, government security and government operations",
        "16 Miscellaneous offences": "Miscellaneous offences"
    }
    return series.astype(str).str.strip().replace(mapping)


# =========================================================
# 3. LOAD DATA
# =========================================================

pop_df = pd.read_excel(POP_FILE)
crime_nov_df = pd.read_csv(CRIME_NOV_2023_FILE)
crime_dec_df = pd.read_csv(CRIME_DEC_2025_FILE)

pop_df = clean_column_names(pop_df)
crime_nov_df = clean_column_names(crime_nov_df)
crime_dec_df = clean_column_names(crime_dec_df)

print("Population columns:", pop_df.columns.tolist())
print("Crime Nov 2023 columns:", crime_nov_df.columns.tolist())
print("Crime Dec 2025 columns:", crime_dec_df.columns.tolist())


# =========================================================
# 4. BASIC CLEANING
# =========================================================

# Numeric columns
for col in ["year", "population"]:
    if col in pop_df.columns:
        pop_df[col] = pd.to_numeric(pop_df[col], errors="coerce")

for col in ["year", "month_number", "number_of_offences"]:
    if col in crime_nov_df.columns:
        crime_nov_df[col] = pd.to_numeric(crime_nov_df[col], errors="coerce")
    if col in crime_dec_df.columns:
        crime_dec_df[col] = pd.to_numeric(crime_dec_df[col], errors="coerce")

# Standardise text columns
for col in ["status", "sex", "age_group", "aboriginal_status", "region"]:
    if col in pop_df.columns:
        pop_df[col] = pop_df[col].astype(str).str.strip()

for col in ["offence_category", "offence_type", "alcohol_involvement",
            "dv_involvement", "reporting_region", "statistical_area_2"]:
    if col in crime_nov_df.columns:
        crime_nov_df[col] = crime_nov_df[col].astype(str).str.strip()
    if col in crime_dec_df.columns:
        crime_dec_df[col] = crime_dec_df[col].astype(str).str.strip()

# Harmonise Dec 2025 offence categories
crime_dec_df["offence_category"] = harmonise_offence_category(crime_dec_df["offence_category"])

# Region keys
pop_df["region_key"] = pop_df["region"].astype(str).str.strip().str.lower()
crime_nov_df["region_key"] = crime_nov_df["reporting_region"].astype(str).str.strip().str.lower()
crime_dec_df["region_key"] = crime_dec_df["reporting_region"].astype(str).str.strip().str.lower()

# Harmonise regions
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
crime_nov_df["region_key"] = crime_nov_df["region_key"].replace(REGION_MAP)
crime_dec_df["region_key"] = crime_dec_df["region_key"].replace(REGION_MAP)

# Alcohol / DV
crime_nov_df["alcohol_flag"] = standardise_yes_no(crime_nov_df["alcohol_involvement"])
crime_nov_df["dv_flag"] = standardise_yes_no(crime_nov_df["dv_involvement"])
crime_dec_df["alcohol_flag"] = standardise_yes_no(crime_dec_df["alcohol_involvement"])
crime_dec_df["dv_flag"] = standardise_yes_no(crime_dec_df["dv_involvement"])

# Drop unusable rows
pop_df = pop_df.dropna(subset=["year", "region_key", "population"])
crime_nov_df = crime_nov_df.dropna(subset=["year", "month_number", "region_key", "number_of_offences"])
crime_dec_df = crime_dec_df.dropna(subset=["year", "month_number", "region_key", "number_of_offences"])


# =========================================================
# 5. DATA QUALITY CHECKS
# =========================================================

# 5.1 Overlap check
overlap_check = crime_nov_df.merge(
    crime_dec_df,
    on=["year", "month_number", "reporting_region", "offence_category", "offence_type"],
    how="inner"
)
print("\n=== OVERLAP CHECK ===")
print("Overlapping rows:", len(overlap_check))

# 5.2 Combine crime files
crime_df = pd.concat([crime_nov_df, crime_dec_df], ignore_index=True)

# 5.3 Missing month check
month_check = (
    crime_df.groupby(["year", "month_number"], as_index=False)["number_of_offences"]
    .sum()
    .sort_values(["year", "month_number"])
)

all_year_month = pd.MultiIndex.from_product(
    [sorted(month_check["year"].dropna().unique()), range(1, 13)],
    names=["year", "month_number"]
).to_frame(index=False)

missing_months = all_year_month.merge(
    month_check[["year", "month_number"]],
    on=["year", "month_number"],
    how="left",
    indicator=True
)

missing_months = missing_months.loc[
    missing_months["_merge"] == "left_only",
    ["year", "month_number"]
]

print("\n=== MISSING YEAR-MONTH CHECK ===")
print(missing_months.to_string(index=False))

# 5.4 Alcohol / DV missing rates
alcohol_missing_rate = crime_df["alcohol_flag"].isna().mean() * 100
dv_missing_rate = crime_df["dv_flag"].isna().mean() * 100

print("\n=== MISSING / UNRECORDED FLAG RATES ===")
print(f"Alcohol involvement missing/unrecorded: {alcohol_missing_rate:.2f}%")
print(f"DV involvement missing/unrecorded: {dv_missing_rate:.2f}%")

# 5.5 Align population years to crime years
min_year = crime_df["year"].min()
max_year = crime_df["year"].max()
pop_df = pop_df[(pop_df["year"] >= min_year) & (pop_df["year"] <= max_year)].copy()


# =========================================================
# 6. POPULATION FEATURES
# =========================================================

# Total population: use one consistent slice to reduce double count
# Here: sum by region-year-age_group-aboriginal_status, then sum once
pop_total_base = (
    pop_df.groupby(["region_key", "year", "age_group", "aboriginal_status"], as_index=False)["population"]
    .sum()
)
population_total = (
    pop_total_base.groupby(["region_key", "year"], as_index=False)["population"]
    .sum()
    .rename(columns={"population": "population_total"})
)

# Young ratio
pop_df["is_young_group"] = pop_df["age_group"].apply(is_young_group)
young_base = (
    pop_df.loc[pop_df["is_young_group"] == 1]
    .groupby(["region_key", "year", "age_group", "aboriginal_status"], as_index=False)["population"]
    .sum()
)
young_population = (
    young_base.groupby(["region_key", "year"], as_index=False)["population"]
    .sum()
    .rename(columns={"population": "young_population"})
)

# Male ratio
male_base = (
    pop_df.loc[pop_df["sex"].str.lower() == "male"]
    .groupby(["region_key", "year", "age_group", "aboriginal_status"], as_index=False)["population"]
    .sum()
)
male_population = (
    male_base.groupby(["region_key", "year"], as_index=False)["population"]
    .sum()
    .rename(columns={"population": "male_population"})
)

# Aboriginal ratio
ab_base = (
    pop_df.loc[pop_df["aboriginal_status"].str.lower() == "aboriginal"]
    .groupby(["region_key", "year", "age_group", "sex"], as_index=False)["population"]
    .sum()
)
aboriginal_population = (
    ab_base.groupby(["region_key", "year"], as_index=False)["population"]
    .sum()
    .rename(columns={"population": "aboriginal_population"})
)

pop_features = population_total.merge(young_population, on=["region_key", "year"], how="left")
pop_features = pop_features.merge(male_population, on=["region_key", "year"], how="left")
pop_features = pop_features.merge(aboriginal_population, on=["region_key", "year"], how="left")

for col in ["young_population", "male_population", "aboriginal_population"]:
    pop_features[col] = pop_features[col].fillna(0)

pop_features["young_ratio"] = safe_divide(pop_features["young_population"], pop_features["population_total"])
pop_features["male_ratio"] = safe_divide(pop_features["male_population"], pop_features["population_total"])
pop_features["aboriginal_ratio"] = safe_divide(pop_features["aboriginal_population"], pop_features["population_total"])

pop_features = pop_features.sort_values(["region_key", "year"])
pop_features["population_growth_rate"] = pop_features.groupby("region_key")["population_total"].pct_change()

pop_features["log_population"] = np.log1p(pop_features["population_total"])

pop_features.to_csv(OUTPUT_DIR / "population_features.csv", index=False)


# =========================================================
# 7. CRIME FEATURES
# =========================================================

crime_monthly = (
    crime_df.groupby(["region_key", "year", "month_number"], as_index=False)
    .agg(
        number_of_offences=("number_of_offences", "sum"),
        alcohol_related_offences=("alcohol_flag", lambda x: np.nansum(pd.to_numeric(x, errors="coerce"))),
        dv_related_offences=("dv_flag", lambda x: np.nansum(pd.to_numeric(x, errors="coerce")))
    )
    .sort_values(["region_key", "year", "month_number"])
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

cat_pivot.columns = [
    str(c).strip().lower().replace(" ", "_").replace("-", "_").replace(",", "").replace("/", "_")
    if isinstance(c, str) else c
    for c in cat_pivot.columns
]

offence_cols = [c for c in cat_pivot.columns if c not in ["region_key", "year", "month_number"]]
row_totals = cat_pivot[offence_cols].sum(axis=1)

for col in offence_cols:
    cat_pivot[f"{col}_ratio"] = safe_divide(cat_pivot[col], row_totals)

ratio_cols = ["region_key", "year", "month_number"] + [f"{c}_ratio" for c in offence_cols]
offence_comp = cat_pivot[ratio_cols].copy()

crime_monthly = crime_monthly.merge(
    offence_comp,
    on=["region_key", "year", "month_number"],
    how="left"
)

# Lags + rolling history (best from your code)
crime_monthly["lag_offence_1"] = crime_monthly.groupby("region_key")["number_of_offences"].shift(1)
crime_monthly["lag_offence_2"] = crime_monthly.groupby("region_key")["number_of_offences"].shift(2)
crime_monthly["prev_3mo_avg"] = (
    crime_monthly.groupby("region_key")["number_of_offences"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

crime_monthly.to_csv(OUTPUT_DIR / "crime_monthly_features.csv", index=False)


# =========================================================
# 8. MERGE
# =========================================================

model_df = crime_monthly.merge(
    pop_features,
    on=["region_key", "year"],
    how="left"
)

print("\n=== MERGE QUALITY CHECK ===")
print("Rows in crime_monthly:", len(crime_monthly))
print("Rows in merged model_df:", len(model_df))
print("Rows with missing population after merge:", model_df["population_total"].isna().sum())

# Keep November 2023 as true gap
model_df["crime_rate"] = safe_divide(model_df["number_of_offences"], model_df["population_total"])
model_df["month"] = model_df["month_number"]
model_df["year_num"] = model_df["year"]
model_df["year_index"] = model_df["year"] - model_df["year"].min()

model_df.to_csv(OUTPUT_DIR / "merged_before_missing_handling.csv", index=False)


# =========================================================
# 9. MISSING HANDLING
# =========================================================

for col in ["population_growth_rate", "alcohol_ratio", "dv_ratio",
            "lag_offence_1", "lag_offence_2", "prev_3mo_avg"]:
    if col in model_df.columns:
        model_df[col] = model_df.groupby("region_key")[col].transform(lambda x: x.fillna(x.median()))
        model_df[col] = model_df[col].fillna(model_df[col].median())

ratio_cols_all = [c for c in model_df.columns if c.endswith("_ratio")]
for col in ratio_cols_all:
    model_df[col] = model_df[col].fillna(0)

model_df = model_df.dropna(subset=["population_total", "number_of_offences", "crime_rate"])
model_df.to_csv(OUTPUT_DIR / "merged_model_dataset.csv", index=False)

print("\nFinal modelling dataset saved.")


# =========================================================
# 10. DESCRIPTIVE SUMMARIES
# =========================================================

summary_cols = [
    "number_of_offences", "crime_rate", "population_total",
    "young_ratio", "male_ratio", "aboriginal_ratio",
    "population_growth_rate", "alcohol_ratio", "dv_ratio"
]
summary_cols = [c for c in summary_cols if c in model_df.columns]

summary_stats = model_df[summary_cols].describe().T
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
# 11. EDA (6 best charts)
# =========================================================

# 11.1 Total crime over time -> RQ3
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

# 11.2 Population vs average crime rate -> RQ2
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

# 11.3 Crime rate distribution by region -> RQ2
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

# 11.4 Monthly seasonality -> RQ3
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

# 11.5 Correlation heatmap -> RQ2 + RQ3
corr_cols = [
    "number_of_offences", "crime_rate", "population_total", "log_population",
    "young_ratio", "male_ratio", "aboriginal_ratio",
    "population_growth_rate", "alcohol_ratio", "dv_ratio",
    "lag_offence_1", "lag_offence_2", "prev_3mo_avg"
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

# 11.6 Lagged offence vs current offence -> RQ3
plt.figure(figsize=(8, 5))
plt.scatter(model_df["lag_offence_1"], model_df["number_of_offences"])
plt.title("Lagged Offence Count vs Current Offence Count")
plt.xlabel("Lagged Offence Count (t-1)")
plt.ylabel("Current Offence Count")
plt.tight_layout()
plt.show()


# =========================================================
# 12. RQ2 - EXPLANATORY REGRESSION
# =========================================================

rq2_features = [
    "log_population",
    "young_ratio",
    "male_ratio",
    "aboriginal_ratio",
    "population_growth_rate",
    "alcohol_ratio",
    "dv_ratio",
    "year_num"
]
rq2_features = [c for c in rq2_features if c in model_df.columns]

rq2_df = model_df.dropna(subset=rq2_features + ["crime_rate"]).copy()
rq2_df = rq2_df.sort_values(["year", "month"])

split_index_rq2 = int(len(rq2_df) * 0.8)
train_rq2 = rq2_df.iloc[:split_index_rq2]
test_rq2 = rq2_df.iloc[split_index_rq2:]

X_train_rq2 = train_rq2[rq2_features]
y_train_rq2 = train_rq2["crime_rate"]
X_test_rq2 = test_rq2[rq2_features]
y_test_rq2 = test_rq2["crime_rate"]

rq2_model = LinearRegression()
rq2_model.fit(X_train_rq2, y_train_rq2)
rq2_pred = rq2_model.predict(X_test_rq2)

rq2_metrics = evaluate_regression(y_test_rq2, rq2_pred, "RQ2_MultipleLinearRegression")
print("\nRQ2 metrics:", rq2_metrics)

rq2_coef_df = pd.DataFrame({
    "feature": X_train_rq2.columns,
    "coefficient": rq2_model.coef_
}).sort_values("coefficient", key=np.abs, ascending=False)

rq2_coef_df.to_csv(OUTPUT_DIR / "rq2_coefficients.csv", index=False)


# =========================================================
# 13. RQ3 - PREDICTIVE REGRESSION
# =========================================================

# Include strongest features from both versions
offence_ratio_cols = [
    c for c in model_df.columns
    if c.endswith("_ratio") and c not in [
        "young_ratio", "male_ratio", "aboriginal_ratio", "alcohol_ratio", "dv_ratio"
    ]
]

rq3_features = [
    "log_population",
    "young_ratio",
    "male_ratio",
    "aboriginal_ratio",
    "population_growth_rate",
    "alcohol_ratio",
    "dv_ratio",
    "lag_offence_1",
    "lag_offence_2",
    "prev_3mo_avg",
    "month",
    "year_index"
] + offence_ratio_cols[:5]

rq3_features = [c for c in rq3_features if c in model_df.columns]

rq3_df = model_df.dropna(subset=rq3_features + ["number_of_offences"]).copy()
rq3_df = rq3_df.sort_values(["year", "month"])

# Time-aware split (important for prediction)
split_index_rq3 = int(len(rq3_df) * 0.8)
train_rq3 = rq3_df.iloc[:split_index_rq3]
test_rq3 = rq3_df.iloc[split_index_rq3:]

X_train_rq3 = train_rq3[rq3_features]
y_train_rq3 = train_rq3["number_of_offences"]
X_test_rq3 = test_rq3[rq3_features]
y_test_rq3 = test_rq3["number_of_offences"]

# Models
rq3_lr = LinearRegression()
rq3_lr.fit(X_train_rq3, y_train_rq3)
rq3_lr_pred = rq3_lr.predict(X_test_rq3)
rq3_lr_metrics = evaluate_regression(y_test_rq3, rq3_lr_pred, "RQ3_LinearRegression")

rq3_ridge = Ridge(alpha=1.0)
rq3_ridge.fit(X_train_rq3, y_train_rq3)
rq3_ridge_pred = rq3_ridge.predict(X_test_rq3)
rq3_ridge_metrics = evaluate_regression(y_test_rq3, rq3_ridge_pred, "RQ3_Ridge")

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

rf_importance_df = pd.DataFrame({
    "feature": X_train_rq3.columns,
    "importance": rq3_rf.feature_importances_
}).sort_values("importance", ascending=False)

rf_importance_df.to_csv(OUTPUT_DIR / "rq3_rf_feature_importance.csv", index=False)


# =========================================================
# 14. REGRESSION VISUALISATIONS
# =========================================================

# RQ2 coefficients
plt.figure(figsize=(10, 5))
plt.bar(rq2_coef_df["feature"], rq2_coef_df["coefficient"])
plt.xticks(rotation=45, ha="right")
plt.title("Regression Coefficients (RQ2)")
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.tight_layout()
plt.show()

# RQ3 actual vs predicted - Linear
plt.figure(figsize=(8, 5))
plt.scatter(y_test_rq3, rq3_lr_pred, alpha=0.6)
plt.plot(
    [y_test_rq3.min(), y_test_rq3.max()],
    [y_test_rq3.min(), y_test_rq3.max()],
    linestyle="--"
)
plt.title("Actual vs Predicted (Linear Regression)")
plt.xlabel("Actual Offence Count")
plt.ylabel("Predicted Offence Count")
plt.tight_layout()
plt.show()

# RQ3 actual vs predicted - Random Forest
plt.figure(figsize=(8, 5))
plt.scatter(y_test_rq3, rq3_rf_pred, alpha=0.6)
plt.plot(
    [y_test_rq3.min(), y_test_rq3.max()],
    [y_test_rq3.min(), y_test_rq3.max()],
    linestyle="--"
)
plt.title("Actual vs Predicted (Random Forest)")
plt.xlabel("Actual Offence Count")
plt.ylabel("Predicted Offence Count")
plt.tight_layout()
plt.show()

# Residual plot
residuals = y_test_rq3 - rq3_lr_pred
plt.figure(figsize=(8, 5))
plt.scatter(rq3_lr_pred, residuals, alpha=0.6)
plt.axhline(y=0, linestyle="--")
plt.title("Residual Plot (Linear Regression)")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.tight_layout()
plt.show()

# Feature importance
plt.figure(figsize=(10, 5))
plt.bar(rf_importance_df["feature"], rf_importance_df["importance"])
plt.xticks(rotation=45, ha="right")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()


# =========================================================
# 15. MODEL METRICS TABLE
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

# MAE comparison
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
# 16. STATISTICAL TESTING
# =========================================================

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
# 17. SAVE EXTRA OUTPUTS
# =========================================================

model_df.head(50).to_csv(OUTPUT_DIR / "model_df_sample.csv", index=False)
rq2_coef_df.head(10).to_csv(OUTPUT_DIR / "rq2_top_coefficients.csv", index=False)
rf_importance_df.head(10).to_csv(OUTPUT_DIR / "rq3_top_rf_importance.csv", index=False)

print("\n================ FINAL SUMMARY ================")
print("Files created in:", OUTPUT_DIR.resolve())
print("- population_features.csv")
print("- crime_monthly_features.csv")
print("- merged_before_missing_handling.csv")
print("- merged_model_dataset.csv")
print("- descriptive_summary.csv")
print("- region_summary.csv")
print("- rq2_coefficients.csv")
print("- rq3_rf_feature_importance.csv")
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
