# =====================================================
# NSQIP 2022 Dataset Cleaning + Correlation Analysis
# =====================================================

import os
import numpy as np
import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, pointbiserialr, chi2_contingency
from matplotlib.patches import Patch

# =====================================================
# Load and Clean Raw Dataset (.sav)
# =====================================================
file_2022 = r"D:\NSQIP2022.sav"

if not os.path.exists(file_2022):
    raise FileNotFoundError(f"File not found: {file_2022}")

try:
    df_2022, meta_2022 = pyreadstat.read_sav(file_2022, encoding="utf-8")
except pyreadstat._readstat_parser.ReadstatError:
    print("UTF-8 failed. Retrying with Windows-1252 encoding...")
    df_2022, meta_2022 = pyreadstat.read_sav(file_2022, encoding="windows-1252")

print(f" Loaded NSQIP 2022 dataset. Shape: {df_2022.shape}")

# Replace invalid entries
df_2022.replace([-99, -999, " ", "", "NA", "NaN", "nan"], np.nan, inplace=True)

# Drop columns with >30% NaN
missing_percent = df_2022.isna().mean() * 100
cols_to_drop = [col for col in missing_percent.index if missing_percent[col] > 30 and col != "DOpertoD"]
df_2022_clean = df_2022.drop(columns=cols_to_drop)
print(f"Dropped {len(cols_to_drop)} columns with >30% NaN values.")

# Drop columns with >30% 'NULL'
null_percent = (df_2022_clean == "NULL").sum() / len(df_2022_clean) * 100
cols_with_many_nulls = null_percent[null_percent > 30].index.tolist()
df_2022_clean.drop(columns=cols_with_many_nulls, inplace=True)
print(f"Dropped {len(cols_with_many_nulls)} columns with >30% 'NULL' values.")
print(f"Shape after cleaning: {df_2022_clean.shape}")

# --- Filter by age >= 50 ---
age_col = next((c for c in df_2022_clean.columns if c.lower().startswith("age")), None)
if not age_col:
    raise KeyError("No column starting with 'Age' found.")

df_2022_clean[age_col] = (
    df_2022_clean[age_col]
    .astype(str)
    .str.replace("+", "", regex=False)
    .str.extract(r"(\d+)", expand=False)
    .astype(float)
)

rows_before = len(df_2022_clean)
df_2022_clean = df_2022_clean[df_2022_clean[age_col] >= 50]
print(f"Dropped {rows_before - len(df_2022_clean)} rows with {age_col} < 50. New shape: {df_2022_clean.shape}")

# --- Remove redundant columns ---
cols_to_remove = [
    "POSTOP_COVID", "RETURNOR", "REOPERATION1", "REOPERATION2", "READMISSION1",
    "READMISSION2", "READMISSION3", "READMISSION4", "READMISSION5",
    "MORTPROB", "MORBPROB", "DISFXNSTAT", "BLEED_UNITS_TOT", "PUFYEAR"
]
existing_remove = [c for c in cols_to_remove if c in df_2022_clean.columns]
df_2022_clean.drop(columns=existing_remove, inplace=True)
print(f"Dropped {len(existing_remove)} redundant columns.")

# --- Create 'Complication' variable ---
complication_cols = ["CDARREST", "DOpertoD", "DCDMI", "CNSCVA", "CDMI"]
existing_comp = [c for c in complication_cols if c in df_2022_clean.columns]

if not existing_comp:
    raise KeyError("No complication columns found.")
print(f"Using complication columns: {existing_comp}")

if "DOpertoD" in df_2022_clean.columns:
    df_2022_clean["DOpertoD"] = df_2022_clean["DOpertoD"].replace(-99, 0)

df_2022_clean["Complication"] = df_2022_clean[existing_comp].apply(
    lambda r: 1 if any(pd.to_numeric(r, errors="coerce").fillna(0) != 0) else 0, axis=1
)
print("Created 'Complication' column.")
print(df_2022_clean["Complication"].value_counts())

df_2022_clean.drop(columns=existing_comp, inplace=True, errors="ignore")
for c in ["AdmYR", "OperYR"]:
    if c in df_2022_clean.columns:
        df_2022_clean.drop(columns=c, inplace=True)

# Save cleaned dataset
output_dir = r"D:\Files\AUB Semesters\Semesters\Year 2\Cleaned_datasets"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "df_2022_clean.csv")
df_2022_clean.to_csv(output_path, index=False)
print(f"Cleaned dataset saved at: {output_path}")

# =====================================================
# Load Cleaned Dataset and Handle Missing Values
# =====================================================
file_path = output_path
df = pd.read_csv(file_path)
print(f" Loaded cleaned dataset. Shape: {df.shape}")

df.replace(["", " ", "NULL", "NaN", "nan", "NA"], pd.NA, inplace=True)
if "PRSEPSIS" in df.columns:
    none_count = (df["PRSEPSIS"] == "None").sum()
    print(f"  Keeping {none_count} rows where PRSEPSIS == 'None'.")

missing_counts = df.isna().sum()
missing_summary = (
    pd.DataFrame({
        "Missing_Count": missing_counts,
        "Missing_%": (missing_counts / len(df) * 100).round(2)
    })
    .sort_values(by="Missing_%", ascending=False)
)
print("\n Missing values (top 20):")
print(missing_summary.head(20))

# =====================================================
# Correlation Analysis (Feature Relationships)
# =====================================================
def classify_columns(df):
    categorical, continuous = [], []
    for col in df.columns:
        if df[col].dtype in ["object", "bool"] or df[col].nunique() < 10:
            categorical.append(col)
        else:
            continuous.append(col)
    return categorical, continuous

def cramers_v(x, y):
    cm = pd.crosstab(x, y)
    chi2 = chi2_contingency(cm, correction=False)[0]
    n = cm.sum().sum()
    phi2 = chi2 / n
    r, k = cm.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1))/(n - 1))
    rcorr = r - ((r - 1)**2)/(n - 1)
    kcorr = k - ((k - 1)**2)/(n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

def pearson_corr(x, y):
    try:
        return abs(pearsonr(x, y)[0])
    except Exception:
        return np.nan

def point_biserial_corr(cat, cont):
    try:
        return abs(pointbiserialr(cat, cont)[0])
    except Exception:
        return np.nan

categorical_cols, continuous_cols = classify_columns(df)
print(f"\n🔹 Categorical: {len(categorical_cols)} | 🔹 Continuous: {len(continuous_cols)}")

high_corr = []
for i, c1 in enumerate(continuous_cols):
    for c2 in continuous_cols[i + 1:]:
        r = pearson_corr(df[c1], df[c2])
        if r >= 0.6:
            high_corr.append((c1, c2, r, "Pearson"))

for cat in categorical_cols:
    for cont in continuous_cols:
        r = point_biserial_corr(df[cat].astype(float, errors="ignore"), df[cont])
        if r >= 0.6:
            high_corr.append((cat, cont, r, "Point-Biserial"))

for i, c1 in enumerate(categorical_cols):
    for c2 in categorical_cols[i + 1:]:
        r = cramers_v(df[c1], df[c2])
        if r >= 0.6:
            high_corr.append((c1, c2, r, "Cramér’s V"))

corr_df = pd.DataFrame(high_corr, columns=["Feature_1", "Feature_2", "Correlation", "Method"])
if corr_df.empty:
    print("\n No strong correlations (|r| ≥ 0.6).")
else:
    print("\n Highly correlated pairs:")
    print(corr_df.sort_values(by="Correlation", ascending=False))

# Drop redundant columns
to_drop = [c for c in ["DPRWBC", "DPRBUN", "DPRHCT", "DPRPLATE", "CaseID", "INOUT", "DPRCREAT", "SURGSPEC"] if c in df.columns]
df.drop(columns=to_drop, inplace=True, errors="ignore")
print(f"\n🧹 Dropped {len(to_drop)} redundant columns: {to_drop}")
print(f"Updated dataset shape: {df.shape}")

# =====================================================
#  Feature-to-Outcome Correlation Visualization
# =====================================================
target = "Complication"
if target not in df.columns:
    raise KeyError(f"Column '{target}' not found.")

categorical_cols = [c for c in df.columns if df[c].dtype in ["object", "bool"] or df[c].nunique() < 10 and c != target]
continuous_cols = [c for c in df.columns if c not in categorical_cols + [target]]

corrs = []
for col in categorical_cols:
    try:
        corrs.append((col, cramers_v(df[col], df[target]), "Categorical"))
    except Exception:
        pass

for col in continuous_cols:
    try:
        corrs.append((col, abs(pointbiserialr(df[col], df[target])[0]), "Continuous"))
    except Exception:
        pass

corr_df = pd.DataFrame(corrs, columns=["Feature", "Correlation", "Type"]).dropna().sort_values(by="Correlation", ascending=False)

plt.figure(figsize=(18, 10))
colors = corr_df["Type"].map({"Categorical": "#0072B2", "Continuous": "#009E73"})
bars = plt.bar(corr_df["Feature"], corr_df["Correlation"], color=colors, edgecolor="black")
plt.xticks(rotation=75, ha="right", fontsize=9)
plt.ylabel("Correlation Coefficient (0–1)")
plt.title("Feature–Outcome Correlation with 'Complication'")
plt.grid(axis="y", linestyle="--", alpha=0.5)

for bar, val in zip(bars, corr_df["Correlation"]):
    plt.text(bar.get_x() + bar.get_width()/2, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=8)

plt.legend(handles=[Patch(facecolor="#0072B2", label="Categorical"), Patch(facecolor="#009E73", label="Continuous")], loc="upper right")
plt.tight_layout()
plt.show()