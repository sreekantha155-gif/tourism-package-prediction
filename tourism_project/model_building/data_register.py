from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tourism.csv"
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
raw_shape = df.shape
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
missing_columns = sorted(set(EXPECTED_COLUMNS) - set(df.columns))
if missing_columns:
    raise ValueError(f"Dataset is missing expected columns: {missing_columns}")
if df.empty:
    raise ValueError("Dataset contains no rows.")
if df["CustomerID"].isna().any():
    raise ValueError("CustomerID contains missing values.")
if df["CustomerID"].duplicated().any():
    print(f"WARNING: duplicate CustomerID count: {int(df['CustomerID'].duplicated().sum())}")
if df["ProdTaken"].isna().any():
    print(f"WARNING: missing target labels: {int(df['ProdTaken'].isna().sum())}")
if not df["ProdTaken"].dropna().isin([0, 1]).all():
    raise ValueError("ProdTaken must contain binary labels 0 and 1 only.")
print("DATASET REGISTRATION AND VALIDATION SUCCESSFUL")
print(f"Raw shape: {raw_shape[0]:,} rows x {raw_shape[1]} columns")
print(f"Validated shape (excluding index-like columns): {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Duplicate rows: {int(df.duplicated().sum())}")
print("Missing values by column:")
print(df.isna().sum().to_string())
print("Target distribution (count):")
print(df["ProdTaken"].value_counts(dropna=False).sort_index().to_string())
print("Target distribution (proportion):")
print(df["ProdTaken"].value_counts(normalize=True, dropna=False).sort_index().round(3).to_string())
print("Numeric summary:")
print(df.select_dtypes(include="number").describe().round(2).to_string())
