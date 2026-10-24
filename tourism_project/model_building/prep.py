from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "tourism_project/data/tourism.csv"
OUT_DIR = ROOT / "model_building"
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
required = {"CustomerID", TARGET}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
df = df.drop_duplicates().copy()
# Standardize known inconsistent category labels before splitting.
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype("string").str.strip().replace({
        "Fe Male": "Female", "Fe-Male": "Female", "FEMALE": "Female", "MALE": "Male"
    })
if "MaritalStatus" in df.columns:
    df["MaritalStatus"] = df["MaritalStatus"].astype("string").str.strip().replace({
        "Unmarried": "Single", "unmarried": "Single", "SINGLE": "Single"
    })
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df[df[TARGET].isin([0, 1])].copy()
# CustomerID is an identifier, not a predictive feature.
X = df.drop(columns=[TARGET, "CustomerID"], errors="ignore")
y = df[TARGET].astype(int)
# Convert obvious numeric columns to numeric where possible.
for col in X.columns:
    if col in ["Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
               "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
               "Passport", "PitchSatisfactionScore", "OwnCar",
               "NumberOfChildrenVisiting", "MonthlyIncome"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
X_train.to_csv(OUT_DIR / "Xtrain.csv", index=False)
X_test.to_csv(OUT_DIR / "Xtest.csv", index=False)
y_train.to_frame(TARGET).to_csv(OUT_DIR / "ytrain.csv", index=False)
y_test.to_frame(TARGET).to_csv(OUT_DIR / "ytest.csv", index=False)
print(f"Rows after cleaning: {len(df)}")
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
print("Target distribution:")
print(y.value_counts(normalize=True).rename("proportion").round(3))
