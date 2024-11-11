import os

import pandas as pd

src_file = "datasets/tjh/raw/time_series_375_prerpocess_en.xlsx"
dst_dir = "datasets/tjh/raw"
standard_data = "standard_data.csv"
formatted_data = "formatted_data.csv"

# Read in original data
df = pd.read_excel(src_file, engine="openpyxl")

# Rename columns
df = df.rename(
    columns={
        "PATIENT_ID": "PatientID",
        "outcome": "Outcome",
        "gender": "Gender",
        "age": "Age",
        "RE_DATE": "RecordTime",
        "Admission time": "AdmissionTime",
        "Discharge time": "DischargeTime",
    }
)

# Fill PatientID column
df["PatientID"] = df["PatientID"].ffill()

# Change the order of columns
basic_records = ["PatientID", "RecordTime", "AdmissionTime", "DischargeTime", "Outcome"]
demographic_features = ["Gender", "Age"]
labtest_features = list(
    set(df.columns) - set(basic_records + demographic_features)
).sort()
df = df[basic_records + demographic_features + labtest_features]

# Export standardized table
os.makedirs(dst_dir, exist_ok=True)
df.to_csv(os.path.join(dst_dir, standard_data), index=False)

print("Finishing standardizing data!")

# Gender transformation: 1--male, 0--female
df["Sex"].replace(2, 0, inplace=True)

# Reserve Y/M/D format for `RecordTime`, `AdmissionTime` and `DischargeTime` columns
df["RecordTime"] = df["RecordTime"].dt.strftime("%Y/%m/%d")
df["DischargeTime"] = df["DischargeTime"].dt.strftime("%Y/%m/%d")
df["AdmissionTime"] = df["AdmissionTime"].dt.strftime("%Y/%m/%d")

# Exclude patients with missing labels in these columns
df = df.dropna(
    subset=["PatientID", "RecordTime", "AdmissionTime", "DischargeTime"], how="any"
)

# Drop columns whose values are all NaN ('2019-nCoV nucleic acid detection')
df = df.drop(columns=["2019-nCoV nucleic acid detection"])

# Merge data by PatientID and RecordTime
df = df.groupby(
    ["PatientID", "RecordTime", "AdmissionTime", "DischargeTime"],
    dropna=True,
    as_index=False,
).mean()

# Calculate LOS (Length of Stay) in days and insert it after the column `Outcome`
df.insert(
    5,
    "LOS",
    (pd.to_datetime(df["DischargeTime"]) - pd.to_datetime(df["RecordTime"])).dt.days,
)

# Notice: Set negative LOS values to 0
df["LOS"] = df["LOS"].apply(lambda x: 0 if x < 0 else x)

# Export formatted table
os.makedirs(dst_dir, exist_ok=True)
df.to_csv(os.path.join(dst_dir, formatted_data), index=False)

print("Finishing formatting data!")
