import os

import pandas as pd

src_file = "data/standard_data.csv"
dst_dir = "data"
dst_file = "formatted_data.csv"

# Read in data
df = pd.read_csv(src_file)

# Format data values

# Gender transformation: 1--male, 0--female
df["Sex"].replace(2, 0, inplace=True)

# Reserve y-m-d precision for `RecordTime`, `AdmissionTime` and `DischargeTime` columns
df["RecordTime"] = df["RecordTime"].dt.strftime("%Y-%m-%d")
df["DischargeTime"] = df["DischargeTime"].dt.strftime("%Y-%m-%d")
df["AdmissionTime"] = df["AdmissionTime"].dt.strftime("%Y-%m-%d")

# Clean data

# Exclude patients with missing labels
df = df.dropna(subset=["PatientID", "RecordTime", "DischargeTime"], how="any")

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
df.to_csv(os.path.join(dst_dir, dst_file), index=False)

print("Data formatting is done!")
