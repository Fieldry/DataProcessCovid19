import os

import pandas as pd

src_file = "time_series_375_prerpocess_en.xlsx"
dst_dir = "data"
dst_file = "standard_data.csv"

# Read in original data
df = pd.read_excel(src_file)

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
labtest_features = list(set(df.columns) - set(basic_records + demographic_features))
df = df[basic_records + demographic_features + labtest_features]

# Export standardized table
os.makedirs(dst_dir, exist_ok=True)
df.to_csv(os.path.join(dst_dir, dst_file), index=False)

print("Data preparation is done!")
