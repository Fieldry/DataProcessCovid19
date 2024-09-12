# Import packages and define tool functions

import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

src_file = "data/formatted_data.csv"
dst_dir = "processed"


def calculate_data_existing_length(data):
    res = 0
    for i in data:
        if not pd.isna(i):
            res += 1
    return res


def fill_missing_value(data, to_fill_value=0):
    data_len = len(data)
    data_exist_len = calculate_data_existing_length(data)
    if data_len == data_exist_len:
        return data
    elif data_exist_len == 0:
        # data = [to_fill_value for _ in range(data_len)]
        for i in range(data_len):
            data[i] = to_fill_value
        return data
    if pd.isna(data[0]):
        # find the first non-nan value's position
        not_na_pos = 0
        for i in range(data_len):
            if not pd.isna(data[i]):
                not_na_pos = i
                break
        # fill element before the first non-nan value with median
        for i in range(not_na_pos):
            data[i] = to_fill_value
    # fill element after the first non-nan value
    for i in range(1, data_len):
        if pd.isna(data[i]):
            data[i] = data[i - 1]
    return data


def forward_fill_pipeline(
    df: pd.DataFrame,
    default_fill: pd.DataFrame,
    demographic_features: list[str],
    labtest_features: list[str],
    target_features: list[str],
):
    grouped = df.groupby("PatientID")

    all_x = []
    all_y = []
    all_pid = []

    for name, group in grouped:
        sorted_group = group.sort_values(by=["RecordTime"], ascending=True)
        patient_x = []
        patient_y = []

        for f in ["Age"] + labtest_features:
            to_fill_value = default_fill[f]
            # take median patient as the default to-fill missing value
            fill_missing_value(sorted_group[f].values, to_fill_value)

        for _, v in sorted_group.iterrows():
            y = []
            for f in target_features:
                y.append(v[f])
            patient_y.append([v["Outcome"], v["LOS"]])
            x = []
            for f in demographic_features + labtest_features:
                x.append(v[f])
            patient_x.append(x)
        all_x.append(patient_x)
        all_y.append(patient_y)
        all_pid.append(name)
    return all_x, all_y, all_pid


def filter_outlier(element):
    if pd.isna(element):
        return 0
    elif np.abs(float(element)) > 1e4:
        return 0
    else:
        return element


def normalize_dataframe(train_df, val_df, test_df, normalize_features):
    # Calculate the quantiles
    q_low = train_df[normalize_features].quantile(0.05)
    q_high = train_df[normalize_features].quantile(0.95)

    # Filter the DataFrame based on the quantiles
    filtered_df = train_df[
        (train_df[normalize_features] > q_low) & (train_df[normalize_features] < q_high)
    ]

    # Calculate the mean and standard deviation and median of the filtered data, also the default fill value
    train_mean = filtered_df[normalize_features].mean()
    train_std = filtered_df[normalize_features].std()
    train_median = filtered_df[normalize_features].median()
    default_fill: pd.DataFrame = (train_median - train_mean) / (train_std + 1e-12)

    # LOS info
    los_info = {
        "los_mean": train_mean["LOS"].item(),
        "los_std": train_std["LOS"].item(),
        "los_median": train_median["LOS"].item(),
    }

    # Calculate large los and threshold (optional, designed for covid-19 benchmark)
    los_array = train_df.groupby("PatientID")["LOS"].max().values
    los_p95 = np.percentile(los_array, 95)
    los_p5 = np.percentile(los_array, 5)
    filtered_los = los_array[(los_array >= los_p5) & (los_array <= los_p95)]
    los_info.update(
        {"large_los": los_p95.item(), "threshold": filtered_los.mean().item() * 0.5}
    )

    # Z-score normalize the train, val, and test sets with train_mean and train_std
    train_df.loc[:, normalize_features] = (
        train_df.loc[:, normalize_features] - train_mean
    ) / (train_std + 1e-12)
    val_df.loc[:, normalize_features] = (
        val_df.loc[:, normalize_features] - train_mean
    ) / (train_std + 1e-12)
    test_df.loc[:, normalize_features] = (
        test_df.loc[:, normalize_features] - train_mean
    ) / (train_std + 1e-12)

    train_df.loc[:, normalize_features] = train_df.loc[:, normalize_features].applymap(
        filter_outlier
    )
    val_df.loc[:, normalize_features] = val_df.loc[:, normalize_features].applymap(
        filter_outlier
    )
    test_df.loc[:, normalize_features] = test_df.loc[:, normalize_features].applymap(
        filter_outlier
    )

    return train_df, val_df, test_df, default_fill, los_info, train_mean, train_std


# Read in data
df = pd.read_csv(src_file)

basic_records = ["PatientID", "RecordTime", "AdmissionTime", "DischargeTime"]
demographic_features = ["Gender", "Age"]
target_features = ["Outcome", "LOS"]
labtest_features = list(set(df.columns) - set(basic_records + demographic_features))
seed = 42
num_folds = 10

# Group the dataframe by patient ID
grouped = df.groupby("PatientID")

# Split the patient IDs into train/val/test sets
patients = np.array(list(grouped.groups.keys()))
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)

for fold, (train_val_index, test_index) in enumerate(
    kf.split(patients, df.groupby("PatientID")["Outcome"].first())
):
    # Get the train/val/test patient IDs for the current fold
    train_val_patients, test_patients = patients[train_val_index], patients[test_index]

    # Split the train_val_patients into train/val sets
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=1 / (num_folds - 1),
        random_state=seed,
        stratify=df[df["PatientID"].isin(train_val_patients)]
        .groupby("PatientID")["Outcome"]
        .first(),
    )

    # Create train, val, and test dataframes for the current fold
    train_df = df[df["PatientID"].isin(train_patients)]
    val_df = df[df["PatientID"].isin(val_patients)]
    test_df = df[df["PatientID"].isin(test_patients)]

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    fold_dir = os.path.join(dst_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Calculate the mean and std of the train set (include age, lab test features, and LOS) on the data in 5% to 95% quantile range
    normalize_features = ["Age"] + labtest_features + ["LOS"]

    # Normalize data
    train_df, val_df, test_df, default_fill, los_info, train_mean, train_std = (
        normalize_dataframe(train_df, val_df, test_df, normalize_features)
    )

    # Drop rows if all features are recorded NaN
    train_df = train_df.dropna(axis=0, how="all", subset=normalize_features)
    val_df = val_df.dropna(axis=0, how="all", subset=normalize_features)
    test_df = test_df.dropna(axis=0, how="all", subset=normalize_features)

    # Forward Imputation after grouped by PatientID
    # Notice: if a patient has never done certain lab test, the imputed value will be the median value calculated from train set
    train_x, train_y, train_pid = forward_fill_pipeline(
        train_df, default_fill, demographic_features, labtest_features
    )
    val_x, val_y, val_pid = forward_fill_pipeline(
        val_df, default_fill, demographic_features, labtest_features
    )
    test_x, test_y, test_pid = forward_fill_pipeline(
        test_df, default_fill, demographic_features, labtest_features
    )

    # Save the imputed dataset to pickle file
    pd.to_pickle(train_x, os.path.join(fold_dir, "train_x.pkl"))
    pd.to_pickle(train_y, os.path.join(fold_dir, "train_y.pkl"))
    pd.to_pickle(train_pid, os.path.join(fold_dir, "train_pid.pkl"))
    pd.to_pickle(val_x, os.path.join(fold_dir, "val_x.pkl"))
    pd.to_pickle(val_y, os.path.join(fold_dir, "val_y.pkl"))
    pd.to_pickle(val_pid, os.path.join(fold_dir, "val_pid.pkl"))
    pd.to_pickle(test_x, os.path.join(fold_dir, "test_x.pkl"))
    pd.to_pickle(test_y, os.path.join(fold_dir, "test_y.pkl"))
    pd.to_pickle(test_pid, os.path.join(fold_dir, "test_pid.pkl"))
    pd.to_pickle(los_info, os.path.join(fold_dir, "los_info.pkl"))
