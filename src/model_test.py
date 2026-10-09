import pandas as pd

df = pd.read_parquet(
    "data/processed/01_M01_sensor_ttf.parquet"
)

# print(df.shape)
# print(df.columns.tolist())
# print(df.head())

# print(df.columns.tolist())

# print(df.iloc[:, -3:].head(10))
# print(df.iloc[:, -3:].describe())

###############
# ttf_cols = [
#     "TTF_FlowCool Pressure Dropped Below Limit",
#     "TTF_Flowcool Pressure Too High Check Flowcool Pump",
#     "TTF_Flowcool leak"
# ]

# fault_names = {
#     "TTF_FlowCool Pressure Dropped Below Limit":
#         "Pressure Dropped Below Limit",

#     "TTF_Flowcool Pressure Too High Check Flowcool Pump":
#         "Pressure Too High",

#     "TTF_Flowcool leak":
#         "Flowcool Leak"
# }

# # 가장 가까운 고장까지 남은 시간
# df["min_ttf"] = df[ttf_cols].min(axis=1, skipna=True)

# # 가장 가까운 고장의 종류
# df["fault_type"] = (
#     df[ttf_cols]
#     .idxmin(axis=1, skipna=True)
#     .map(fault_names)
# )


# print(
#     df[
#         ttf_cols + ["min_ttf", "fault_type"]
#     ].head(20)
# )

# print(df["fault_type"].value_counts(dropna=False))
# print(df["min_ttf"].describe())


# # min_ttf가 작은 데이터를 확인
# near_fault = df.nsmallest(
#     100,
#     "min_ttf"
# )

# print(
#     near_fault[
#         [
#             "time",
#             "Tool",
#             "FLOWCOOLFLOWRATE",
#             "FLOWCOOLPRESSURE",
#             "min_ttf",
#             "fault_type"
#         ]
#     ]
# )

# print(df[df["min_ttf"] == 0][
#     [
#         "time",
#         "Tool",
#         "min_ttf",
#         "fault_type"
#     ]
# ])



ttf_cols = [
    "TTF_FlowCool Pressure Dropped Below Limit",
    "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "TTF_Flowcool leak"
]

all_nan_mask = df[ttf_cols].isna().all(axis=1)

print("전체 행:", len(df))
print("TTF 전체 NaN:", all_nan_mask.sum())
print("비율:", all_nan_mask.mean() * 100, "%")

all_nan_mask = df[ttf_cols].isna().all(axis=1)

nan_rows = df.loc[
    all_nan_mask,
    ["time", "Tool"] + ttf_cols
]

print(nan_rows.head(20))
print("\n마지막 NaN:")
print(nan_rows.tail(20))

print("\nNaN index 범위:")
print("첫 index:", nan_rows.index.min())
print("마지막 index:", nan_rows.index.max())

nan_idx = df.index[all_nan_mask]

groups = (
    nan_idx.to_series()
    .diff()
    .ne(1)
    .cumsum()
)

nan_ranges = (
    nan_idx.to_series()
    .groupby(groups)
    .agg(["min", "max", "count"])
)

print(nan_ranges)

boundary = df.loc[
    3052915:3052935,
    ["time"] + ttf_cols
]

print(boundary.to_string())

print("\nTTF 마지막 유효 행")
print(
    df.loc[
        3052925,
        ["time"] + ttf_cols
    ]
)

df["min_ttf"] = df[ttf_cols].min(axis=1)

print(
    df.loc[
        3052915:3052935,
        ["time", "min_ttf"] + ttf_cols
    ].to_string()
)

for col in ttf_cols:

    valid = df[col].notna()

    last_idx = df.index[valid].max()

    print("\n", "=" * 70)
    print(col)
    print("유효값 개수 :", valid.sum())
    print("마지막 index:", last_idx)
    print("마지막 time :", df.loc[last_idx, "time"])
    print("마지막 TTF  :", df.loc[last_idx, col])

    print("\n마지막 10개:")
    print(
        df.loc[
            max(0, last_idx - 9):last_idx,
            ["time", col]
        ].to_string(index=False)
    )

    import pandas as pd
from pathlib import Path

# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

processed_path = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "01_M01_sensor_ttf.parquet"
)

fault_path = (
    PROJECT_ROOT
    / "data"
    / "train"
    / "train_faults"
    / "01_M01_train_fault_data.csv"
)


# ============================================================
# 2. 데이터 불러오기
# ============================================================

df = pd.read_parquet(processed_path)

fault_df = pd.read_csv(fault_path)

print("Sensor + TTF shape:", df.shape)
print("Fault shape       :", fault_df.shape)

print("\nFault columns:")
print(fault_df.columns.tolist())

print("\n전체 Fault 데이터:")
print(fault_df.to_string(index=False))

import matplotlib.pyplot as plt

fault_name = "FlowCool Pressure Dropped Below Limit"

fault_times = (
    fault_df.loc[
        fault_df["fault_name"] == fault_name,
        "time"
    ]
    .sort_values()
    .tolist()
)

print("고장 발생 횟수:", len(fault_times))
print("고장 시각:")
print(fault_times)

window = 100000


# ============================================================
# FLOWCOOLPRESSURE
# ============================================================

plt.figure(figsize=(14, 7))

for i, fault_time in enumerate(fault_times):

    temp = df[
        (df["time"] >= fault_time - window) &
        (df["time"] < fault_time)
    ].copy()

    if temp.empty:
        continue

    # 고장 시점을 0으로 맞춤
    temp["time_to_fault"] = temp["time"] - fault_time

    plt.plot(
        temp["time_to_fault"],
        temp["FLOWCOOLPRESSURE"],
        alpha=0.5,
        linewidth=0.8,
        label=f"Fault {i+1}"
    )

plt.axvline(
    0,
    linestyle="--",
    label="Fault"
)

plt.xlabel("Time to Fault")
plt.ylabel("FLOWCOOLPRESSURE")
plt.title(
    "FLOWCOOLPRESSURE before "
    "FlowCool Pressure Dropped Below Limit"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()


# ============================================================
# FLOWCOOLFLOWRATE
# ============================================================

plt.figure(figsize=(14, 7))

for i, fault_time in enumerate(fault_times):

    temp = df[
        (df["time"] >= fault_time - window) &
        (df["time"] < fault_time)
    ].copy()

    if temp.empty:
        continue

    temp["time_to_fault"] = temp["time"] - fault_time

    plt.plot(
        temp["time_to_fault"],
        temp["FLOWCOOLFLOWRATE"],
        alpha=0.5,
        linewidth=0.8,
        label=f"Fault {i+1}"
    )

plt.axvline(
    0,
    linestyle="--",
    label="Fault"
)

plt.xlabel("Time to Fault")
plt.ylabel("FLOWCOOLFLOWRATE")
plt.title(
    "FLOWCOOLFLOWRATE before "
    "FlowCool Pressure Dropped Below Limit"
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.show()