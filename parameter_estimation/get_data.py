import pandas as pd
import numpy as np
import copy


def form_data_dictionary(df_fname, start_date, end_date):
    area_df = pd.read_csv(df_fname)
    area_df["date"] = pd.to_datetime(area_df["date"], format="%Y-%m-%d")

    area_subset_df = area_df[
        (area_df["date"] >= start_date) & (area_df["date"] <= end_date)
    ]

    num_datapoints = len(area_subset_df)
    days = np.arange(1, num_datapoints, 1)
    day_ints = np.array([[t - 1, t] for t in days])

    data_dict = {}
    data_dict["occ"] = {
        "days": days,
        "data": np.flipud(area_subset_df["hospitalCases"].to_numpy())[1:],
    }
    data_dict["adm"] = {
        "days": day_ints,
        "data": np.flipud(area_subset_df["newAdmissions"].to_numpy())[1:],
    }
    data_dict["dis"] = {
        "days": day_ints,
        "data": np.flipud(area_subset_df["newDischarges"].to_numpy())[1:],
    }
    data_dict["dhp"] = {
        "days": day_ints,
        "data": np.flipud(area_subset_df["hospitalDeaths"].to_numpy())[1:],
    }
    data_dict["dnh"] = {
        "days": day_ints,
        "data": np.flipud(area_subset_df["otherDeaths"].to_numpy())[1:],
    }

    return copy.deepcopy(data_dict), days[0], days[-1]


def get_population_size(region_code):
    fname = "parameter_estimation/data_management/ukpopestimatesmid2020on2021geography.xls"
    data_frame = pd.read_excel(fname, header=7, sheet_name="MYE4")

    red_data_frame = data_frame[data_frame["Code"] == region_code]
    return int(red_data_frame["Mid-2020"])
