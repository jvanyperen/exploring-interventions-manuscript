import pandas as pd
import numpy as np

SMOOTHING_LENGTH = 7


def add_deaths_discharges(area_fname, area_code, death_fname, lookup_fname):
    area_df = pd.read_csv(area_fname)
    area_df = split_deaths(area_df, area_code, death_fname, lookup_fname)
    area_df = get_discharges(area_df)

    area_df.to_csv(area_fname, index=False, index_label=False)


def get_discharges(area_df):
    num_rows = len(area_df)
    discharges = np.zeros(num_rows)

    td = (
        area_df["newAdmissions"][1:]
        + area_df["hospitalCases"][1:]
        - area_df["hospitalDeaths"][1:]
    ).to_numpy()
    tm = area_df["hospitalCases"][:-1].to_numpy()
    dis = td - tm

    discharges[1:] = np.maximum(dis, 0.0)

    discharges[SMOOTHING_LENGTH + 1 :] = moving_average(discharges[1:])
    discharges[: SMOOTHING_LENGTH + 1] = np.nan

    area_df["newDischarges"] = discharges

    return area_df


def split_deaths(area_df, area_code, death_fname, lookup_fname):
    death_proportion = get_death_proportion(
        area_code, death_fname, lookup_fname
    )
    area_df["hospitalDeaths"] = np.round(
        area_df["newDeaths28DaysByDeathDate"] * death_proportion
    )
    area_df["otherDeaths"] = (
        area_df["newDeaths28DaysByDeathDate"] - area_df["hospitalDeaths"]
    )
    return area_df


def get_death_proportion(area_code, death_fname, lookup_fname):
    ons_death_df = pd.read_excel(
        death_fname, header=3, sheet_name="Occurrences - All data"
    )

    covid_df = ons_death_df[ons_death_df["Cause of death"] == "COVID 19"]

    ltla_codes = get_LTLA_codes(area_code, lookup_fname)

    ltla_df = covid_df[covid_df["Area code"].isin(ltla_codes)]

    hosp_deaths = sum(
        ltla_df[ltla_df["Place of death"].isin(["Hospital"])][
            "Number of deaths"
        ]
    )
    home_deaths = sum(
        ltla_df[ltla_df["Place of death"].isin(["Home"])]["Number of deaths"]
    )

    return hosp_deaths / (hosp_deaths + home_deaths)


def get_LTLA_codes(area_code, lookup_fname):
    lookup_df = pd.read_excel(lookup_fname, header=4)
    la_codes = lookup_df[lookup_df["Region code"] == area_code][
        "LA code"
    ].to_list()

    if not la_codes:
        return lookup_df[lookup_df["Region name"] != "Wales"][
            "LA code"
        ].to_list()
    return la_codes


def moving_average(arr):
    cum_arr = np.cumsum(arr)
    return np.round(
        (cum_arr[SMOOTHING_LENGTH:] - cum_arr[:-SMOOTHING_LENGTH])
        / SMOOTHING_LENGTH
    )
