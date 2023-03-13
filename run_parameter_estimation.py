import datetime
import json
import numpy as np

import parameter_estimation.data_management.coronavirus_dashboard_data as cdd
import parameter_estimation.data_management.combine_datasets as cd
import parameter_estimation.get_data as gd
import Rt_formula as Rf
import parameter_estimation.fit_procedure as fp

REGION_CODES = [
    ["E12000002", "E12000008", "E92000001"],
    ["E40000010", "E40000005", "E92000001"],
]
METRICS = [["newDeaths28DaysByDeathDate"], ["newAdmissions", "hospitalCases"]]
REGION_TYPES = [
    ["region", "region", "nation"],
    ["nhsRegion", "nhsRegion", "nation"],
]
REGION_NAMES = ["northWest", "southEast", "england"]

DEATH_FNAME = "parameter_estimation/data_management/lahbtablesweek01to532020datawk232021.xlsx"
LOOKUP_FNAME = "parameter_estimation/data_management/lasregionew2021lookup.xlsx"

START_DATE = datetime.datetime.strptime("23/03/2020", "%d/%m/%Y")
END_DATE = datetime.datetime.strptime("30/06/2020", "%d/%m/%Y")


def parameter_information(
    region_name, parameters_info=True, initial_conditions_info=True
):
    model_file = f"{region_name}.json"

    with open(model_file, "r") as json_file:
        model_dict = json.load(json_file)

    params = np.array(model_dict["params"])
    initial_conditions = np.array(model_dict["initial_conditions"])
    population_size = model_dict["population_size"]

    if parameters_info:
        Rt = Rf.calculate_Rt(params, initial_conditions[0], population_size)
        print(f"R_t = {Rt:.3f}")
        print(f"gamma_H^-1 = {1.0/params[5]:.2f} days")
        print(f"m_U = {params[7]:.4f}")
        print(f"mu_H^-1 = {1.0/(params[5]*params[6]):.2f} days")

    if initial_conditions_info:
        print(f"E_0 (% N) = {100.0*initial_conditions[1]/population_size:.4f}")
        print(f"U_0 (% N) = {100.0*initial_conditions[2]/population_size:.4f}")
        print(f"I_0 (% N) = {100.0*initial_conditions[3]/population_size:.4f}")


def produce_complete_data():
    print("adding approximate daily deaths and discharges")
    for rn, rc in zip(REGION_NAMES, REGION_CODES[0]):
        fname = f"parameter_estimation/data_management/{rn}.csv"
        cd.add_deaths_discharges(fname, rc, DEATH_FNAME, LOOKUP_FNAME)


def get_data():
    print("Getting data")
    cdd.get_coronavirus_data(REGION_TYPES, REGION_CODES, REGION_NAMES, METRICS)


def fit_data():
    for rn, rc in zip(REGION_NAMES, REGION_CODES[0]):
        print(f"Fitting {rn}")
        fname = f"parameter_estimation/data_management/{rn}.csv"
        data_dict, *_ = gd.form_data_dictionary(fname, START_DATE, END_DATE)
        population_size = gd.get_population_size(rc)
        params, initial_condition = fp.parameter_estimation(
            data_dict, population_size
        )

        region_dict = {
            "params": list(params),
            "initial_conditions": list(initial_condition),
            "population_size": population_size,
        }

        pname = f"outputs/{rn}.json"

        with open(pname, "w") as json_file:
            json.dump(region_dict, json_file)


if __name__ == "__main__":
    get_data()
    produce_complete_data()
    fit_data()
