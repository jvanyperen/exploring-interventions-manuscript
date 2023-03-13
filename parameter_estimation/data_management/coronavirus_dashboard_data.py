from requests import get
import pandas as pd


def get_coronavirus_data(area_types, area_codes, area_names, metrics):
    area_dfs = {}

    for m_idx, metric_lists in enumerate(metrics):
        for idx, (at, ac) in enumerate(
            zip(area_types[m_idx], area_codes[m_idx])
        ):
            if m_idx > 0:
                df = hit_dashboard(at, ac, metric_lists)
                area_dfs[idx] = pd.merge(
                    area_dfs[idx], df, how="inner", on="date", sort=False
                )
            else:
                area_dfs[idx] = hit_dashboard(at, ac, metric_lists)

    for idx, fn in enumerate(area_names):
        fname = f"parameter_estimation/data_management/{fn}.csv"
        area_dfs[idx].to_csv(
            fname,
            index=False,
            index_label=False,
            columns=["date", *[m for ml in metrics for m in ml]],
        )


def hit_dashboard(area_type, area_code, metric_list):
    metric_url = ""
    for metric in metric_list:
        metric_url += f"metric={metric}&"
    url = (
        "https://api.coronavirus.data.gov.uk/v2/data?"
        f"areaType={area_type}&areaCode={area_code}&{metric_url[:-1]}"
    )

    response_json = get_data(url)
    return pd.DataFrame(response_json["body"])


def get_data(url):
    response = get(url, timeout=10)

    if response.status_code >= 400:
        raise RuntimeError(f"Request failed: { response.text }")

    return response.json()
