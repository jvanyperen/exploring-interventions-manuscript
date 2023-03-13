# Data Sources

Table of contents

1. [Coronavirus Dashboard](#coronavirus-dashboard)

   1. [Admissions](#admissions)
   2. [Beds Occupied](#beds-occupied)
   3. [Discharges](#discharges)

2. [Office for National Statistics](#office-for-national-statistics)

   1. [Deaths in Hopsital](#deaths-ons-deaths)
   2. [Area Codes](#area-codes)
   3. [Population Estimates](#population-estimates)

# Coronavirus Dashboard

[Coronavirus Dashboard](https://coronavirus.data.gov.uk/) is maintained by the UK Government, see this page for more information: [About](https://coronavirus.data.gov.uk/about) for further information. To download data see this page: [Download data](https://coronavirus.data.gov.uk/details/download).

If you want to automate, you can use the API instead. To do this in Python, import the [requests](https://requests.readthedocs.io/en/latest/) module and use the web API, information can be found here: [Web Api](https://coronavirus.data.gov.uk/details/developers-guide/main-api). Note the fair use policy here: [Fair Use](https://coronavirus.data.gov.uk/details/download#fair-usage-policy), which we restate here: (true as of _insert date here_)

1. Throttling: Each user is limited to 10 download requests per any 100–second period, with a maximum rate limit of 100 per hour.
2. Metric Limit: Each download request may contain up to a maximum number of 5 metrics. This excludes the default metrics.
3. Freshness: Identical requests are only refreshed once every 150 seconds.

---

## Admissions

Summary: Daily number of new admissions to hospital of patients with COVID-19.

Webpage: [Admissions](https://coronavirus.data.gov.uk/metrics/doc/newAdmissions)

To download using API (if you click the hyperlink it will download a csv file that contains the data for the South East region): [New Admissions](https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsRegion&areaCode=E40000005&metric=newAdmissions&format=csv)

---

## Beds Occupied

Summary: Daily numbers of confirmed COVID-19 patients in hospital.

Webpage: [Beds occupied](https://coronavirus.data.gov.uk/metrics/doc/hospitalCases)

To download using API (if you click the hyperlink it will download a csv file that contains the data for the South East region): [Beds occupied](https://api.coronavirus.data.gov.uk/v2/data?areaType=nhsRegion&areaCode=E40000005&metric=hospitalCases&format=csv)

---

## Deaths

Summary: Daily numbers of people who died within 28 days of being identified as a COVID-19 case by a positive test. Data are shown by date of death.

Webpage: [Deaths](https://coronavirus.data.gov.uk/metrics/doc/newDeaths28DaysByDeathDate)

To download using API (if you click the hyperlink it will download a csv file that contains the data for the South East region): [Deaths](https://api.coronavirus.data.gov.uk/v2/data?areaType=region&areaCode=E12000008&metric=changeInNewDeaths28DaysByDeathDate&format=csv)

---

# Office for National Statistics

The [Office for National Statistics](https://www.ons.gov.uk/) is a Government department.

---

## Deaths {#ons-deaths}

Summary: At the beginning of the pandemic, the ONS started to record whether COVID-19 was mentioned on the death certificate. The ONS was already recording deaths by place of occurence at a Local Tier Local Authority level (to be able to differentiate between a death in hospital and a death elsewhere). Since we use data corresponding to the first few months of the pandemic, we use the "2020 edition of this dataset", and use the "Occurrences" dataset rather than the "Registrations".

Webpage: [Deaths](https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/causesofdeath/datasets/deathregistrationsandoccurrencesbylocalauthorityandhealthboard)

---

## Area Codes

Summary: In order to collect the deaths for each of the regions in the analysis, we needed to collect all the Area Codes of the Local Tier Local Authorities that are included in that region. The ONS provide a lookup table for this purpose, and we use the "June 2020 edition of this dataset 2021 local authority boundaries".

Webpage: [Area Codes](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/migrationwithintheuk/datasets/userinformationenglandandwaleslocalauthoritytoregionlookup)

---

## Population Estimates

Summary: For an infectious disease model, it is very helpful to have an estimate on the population size of the hospitals and local authorities (as data reporting and splitting entities). The ONS collects mid-year population estimates for the UK, and splits it by administrative area (wanted), age (not wanted) and sex (not wanted). For the most up-to-date estimates, we use the "Mid-2020 edition" and use the "Summary" (MYE4) dataset.

Webpage: [Population estimates](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland)
