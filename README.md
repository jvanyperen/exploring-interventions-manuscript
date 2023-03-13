# A hospital demand and capacity intervention approach for COVID-19

[James Van Yperen](https://profiles.sussex.ac.uk/p311115-james-van-yperen)<sup>1,+</sup>, [ORCID-ID](https://orcid.org/0000-0002-0513-1494): 0000-0002-0513-1494  
[Eduard Campillo-Funollet](https://www.lancaster.ac.uk/maths/people/eduard-campillo-funollet)<sup>2,3</sup>, [ORCID-ID](https://orcid.org/0000-0001-7021-1610): 0000-0001-7021-1610  
Rebecca Inkpen<sup>1</sup>  
[Anjum Memon](https://www.bsms.ac.uk/about/contact-us/staff/professor-anjum-memon.aspx)<sup>4</sup>, [ORCID-ID](https://orcid.org/0000-0001-8256-3015): 0000-0001-8256-3015  
[Anotida Madzvamuse](https://www.math.ubc.ca/user/3665)<sup>1,5,6,7,\*</sup>, [ORCID-ID](https://orcid.org/0000-0002-9511-8903): 0000-0002-9511-8903

<sup>1</sup>: Department of Mathematics, School of Mathematical and Physical Sciences, University of Sussex, Brighton, United Kingdom  
<sup>2</sup>: Department of Mathematics, School of Mathematical, Statistical and Actuarial Sciences, University of Kent, Canterbury, United Kingdom  
<sup>3</sup>: Department of Mathematics and Statistics, Lancaster University, Lancaster, United Kingdom  
<sup>4</sup>: Department of Primary Care and Public Health, Brighton and Sussex Medical School, Brighton, United Kingdom  
<sup>5</sup>: Department of Mathematics, University of Johannesburg, Johannesburg, South Africa  
<sup>6</sup>: Department of Mathematics, University of British Columbia, Vancouver, Canada  
<sup>7</sup>: Department of Mathematics, University of Pretoria, Pretoria, South Africa  
<sup>\*</sup>: Corresponding author: am823@math.ubc.ca  
<sup>+</sup>: Corresponding author: j.vanyperen@sussex.ac.uk

---

Table of contents

1. [Setup](#setup)

2. [Data and parameter estimation](#data-and-parameter-estimation)

3. [Tables and figures](#tables-and-figures)

4. [Supplementary material](#supplementary-material)

---

# Setup

In order to run the code, [conda](https://docs.conda.io/en/latest/) needs to be installed.

With conda installed, open up a terminal and run

```
conda env create -f environment.yaml
```

Once it has finished installing the virtual environment, run

```
conda activate exploring-interventions
```

---

# Data and parameter estimation

The data used for the analysis is contained in the director: parameter_estimation/data_management. Further information about the different datasets can be found in: parameter_estimation/data_management/data_sources.md.

Note: the data was collected on 19/12/22 - if you run the script to gather data it will be updated to contain the most recent data.

In order to generate your own versions of the dataset, run

```
python run_parameter_estimation.py
```

If you want to generate datasets that are different from the regions provided, then go into run_parameter_estimation.py and add the appropriate codes, region type and name to the constant variables

```
REGION_CODES
REGION_TYPES
REGION_NAMES
```

The region names are written in camelCase. The region type is associated to the [Coronavirus Dashboard](https://coronavirus.data.gov.uk/), see parameter_estimation/data_management/data_sources.md for futher details. To find the associated region code, either try an online download using the online [API](https://coronavirus.data.gov.uk/details/download), or go into the Excel file parameter_estimation/data_management/lasregionew2021lookup.xlsx.

By running run_parameter_estimation.py, you will automatically run the parameter estimation method.

---

# Tables and figures

In order to generate the tables and figures in the manuscripts, open files paper_tables.py and paper_figures.py. In the files, scroll to the bottom to find the statement

```
if __name__ == "__main__":
```

Beneath this statement, uncomment the figure or table you want to see. The figure will save in the figures directory and a window will open the plot once it has been generated. The results for the tables will be printing into the terminal you used to run the script. In order to see the results quicker, we have pre-computed some of the simulations which took a significant amount of time to run. These can be found in the outputs directory. You can run all the analysis yourself by either removing the files from the outputs directory, or changing their names.

---

# Supplementary material

In order to generate the tables and figures in the supplementary material, open supplementary_tables.py and supplementary_figures.py. Then, follow the same directions as in [Tables and figures](#tables-and-figures).
