# Executive Functioning in Spider Phobia: Analysis Repository

This repository contains all scripts and resources for the analyses described in the paper "Executive functioning does not predict exposure therapy outcomes and is not impaired in spider phobia."

The repository is structured into two main sections, each aligned with the study's analytical objectives: **Group Comparison** and **Machine Learning**. It is designed to document our exact calculations and analyses, allowing other researchers to retrace our work, rather than re-run our analyses, as our data cannot be shared. Researchers are encouraged to integrate our provided code into their own analyses, which may be particularly relevant for our adaptation of the SSRTcalc function (see "Function_adaptive_SSRT.R"), which implements the replacement of go-trials. For a concise summary of the analysis steps and their results, please refer to the accompanying HTML file for each script.

*HTML*: To inspect the `.html` files, simply download them or clone the repository using the GitHub App, then open them in your default browser (double-click the file).

---

## Folder Structure

### 1. Group Comparison: Executive Functions
This section contains scripts and resources for comparing executive functioning performance between participants with spider phobia and healthy controls as well as responders and nonresponders.
- **Group Comparison_Healthy Control Patients.rmd**: Compares executive functioning between patients having spider phobia and healthy controls.
- **Group Comparison_Healthy Control Patients_ANCOVA.rmd**: Controls the comparison of executive functioning between patients having spider phobia and healthy controls for confounders.
- **Group Comparison_Response Nonresponse.rmd**: Compares executive functioning between responders and nonresponders.
- **Group Comparison_Pre Post.rmd (only exploratory)**: Compares executive functioning between pre- and post-treatment.




### 2. Machine Learning: Response Prediction
This section contains scripts for building and testing machine learning models to predict treatment response based on baseline executive functions and clinical variables.

- **ML_preprocessing.Rmd**: Preprocesses the data for the subsequent machine learning pipeline.

- **Classifier_treatment_response.py**: Runs a classification machine learning pipeline.
  
- **Regressor_treatment_outcome.py**: Runs a regressor machine learning pipeline.

- **Run_scripts_sequentially.py**: Runs python / machine learning scripts at once.


### Others:
- **Function_adaptive_SSRT.Rmd**: Adapts the function "Integration_adaptiveSSD" from the SSRT-calc package (https://github.com/agleontyev/SSRTcalc/tree/master) to replace omissions errors with go-trials.

- **Useful_function.R**: Stores several useful functions used in our R-scripts.

- **Run_scripts_sequentially.py**: Reruns most of our R-scripted analyses sequentially. Ensures that most of our analysis can be reproduced at once.
  

---

## Requirements
- **R** (with packages for statistical analyses and visualization)
- **Python** (with `scikit-learn` and additional packages for machine learning)

