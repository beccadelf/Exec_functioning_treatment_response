# Executive Functions in Spider Phobia: Analysis Repository

This repository contains all scripts and resources for the analyses described in the paper "Executive Functions in Spider Phobia: Relation to Psychopathology and Treatment Response." The goal of this study is twofold: 
1. To assess whether individuals with spider phobia exhibit deficits in executive functions compared to healthy controls.
2. To investigate if baseline executive functions can predict response to exposure therapy for spider phobia.

The repository is structured into two main sections, each aligned with the study's analytical objectives: **Group Comparison** and **Machine Learning**. It is designed to document our exact calculations and analyses, allowing other researchers to retrace our work. For a concise summary of the analysis steps and their results, please refer to the accompanying HTML file for each script.
*HTML*: To inspect the `.html` files, simply download them or clone the repository using the GitHub App, then open them in your default browser (double-click the file).

---

## Folder Structure

### 1. Group Comparison: Executive Functions
This section contains scripts and resources for comparing executive function performance between participants with spider phobia and healthy controls.

- **Group_Comparison.Rmd**: R Markdown file containing t-tests and plotting for group comparisons. The analyses cover the tasks: Spatial 2-Back, Stroop, Number-Letter, and Stop Signal.

- **Exploratory Analyses/**:
  - **Analyses without wrong responses/**: Filters out trials with incorrect, but doing so turned out to not affect analyses.
  - **ANCOVA/**: Contains scripts for performing ANCOVA to control for age, sex, and education.
  - **Pre-Post Validation/**: Validates the consistency of data across time points (baseline and post-intervention).
  - **Testing Normality/**: Checks normality assumptions required for tests in group comparisons.

### 2. Machine Learning: Response Prediction
This section contains scripts for building and testing machine learning models to predict treatment response based on baseline executive functions and clinical variables.

- **ML_preprocessing.Rmd**: 

- **Classifier_treatment_response.py**: 
  
- **Regressor_treatment_outcome.py**: 

- **Run_scripts_sequentially.py**: 

### 3. Results

---

## Requirements
- **R** (with packages for statistical analyses and visualization)
- **Python** (with `scikit-learn` and additional packages for machine learning)

---

## Citation
If using this repository, please cite the original paper:

> Meinke, C., Delfendahl, R., Adam, T., Lueken, U., Beesdo-Baum, K., ... & Hilbert, K. (2024). *Executive Functions in Spider Phobia: Relation to Psychopathology and Treatment Response*. Journal of Anxiety Disorders.
