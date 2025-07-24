# Model Release

We provide a pretrained model for predicting treatment response based on selected sociodemographic and clinical variables. To ensure that your data is preprocessed in the same way as during training, we strongly recommend using the provided script **"[Apply_pretrained_model.py](https://github.com/beccadelf/Exec_functioning_treatment_response/1_Model_Release/Apply_pretrained_model.py)"** when applying the model to your own dataset. 

> ## **ℹ️ Getting Started** 
> Follow these steps to quickly set up the environment and run a first prediction using the provided example data and pretrained model:
> 1. **Clone this repository** to your local machine.
> 2. Make sure you have installed the packages pandas, scikit-learn, and imbalanced-learn.
> 3. **Open a terminal and run the following command to execute the script ("Apply_pretrained_model.py"), specifying the required parameters:**
>     ```python
>     python YOUR_PATH\Apply_pretrained_model.py \
>      --PATH_INPUT_DATA path_to_your_data_folder \
>      --OUTPUT_PATH path_for_results \
>      --MODE predict_only \
>      --X_FILE simulated_data.csv \
>      --Y_FILE simulated_labels.csv
>     ```
> 4. **View results**: after running the script, a "model_performance_evaluation.txt" can be found in the specified output-folder.

# Prepare the environment

1. Clone the repository.
   Alternativelly, all required files may be manually downloaded and the necessary folder structure be reconstructed.
   
2. Make sure all needed requirements are installed.

   a) Manually

      Three additional packages are needed: pandas, scikit-learn, and imbalanced-learn. Install them manually if you have not installed them yet.
    
    b) Automatically for conda-users, run in the terminal:
    
    ```python
    conda env create -f "YOUR_PATH_TO_THE_ENVIRONMENT\environment.yaml"
    ```
    
    c) Automatically for non-conda-users, run in the terminal:
    
    ```python
    pip install -r "requirements.txt".
    ```
    
# Prepare the input data

To use this model, a dataset containing the appropriate set of predictors must be created. The dataset can be in .sav, .csv, or .xlsx format. For reference, a simulated dataset ("simulated_data.csv") demonstrating the required structure is provided in this repository.

**Feature naming:**

The variables will need to be presented with the following name:

-	is_woman - (0 = male, 1 = female)
-	Age
-	Abschluss_Gymnasium - (0 = other, 1 = high education*)
-	FAS - (Fear of spiders questionnaire, total score)
-	BDI_II - (Beck depression inventory II, total score)
-	STAI_T - (State-trait anxiety inventory, trait subscale)

\*  i.e., 12–13 years of school, typically qualifying for university entrance

**Data Preprocessing:**

Missing values need to be recoded as 77777 and 99999 for dichotomous and dimensional variables respectively, and dichotomous variables need to be recoded as 0 and 1. These steps have already been performed on the example data. 

All other preprocessing steps are handled in the provided Python script "Apply_pretrained_model.py". In brief, these include imputation of missing values and the standardization of dimensional variables using the scaler fitted during model training.

# Run the script

To apply the predictive model to your own data, the script **"Apply_pretrained_model.py"** needs to be executed.
To execute the script, **two methods** are available, each requiring the same set of arguments to be provided:

- PATH_INPUT_DATA → The path to the folder which contains features.csv, and labels.csv
- OUTPUT_PATH → The path to the folder where the prediction results should be saved
- MODE → Select "predict_only" (the script can also be used for training and saving a model)
- X_FILE → Name of the features-file. Specify "simulated_features.csv" to use example data
- Y_FILE → Name of the labels-file. Specify "simulated_labels.csv" to use example data

## 1. Run script via the command line (most simple)

Open a terminal (e.g., Powershell on Windows, Terminal on macOS/Linux), and run the following command:

 ```python
 python YOUR_PATH\Apply_pretrained_model.py \
  --PATH_INPUT_DATA path_to_your_data_folder \
  --OUTPUT_PATH path_for_results \
  --MODE inference_only \
  --X_FILE simulated_data.csv \
  --Y_FILE simulated_labels.csv
 ```
## 2. Run script in your IDE (e.g., Spyder)

Change the arguments within the script, in particular specifying the path to the input data (--PATH_INPUT_DATA) and the path to where performance evaluation results should be saved (--OUTPUT_PATH):

```python
# Change the arguments here, when running script in IDE
args = parser.parse_args([
   '--PATH_INPUT_DATA', "PATH_TO_YOUR_INPUT_DATA",
   '--OUTPUT_PATH', "PATH_TO_RESULTS_FOLDER",
   '--MODE', "predict_only",
   '--X_FILE', "simulated_data.csv",
   '--Y_FILE', "simulated_labels.csv",
        ])
```

# Model performance

In a cross-validation with 100 iterations, the model achieved an average balanced accuracy of 0.60 (SD = 0.11), with a permutation-based p-value of p ≈ 0.08. Further performance metrics can be found in "model_performance_crossvalidation.docx".


