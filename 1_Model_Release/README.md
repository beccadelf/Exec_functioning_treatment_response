# Model Release

Introductory sentence. (Mention that to make sure that ... (preprocessing), we highly recommend to use our provided script "Apply_pretrained_model.py" for prediction).

> ## **ℹ️ Getting Started** 
> Follow these steps to quickly set up the program and run a first analysis using the provided example data:
> 1. **Clone this repository** to your local machine.
> 2. Make sure you have installed the packages pandas, scikit-learn, and imbalanced-learn.
> 3. **Open the script "[Apply_pretrained_model.py](https://github.com/beccadelf/Exec_functioning_treatment_response/1_Model_Release/Apply_pretrained_model.py)" and specify required parameters:**
>    1. *--PATH_INPUT_DATA*: specify the path to the "simulated_data.csv" for using example data.
>    2. *--OUTPUT_PATH*: specify a path where prediction results should be saved.
>    3. *--MODE*: select "inference_only".
>    4. *--X_FILE*: "simulated_data.csv" for using example data.
>    5. *--Y_FILE*: "simulated_labels.csv" for using example data.
> 4. **Run the script**
> 5. **View results**: after running the script, a XXX can be found in the specified output-folder.

# Prepare the XXX

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

To use this model, a dataset containing the appropriate set of predictors must be created. The dataset can be in .sav, .csv, or .xlsx format. For reference, a simulated dataset ("simulated_data.csv") demonstrating the required structure is available on GitHub (https://github.com/beccadelf/Exec_functioning_treatment_response).

**Feature naming:**

The variables will need to be presented with the following name:

-	is_woman (0 = male, 1 = female)
-	Age
-	Abschluss_Gymnasium (0 = other, 1 = high education)
-	FAS (Fear of spiders questionnaire, total score)
-	BDI_II (Beck depression inventory II, total score)
-	STAI_T (State-trait anxiety inventory, trait subscale)

**Data Preprocessing:**

Missing values need to be recoded as 77777 and 99999 for dichotomous and dimensional variables respectively. Further preprocessing steps are handled in the provided Python script "Apply_pretrained_model.py".

# Run the script

To XXX, the script **"Apply_pretrained_model.py"** needs to be executed.
To execute the script, **two methods** are available, each requiring the same set of arguments to be provided:

- PATH_INPUT_DATA → The path to the folder which contains features.csv, and labels.csv
- OUTPUT_PATH → The path to the folder where the prediction results should be saved
- MODE → Select "inference_only" (the script can also be used for training and saving a model)
- X_FILE → Name of the features-file. Specify "simulated_features.csv" to use example data
- Y_FILE → Name of the labels-file. Specify "simulated_labels.csv" to use example data

## 1. Run script via the command line (most simple)

Open a terminal (e.g., Powershell on Windows, Terminal on macOS/Linux), and run the following command:

 ```python
 python "YOUR_PATH\Apply_pretrained_model.py" \
  --PATH_INPUT_DATA "path_to_your_data_folder" \
  --OUTPUT_PATH "path_for_results" \
  --MODE inference_only \
  --X_FILE simulated_data.csv \
  --Y_FILE simulated_labels.csv
 ```


