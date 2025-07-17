# Model Release

Introductory sentence.

> ## **ℹ️ Getting Started** 
> Follow these steps to quickly set up the program and run a first analysis using the provided example data:
> 1. **Clone this repository** to your local machine.
> 2. Make sure you have installed the packages pandas, scikit-learn, and imbalanced-learn.
> 3. **Open the script "Apply_pretrained_model.py" and specify required parameters:**
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
