# Run main scripts needed for the analysis sequentially

# 0. Packages and Paths
library(rmarkdown)
library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
source("Useful_functions.R")

# Define the base path
base_path <- "Y:/PsyThera/Projekte_Meinke/Old_projects/Labrotation_Rebecca/0_Datapreparation"
parent_path <- dirname(base_path)

# 1. Taskdata_preprocessing-Skript
# Define the parameters you want to iterate over
RT_trimming_options <- c(TRUE,FALSE)
RT_remove_wrong_options <- c(TRUE,FALSE)

# Create a function to generate the filename based on options chosen for 
# the preprocessing of the reaction time
generate_htmlfilename_meanRTacc<- function(RT_trimming, RT_remove_wrong) {
  trimming_text <- ifelse(RT_trimming == TRUE, "trimmed", "not_trimmed")
  remove_wrong_text <- ifelse(RT_remove_wrong == TRUE, "wrong_removed", "not_removed")
  paste0("Calculate_mean_RT_and_accuracy_", trimming_text,"_", remove_wrong_text, ".html")
}

# Loop over the parameter sets and knit the RMarkdown file for each set
for (RT_trimming in RT_trimming_options) {
  for (RT_remove_wrong in RT_remove_wrong_options){
        params_list <- list(RT_trimming = RT_trimming, RT_remove_wrong =  RT_remove_wrong )
        
        output_filename <- generate_htmlfilename_meanRTacc(RT_trimming, RT_remove_wrong)
      
        rmarkdown::render(
          input = file.path(base_path, "Calc_meanRT_meanacc.Rmd"),
          output_file = output_filename,
          params = params_list,
          envir = new.env()
        )
        cat("Generated file:", output_filename, "\n")
    }
  }

# 2. BIS-Script
outliers_removed_options <- c("yes", "no")
input_data_path_options <- c(
  file.path(base_path, "not_trimmed_not_removed"),
  file.path(base_path, "RT_trimmed_not_removed"),
  file.path(base_path, "not_trimmed_RT_wrong_removed"),
  file.path(base_path, "RT_trimmed_RT_wrong_removed")
)

# Create a function to generate the filename based on parameters
generate_htmlfilename_BIS <- function(outliers_removed) {
  # Get the name of the input data folder as it tells us how the mean RT was calculated
  last_folder <- basename(input_data_path) 
  outliers_text <- ifelse(outliers_removed == "yes", "outliers-removed", "outliers-not-removed")
  paste0("Add_groupinfo_calc_BIS_", last_folder, "_", outliers_text,".html")
}

for (outliers_removed in outliers_removed_options) {
  for (input_data_path in input_data_path_options){
      params_list <- list(outliers_removed = outliers_removed, input_data_path = input_data_path)
      
      output_filename <- generate_htmlfilename_BIS(outliers_removed)
      outliers_text <- ifelse(outliers_removed == "yes", "outliers-removed", "outliers-not-removed")
      
      rmarkdown::render(
        input = file.path(base_path, "Add_groupinfo_calc_BIS.Rmd"),
        output_file = output_filename,
        params = params_list,
        envir = new.env()
      )
      cat("Generated file:", output_filename, "\n")
}
}


# 3. Group comparison and machine learning analyses
# General further processing
inputdata_variants_paths <- c(
  file.path(base_path, "not_trimmed_not_removed/BIS/outliers-not-removed"),
  file.path(base_path, "not_trimmed_not_removed/BIS/outliers-removed"),
  file.path(base_path, "RT_trimmed_RT_wrong_removed/BIS/outliers-not-removed"),
  file.path(base_path, "RT_trimmed_RT_wrong_removed/BIS/outliers-removed")
)

# Generate htmlfilename for the group comparison and machine learning script
generate_htmlfilename_analyses <- function(input_data_path,prefix) {
  # This function creates the html-filename based on the preprocessing of the input data
  # Get the name of the input data folder as it tells us whether outliers were removed
  last_folder <- basename(input_data_path) 
  # Get the second-to-last folder as it tells how the mean RT was calculated
  parent_dir <- dirname(input_data_path)
  grandparent_dir <- dirname(parent_dir)
  third_last_folder <- basename(grandparent_dir)
  paste0(prefix,"_", third_last_folder, "_", last_folder,".html")
}

# Group comparison script (HC vs. patients)
for (input_data_path in inputdata_variants_paths) {
    params_list <- list(input_data_path = input_data_path)
    
    output_filename <- generate_htmlfilename_analyses(input_data_path, prefix = "HC_vs_patients")
    output_path <- file.path(create_results_path(inputdata_path = input_data_path,
                                       output_mainpath = file.path(parent_path,"1_Group_comparison/HC_vs_Pat")),
                             output_filename)
    
    rmarkdown::render(
      input = "Group Comparison_Executive Functions\\Group Comparison_Healthy Controls Patients.Rmd",
      output_file = output_path,
      params = params_list,
      envir = new.env()
    )
    cat("Generated file:", output_filename, "\n")
}

# Group comparison script (Response vs. Nonresponse)
for (input_data_path in inputdata_variants_paths) {
  params_list <- list(input_data_path = input_data_path)
  
  output_filename <- generate_htmlfilename_analyses(input_data_path, prefix = "Response_vs_Nonresponse")
  output_path <- file.path(create_results_path(inputdata_path = input_data_path,
                                               output_mainpath = file.path(parent_path,"1_Group_comparison/R_vs_NR")),
                           output_filename)
  
  rmarkdown::render(
    input = "Group Comparison_Executive Functions\\Group Comparison_Response Nonresponse.Rmd",
    output_file = output_path,
    params = params_list,
    envir = new.env()
  )
  cat("Generated file:", output_filename, "\n")
}

# Machine learning preprocessing
for (input_data_path in inputdata_variants_paths) {
  params_list <- list(input_data_path = input_data_path)
  
  output_filename <- generate_htmlfilename_analyses(input_data_path, prefix = "Machine_learning_preparation")
  output_path <- file.path(create_results_path(inputdata_path = input_data_path,
                                               output_mainpath = file.path(parent_path,"2_Machine_Learning/Feature_Label_Dataframes")),
                           output_filename)
  
  rmarkdown::render(
    input = "Machine Learning_Response Prediction/ML_preprocessing.Rmd",
    output_file = output_path,
    params = params_list,
    envir = new.env()
  )
  cat("Generated file:", output_filename, "\n")
}
