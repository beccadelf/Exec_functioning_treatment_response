# Run all scripts needed for the analysis sequentially

# 0. Packages and Paths
library(rmarkdown)
library(rstudioapi)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Define the base path
base_path <- "Y:/PsyThera/Projekte_Meinke/Old_projects/Labrotation_Rebecca/0_Datapreparation"
parent_path <- dirname(base_path)

# 1. Taskdata_preprocessing-Skript
# Define the parameters you want to iterate over
RT_trimming_options <- c(TRUE,FALSE)
RT_remove_wrong_options <- c(TRUE,FALSE)

# Create a function to generate the filename based on parameters
generate_filename <- function(RT_trimming, RT_remove_wrong) {
  trimming_text <- ifelse(RT_trimming == TRUE, "trimmed", "not_trimmed")
  remove_wrong_text <- ifelse(RT_remove_wrong == TRUE, "wrong_removed", "not_removed")
  #trimmed_suffix <- ifelse(RT_trimming == "TRUE", "RT_trimmed", "not_trimmed")
  #wrong_suffix <- ifelse(RT_remove_wrong == "TRUE", "RT_wrong_removed", "not_removed")
  #path = file.path(basic_results_path,paste(trimmed_suffix,wrong_suffix,sep = "_"),"raw_data")
  paste0("Calculate_mean_RT_and_accuracy_", trimming_text,"_", remove_wrong_text, ".html")
}

# Loop over the parameter sets and render the RMarkdown file for each set
for (RT_trimming in RT_trimming_options) {
  for (RT_remove_wrong in RT_remove_wrong_options){
    if ((RT_trimming == TRUE && RT_remove_wrong == TRUE) || 
        (RT_trimming == FALSE && RT_remove_wrong == FALSE)) {
        params_list <- list(RT_trimming = RT_trimming, RT_remove_wrong =  RT_remove_wrong )
        
        output_filename <- generate_filename(RT_trimming, RT_remove_wrong)
      
        rmarkdown::render(
          input = file.path(base_path, "Calc_meanRT_meanacc.Rmd"),
          output_file = output_filename,
          params = params_list,
          envir = new.env()
        )
        cat("Generated file:", output_filename, "\n")
    }
  }
}

# BIS-Script
outliers_removed_options <- c("yes", "no")
input_data_path_options <- c(
  file.path(base_path, "not_trimmed_not_removed"),
  file.path(base_path, "RT_trimmed_RT_wrong_removed")
)

# Create a function to generate the filename based on parameters
generate_filename_out <- function(outliers_removed) {
  outliers_text <- ifelse(outliers_removed == "yes", "outliers-removed", "outliers-not-removed")
  paste0("Calculate_BIS_", outliers_text,".html")
}

for (outliers_removed in outliers_removed_options) {
  for (input_data_path in input_data_path_options){
      params_list <- list(outliers_removed = outliers_removed, input_data_path = input_data_path)
      
      output_filename <- generate_filename_out(outliers_removed)
      outliers_text <- ifelse(outliers_removed == "yes", "outliers-removed", "outliers-not-removed")
      output_path = file.path(input_data_path, "BIS", outliers_text, output_filename)
      
      rmarkdown::render(
        input = file.path(base_path, "Add_groupinfo_calc_BIS.Rmd"),
        output_file = output_path,
        params = params_list,
        envir = new.env()
      )
      cat("Generated file:", output_filename, "\n")
}
}

# General further processing
inputdata_variants_paths <- c(
  file.path(base_path, "not_trimmed_not_removed/BIS/outliers-not-removed"),
  file.path(base_path, "not_trimmed_not_removed/BIS/outliers-removed"),
  file.path(base_path, "RT_trimmed_RT_wrong_removed/BIS/outliers-not-removed"),
  file.path(base_path, "RT_trimmed_RT_wrong_removed/BIS/outliers-removed")
)

# Group comparison script (HC vs. patients)
generate_filename <- function(input_data_path,prefix) {
  last_folder <- basename(input_data_path)
  # Get the grandparent directory of the path
  parent_dir <- dirname(input_data_path)
  grandparent_dir <- dirname(parent_dir)
  # Get the second-to-last folder
  third_last_folder <- basename(grandparent_dir)
  paste0(prefix, third_last_folder, "_", last_folder,".html")
}

for (input_data_path in inputdata_variants_paths) {
    params_list <- list(input_data_path = input_data_path)
    
    output_filename <- generate_filename(input_data_path, prefix = "HC_vs_patients_")
    output_data_path <- file.path(parent_path, "1_Group_comparison")
    output_path = file.path(output_data_path, output_filename)
    
    rmarkdown::render(
      input = "Group Comparison_Executive Functions\\Group Comparison_Healthy Controls Patients.Rmd",
      output_file = output_path,
      params = params_list,
      envir = new.env()
    )
    cat("Generated file:", output_filename, "\n")
  }

# Machine learning preprocessing
for (input_data_path in inputdata_variants_paths) {
  params_list <- list(input_data_path = input_data_path)
  
  output_filename <- generate_filename(input_data_path, prefix = "Machine_learning_preparation")
  output_data_path <- file.path(parent_path, "2_Machine_learning")
  #TODO change maybe outputpath
  output_path = output_data_path
  
  rmarkdown::render(
    input = "Machine Learning_Response Prediction/ML_preprocessing.Rmd",
    output_file = output_path,
    params = params_list,
    envir = new.env()
  )
  cat("Generated file:", output_filename, "\n")
}
