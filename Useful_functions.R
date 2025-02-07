# Useful Functions

####################################################
# File Management and Settings
####################################################

# This function creates a folder to save the results which is named after the folder-structure of the import-data path
# and returns the corresponding 
create_results_path <- function(inputdata_path, output_mainpath){
  # Get the part of the file-path describing the dealing with outliers
  outliers_part = basename(inputdata_path)
  # Get the part of the file-path describing the processing of RT
  grandparent_dir = dirname(dirname(inputdata_path))
  RTprocessing_part = basename(grandparent_dir)
  
  # Create a new path
  new_folder = paste(RTprocessing_part, outliers_part, sep = "_")
  output_path = file.path(output_mainpath, new_folder)
  dir.create(output_path, recursive = TRUE)
  
  return(output_path)
}

# Function to configure flextable settings and Word document formatting

flextable_settings <- function(
    word_orientation = "portrait" # Orientation: "portrait" or "landscape"
    ){
  # Set flextable defaults
  flextable::set_flextable_defaults(font.family = "Arial",
                         font.size = 8,
                         padding.bottom = 3,
                         padding.top = 3,
                         padding.left = 0.5,
                         paddings.right = 0.5,
                         #theme_fun = "theme_apa",
                         theme_fun = NULL,
                         text.align = "center",
                         line_spacing = 1.5)
  
  # Word document formatting 
  margins <- officer::page_mar(
    bottom = 2 / 2.54,
    top = 2.5 / 2.54,
    right = 2.5 / 2.54,
    left = 1 / 2.54
    #header = 0.5,
    #footer = 0.5,
    #gutter = 0.5
  )
  
  format_table <- officer::prop_section(
    page_size = officer::page_size(orient = word_orientation),
    page_margins = margins
  )
  
  return(format_table)
}

####################################################
# Statistical Analyses
####################################################

# Function to calculate an independent-sample Welch t-test for multiple comparisons 
# and store the results in a dataframe

t_test_mult_cols <- function(df_basis, cols, df_results_columns, grouping_variable) {
  # This function assumes that the grouping variable is 0-1 coded.
  
  # Initialize an empty list to store results
  results_list <- vector("list", length(cols))
  names(results_list) <- cols  # Assign column names dynamically
  #Create empty vector to store raw p-values
  p_values_raw <- numeric(length(cols)) 
  
  for (i in seq_along(cols)) {
    col <- cols[i] #seq_along for sequencing column indices
    
    # Extract data for each group
    group0 <- na.omit(df_basis[df_basis[[grouping_variable]] == 0, col])
    group1 <- na.omit(df_basis[df_basis[[grouping_variable]] == 1, col])
    
    # Perform t-test
    # We use the Welch-test as default, following Delacre et al., 2017 https://pure.tue.nl/ws/portalfiles/portal/80459772/82_534_3_PB.pdf
    results <- t.test(group0, group1, paired = FALSE, var.equal = FALSE)
    #Alternativ using formula method:
    #results <- t.test(df_basis[[col]] ~ df_basis[["Gruppe"]], paired = FALSE, var.equal = FALSE)
    
    # Store raw p-values for BH correction
    p_values_raw[i] <- results$p.value
    
    # Calculate Hedges g based on non-pooled standard-deviations as recommended in https://orbilu.uni.lu/bitstream/10993/57901/1/ES.pdf
    # using the package "effectsize"
    effsize_result <- effectsize::hedges_g(group0, group1,pooled_SD = FALSE)
    effsize <-effsize_result$Hedges_g
    
    # Dynamically assign results to the dataframe
    results_list[[col]] <- c(
      group_1_mean = round(mean(group1), 2),
      group_1_sd = round(sd(group1), 2),
      group_0_mean = round(mean(group0), 2),
      group_0_sd = round(sd(group0), 2),
      df = round(results$parameter[["df"]], 2),
      t_statistic = round(results$statistic[["t"]], 2),
      p_value = round(results$p.value, 2),
      effect_size = round(effsize, 2)
    )
  }
  df_results <- data.frame(do.call(rbind, results_list))
  
  # Adjust p-values using Benjamini-Hochberg method for multiple testing of related tasks
  p_values_adjusted <- p.adjust(p_values_raw, method = "BH")
  df_results$p_value_adjusted <- round(p_values_adjusted, 2)
  
  return(df_results)
}


# Function to perform Levene's test for homogeneity of variance on multiple independent variables

levene_test_mult_cols <- function(df_basis, cols, grouping_variable) {
  df <- data.frame(p_value = numeric(length(cols)))
  rownames(df) <- cols
  
  for (i in seq_along(cols)) {
    col <- cols[i]
    # Remove NA-cases per variable/task
    df_basis_nomissings <- df_basis[!is.na(df_basis[[col]]),]
    df_basis_nomissings[[col]] <- as.numeric(df_basis_nomissings[[col]])
    # Perform Levene's test
    levene_result <- car::leveneTest(df_basis_nomissings[[col]] ~ df_basis_nomissings[[grouping_variable]])
    df[col, "p_value"] <- round(levene_result[1, "Pr(>F)"], 4)
  }
  
  return(df)
}
