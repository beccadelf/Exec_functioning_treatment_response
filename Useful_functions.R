# Useful Functions

####################################################
# File Management and Settings
####################################################

# This function creates a new folder named after the folder-structure the data imported was part of

create_results_folder <- function(inputdata_path, output_mainpath){
  # Get the part of the file-path describing the dealing with outliers
  outliers_part = basename(inputdata_path)
  # Get the part of the file-path describing the processing of RT
  grandparent_dir = dirname(dirname(inputdata_path))
  RTprocessing_part = basename(grandparent_dir)
  
  # Create a new path
  new_folder = paste(RTprocessing_part, outliers_part, sep = "_")
  output_path = file.path(output_mainpath, new_folder)
  dir.create(output_path)
  
  return(output_path)
}

# Function to configure flextable settings and Word document formatting

flextable_settings <- function(
    word_orientation = "portrait" # Orientation: "portrait" or "landscape"
    ){
  # Set flextable defaults
  set_flextable_defaults(font.family = "Arial",
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
  margins <- page_mar(
    bottom = 2 / 2.54,
    top = 2.5 / 2.54,
    right = 2.5 / 2.54,
    left = 2.5 / 2.54
    #header = 0.5,
    #footer = 0.5,
    #gutter = 0.5
  )
  
  format_table <- prop_section(
    page_size = page_size(orient = word_orientation),
    page_margins = margins
  )
  
  return(format_table)
}


####################################################
# Statistical Analyses
####################################################

# Function to calculate an independent-sample t-test for multiple comparisons 
# and store the results in a dataframe

t_test_mult_cols <- function(df_basis, cols, df_results_columns, grouping_variable) {
  # Initialize an empty results dataframe
  df_results <- data.frame(matrix(NA, nrow = length(cols), ncol = length(df_results_columns)))
  colnames(df_results) <- df_results_columns
  rownames(df_results) <- cols
  
  #Create empty vector to store raw p-values
  p_values_raw <- numeric(length(cols)) 
  
  for (i in seq_along(cols)) {
    col <- cols[i] #seq_along for sequencing column indices
    
    # Extract data for each group
    group0 <- na.omit(df_basis[df_basis[[grouping_variable]] == 0, col])
    group1 <- na.omit(df_basis[df_basis[[grouping_variable]] == 1, col])
    
    # Perform t-test
    results <- t.test(group0, group1, paired = FALSE, var.equal = FALSE)
    #Alternativ using formula method:
    #results <- t.test(df_basis[[col]] ~ df_basis[["Gruppe"]], paired = FALSE, var.equal = FALSE)
    
    # Store raw p-values for BH correction
    p_values_raw[i] <- results$p.value
    
    # Calculate Cohen's d using the cohen.d function (effect size)
    cohen_d_result <- cohen.d(group0, group1, hedges.correction = FALSE)
    cohen_d <- cohen_d_result$estimate
    
    # Calculate power using pwr.t.test
    n0 <- length(group0)
    n1 <- length(group1)
    power_result <- pwr.t.test(d = cohen_d, n = min(n0, n1), sig.level = 0.05, type = "two.sample", alternative = "greater")
    power <- power_result$power
    
    # Dynamically assign results to the dataframe
    df_results[col, ] <- c(
      round(mean(group1), 2),  # e.g., group_mean_Patients
      round(sd(group1), 2),    # e.g., sd_Patients
      round(mean(group0), 2),  # e.g., group_mean_HC
      round(sd(group0), 2),    # e.g., sd_HC
      round(results$parameter["df"], 2),  # df
      round(results$statistic, 2),        # t_statistic
      round(results$p.value, 2),          # p_value
      round(cohen_d, 2),                  # cohen_d
      round(power, 2)                     # power
    )
  }
  
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