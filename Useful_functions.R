
# ===========================
# File Management and Settings
# ===========================

# ---- Create results paths ----
# This function creates folders to store the results. The folders are named after the 
# the processing of the data.

create_results_path <- function(inputdata_path, output_mainpath, response_criterion = NULL){
  # inputdata_path: path to the file with the preprocessed taskdata
  # Get the part of the file-path describing the dealing with outliers
  filename = basename(inputdata_path)
  parts_filename = strsplit(filename, "_")[[1]]
  
  # Get the part of the file-path describing the processing of outliers
  outliers_csv = parts_filename[3]
  outliers_part = gsub(".csv","",outliers_csv)
  
  # Get the part of the file-path describing the processing of RT
  RTprocessing_part = parts_filename[2]
  
  # Get new folder-name
  new_folder = paste(RTprocessing_part, outliers_part, sep = "_")
  
  # Create the new folder in the output-directory
  if (!is.null(response_criterion)) {
    output_path = file.path(output_mainpath, new_folder, response_criterion)
  } else {
    output_path = file.path(output_mainpath, new_folder)
  }
  dir.create(output_path, recursive = TRUE)
  
  return(output_path)
}

# ---- Standardize variable naming ----
# Function that uses a centralized label map to rename variables

variable_labels <- c(
  Subject = "Subject",
  Gruppe = "Group",
  Response = "Response",
  Alter = "Age",
  Geschlecht = "Sex (Female)",
  Abschluss = "Education",
  Abschluss_Gymnasium = "Education (High-School)",
  TwoBack_BIS_Foil = "2-Back: BIS Foil",
  TwoBack_BIS_Target = "2-Back: BIS Target",
  TwoBack_BIS_Total = "2-Back: BIS Total",
  Stroop_BIS_Congruent = "Stroop: BIS Congruent",
  Stroop_BIS_Incongruent = "Stroop: BIS Incongruent",
  Stroop_BIS_Diff_Score = "Stroop: BIS Incongruent-Congruent",
  NumberLetter_BIS_Repeat = "Number-Letter: BIS Repeat",
  NumberLetter_BIS_Switch = "Number-Letter: BIS Switch",
  NumberLetter_BIS_Diff_Score = "Number-Letter: BIS Switch-Repeat",
  StopSignal_SSRT = "Stop-Signal: SSRT",
  T1_BAT_FAS_score = "FSQ Score",
  BAT_T1 = "BAT Score",
  T1_BAT_BDI_II_score = "BDI-II Score",
  T1_BAT_STAI_T_score = "STAI-T Score",
  T1_BAT_BIS_11_score = "BIS-11 Score",
  T1_BAT_Kirby_k_score = "Kirby k",
  T1_BAT_CFC_14_score = "CFC-14 Score",
  T1_BAT_SRHI_score = "SRHI Score"
)

# TODO: to be adapted for different use cases
reorder_and_rename_rows <- function(df, col_name, label_map) {
  # Step 1: Create the desired order, keeping only names that exist in the data
  desired_order <- names(label_map)
  desired_order <- desired_order[desired_order %in% df[[col_name]]]
  
  # Step 2: Reorder the dataframe
  df <- df[match(desired_order, df[[col_name]]), ]
  
  # Step 3: Rename the entries
  df[[col_name]] <- label_map[df[[col_name]]]
  
  return(df)
}

# ---- Get relevant sample subset ----
# Function to get specific datasets
get_group_data <- function(data, group = NULL, response = NULL, criterion = "FSQ") {
  out <- data
  
  # Group filter
  if (!is.null(group)) {
    group_map <- c("HC" = 0, "Pat_Pre" = 1, "Pat_Post" = 2)
    selected_groups <- unname(group_map[group])
    
    out <- out %>% filter(Gruppe %in% selected_groups)
  }
  
  return(out)
}

# ---- Add socdem and clinical variables ----
# Function to ...
add_socdem_vars <- function(df, socdem_file_path, vars) {
  clin_socdem_data <- read.csv(socdem_file_path)
  
  # Check that requested columns exist
  missing_vars <- setdiff(vars, colnames(clin_socdem_data))
  if (length(missing_vars) > 0) {
    stop("These variables were not found in socdem file: ", paste(missing_vars, collapse = ", "))
  }
  
  out <- df %>%
    left_join(
      clin_socdem_data %>% select(Subject, all_of(vars)), 
      by = "Subject")
  
  return(out)
}

# ---- Add response variable ----
# Function to ...
add_response <- function(df, socdem_file_path, criterion) {
  clin_socdem_data <- read.csv(socdem_file_path)
  
  # Check that requested columns exist
  missing_vars <- setdiff(criterion, colnames(clin_socdem_data))
  if (length(missing_vars) > 0) {
    stop("These variables were not found in socdem file: ", paste(missing_vars, collapse = ", "))
  }
  
  out <- df %>%
    left_join(
      clin_socdem_data %>% select(Subject, all_of(criterion)), 
      by = "Subject"
      ) %>%
    rename(Response = all_of(criterion))
  
  return(out)
}

# ===========================
# Statistical Analyses
# ===========================

# ---- Dummy code "Abschluss" ----
# Function to create a dummy-variable "Abschluss_Gymnasium" to account for underrepresented 
# and meaningless categories

dummy_code_education <- function(data) {
  data[["Abschluss"]] <- as.factor(data[["Abschluss"]]) 
  levels(data[["Abschluss"]]) <- list("Hauptschule" = "1", "Realschule" = "2", "Gymnasium" = "3", "Anderes" = "4")
  
  data_recoded <- data
  
  # Set level "Anderes" (= other degree) to NA as it is not meaningful
  data_recoded[["Abschluss"]][data_recoded[["Abschluss"]] == "Anderes"] <- NA
  
  # Add Hauptschule to Realschule and create dummy-variable "Gymnasium"
  data_recoded[["Abschluss_Gymnasium"]] <- dplyr::recode(data_recoded[["Abschluss"]], "Hauptschule" = 0, "Realschule" = 0, "Gymnasium" = 1)
  
  return(list(data = data, data_recoded = data_recoded))
}


# ---- t-test for multiple comparisons ----
# Function to calculate an independent-sample Welch t-test for multiple comparisons 
# and store the results in a dataframe

t_test_mult_cols <- function(df_basis, cols, grouping_variable) {
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
    
    # Count missing values
    missings_group0 <- sum(is.na(df_basis[df_basis[[grouping_variable]] == 0, col]))
    missings_group1 <- sum(is.na(df_basis[df_basis[[grouping_variable]] == 1, col]))
    
    # Perform t-test
    # We use the Welch-test as default, following Delacre et al., 2017 https://pure.tue.nl/ws/portalfiles/portal/80459772/82_534_3_PB.pdf
    results <- t.test(group0, group1, paired = FALSE, var.equal = FALSE)
    #Alternativ using formula method:
    #results <- t.test(df_basis[[col]] ~ df_basis[["Gruppe"]], paired = FALSE, var.equal = FALSE)
    
    # Calculate Hedges g based on non-pooled standard-deviations as recommended in https://orbilu.uni.lu/bitstream/10993/57901/1/ES.pdf
    # using the package "effectsize"
    effsize_result <- effectsize::hedges_g(group0, group1,pooled_SD = FALSE)
    effsize <-effsize_result$Hedges_g
    
    # Set rounding precision based on variable name
    if (col == "T1_BAT_Kirby_k_score") {
      decimal_places <- 3
    } else if (col %in% c("NumberLetter_BIS_Repeat", "NumberLetter_BIS_Switch", "NumberLetter_BIS_Diff_Score",
                          "Stroop_BIS_Congruent", "Stroop_BIS_Incongruent", "Stroop_BIS_Diff_Score",
                          "TwoBack_BIS_Foil", "TwoBack_BIS_Target", "TwoBack_BIS_Total", "SSRT"))
      {
      decimal_places <- 2
    } else { # for questionnaire scores, age
      decimal_places <- 1
    }
    
    # Dynamically assign results to the dataframe
    results_list[[col]] <- c(
      group_0_mean = formatC(mean(group0), format = "f", digits = decimal_places),
      group_0_sd = formatC(sd(group0), format = "f", digits = decimal_places),
      group_1_mean = formatC(mean(group1), format = "f", digits = decimal_places),
      group_1_sd = formatC(sd(group1), format = "f", digits = decimal_places),
      statistic = round(results$statistic[["t"]], 2),
      df = round(results$parameter[["df"]], 1),
      p_value = round(results$p.value, 2),
      effect_size = round(effsize, 2),
      missings_group0 = missings_group0,
      missings_group1 = missings_group1
    )
    
    # Store raw p-values for BH correction
    p_values_raw[i] <- results$p.value
  }
  df_results <- data.frame(do.call(rbind, results_list))
  
  # Adjust p-values using Benjamini-Hochberg method for multiple testing of related tasks
  p_values_adjusted <- p.adjust(p_values_raw, method = "BH")
  df_results$p_value_adjusted <- round(p_values_adjusted, 2)
  
  return(df_results)
}


# ---- Chi-square test for multiple comparisons ----
# Function to calculate a Chi-square test for multiple comparisons and store the results in a dataframe

chi_sq_test_mult_cols <- function(df_basis, cols, grouping_variable){
  # This function assumes that the grouping variable is 0-1 coded and cols are categorical
  
  results_list <- vector("list", length(cols))
  names(results_list) <- cols
  p_values_raw <- numeric(length(cols))
  
  for (i in seq_along(cols)){
    col <- cols[i]
    
    # Build contingency table (removes NAs automatically)
    table_data <- table(df_basis[[col]], df_basis[[grouping_variable]])
    
    # Perform Chi-square test
    test_result <- chisq.test(table_data)
    
    # Group-wise data
    group0_data <- df_basis[df_basis[[grouping_variable]] == 0, col]
    group1_data <- df_basis[df_basis[[grouping_variable]] == 1, col]
    
    # Count missings
    missings_group0 <- sum(is.na(group0_data))
    missings_group1 <- sum(is.na(group1_data))
    
    # Get counts and percentages of category 1 in each group
    ## Group 0
    total_group0 <- length(group0_data)
    n_1_group0 <- sum(group0_data, na.rm = TRUE)
    pct_1_group0 <- round(100 * n_1_group0 / total_group0, 0)
    
    ## Group 1
    total_group1 <- length(group1_data)
    n_1_group1 <- sum(group1_data, na.rm = TRUE)
    pct_1_group1 <- round(100 * n_1_group1 / total_group1, 0)
    
    # Store raw p-values
    p_values_raw[i] <- test_result$p.value
    
    # Store results in a list
    results_list[[col]] <- c(
      n_1_group0 = n_1_group0,
      pct_1_group0 = pct_1_group0,
      n_1_group1 = n_1_group1,
      pct_1_group1 = pct_1_group1,
      statistic = round(unname(test_result$statistic), 2),
      df = unname(test_result$parameter),
      p_value = round(test_result$p.value, 2),
      missings_group0 = missings_group0,
      missings_group1 = missings_group1
    )
  }
  
  df_results <- data.frame(do.call(rbind, results_list), stringsAsFactors = FALSE)
  
  # Adjust p-values
  p_values_adjusted <- p.adjust(p_values_raw, method = "BH")
  df_results$p_value_adjusted <- round(p_values_adjusted, 2)
  
  return(df_results)
}


# ---- Levene's test on multiple IVs ----
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


# ===========================
# Results tables (for publication)
# ===========================

# ---- Flextable & Word settings ----
# Function to configure flextable settings and Word document formatting

flextable_settings <- function(word_orientation = "portrait"){ # Orientation: "portrait" or "landscape"
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

# ---- Create and save flextable ----
# Function to create and save a flextable, using formatting according to APA

create_save_flextable <- function(
    table_pub, results_path, file_name) {
  
  format_flextable_portrait <- flextable_settings(word_orientation = "portrait")
  format_flextable_landscape <- flextable_settings(word_orientation = "landscape")
  
  ft <- flextable(table_pub)
  
  # Set table properties
  ft <- set_table_properties(ft, width = 1, layout = "autofit")
  
  # Header in bold
  ft <- bold(ft, bold = TRUE, part = "header")
  
  # Alignments
  ft <- align(ft, j = 1, align = "left", part = "all") # first column
  ft <- align(ft, j = 2:ncol(table_pub), align = "center", part = "all") # rest
  
  # Export flextable
  save_as_docx(
    ft,
    path = file.path(results_path, file_name), 
    pr_section = format_flextable_portrait)
}


# ---- Concatenate test results ----
# Function to concatenate (t-)test results for dimensional and categorical variables 
# in a standardized way 

prepare_ttest_table <- function(ttest_table, var_type = c("dimensional", "categorical"), group_0, group_1) {
  type <- match.arg(var_type)
  
  # Convert rownames (dependent variables) to separate column
  t_test_table_pub <- cbind(dependent_variables = rownames(ttest_table), ttest_table)
  
  # Combine group-wise descriptives and reduce dataframe
  if (type == "dimensional") {
    # Format mean (SD)
    t_test_table_pub <- t_test_table_pub %>%
      mutate(
        group_0 = paste0(group_0_mean, " (", group_0_sd, ")"),
        group_1 = paste0(group_1_mean, " (", group_1_sd, ")"),
        `Comparison` = paste0("t(", df, ") = ", statistic, ", p = ", format(p_value_adjusted, nsmall = 2))
        ) %>%
      select(dependent_variables, group_0, group_1, Comparison, missings_group0, missings_group1)
    
  } else if (type == "categorical") {
    # Format count (percentage)
    t_test_table_pub <- t_test_table_pub %>%
      mutate(
        group_0 = paste0(n_1_group0, " (", pct_1_group0, "%)"),
        group_1 = paste0(n_1_group1, " (", pct_1_group1, "%)"),
        `Comparison` = paste0("χ²(", df, ") = ", statistic, ", p = ", format(p_value_adjusted, nsmall = 2))
      ) %>%
      select(dependent_variables, group_0, group_1, Comparison, missings_group0, missings_group1)
  }
  
  # Rename columns
  colnames(t_test_table_pub) <- c("Variable", group_0, group_1, "Statistic", paste0("Missings_", group_0), paste0("Missings_", group_1))
  
  return(t_test_table_pub)
}


# ---- Generate characteristics table (task-wise) ----

generate_characteristics_table <- function(data, task_name, grouping_variable, group_labels) {
  data %>%
    select({{ grouping_variable }}, Alter, Geschlecht, Abschluss_Gymnasium, 
           T1_BAT_FAS_score, BAT_T1, T1_BAT_BDI_II_score, T1_BAT_STAI_T_score) %>%
    mutate({{ grouping_variable }} := factor({{ grouping_variable }}, levels = c(0, 1), labels = group_labels),
           Geschlecht = factor(Geschlecht, levels = c(0, 1), labels = c("Male", "Female"))) %>%
    tbl_summary(by = {{ grouping_variable }},  
                digits = list(T1_BAT_BDI_II_score = c(1, 1),
                              T1_BAT_FAS_score = c(1, 1),
                              BAT_T1 = c(1, 1),
                              T1_BAT_STAI_T_score= c(1, 1)),
                label = list(
                  Alter ~ "Age",
                  Geschlecht ~ "Sex",
                  Abschluss_Gymnasium ~ "High Education",
                  T1_BAT_FAS_score ~ "FSQ Score",
                  BAT_T1 ~ "BAT Score",
                  T1_BAT_BDI_II_score ~ "BDI-II Score",
                  T1_BAT_STAI_T_score ~ "STAI-T Score"
                ),
                statistic = list(
                  all_continuous() ~ "{mean} ({sd})",
                  all_categorical() ~ "{n} ({p}%)"
                ),
                missing = "ifany" # or "no" to not display
    ) %>%
    modify_header(label ~ "**Characteristic**") %>%
    modify_spanning_header(
      starts_with("T1_BAT_") ~ "**Baseline Assessment**"
    ) %>%
    # Remove the row for "Male" in sex
    modify_table_body(
      ~ .x %>% filter(!(variable == "Geschlecht" & label == "Male"))
    ) 
}
