## Function: Calc SSRT with 
#This function is based on the function https://github.com/agleontyev/SSRTcalc/blob/master/R/Integration_adaptiveSSD.R
#for calculating the SSD based on integration.

integration_adaptiveSSD_more_info <- function(df, stop_col, rt_col, acc_col, ssd_col, response_col) {
  
  # Get go-trials and replace omission errors with maximum RT
  go_trials = df[ which(df[,stop_col] == 0),]
  max_RT <- max(go_trials$RT)
  # Get number of omission errors
  prob_omission <- nrow(go_trials[is.na(go_trials[response_col]), ]) / nrow(go_trials)
  go_trials_clean <- go_trials
  go_trials_clean[is.na(go_trials[response_col]),c("RT")] <- max_RT
  
  # Prepare stop-trials
  stop_trials <- df[ which(df[,stop_col]==1), ]
  # Get number of correct stop-trials
  stop_count <- sum(stop_trials[,acc_col])
  # Get probability of responding on a stop trials
  prob_stopresponse = (nrow(stop_trials)-stop_count)/nrow(stop_trials)
  
  # Calculate the difference between mean RT for unsuccesful stop-trials and than the mean RT for go-trials
  mean_RT_go <- mean(go_trials[!is.na(go_trials[response_col]),c("RT")])
  mean_RT_unsucces_stop <- mean(stop_trials[!is.na(stop_trials[response_col]),c("RT")])
  diff_go_stop <- mean_RT_go - mean_RT_unsucces_stop
  
  overall_prob = 1 - stop_count/nrow(stop_trials)
  df1 <- go_trials_clean[order(go_trials_clean[,rt_col], na.last = NA) , ]
  nrt <- length(df1[,rt_col])
  nthindex = as.integer(round(nrt*overall_prob))
  meanssd = mean(stop_trials[, ssd_col], na.rm =TRUE)
  nthrt <- df1[,rt_col][nthindex]
  ssrt_raw <- nthrt - meanssd
  
  if(isTRUE(ssrt_raw <= 0)){
    ssrt = NA
  } else {
    ssrt = ssrt_raw
  }
  # Save only single values!
  list_results = list(
    ssrt = ssrt,
    stop_count = stop_count,
    diff_go_stop = diff_go_stop,
    prob_stopresponse =  prob_stopresponse,
    prob_omission = prob_omission,
    overall_prob = overall_prob,
    meanssd = meanssd,
    nthindex = nthindex,
    nthrt = nthrt,
    ssrt_raw = ssrt_raw
  )
  return(list_results)
}