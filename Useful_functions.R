# Useful Functions

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