# ========================
# Configuration Section
# ========================
# Directory and file paths
BASE_DIR <- "neuron_interpretability"
COR_PLOT_DIR <- file.path(BASE_DIR, "cor_plot")
COR_JSON_DIR <- file.path(BASE_DIR, "cor_json")

# Gene set and annotation files
GENE_ATLAS_DIR <- file.path(BASE_DIR, "gene_atlas")
PATHWAY_FILE <- file.path(GENE_ATLAS_DIR, "tissue.specific.genes.RData")
ROWNUMBERS_FILE <- file.path(BASE_DIR, "input_data/gene_name_60498.csv")  # 60,498 genes
# ROWNUMBERS_FILE <- file.path(BASE_DIR, "together_work/vector/bigmodel/gene_name_19594.csv")  # 19,594 genes

# Analysis-specific paths
VECTOR_PATH <- file.path(BASE_DIR, "vector_outputs")
MEDIAN_PATH <- file.path(BASE_DIR, "training_ouputs")

# Create output directories if they don't exist
dir.create(COR_PLOT_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(COR_JSON_DIR, showWarnings = FALSE, recursive = TRUE)

# ========================
# Library Imports
# ========================
library(GSVA)
library(data.table)
library(dplyr)
library(ggplot2)
library(fgsea)
library(parallel)
library(foreach)
library(doParallel)
library(jsonlite)

# ========================
# Data Loading Functions
# ========================
load_data <- function() {
    # """
    # Load required data for analysis
    # 
    # Returns:
    # pathway_standard: Tissue-specific gene sets
    # rownames: Gene identifiers
    # """
  # Load tissue-specific gene sets
  load(PATHWAY_FILE)
  pathway_standard <- tissue.specific.genes
  
  # Load gene identifiers
  rownames <- fread(ROWNUMBERS_FILE, header = FALSE, col.names = c('RowNames'))
  
  return(list(pathway_standard = pathway_standard, rownames = rownames))
}

# ========================
# Analysis Functions
# ========================
run_analysis <- function(i, gene_data, output, rowname) {
    # """
    # Perform enrichment analysis for a single gene set
    # 
    # Args:
    #     i: Index of current analysis
    #     gene_data: Gene expression data
    #     output: Reference values for correlation
    #     rowname: Row names for matching
    #     
    # Returns:
    #     Correlation coefficient between enrichment scores and reference values
    # """
  cat(sprintf("Processing sample %d...\n", i))
  
  # Preprocess gene data
  gene_data_x <- cbind(as.data.frame(rownames), gene_data)
  colnames(gene_data_x) <- c("RowNames", "Value")
  
  gene_data_x <- gene_data_x %>%
    group_by(RowNames) %>%
    summarise(Value = mean(Value, na.rm = TRUE)) %>%
    arrange(Value)
  
  x <- setNames(gene_data_x$Value, gene_data_x$RowNames)
  
  # Run gene set enrichment analysis
  fgseaRes <- fgsea(pathways = pathway_standard,
                    stats    = x,
                    eps      = 0.0,
                    minSize  = 15,
                    maxSize  = 500)
  
  # Prepare results
  result_df <- data.frame(
    Tissue = gsub(" ", "", fgseaRes$pathway),
    MeanScore = fgseaRes$ES
  )
  
  # Merge with reference values
  column <- data.frame(Column_Value = output, Row_Name = rowname)
  merged_df <- merge(result_df, column, by.x = "Tissue", by.y = "Row_Name")
  
  # Calculate correlation
  correlation <- cor(merged_df$MeanScore, merged_df$Column_Value)
  
  return(correlation)
}

perform_layer_analysis <- function(layer_name, num_samples, vector_path, median_path) {
    # """
    # Perform analysis for a specific layer
    # 
    # Args:
    #     layer_name: Name of the layer (e.g., "encoder_layer4")
    #     num_samples: Number of samples to process
    #     vector_path: Path to vector data
    #     median_path: Path to median data
    #     
    # Returns:
    #     List of correlation results
    # """
  cat(sprintf("\nStarting analysis for %s...\n", layer_name))
  
  # Load data
  vector_file <- file.path(vector_path, sprintf("%s_weight.csv", layer_name))
  median_file <- file.path(median_path, sprintf("%s_grouped_median.csv", layer_name))
  
  vector_data <- read.csv(vector_file)
  median_data <- read.csv(median_file, row.names = 1)
  
  # Run analysis in parallel
  results <- foreach(i = 1:num_samples, .combine = c) %dopar% {
    run_analysis(i, t(vector_data[i, ]), median_data)
  }
  
  cat(sprintf("Completed analysis for %s\n", layer_name))
  return(results)
}

# ========================
# Visualization Functions
# ========================
save_correlation_plot <- function(results, names, filename) {
    # """
    # Save correlation results as boxplot
    # 
    # Args:
    #     results: List of result vectors
    #     names: Names for each result set
    #     filename: Output file path
    # """
  cat(sprintf("Saving correlation plot to %s...\n", filename))
  
  pdf(filename, width = 12, height = 6)
  boxplot(results, names = names)
  dev.off()
}

save_correlation_json <- function(results, filename) {
    # """
    # Save correlation results as JSON
    # 
    # Args:
    #     results: List of result vectors
    #     filename: Output file path
    # """
  cat(sprintf("Saving correlation results to %s...\n", filename))
  write_json(results, filename, pretty = TRUE)
}

# ========================
# Main Execution
# ========================
# Step 1: Initialize parallel processing
cat("Initializing parallel processing...\n")
num_cores <- 12  # Adjust based on available resources
cl <- makeCluster(num_cores)
registerDoParallel(cl)
clusterEvalQ(cl, {
  library(dplyr)
  library(fgsea)
})

# Step 2: Load required data
cat("Loading pathway and gene data...\n")
data_objects <- load_data()
pathway_standard <- data_objects$pathway_standard
rownames <- data_objects$rownames

# Step 3: Perform analyses
cat("\nStarting layer analyses...\n")

# Encoder analyses
results_encoder_0 <- perform_layer_analysis("encoder_layer4", 300, VECTOR_PATH, MEDIAN_PATH)
results_encoder_4 <- perform_layer_analysis("encoder_layer8", 300, VECTOR_PATH, MEDIAN_PATH)
results_encoder_8 <- perform_layer_analysis("encoder_last_layer", 300, VECTOR_PATH, MEDIAN_PATH)

# Decoder analyses
results_decoder_0 <- perform_layer_analysis("decoder_layer0", 32, VECTOR_PATH, MEDIAN_PATH)
results_decoder_4 <- perform_layer_analysis("decoder_layer4", 300, VECTOR_PATH, MEDIAN_PATH)
results_decoder_8 <- perform_layer_analysis("decoder_layer8", 300, VECTOR_PATH, MEDIAN_PATH)

# Step 4: Save results
cat("\nSaving results...\n")

# Prepare results list
results_list <- list(
  Encoder_0 = results_encoder_0,
  Encoder_4 = results_encoder_4,
  Encoder_8 = results_encoder_8,
  Decoder_0 = results_decoder_0,
  Decoder_4 = results_decoder_4,
  Decoder_8 = results_decoder_8
)

# Save visualizations and data
plot_filename <- file.path(COR_PLOT_DIR, "cor_plot.pdf")
save_correlation_plot(
  results_list,
  c('Encoder_0', 'Encoder_4', 'Encoder_8', 'Decoder_0', 'Decoder_4', 'Decoder_8'),
  plot_filename
)

json_filename <- file.path(COR_JSON_DIR, "cor_plot.json")
save_correlation_json(results_list, json_filename)

# Step 5: Cleanup
cat("Stopping parallel cluster...\n")
stopCluster(cl)

cat("\nAnalysis pipeline execution complete!\n")