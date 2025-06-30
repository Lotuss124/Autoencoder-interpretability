"""
Autoencoder Analysis Pipeline

This script performs two main tasks:
1. Processes encoder outputs by tissue type (from a combined file to individual tissue files)
2. Computes decoder Jacobian matrices for each tissue type

Pipeline:
all_encoder.csv -> tissue-specific CSV files -> Jacobian matrices for each tissue
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path

# ========================
# Configuration Section
# ========================
class Config:
    # Directory structure configuration
    BASE_DIR = Path("neuron_interpretability")
    INPUT_DIR = BASE_DIR / "input_data"
    OUTPUT_DIR = BASE_DIR / "jacobian_matrices"
    ENCODER_TISSUE_DIR = BASE_DIR / "encoder_tissue_outputs"
    
    # File paths
    ENCODER_OUTPUT_FILE = INPUT_DIR / "encoder_layer_outputs" / "encoder_layer_9_output.csv"     # bottleneck layer outputs
    TISSUE_LABEL_FILE = INPUT_DIR / "tissue_appear_times.csv"    # Tissue names
    TSNE_LABEL_FILE = INPUT_DIR / "tsne_label.csv"       # t-SNE cluster labels
    MODEL_PATH = BASE_DIR / "models" / "Autoencoder.pth"    # Trained autoencoder

    # Create output directories if they don't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ENCODER_TISSUE_DIR.mkdir(parents=True, exist_ok=True)

# ========================
# Data Processing Functions
# ========================
def process_tissue_data():
    """
    Split  encoder outputs into tissue-specific files
    
    Reads:
    - Bottleneck layer outputs (encoder_layer_9_output.csv)
    - Tissue labels (tissue_appear_times.csv)
    - t-SNE cluster labels (tsne_label.csv)
    
    Writes:
    - Individual CSV files for each tissue type
    """
    print("Processing tissue data...")
    
    # Load data files
    df = pd.read_csv(Config.ENCODER_OUTPUT_FILE).drop(df.columns[0], axis=1)
    tissue_labels = pd.read_csv(Config.TISSUE_LABEL_FILE, index_col=0)
    tsne_labels = pd.read_csv(Config.TSNE_LABEL_FILE).iloc[:, 1]
    
    print(f"Loaded data with shape: {df.shape}")
    print(f"Found {len(tissue_labels)} tissue types")
    
    # Process each tissue type
    tissue_names = tissue_labels.index  # 无需转为list
    for i, tissue_name in enumerate(tissue_names, 1):  # 从1开始计数
        print(f"{i}. Processing {tissue_name}...")
        
        # Filter data for current tissue
        tissue_data = df[tsne_labels == tissue_name]
        
        # Save to tissue-specific file
        output_path = Config.ENCODER_TISSUE_DIR / f"encoder_{tissue_name}_data.csv"
        tissue_data.to_csv(output_path, index=True)
        
    print("Tissue data processing complete!\n")

# ========================
# Autoencoder Analysis Functions
# ========================
def compute_jacobian_matrices():
    """
    Compute decoder Jacobian matrices for each tissue type
    
    For each tissue:
    1. Loads the tissue-specific encoder outputs
    2. Computes Jacobian matrices (decoder gradients)
    3. Saves matrices for further analysis
    """
    print("Computing Jacobian matrices...")
    
    # Load autoencoder model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(Config.MODEL_PATH).to(device)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded on {device}")
    
    # Load tissue names
    tissue_labels = pd.read_csv(Config.TISSUE_LABEL_FILE, index_col=0)
    
    for tissue_name, _ in tissue_labels.iterrows():
        print(f"\nProcessing {tissue_name}...")
        
        # Load tissue data
        input_path = Config.BASE_DIR / "encoder_matrix" / f"encoder_{tissue_name}_data.csv"
        data = pd.read_csv(input_path, index_col=0)
        data_tensor = torch.tensor(data.values.astype(np.float32), requires_grad=True).to(device)
        
        # Compute Jacobian matrices
        grad_matrices = []
        for i, sample in enumerate(data_tensor):
            jacobian = compute_sample_jacobian(model, sample.unsqueeze(0))
            grad_matrices.append(jacobian)
            
            if (i+1) % 10 == 0:
                print(f"  Processed sample {i+1}/{len(data_tensor)}")
        
        # Save results
        output_path = Config.OUTPUT_DIR / f"matrices_{tissue_name}.pth"
        torch.save(grad_matrices, output_path)
        print(f"Saved {len(grad_matrices)} matrices to {output_path}")

def compute_sample_jacobian(model, sample):
    """
    Compute Jacobian matrix for a single sample
    
    Args:
        model: Autoencoder model
        sample: Input sample (encoder output)
        
    Returns:
        Jacobian matrix (gradients of decoder output w.r.t. input)
    """
    # Compute gradients
    jacobian = torch.autograd.functional.jacobian(model.decoder, sample)
    
    # Reshape based on model architecture
    # Note: Adjust these dimensions according to your model
    output_dim = 60498  # Should match decoder output dimension
    latent_dim = 64     # Should match encoder bottleneck dimension
    return jacobian.view(output_dim, latent_dim).t()

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    # Step 1: Process tissue data
    process_tissue_data()
    
    # Step 2: Compute Jacobian matrices
    compute_jacobian_matrices()
    
    print("\nPipeline execution complete!")