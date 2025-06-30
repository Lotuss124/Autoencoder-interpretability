"""
Autoencoder Training Pipeline with Layer Output Analysis

This script trains a symmetric autoencoder model for transcriptomic data analysis,
and processes the layer outputs to generate tissue-specific median values.

Pipeline:
GTEx data -> Preprocessing -> Autoencoder Training -> Layer Output Extraction -> Median Calculation by Tissue
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import os

# ========================
# Configuration Section
# ========================
class Config:
    # Directory structure configuration
    BASE_DIR = Path("neuron_interpretability")
    INPUT_DIR = BASE_DIR / "input_data"
    OUTPUT_DIR = BASE_DIR / "training_outputs"
    
    # File paths
    RAW_DATA_FILE = INPUT_DIR / "gtex_RSEM_gene_tpm.csv"          # Raw GTEx TPM data
    TISSUE_LABEL_FILE = INPUT_DIR / "tissue_name.csv"             # Tissue names
    MODEL_SAVE_DIR = BASE_DIR / "models"                          # Model save location
    
    # Layers to save and analyze (1-based indexing)
    SAVE_LAYERS = [3, 6, 7, 12, 15]  # Save specific encoder/decoder layers

    # Create directories if they don't exist
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# Model Architecture
# ========================
class Autoencoder(nn.Module):
    """Symmetric Autoencoder Architecture for Transcriptomic Data (60498→2048→512→64→512→2048→60498)"""
    def __init__(self):
        super().__init__()
        
        # Encoder Network (Gene Expression → Latent Space)
        # ====== ENCODER ARCHITECTURE ====== (Copy & paste into skeleton)
        self.encoder = nn.Sequential(
            # Layer 1: 60498 (genes) → 2048
            nn.Linear(60498, 2048),
            nn.BatchNorm1d(2048),           # Batch normalization
            nn.LeakyReLU(0.2), # Paper-specified α=0.2
            nn.Dropout(0.3),                # Paper-specified dropout rate
            
            # Layer 2: 2048 → 512
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Bottleneck Layer: 512 → 64 (no activation/dropout)
            nn.Linear(512, 64)  # Direct latent representation output
        )
        
        # Decoder Network (Latent Space → Gene Expression)
        # ====== DECODER ARCHITECTURE ====== (Should mirror encoder)
        self.decoder = nn.Sequential(
            # Layer 1: 64 → 512
            nn.Linear(64, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Layer 2: 512 → 2048
            nn.Linear(512, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2048, 60498)  
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Kaiming Uniform Initialization for LeakyReLU"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass through autoencoder"""
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon
    
    def get_layer_outputs(self, x):
        """Extract intermediate layer activations"""
        encoder_outputs = []
        for layer in self.encoder:
            x = layer(x)
            encoder_outputs.append(x)
        
        decoder_outputs = []
        for layer in self.decoder:
            x = layer(x)
            decoder_outputs.append(x)
        
        return encoder_outputs, decoder_outputs

def process_layer_outputs(csv_file_path, output_file_path):
    """
    Process layer output CSV files to calculate tissue-specific medians
    
    Args:
        csv_file_path: Path to input CSV file (layer outputs)
        output_file_path: Path to save processed median values
    """
    try:
        # Read CSV file with tissue labels in first column
        df = pd.read_csv(csv_file_path)
        
        # Group by tissue name and calculate median
        grouped_median = df.groupby(df.columns[0]).median(numeric_only=True)
        
        # Reset index to make tissue names a column
        grouped_median.reset_index(inplace=True)
        
        # Save results
        grouped_median.to_csv(output_file_path, index=False)
        
        print("Processing successful! Results:")
        print(grouped_median)
        print(f"Results saved to: {output_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found - {csv_file_path}")
    except Exception as e:
        print(f"Error processing {csv_file_path}: {e}")

def save_specified_layers_outputs(model, output_dir, tsne_label_path, layers_to_save):
    """
    Save outputs of specified model layers and automatically process them
    
    Args:
        model: Trained autoencoder model
        output_dir: Directory to save layer outputs
        tsne_label_path: Path to tissue labels CSV
        layers_to_save: List of layer indices to process
    """
    model.eval()
    _, _, full_data = load_and_preprocess_data()
    full_dataset = TensorDataset(full_data, full_data)
    data_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    
    # Load tissue labels
    tsne_labels = pd.read_csv(tsne_label_path, index_col=0)
    sample_names = tsne_labels.iloc[:, 0]
    
    if len(sample_names) != len(data_loader.dataset):
        raise ValueError("Label count does not match dataset size")

    # Initialize storage
    layer_outputs = {
        'encoder': [[] for _ in model.encoder],
        'decoder': [[] for _ in model.decoder]
    }

    # Process batches
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            enc_outs, dec_outs = model.get_layer_outputs(x)
            
            for i, out in enumerate(enc_outs):
                layer_outputs['encoder'][i].append(out.cpu().numpy().squeeze())
            for i, out in enumerate(dec_outs):
                layer_outputs['decoder'][i].append(out.cpu().numpy().squeeze())

    # Save and process specified layers
    for layer_idx in layers_to_save:
        if 1 <= layer_idx <= len(model.encoder):
            layer_type = 'encoder'
            data = np.concatenate(layer_outputs['encoder'][layer_idx-1], axis=0)
            layer_num = layer_idx
        elif len(model.encoder) < layer_idx <= (len(model.encoder) + len(model.decoder)):
            layer_type = 'decoder'
            data = np.concatenate(layer_outputs['decoder'][layer_idx - len(model.encoder) - 1], axis=0)
            layer_num = layer_idx - len(model.encoder)
        else:
            print(f"Invalid layer index {layer_idx}. Skipping.")
            continue

        # Save raw layer outputs
        raw_output_path = os.path.join(output_dir, f"{layer_type}_layer_{layer_num}_output.csv")
        pd.DataFrame(data, index=sample_names).to_csv(raw_output_path, index_label="Tissue_name")
        print(f"Saved {layer_type} layer {layer_num} output to {raw_output_path}")

        # Process the layer outputs to calculate tissue medians
        median_output_path = os.path.join(output_dir, f"{layer_type}_grouped_median_{layer_num}.csv")
        process_layer_outputs(raw_output_path, median_output_path)

# ========================
# Data Processing Functions
# ========================
def load_and_preprocess_data():
    """Load and preprocess GTEx data with log2(TPM + 1) transformation"""
    print("Loading and preprocessing data...")
    
    # Load and transpose data
    data = pd.read_csv(Config.RAW_DATA_FILE, sep="\t", index_col=0).T
    
    # Apply log2 transformation
    data = np.log2(np.exp2(data) - 0.001 + 1).astype(np.float16)
    
    # Convert to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Train-test split
    train_data, test_data = train_test_split(data_tensor, test_size=0.2, random_state=42)
    full_data = ConcatDataset([train_data, test_data])

    return train_data, test_data, full_data

# ========================
# Training Functions
# ========================
def train_model():
    """Main training procedure with layer output analysis"""
    print(f"Using device: {device}")
    
    # Load data
    train_data, test_data, _ = load_and_preprocess_data()
    
    # Create dataloaders
    train_dataset = TensorDataset(train_data, train_data)
    test_dataset = TensorDataset(test_data, test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    # Initialize model
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(100):
        model.train()
        train_loss = 0.0

        for (inputs, targets) in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            _, reconstructions = model(inputs)
            loss = criterion(reconstructions, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Validation
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for (inputs, targets) in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                _, reconstructions = model(inputs)
                test_loss += criterion(reconstructions, targets).item() * inputs.size(0)
        
        # Calculate epoch statistics
        train_loss /= len(train_loader.dataset)
        test_loss /= len(test_loader.dataset)

        print(f"Epoch {epoch+1:03d}/100 | "
              f"Train Loss: {train_loss:.6f} | "
              f"Test Loss: {test_loss:.6f}")

    # Save model and process layer outputs
    save_specified_layers_outputs(
        model=model,
        output_dir=Config.OUTPUT_DIR,
        tsne_label_path=Config.TISSUE_LABEL_FILE,
        layers_to_save=Config.SAVE_LAYERS
    )
    
    # Save final model
    torch.save({'model_state_dict': model.state_dict()}, "models/Autoencoder.pth")
    print("Training complete!")

# ========================
# Main Execution
# ========================
if __name__ == "__main__":
    train_model()