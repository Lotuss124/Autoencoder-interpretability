import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

class Config:
    # Directory structure configuration
    BASE_DIR = Path("neuron_interpretability")
    OUTPUT_DIR = BASE_DIR / "vector_outputs"

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Replace this with your actual encoder architecture
        # Example structure (delete and replace with yours):
        # self.encoder = nn.Sequential(
        #     nn.Linear(input_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Dropout(p),
        #     ...
        # )
        self.encoder = nn.Sequential()  # Your encoder here
        
        # Replace this with your actual decoder architecture
        # Example structure (delete and replace with yours):
        # self.decoder = nn.Sequential(
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.LeakyReLU(),
        #     nn.Dropout(p),
        #     ...
        # )
        self.decoder = nn.Sequential()  # Your decoder here

        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        return encoder, decoder

model = Autoencoder()
    
# Part 1: Zero out biases for all Linear layers
for layer in model.modules():
    if isinstance(layer, nn.Linear):
        layer.bias.data.zero_()  # Equivalent to layer.bias.data.fill_(0)

# Part 2: Zero out biases for all BatchNorm1d layers
for layer in model.modules():
    if isinstance(layer, nn.BatchNorm1d):
        layer.bias.data.zero_()

def get_layer_weights(layer_type: str, layer_idx: int) -> pd.DataFrame:
    """
    Extract weights from specified layer and apply processing
    
    Args:
        layer_type: 'encoder' or 'decoder'
        layer_idx: Index of layer to extract
        
    Returns:
        DataFrame containing processed weights with source layer metadata
    """
    with torch.no_grad():
        # Get specified layer
        layer = model.decoder[layer_idx] if layer_type == "decoder" else model.encoder[layer_idx]
        
        # Process weights (ReLU + scaling)
        weights = torch.relu(layer.weight.detach())
        df = pd.DataFrame(weights.numpy().T * 10000)
        
        # Store source layer info
        df.attrs["source_layer"] = (layer_type, layer_idx)  
        return df

def process_layer(layer_type: str, input_data: pd.DataFrame, start_idx: int):
    """
    Process layer outputs with automatic filename generation
    
    Args:
        layer_type: Type of layer ('encoder' or 'decoder')
        input_data: Input DataFrame with source layer metadata
        start_idx: Starting layer index for processing
    """
    # Get source layer info
    source_type, source_idx = input_data.attrs.get("source_layer", ("unknown", -1))
    
    with torch.no_grad():
        # Convert input to tensor
        if not (source_type == "encoder" and source_idx == 8):  # Assuming layer 8 is bottleneck
            input_data = input_data.loc[:299, :]

        tensor_data = torch.tensor(input_data.values, dtype=torch.float32)
        
        # Select layers to process
        layers = model.decoder[start_idx:] if layer_type == "decoder" else model.encoder[start_idx:]
        output = layers(tensor_data)
        
        # Generate filename
        if start_idx == 0 and layer_type == "decoder":
            filename = "encoder_last_layer_weight.csv"
        else:
            filename = f"{source_type}_layer{source_idx}_weight.csv"
            
        # Save results
        pd.DataFrame(output.detach().numpy()).to_csv(filename, index=False)

# ======================================================
# Functional Pairing Explanation:
#
# get_layer_weights() extracts the weight matrix from a SPECIFIED Linear layer
#   ↓
# process_layer() takes those weights and feeds them to THE INPUT OF THE NEXT Linear layer
#
# Key Points:
# 1. get_layer_weights() always targets Linear layers only
# 2. process_layer() always processes through:
#    - From: The output side of the current Linear layer 
#    - To: The input side of the next Linear layer
# 3. Layer indexing starts at 0 for all layer types
# ======================================================
weights = get_layer_weights("decoder", 0)  # Internally records ("decoder", 0)
process_layer("decoder", weights, start_idx=3)  # Auto-generates decoder_layer0_weight.csv