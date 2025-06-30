"""
Statistical Analysis of Neural Network Correlation Coefficients with/without Batch Normalization

This script performs a linear mixed-effects model analysis comparing correlation coefficients
between neural network models with and without batch normalization (BN).

Key Features:
- Automatically loads all available JSON files from specified directories
- Performs Fisher's z-transformation for normality
- Fits linear mixed-effects models with random intercepts
- Provides model diagnostics and effect size calculations
"""

import json
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def load_real_data(with_bn_root="neuron_interpretability/data/experiment_1/with_bn/cor_json",
                   without_bn_root="neuron_interpretability/data/experiment_1/without_bn/cor_json",
                   n_neurons=64):
    """
    Load correlation coefficient data from all JSON files in specified directories
    
    Parameters:
    - bn_root: Path to directory with BN group JSON files (default "data/with_bn/json")
    - no_bn_root: Path to directory with no-BN group JSON files (default "data/wiithout_bn/json")
    - n_neurons: Number of neurons per model (default 64)
    
    Returns:
    - DataFrame containing correlation coefficients with metadata
    - Also prints loading summary statistics
    
    Raises:
    - FileNotFoundError if specified directories don't exist
    """
    data = []
    with_bn_files_loaded = 0
    without_bn_files_loaded = 0
    
    # Process BN group files
    if os.path.exists(with_bn_root):
        for filename in os.listdir(with_bn_root):
            if filename.endswith("_cor.json"):
                model_id = filename.replace("_cor.json", "")
                json_path = os.path.join(with_bn_root, filename)
                
                with open(json_path, 'r') as f:
                    cor_data = json.load(f)
                    coefficients = cor_data.get("Decoder_0", [])    #Assuming "Decoder_0" is the target layer
                    
                    for neuron_idx in range(n_neurons):
                        data.append({
                            "model_id": f"bn_{model_id}",
                            "group": "with_BN",
                            "neuron_position": neuron_idx,
                            "correlation_coef": coefficients[neuron_idx] if neuron_idx < len(coefficients) else np.nan
                        })
                    with_bn_files_loaded += 1
    else:
        raise FileNotFoundError(f"BN directory not found: {with_bn_root}")
    
    # Process no-BN group files
    if os.path.exists(without_bn_root):
        for filename in os.listdir(without_bn_root):
            if filename.endswith("_cor.json"):
                model_id = filename.replace("_cor.json", "")
                json_path = os.path.join(without_bn_root, filename)
                
                with open(json_path, 'r') as f:
                    cor_data = json.load(f)
                    coefficients = cor_data.get("Decoder_0", [])
                    
                    for neuron_idx in range(n_neurons):
                        data.append({
                            "model_id": f"no_bn_{model_id}",
                            "group": "without_BN",
                            "neuron_position": neuron_idx,
                            "correlation_coef": coefficients[neuron_idx] if neuron_idx < len(coefficients) else np.nan
                        })
                    without_bn_files_loaded += 1
    else:
        raise FileNotFoundError(f"No-BN directory not found: {without_bn_root}")
    
    # Create DataFrame and print summary
    df = pd.DataFrame(data)
    
    return df

def main():
    # Load and prepare data
    df = load_real_data()
    
    # Data preprocessing
    df['z_coef'] = np.arctanh(df['correlation_coef'])  # Fisher's z-transform
    df["model_id"] = df["model_id"].astype("category")
    df["group"] = pd.Categorical(
        df["group"], 
        categories=["without_BN", "with_BN"],  # Reference level: without_BN
        ordered=False
    )

    # Fit linear mixed-effects model
    print("\nFitting linear mixed-effects model...")
    model = smf.mixedlm(
        "z_coef ~ group", 
        groups="model_id",  # Grouping variable
        re_formula="~1",    # Random intercept
        data=df
    )
    result = model.fit(reml=True)  # Use REML estimation
    
    # Save and display results
    print("\n=== Model Results ===")
    print(result.summary())

if __name__ == "__main__":
    main()