import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import statsmodels.formula.api as smf

def load_neuron_correlations(
    base_path: str,
    activation_funcs: List[str],
    bn_positions: List[str],
    model_range: Tuple[int, int]
) -> pd.DataFrame:
    """
    Load neuron correlation data from JSON files and organize into DataFrame.
    
    Args:
        base_path: Root directory containing experiment data
        experiment_num: Experiment number (subdirectory name)
        activation_funcs: List of activation functions to include
        bn_positions: List of batch norm positions to include
        model_range: Tuple of (start, end) model numbers to include
    
    Returns:
        DataFrame with columns:
        | activation | bn_position | model_num | neuron_idx | correlation |
    """
    records = []

    for act in activation_funcs:
        for bn in bn_positions:
            for model_id in range(model_range[0], model_range[1] + 1):
                file_path = Path(base_path) / act / bn / "cor_json" / f"model_{model_id}_cor.json"

                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Convert correlations to float (handling "NA" values)
                neuron_corrs = [
                    0 if x == "NA" else float(x) 
                    for x in data["Decoder_0"]  # Assuming "Decoder_0" is the target layer
                ]
                
                # Record each neuron's data
                for neuron_idx, corr in enumerate(neuron_corrs):
                    records.append({
                        "activation": act,
                        "bn_position": bn,
                        "model_num": model_id,
                        "neuron_idx": neuron_idx,
                        "correlation": corr
                    })
    
    return pd.DataFrame(records)

def fisher_z_transform(corr: float) -> float:
    """Fisher z-transform with robust boundary handling"""
    corr = np.clip(corr, -0.99999, 0.99999)
    return 0.5 * np.log((1 + corr) / (1 - corr))

def analyze_neuron_correlations(
    df: pd.DataFrame,
    output_path: Optional[str] = None
) -> smf.MixedLMResults:
    """
    Perform mixed effects modeling on neuron correlation data.
    
    Args:
        df: Input DataFrame from load_neuron_correlations()
        output_path: Optional path to save results
    
    Returns:
        Fitted mixed model results
    """
    # Create unique model identifier
    df["unique_model_id"] = (
        df["activation"] + "_" + 
        df["bn_position"] + "_" + 
        df["model_num"].astype(str)
    )
    
    # Apply Fisher z-transform
    df["z_correlation"] = df["correlation"].apply(fisher_z_transform)
    
    # Fit mixed effects model
    model = smf.mixedlm(
        formula="z_correlation ~ activation * bn_position",
        data=df,
        groups=df["unique_model_id"],
        re_formula="~1"
    ).fit(reml=False)
    
    # Output results
    if output_path:
        with open(output_path, 'w') as f:
            f.write(model.summary().as_text())
    
    return model

if __name__ == "__main__":
    # Example usage - parameters should be set by user
    df = load_neuron_correlations(
        base_path="neuron_interpretability/data/experiment_2",
        activation_funcs=["LeakyReLU", "Tanh", "Sigmoid", "ReLU", "Swish"],
        bn_positions=["pre-activation_bn", "post-activation_bn"],
        model_range=(1, 20)
    )
    
    # Run analysis
    model = analyze_neuron_correlations(
        df,
        output_path="/path/to/save/results.txt"
    )