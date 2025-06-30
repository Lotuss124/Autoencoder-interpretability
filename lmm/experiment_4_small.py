import json
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Optional, Tuple

def load_model_correlations(
    large_model_dir: str,
    small_model_dir: str,
    n_neurons: int = 64,
    model_range: Tuple[int, int] = (1, 20)
) -> pd.DataFrame:
    """
    Load neuron correlation data comparing large and small models.
    
    Args:
        large_model_dir: Directory containing large model JSON files
        small_model_dir: Directory containing small model JSON files
        n_neurons: Number of neurons per model
        model_range: Range of model numbers to include (start, end)
        
    Returns:
        DataFrame with correlation data containing columns:
        - model_id: Unique model identifier
        - group: Model type ("large" or "small")
        - neuron_position: Neuron index
        - correlation_coef: Correlation coefficient
    """
    data = []
    
    def load_model_data(directory: str, model_type: str):
        for model_num in range(model_range[0], model_range[1] + 1):
            json_file = os.path.join(directory, f"model_{model_num}_cor.json")
            
            if not os.path.exists(json_file):
                print(f"Warning: File not found - {json_file}")
                continue
                
            with open(json_file, 'r') as f:
                cor_data = json.load(f)
                coefficients = cor_data.get("Decoder_0", [])    # Assuming "Decoder_0" is the target layer
                
                for neuron_idx in range(n_neurons):
                    coef = coefficients[neuron_idx] if neuron_idx < len(coefficients) else np.nan
                    data.append({
                        "model_id": f"{model_type}_model_{model_num}",
                        "group": model_type,
                        "neuron_position": neuron_idx,
                        "correlation_coef": float(coef) if coef != "NA" else np.nan
                    })
    
    load_model_data(large_model_dir, "large")
    load_model_data(small_model_dir, "small")
    
    return pd.DataFrame(data)


def analyze_model_comparison(
    df: pd.DataFrame,
    output_dir: Optional[str] = None
) -> smf.MixedLMResults:
    """
    Analyze and compare correlation data between model types.
    
    Args:
        df: Input DataFrame from load_model_correlations()
        output_dir: Directory to save results (optional)
        plot: Whether to generate visualization
        
    Returns:
        Fitted mixed model results
    """
    # Data preprocessing
    df['z_coef'] = np.arctanh(df['correlation_coef'])
    df['group'] = pd.Categorical(
        df['group'],
        categories=["small", "large"],
        ordered=False
    )
    
    # Mixed effects modeling
    model = smf.mixedlm(
        "z_coef ~ group",
        groups="model_id",
        re_formula="~1",
        data=df
    )
    result = model.fit(reml=True)
    
    # Save results if output directory specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model results
        with open(os.path.join(output_dir, "model_results.txt"), 'w') as f:
            f.write(result.summary().as_text())
    
    return result


if __name__ == "__main__":
    # Example configuration - users should set these paths
    config = {
        "large_model_dir": "neuron_interpretability/data/experiment_1/with_bn/cor_json",
        "small_model_dir": "neuron_interpretability/data/experiment_4/small_data/cor_json",
        "output_dir": "/path/to/save/results",
        "n_neurons": 64,
        "model_range": (1, 20)
    }
    
    # Load data
    df = load_model_correlations(
        large_model_dir=config["large_model_dir"],
        small_model_dir=config["small_model_dir"],
        n_neurons=config["n_neurons"],
        model_range=config["model_range"]
    )
    
    # Analyze data
    results = analyze_model_comparison(
        df=df,
        output_dir=config["output_dir"]
    )