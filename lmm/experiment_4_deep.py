import json
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import Optional

def load_model_correlations(
    shallow_path: str,
    deep_path: str,
    n_neurons: int = 64,
    model_range: tuple = (1, 20)
) -> pd.DataFrame:
    """
    Load neuron correlation data from shallow and deep models.
    
    Args:
        shallow_path: Directory containing shallow model JSON files
        deep_path: Directory containing deep model JSON files
        n_neurons: Number of neurons expected per model
        model_range: Tuple of (start, end) model numbers to include
    
    Returns:
        DataFrame containing correlation data with columns:
        - model_id: Unique model identifier
        - group: Model type ("shallow" or "deep")
        - neuron_position: Neuron index
        - correlation_coef: Correlation coefficient
    """
    data = []
    
    # Helper function to load data for either model type
    def load_models(directory: str, model_type: str):
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
    
    # Load both model types
    load_models(shallow_path, "shallow")
    load_models(deep_path, "deep")
    
    return pd.DataFrame(data)


def analyze_model_comparison(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> smf.MixedLMResults:
    """
    Analyze and compare correlation data between model types.
    
    Args:
        df: Input DataFrame from load_model_correlations()
        output_dir: Optional directory to save results
        plot: Whether to generate visualization
    
    Returns:
        Fitted mixed model results
    """
    # Data preprocessing
    df['z_coef'] = np.arctanh(df['correlation_coef'])
    df['group'] = pd.Categorical(
        df['group'],
        categories=["shallow", "deep"],
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
    # Example usage - parameters should be configured by user
    shallow_data_path = "neuron_interpretability/data/experiment_1/with_bn/cor_json"
    deep_data_path = "neuron_interpretability/data/experiment_4/deep_model/cor_json"
    results_directory = "/path/to/save/results"
    
    # Load data
    df = load_model_correlations(
        shallow_path=shallow_data_path,
        deep_path=deep_data_path,
        n_neurons=64,
        model_range=(1, 20)
    )
    
    # Analyze data
    results = analyze_model_comparison(
        df=df,
        output_dir=results_directory
    )