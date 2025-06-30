import json
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from typing import List, Optional

def load_neuron_correlations(
    root_dir: str,
    dropout_rates: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8],
    bn_conditions: List[str] = ["with_bn", "without_bn"],
    model_range: tuple = (1, 20),
    n_neurons: int = 64
) -> pd.DataFrame:
    """
    Load neuron correlation data from JSON files.
    
    Args:
        root_dir: Root directory containing experiment data
        dropout_rates: List of dropout rates to include
        bn_conditions: List of batch norm conditions ("bn"/"no_bn")
        model_range: Tuple of (start, end) model numbers to include
        target_layer: Specific neural network layer to analyze
        n_neurons: Number of neurons expected per model
    
    Returns:
        DataFrame containing correlation data with columns:
        - model_id: Unique model identifier
        - bn: Batch norm presence (1/0)
        - group: Combination of dropout and bn condition
        - neuron_position: Neuron index
        - correlation_coef: Correlation coefficient
        - dropout_rate: Numeric dropout rate
    """
    data = []
    
    for rate in dropout_rates:
        dropout_dir = os.path.join(root_dir, f"dropout_{rate}")
        for bn in bn_conditions:
            bn_dir = os.path.join(dropout_dir, bn)
            for model_num in range(model_range[0], model_range[1] + 1):
                json_path = os.path.join(bn_dir, f"cor_json/model_{model_num}_cor.json")

                if not os.path.exists(json_path):
                    continue
                
                with open(json_path, 'r') as f:
                    cor_data = json.load(f)
                    coefficients = cor_data.get("Decoder_0", [])    #   Assuming "Decoder_0" is the target layer
                    
                    for neuron_idx in range(n_neurons):
                        coef = coefficients[neuron_idx] if neuron_idx < len(coefficients) else np.nan
                        data.append({
                            "model_id": f"dropout_{rate}_{bn}_model_{model_num}",
                            "bn": 1 if bn == "with_bn" else 0,
                            "group": f"dropout_{rate}_{bn}",
                            "neuron_position": neuron_idx,
                            "correlation_coef": float(coef) if coef != "NA" else np.nan,
                            "dropout_rate": rate
                        })
    
    df = pd.DataFrame(data)
    return df

def analyze_correlations(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
) -> smf.MixedLMResults:
    """
    Analyze neuron correlation data using mixed effects modeling.
    
    Args:
        df: Input DataFrame from load_neuron_correlations()
        output_dir: Optional directory to save results
        plot: Whether to generate summary plots
    
    Returns:
        Fitted mixed model results
    """
    # Data preprocessing
    df['z_coef'] = np.arctanh(df['correlation_coef'])
    df['dropout_factor'] = pd.Categorical(
        df['dropout_rate'],
        categories=sorted(df['dropout_rate'].unique()),
        ordered=True
    )
    
    # Mixed effects modeling
    model = smf.mixedlm(
        "z_coef ~ C(dropout_factor) * bn",
        groups="model_id",
        re_formula="~1",
        data=df
    )
    result = model.fit(reml=False)
    
    # Save model results
    if output_dir:
        with open(os.path.join(output_dir, "model_results.txt"), 'w') as f:
            f.write(result.summary().as_text())
    
    return result

if __name__ == "__main__":
    # Example usage - parameters should be configured by the user
    data_dir = "neuron_interpretability/data/experiment_3"  # Set your data directory
    output_directory = "/path/to/save/results"  # Set your output directory
    
    # Load data
    df = load_neuron_correlations(
        root_dir=data_dir,
        dropout_rates=[0.0, 0.2, 0.4, 0.6, 0.8],
        bn_conditions=["with_bn", "without_bn"],
        model_range=(1, 20)
    )
    
    # Analyze data
    results = analyze_correlations(
        df=df,
        output_dir=output_directory
    )