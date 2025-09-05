import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math
import pandas as pd

def _get_title_suffix(model_name, use_hadamard, exclude_tokens=None):
    """Helper to create a title suffix for plots."""
    hadamard_str = " - Hadamard Transformed" if use_hadamard else ""
    exclude_str = ""
    if exclude_tokens:
        exclude_str = f"\n(Excluding: {', '.join(map(repr, exclude_tokens))[:50]}...)"
    return f"({model_name}{hadamard_str}){exclude_str}"

def _get_filename_suffix(use_hadamard, exclude_tokens=None):
    """Helper to create a filename suffix for plots."""
    hadamard_suffix = "_hadamard" if use_hadamard else ""
    exclude_suffix = "_excluded" if exclude_tokens else ""
    return f"{hadamard_suffix}{exclude_suffix}"

def plot_layer_errors(layer_errors, model_name: str, bits: int, granularity: str, use_hadamard: bool = False, exclude_tokens: list = None):
    """
    Plots the average quantization error for each layer.

    Args:
        layer_errors (dict): A dictionary mapping layer index to a list of errors.
        model_name (str): The name of the model for the plot title.
        bits (int): The number of bits used for quantization.
        granularity (str): The quantization granularity.
    """
    int_keys = sorted([k for k in layer_errors.keys() if isinstance(k, int)])
    
    if not int_keys:
        print("No layer data to plot.")
        return

    avg_errors = [np.mean(layer_errors[k]) for k in int_keys]

    plt.figure(figsize=(15, 7))
    plt.plot(int_keys, avg_errors, marker='o', linestyle='-')
    title_suffix = _get_title_suffix(model_name, use_hadamard, exclude_tokens)
    plt.title(f'Layer-wise Average Activation Quantization Error\n({model_name} - {bits}-bit, {granularity}) {title_suffix}')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xticks(int_keys)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    filename = f"layer_errors_{model_name.replace('/', '_')}_{bits}bit_{granularity}{_get_filename_suffix(use_hadamard, exclude_tokens)}.png"
    plt.savefig(filename)
    print(f"Saved layer-wise error plot to {filename}")
    plt.close()

def plot_module_errors(module_errors, model_name: str, bits: int, granularity: str, use_hadamard: bool = False, exclude_tokens: list = None):
    """
    Plots the quantization error for each module type across all layers.

    Args:
        module_errors (dict): A dictionary mapping full module name to its error.
        model_name (str): The name of the model for the plot title.
        bits (int): The number of bits used for quantization.
        granularity (str): The quantization granularity.
    """
    # Restructure data: {module_type: {layer_index: error}}
    grouped_by_module = defaultdict(dict)
    for name, error in module_errors.items():
        parts = name.split('.')
        if 'layers' in parts and len(parts) > 3:
            try:
                layer_index = int(parts[2])
                module_type = ".".join(parts[3:]) # e.g., 'mlp.down_proj', 'self_attn.q_proj'
                grouped_by_module[module_type][layer_index] = error
            except (ValueError, IndexError):
                continue
    
    if not grouped_by_module:
        print("No module data to plot.")
        return

    plt.figure(figsize=(15, 7))
    for module_type, layer_data in sorted(grouped_by_module.items()):
        if layer_data:
            layers = sorted(layer_data.keys())
            errors = [layer_data[l] for l in layers]
            plt.plot(layers, errors, marker='o', linestyle='--', label=module_type)

    title_suffix = _get_title_suffix(model_name, use_hadamard, exclude_tokens)
    plt.title(f'Module-wise Activation Quantization Error Across Layers\n({model_name} - {bits}-bit, {granularity}) {title_suffix}')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend

    filename = f"module_errors_{model_name.replace('/', '_')}_{bits}bit_{granularity}{_get_filename_suffix(use_hadamard, exclude_tokens)}.png"
    plt.savefig(filename)
    print(f"Saved module-wise error plot to {filename}")
    plt.close()


def plot_top_token_errors_by_module(module_top_tokens, model_name, bits, granularity, use_hadamard: bool = False, exclude_tokens: list = None):
    """
    Plots the top K tokens with the highest average quantization error for multiple modules in subplots.

    Args:
        module_top_tokens (dict): Dict mapping module type to a dict containing 'top_tokens' and 'median_mse'.
                                  'top_tokens' is a list of tuples (avg_error, std_dev, count, token_text).
        model_name (str): The name of the model for the plot title.
        bits (int): The number of bits used for quantization.
        granularity (str): The quantization granularity.
    """
    if not module_top_tokens:
        print("No token error data to plot.")
        return

    num_modules = len(module_top_tokens)
    ncols = 2
    nrows = math.ceil(num_modules / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 7 * nrows), squeeze=False)
    title_suffix = _get_title_suffix(model_name, use_hadamard, exclude_tokens)
    fig.suptitle(f'Top Tokens with Highest Avg. Activation Quantization Error ({model_name} - {bits}-bit, {granularity}) {title_suffix}', fontsize=20, y=0.96)

    # Define a color cycle for the subplots
    colors = plt.get_cmap('tab10').colors 
    sorted_module_names = sorted(module_top_tokens.keys())

    for i, module_name in enumerate(sorted_module_names):
        ax = axes[i // ncols, i % ncols]
        plot_data = module_top_tokens[module_name]
        top_tokens = plot_data.get("top_tokens", [])
        median_mse = plot_data.get("median_mse", 0.0)
        
        if not top_tokens:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(module_name)
            continue

        top_tokens.reverse()  # Reverse for horizontal bar chart
        
        avg_errors = [item[0] for item in top_tokens]
        std_devs = [item[1] for item in top_tokens]
        counts = [item[2] for item in top_tokens]
        token_texts = [repr(item[3]) for item in top_tokens]

        # Create y-axis labels that include the token count
        y_labels = [f"{text} (n={count})" for text, count in zip(token_texts, counts)]

        # Plot the bars WITHOUT error whiskers
        bars = ax.barh(y_labels, avg_errors, color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_xlabel(f'Average MSE (Median of all tokens: {median_mse:.4f})')
        ax.set_title(f'Module: {module_name}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='x', linestyle='--', alpha=0.6)

        # Add text labels for both average MSE and standard deviation next to the bars
        for bar, std_val in zip(bars, std_devs):
            width = bar.get_width()
            ax.text(width, 
                    bar.get_y() + bar.get_height() / 2,
                    f'±{std_val:.4f}',
                    va='center',
                    ha='left')

    # Hide any unused subplots
    for i in range(num_modules, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout(rect=[0, 0, 1, 0.94]) # Adjust layout for suptitle

    filename = f"top_token_errors_by_module_{model_name.replace('/', '_')}_{bits}bit_{granularity}{_get_filename_suffix(use_hadamard, exclude_tokens)}.png"
    plt.savefig(filename)
    print(f"Saved top token errors by module plot to {filename}")
    plt.close()

def plot_layer_magnitudes(layer_mags, model_name, use_hadamard: bool = False, exclude_tokens: list = None):
    """Plots the average max activation magnitude for each layer."""
    if not layer_mags:
        print("No layer magnitude data to plot.")
        return
        
    layers = sorted(layer_mags.keys())
    magnitudes = [layer_mags[l] for l in layers]

    plt.figure(figsize=(15, 7))
    plt.plot(layers, magnitudes, marker='o', linestyle='-')
    title_suffix = _get_title_suffix(model_name, use_hadamard, exclude_tokens)
    plt.title(f'Layer-wise Average Max Activation Magnitude {title_suffix}')
    plt.xlabel('Layer Index')
    plt.ylabel('Average Max Magnitude')
    plt.xticks(layers)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    filename = f"layer_magnitudes_{model_name.replace('/', '_')}{_get_filename_suffix(use_hadamard, exclude_tokens)}.png"
    plt.savefig(filename)
    print(f"Saved layer magnitude plot to {filename}")
    plt.close()

def plot_module_magnitudes(module_mags, model_name, use_hadamard: bool = False, exclude_tokens: list = None):
    """Plots the average max activation magnitude for each module type."""
    if not module_mags:
        print("No module magnitude data to plot.")
        return

    sorted_modules = sorted(module_mags.keys())
    magnitudes = [module_mags[m] for m in sorted_modules]
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(sorted_modules)))

    plt.figure(figsize=(12, 8))
    plt.bar(sorted_modules, magnitudes, color=colors)
    title_suffix = _get_title_suffix(model_name, use_hadamard, exclude_tokens)
    plt.title(f'Module-wise Average Max Activation Magnitude {title_suffix}')
    plt.xlabel('Module Type')
    plt.ylabel('Average Max Magnitude (across all layers)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = f"module_magnitudes_{model_name.replace('/', '_')}{_get_filename_suffix(use_hadamard, exclude_tokens)}.png"
    plt.savefig(filename)
    print(f"Saved module magnitude plot to {filename}")
    plt.close()

def plot_top_token_magnitudes_by_module(module_top_tokens, model_name, use_hadamard: bool = False, exclude_tokens: list = None):
    """Plots the top K tokens with the highest average max magnitude for multiple modules."""
    if not module_top_tokens:
        print("No token magnitude data to plot.")
        return

    num_modules = len(module_top_tokens)
    ncols = 2
    nrows = math.ceil(num_modules / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 7 * nrows), squeeze=False)
    #fig.suptitle(f'Top Tokens by Average Max Activation Magnitude ({model_name})', fontsize=20, y=0.96)
    title_suffix = _get_title_suffix(model_name, use_hadamard, exclude_tokens)
    fig.suptitle(f'Top Tokens by Max Activation Magnitude {title_suffix}', fontsize=20, y=0.96)

    colors = plt.get_cmap('plasma')(np.linspace(0, 1, num_modules))
    sorted_module_names = sorted(module_top_tokens.keys())

    for i, module_name in enumerate(sorted_module_names):
        ax = axes[i // ncols, i % ncols]
        top_tokens = module_top_tokens[module_name]
        top_tokens.reverse()

        mags = [item[0] for item in top_tokens]
        counts = [item[1] for item in top_tokens]
        token_texts = [repr(item[2]) for item in top_tokens]
        y_labels = [f"{text} (n={count})" for text, count in zip(token_texts, counts)]

        ax.barh(y_labels, mags, color=colors[i], alpha=0.8)
        #ax.set_xlabel('Average Max Magnitude')
        ax.set_xlabel('Max Magnitude')
        ax.set_title(f'Module: {module_name}')
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    for i in range(num_modules, nrows * ncols):
        fig.delaxes(axes.flatten()[i])
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    filename = f"top_token_magnitudes_by_module_{model_name.replace('/', '_')}{_get_filename_suffix(use_hadamard, exclude_tokens)}.png"
    plt.savefig(filename)
    print(f"Saved top token magnitude plot to {filename}")
    plt.close()

def plot_module_magnitudes_per_layer(module_mags_data, model_name, use_hadamard: bool = False, exclude_tokens: list = None):
    """
    Plots the average max activation magnitude for each module type across all layers.
    Groups redundant modules (q_proj, k_proj, v_proj and gate_proj, up_proj) for clarity.
    """
    if not module_mags_data:
        print("No per-layer module magnitude data to plot.")
        return
        
    # --- Grouping Logic ---
    plot_data = defaultdict(dict)
    
    # Use q_proj as the representative for q, k, and v
    if 'self_attn.q_proj' in module_mags_data:
        plot_data['self_attn.qkv_proj'] = module_mags_data['self_attn.q_proj']
    
    # Use gate_proj as the representative for gate and up
    if 'mlp.gate_proj' in module_mags_data:
        plot_data['mlp.gate_up_proj'] = module_mags_data['mlp.gate_proj']
        
    # Copy over the other non-redundant modules
    for module_type, layer_data in module_mags_data.items():
        if all(proj not in module_type for proj in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']):
            plot_data[module_type] = layer_data

    # --- Plotting ---
    plt.figure(figsize=(18, 9))
    colors = plt.get_cmap('tab10').colors 

    for i, (module_type, layer_data) in enumerate(sorted(plot_data.items())):
        if layer_data:
            layers = sorted(layer_data.keys())
            magnitudes = [layer_data[l] for l in layers]
            plt.plot(layers, magnitudes, marker='o', linestyle='--', label=module_type, color=colors[i % len(colors)])

    title_suffix = _get_title_suffix(model_name, use_hadamard, exclude_tokens)
    plt.title(f'Module-wise Average Max Activation Magnitude Across Layers {title_suffix}', fontsize=16)
    plt.xlabel('Layer Index')
    plt.ylabel('Average Max Magnitude')
    plt.xticks(sorted(layers))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    filename = f"module_magnitudes_per_layer_{model_name.replace('/', '_')}{_get_filename_suffix(use_hadamard, exclude_tokens)}.png"
    plt.savefig(filename)
    print(f"Saved per-layer module magnitude plot to {filename}")
    plt.close()

def plot_activation_kurtosis(kurtosis_data, model_name, use_hadamard: bool = False):
    """
    Plots the average activation kurtosis for each layer and module type.

    Args:
        kurtosis_data (dict): A dict containing 'per_layer' and 'per_module' kurtosis results.
        model_name (str): The name of the model for plot titles.
    """
    per_layer_kurtosis = kurtosis_data.get('per_layer', {})
    per_module_kurtosis = kurtosis_data.get('per_module', {})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14))
    title_suffix = _get_title_suffix(model_name, use_hadamard)
    fig.suptitle(f'Activation Kurtosis Analysis {title_suffix}', fontsize=20, y=0.97)

    # Plot 1: Per-Layer Kurtosis
    if per_layer_kurtosis:
        layers = sorted(per_layer_kurtosis.keys())
        kurtosis_values = [per_layer_kurtosis[l] for l in layers]
        ax1.plot(layers, kurtosis_values, marker='o', linestyle='-', color='royalblue')
        ax1.set_title('Layer-wise Average Activation Kurtosis')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Average Kurtosis (Fisher)')
        ax1.set_xticks(layers)
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot 2: Per-Module Kurtosis
    if per_module_kurtosis:
        sorted_modules = sorted(per_module_kurtosis.keys())
        kurtosis_values = [per_module_kurtosis[m] for m in sorted_modules]
        colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(sorted_modules)))
        ax2.bar(sorted_modules, kurtosis_values, color=colors)
        ax2.set_title('Module-wise Average Activation Kurtosis')
        ax2.set_xlabel('Module Type')
        ax2.set_ylabel('Average Kurtosis (Fisher)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = f"activation_kurtosis_{model_name.replace('/', '_')}{_get_filename_suffix(use_hadamard)}.png"
    plt.savefig(filename)
    print(f"Saved activation kurtosis plot to {filename}")
    plt.close()

def plot_top_token_kurtosis(data, title_prefix, filename_prefix, model_name, use_hadamard: bool = False):
    """
    A generic plotter for top token kurtosis by module or by layer.
    """
    if not data:
        print(f"No data to plot for {title_prefix}.")
        return

    num_items = len(data)
    ncols = 2
    nrows = math.ceil(num_items / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 7 * nrows), squeeze=False)
    title_suffix = _get_title_suffix(model_name, use_hadamard)
    fig.suptitle(f'Top Tokens with Highest Activation Kurtosis {title_suffix}', fontsize=20, y=0.96)

    colors = plt.get_cmap('cividis').colors
    sorted_item_names = sorted(data.keys())

    for i, item_name in enumerate(sorted_item_names):
        ax = axes[i // ncols, i % ncols]
        top_tokens = data[item_name]
        top_tokens.reverse()

        kurt_values = [item[0] for item in top_tokens]
        counts = [item[1] for item in top_tokens]
        token_texts = [repr(item[2]) for item in top_tokens]
        y_labels = [f"{text} (n={count})" for text, count in zip(token_texts, counts)]
        
        ax.barh(y_labels, kurt_values, color=colors[i * (len(colors)//num_items) % len(colors)], alpha=0.8)
        ax.set_xlabel('Average Kurtosis (Fisher)')
        ax.set_title(f'{title_prefix}: {item_name}')
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    for i in range(num_items, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    filename = f"{filename_prefix}_{model_name.replace('/', '_')}{_get_filename_suffix(use_hadamard)}.png"
    plt.savefig(filename)
    print(f"Saved {title_prefix} kurtosis plot to {filename}")
    plt.close()

def plot_down_proj_spikes(data, model_name, use_hadamard: bool = False):
    """
    Plots the top 5 token activation magnitudes for each down_proj layer.
    """
    if not data:
        print("No down_proj spike data to plot.")
        return

    num_layers = len(data)
    ncols = 4
    nrows = math.ceil(num_layers / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 4 * nrows), squeeze=False)
    title_suffix = _get_title_suffix(model_name, use_hadamard)
    fig.suptitle(f'Top 5 Token Activation Magnitudes in down_proj Layers {title_suffix}', fontsize=20, y=0.97)

    sorted_layers = sorted(data.keys())

    for i, layer_idx in enumerate(sorted_layers):
        ax = axes[i // ncols, i % ncols]
        top_tokens = data[layer_idx]
        top_tokens.reverse()

        mags = [item[0] for item in top_tokens]
        token_texts = [repr(item[1]) for item in top_tokens]

        ax.barh(token_texts, mags, color='firebrick', alpha=0.8)
        ax.set_title(f'Layer {layer_idx}')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='x', linestyle='--', alpha=0.6)

    for i in range(num_layers, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    filename = f"down_proj_spikes_{model_name.replace('/', '_')}{_get_filename_suffix(use_hadamard)}.png"
    plt.savefig(filename)
    print(f"Saved down_proj activation spike plot to {filename}")
    plt.close()

def plot_prompt_spikes(data, token_labels, model_name, layers_to_plot=None, use_hadamard: bool = False):
    """
    Plots the activation magnitude for each token in a prompt across different layers.
    """
    if not data:
        print("No prompt spike data to plot.")
        return

    num_modules = len(data)
    ncols = 2
    nrows = math.ceil(num_modules / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 7 * nrows), squeeze=False)
    title_suffix = _get_title_suffix(model_name, use_hadamard)
    fig.suptitle(f'Per-Token Activation Magnitude for Prompt {title_suffix}', fontsize=20, y=0.96)

    colors = plt.get_cmap('viridis')
    sorted_module_names = sorted(data.keys())

    for i, module_name in enumerate(sorted_module_names):
        ax = axes[i // ncols, i % ncols]
        layer_data = data[module_name]
        
        # Filter layers if a list is provided
        if layers_to_plot:
            layer_data_to_plot = {l: d for l, d in layer_data.items() if l in layers_to_plot}
        else:
            layer_data_to_plot = layer_data
            
        num_layers_to_plot = len(layer_data_to_plot)
        if num_layers_to_plot == 0:
            ax.text(0.5, 0.5, "No specified layers to plot", ha='center', va='center')
            ax.set_title(f'Module Type: {module_name}')
            continue
            
        layer_colors = colors(np.linspace(0, 1, num_layers_to_plot))

        for j, (layer_idx, mags) in enumerate(sorted(layer_data_to_plot.items())):
            ax.plot(mags, marker='.', linestyle='-', label=f'Layer {layer_idx}', color=layer_colors[j], alpha=0.7)

        ax.set_title(f'Module Type: {module_name}')
        ax.set_xlabel('Token Position in Prompt')
        ax.set_ylabel('Max Activation Magnitude')
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels([repr(l) for l in token_labels], rotation=45, ha='right')
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        if num_layers_to_plot <= 10: # Only show legend if not too cluttered
            ax.legend()

    for i in range(num_modules, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    filename = f"prompt_spikes_{model_name.replace('/', '_')}{_get_filename_suffix(use_hadamard)}.png"
    plt.savefig(filename)
    print(f"Saved prompt spike plot to {filename}")
    plt.close()

def plot_token_occurrence_magnitudes(data, target_token_str, model_name, use_hadamard: bool = False):
    """
    Plots the average max magnitude of a target token vs. its occurrence number.
    """
    if not data:
        print(f"No occurrence data to plot for token: {repr(target_token_str)}")
        return

    plt.figure(figsize=(18, 9))
    colors = plt.get_cmap('tab10').colors

    for i, (module_type, occurrences_data) in enumerate(sorted(data.items())):
        if occurrences_data:
            occurrences = sorted(occurrences_data.keys())
            avg_mags = [occurrences_data[o]['avg_mag'] for o in occurrences]
            counts = [occurrences_data[o]['count'] for o in occurrences]
            
            # Create labels with counts for the legend
            label = f"{module_type} (n={sum(counts)})"
            plt.plot(occurrences, avg_mags, marker='o', linestyle='-', label=label, color=colors[i % len(colors)])

    title_suffix = _get_title_suffix(model_name, use_hadamard)
    plt.title(f"Activation Magnitude of Token {repr(target_token_str)} vs. Occurrence Number {title_suffix}", fontsize=16)
    plt.xlabel("Occurrence Number in Sequence")
    plt.ylabel("Average Max Magnitude")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    filename = f"token_occurrence_magnitude_{model_name.replace('/', '_')}_{target_token_str.encode('unicode_escape').decode()}{_get_filename_suffix(use_hadamard)}.png"
    plt.savefig(filename)
    print(f"Saved token occurrence magnitude plot to {filename}")
    plt.close()


def plot_bops_analysis(bops_data, model_name, bits):
    """
    Generates plots for BOPs analysis.

    Args:
        bops_data (dict): Dict with per-module BOPs for each layer.
        model_name (str): Name of the model for titles.
        bits (int): Bit-width for weights and activations.
    """
    # --- Plot 1: Layer-wise Stacked Bar Chart ---
    df = pd.DataFrame(bops_data).T.fillna(0)
    # Ensure layers are sorted numerically, with 'head' at the end
    numeric_layers = sorted([l for l in df.index if isinstance(l, int)])
    head_layer = ['head'] if 'head' in df.index else []
    
    df = df.reindex(numeric_layers)
    df = df.reindex(sorted(df.columns), axis=1) # Sort columns for consistent color
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 18))
    fig.suptitle(f'Bit-Operations (BOPs) Analysis ({model_name} - {bits}-bit)', fontsize=20, y=0.97)

    df.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
    ax1.set_title('Layer-wise BOPs Distribution')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Giga-BOPs per Token')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    
    # --- Plot 2: Module-wise BOPs Across Layers ---
    for module in df.columns:
        ax2.plot(df.index.astype(str), df[module], marker='o', linestyle='--', label=module)
    
    ax2.set_title('Module-wise BOPs Across Layers')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Giga-BOPs per Token')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    filename = f"bops_analysis_{model_name.replace('/', '_')}_{bits}bit.png"
    plt.savefig(filename)
    print(f"Saved BOPs analysis plot to {filename}")
    plt.close()

def plot_fisher_information(fisher_data, model_name, agg: str = 'average'):
    """
    Generates plots for Fisher Information analysis.

    Args:
        fisher_data (dict): Dict with per-layer and per-module-per-layer data.
        model_name (str): Name of the model for titles.
    """
    per_layer_fisher = fisher_data.get('per_layer', {})
    per_module_fisher = fisher_data.get('per_module', {})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 18))
    fig.suptitle(f'Activation Fisher Information ({agg}) Analysis ({model_name})', fontsize=20, y=0.97)

    # --- Plot 1: Per-Layer Total Fisher Information ---
    if per_layer_fisher:
        layers = sorted(per_layer_fisher.keys())
        fisher_values = [per_layer_fisher[l] for l in layers]
        ax1.bar(layers, fisher_values, color='darkcyan')
        ax1.set_title(f'Layer-wise Total Activation Fisher Information ({agg})')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Fisher Information (Sum of Squared Gradients)')
        ax1.set_xticks(layers)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 2: Per-Module Fisher Information Across Layers ---
    if per_module_fisher:
        plot_data = defaultdict(dict)
        if 'self_attn.q_proj' in per_module_fisher: plot_data['self_attn.qkv_proj'] = per_module_fisher['self_attn.q_proj']
        if 'mlp.gate_proj' in per_module_fisher: plot_data['mlp.gate_up_proj'] = per_module_fisher['mlp.gate_proj']
        for mtype, ldata in per_module_fisher.items():
            if all(p not in mtype for p in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']):
                plot_data[mtype] = ldata
        
        all_layers = set(); [all_layers.update(ld.keys()) for ld in plot_data.values()]
        sorted_layers = sorted(list(all_layers))
        colors = plt.get_cmap('tab10').colors
        
        for i, (mtype, ldata) in enumerate(sorted(plot_data.items())):
            values = [ldata.get(l, np.nan) for l in sorted_layers]
            ax2.plot(sorted_layers, values, marker='o', linestyle='--', label=mtype, color=colors[i % len(colors)])

        ax2.set_title(f'Module-wise Activation Fisher Information ({agg}) Across Layers')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Fisher Information')
        ax2.set_xticks(sorted_layers)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    filename = f"fisher_information_{agg}_{model_name.replace('/', '_')}.png"
    plt.savefig(filename)
    print(f"Saved Fisher Information analysis plot to {filename}")
    plt.close()

def plot_max_median_ratio(ratio_data, model_name):
    """
    Generates plots for Max-to-Median Ratio analysis.
    """
    per_layer_ratio = ratio_data.get('per_layer', {})
    per_module_ratio = ratio_data.get('per_module', {})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 18))
    fig.suptitle(f'Activation Max-to-Median Ratio Analysis ({model_name})', fontsize=20, y=0.97)

    # --- Plot 1: Per-Layer Max-to-Median Ratio ---
    if per_layer_ratio:
        layers = sorted(per_layer_ratio.keys())
        ratios = [per_layer_ratio[l] for l in layers]
        ax1.bar(layers, ratios, color='purple')
        ax1.set_title('Layer-wise Activation Max-to-Median Ratio')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Max / Median Ratio')
        ax1.set_xticks(layers)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 2: Per-Module Max-to-Median Ratio Across Layers ---
    if per_module_ratio:
        plot_data = defaultdict(dict)
        if 'self_attn.q_proj' in per_module_ratio: plot_data['self_attn.qkv_proj'] = per_module_ratio['self_attn.q_proj']
        if 'mlp.gate_proj' in per_module_ratio: plot_data['mlp.gate_up_proj'] = per_module_ratio['mlp.gate_proj']
        for mtype, ldata in per_module_ratio.items():
            #if all(p not in mtype for p in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']):
            plot_data[mtype] = ldata
        
        all_layers = set(); [all_layers.update(ld.keys()) for ld in plot_data.values()]
        sorted_layers = sorted(list(all_layers))
        colors = plt.get_cmap('tab10').colors
        
        for i, (mtype, ldata) in enumerate(sorted(plot_data.items())):
            values = [ldata.get(l, np.nan) for l in sorted_layers]
            ax2.plot(sorted_layers, values, marker='o', linestyle='--', label=mtype, color=colors[i % len(colors)])

        ax2.set_title('Module-wise Activation Max-to-Median Ratio Across Layers')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Max / Median Ratio')
        ax2.set_xticks(sorted_layers)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    filename = f"max_median_ratio_{model_name.replace('/', '_')}.png"
    plt.savefig(filename)
    print(f"Saved Max-to-Median Ratio analysis plot to {filename}")
    plt.close()

def plot_fgmp_sensitivity(sensitivity_data, model_name, high_prec_bits, low_prec_bits, block_size=None):
    """
    Generates plots for FGMP sensitivity analysis.
    """
    per_layer_sensitivity = sensitivity_data.get('per_layer', {})
    per_module_sensitivity = sensitivity_data.get('per_module', {})

    block_size_str = f", Block Size: {block_size}" if block_size is not None else ""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 18))
    fig.suptitle(f'FGMP Activation Sensitivity Analysis ({model_name} - FP{high_prec_bits} vs FP{low_prec_bits}{block_size_str})', fontsize=20, y=0.97)

    # --- Plot 1: Per-Layer Total Sensitivity ---
    if per_layer_sensitivity:
        layers = sorted(per_layer_sensitivity.keys())
        sensitivity_values = [per_layer_sensitivity[l] for l in layers]
        ax1.bar(layers, sensitivity_values, color='teal')
        ax1.set_title('Layer-wise Average Block Sensitivity (Impact Score)')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Average Block Impact Score')
        ax1.set_xticks(layers)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # --- Plot 2: Per-Module Sensitivity Across Layers ---
    if per_module_sensitivity:
        plot_data = defaultdict(dict)
        if 'self_attn.q_proj' in per_module_sensitivity: plot_data['self_attn.qkv_proj'] = per_module_sensitivity['self_attn.q_proj']
        if 'mlp.gate_proj' in per_module_sensitivity: plot_data['mlp.gate_up_proj'] = per_module_sensitivity['mlp.gate_proj']
        for mtype, ldata in per_module_sensitivity.items():
            if all(p not in mtype for p in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']):
                plot_data[mtype] = ldata
        
        all_layers = set(); [all_layers.update(ld.keys()) for ld in plot_data.values()]
        sorted_layers = sorted(list(all_layers))
        colors = plt.get_cmap('tab10').colors
        
        for i, (mtype, ldata) in enumerate(sorted(plot_data.items())):
            values = [ldata.get(l, np.nan) for l in sorted_layers]
            ax2.plot(sorted_layers, values, marker='o', linestyle='--', label=mtype, color=colors[i % len(colors)])

        ax2.set_title('Module-wise Average Block Sensitivity (Impact Score) Across Layers')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Average Block Impact Score')
        ax2.set_xticks(sorted_layers)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    block_size_suffix = f"_bs{block_size}" if block_size is not None else ""
    filename = f"fgmp_sensitivity_{model_name.replace('/', '_')}_{high_prec_bits}_{low_prec_bits}{block_size_suffix}.png"
    plt.savefig(filename)
    print(f"Saved FGMP sensitivity analysis plot to {filename}")
    plt.close()

def plot_propagated_error_variance(pev_data, model_name, low_prec_bits):
    """
    Generates plots for Propagated Error Variance (PEV) analysis.

    Args:
        pev_data (dict): A dict containing 'per_layer' and 'per_module' PEV results.
        model_name (str): The name of the model for plot titles.
        low_prec_bits (int): The low-precision bit width used in the analysis.
    """
    per_layer_pev = pev_data.get('per_layer', {})
    per_module_pev = pev_data.get('per_module', {})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 18))
    fig.suptitle(f'Propagated Error Variance (PEV) Analysis ({model_name} - FP{low_prec_bits})', fontsize=20, y=0.97)

    # --- Plot 1: Per-Layer Total Propagated Error Variance ---
    if per_layer_pev:
        layers = sorted(per_layer_pev.keys())
        pev_values = [per_layer_pev[l] for l in layers]
        
        ax1.bar(layers, pev_values, color='mediumseagreen')
        ax1.set_title('Layer-wise Total Propagated Error Variance (PEV Score)')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('PEV Score (Total Output Variance)')
        ax1.set_xticks(layers)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        ax1.set_yscale('log') # PEV scores can have a large dynamic range, log scale is often better
        ax1.set_ylabel('PEV Score (Log Scale)')


    # --- Plot 2: Per-Module Propagated Error Variance Across Layers ---
    if per_module_pev:
        # Group similar modules for cleaner plotting
        plot_data = defaultdict(dict)
        if 'self_attn.q_proj' in per_module_pev:
            plot_data['self_attn.qkv_proj'] = per_module_pev['self_attn.q_proj']
        if 'mlp.gate_proj' in per_module_pev:
            plot_data['mlp.gate_up_proj'] = per_module_pev['mlp.gate_proj']
        
        # Copy over other modules
        for module_type, layer_data in per_module_pev.items():
            if all(proj not in module_type for proj in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']):
                plot_data[module_type] = layer_data

        all_layers = set()
        for layer_data in plot_data.values():
            all_layers.update(layer_data.keys())
        
        sorted_layers = sorted(list(all_layers))
        colors = plt.get_cmap('tab10').colors
        
        for i, (module_type, layer_data) in enumerate(sorted(plot_data.items())):
            # Get values for each layer, using np.nan for missing data to create gaps in the line
            values = [layer_data.get(layer, np.nan) for layer in sorted_layers]
            ax2.plot(sorted_layers, values, marker='o', linestyle='--', label=module_type, color=colors[i % len(colors)])

        ax2.set_title('Module-wise PEV Score Across Layers')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('PEV Score')
        ax2.set_xticks(sorted_layers)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        ax2.set_yscale('log') # Use log scale here as well
        ax2.set_ylabel('PEV Score (Log Scale)')


    # Adjust layout to make room for titles and legends
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    filename = f"pev_sensitivity_{model_name.replace('/', '_')}_{low_prec_bits}.png"
    plt.savefig(filename)
    print(f"Saved PEV sensitivity analysis plot to {filename}")
    plt.close()
