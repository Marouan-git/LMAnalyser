import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math

def plot_layer_errors(layer_errors, model_name: str, bits: int, granularity: str):
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
    plt.title(f'Layer-wise Average Activation Quantization Error\n({model_name} - {bits}-bit, {granularity})')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.xticks(int_keys)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    filename = f"layer_errors_{model_name.replace('/', '_')}_{bits}bit_{granularity}.png"
    plt.savefig(filename)
    print(f"Saved layer-wise error plot to {filename}")
    plt.close()

def plot_module_errors(module_errors, model_name: str, bits: int, granularity: str):
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

    plt.title(f'Module-wise Activation Quantization Error Across Layers\n({model_name} - {bits}-bit, {granularity})')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend

    filename = f"module_errors_{model_name.replace('/', '_')}_{bits}bit_{granularity}.png"
    plt.savefig(filename)
    print(f"Saved module-wise error plot to {filename}")
    plt.close()


def plot_top_token_errors_by_module(module_top_tokens, model_name, bits, granularity):
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
    fig.suptitle(f'Top Tokens with Highest Avg. Activation Quantization Error ({model_name} - {bits}-bit, {granularity})', fontsize=20, y=0.96)

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
    
    filename = f"top_token_errors_by_module_{model_name.replace('/', '_')}_{bits}bit_{granularity}.png"
    plt.savefig(filename)
    print(f"Saved top token errors by module plot to {filename}")
    plt.close()

def plot_layer_magnitudes(layer_mags, model_name):
    """Plots the average max activation magnitude for each layer."""
    if not layer_mags:
        print("No layer magnitude data to plot.")
        return
        
    layers = sorted(layer_mags.keys())
    magnitudes = [layer_mags[l] for l in layers]

    plt.figure(figsize=(15, 7))
    plt.plot(layers, magnitudes, marker='o', linestyle='-')
    plt.title(f'Layer-wise Average Max Activation Magnitude ({model_name})')
    plt.xlabel('Layer Index')
    plt.ylabel('Average Max Magnitude')
    plt.xticks(layers)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    filename = f"layer_magnitudes_{model_name.replace('/', '_')}.png"
    plt.savefig(filename)
    print(f"Saved layer magnitude plot to {filename}")
    plt.close()

def plot_module_magnitudes(module_mags, model_name):
    """Plots the average max activation magnitude for each module type."""
    if not module_mags:
        print("No module magnitude data to plot.")
        return

    sorted_modules = sorted(module_mags.keys())
    magnitudes = [module_mags[m] for m in sorted_modules]
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(sorted_modules)))

    plt.figure(figsize=(12, 8))
    plt.bar(sorted_modules, magnitudes, color=colors)
    plt.title(f'Module-wise Average Max Activation Magnitude ({model_name})')
    plt.xlabel('Module Type')
    plt.ylabel('Average Max Magnitude (across all layers)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = f"module_magnitudes_{model_name.replace('/', '_')}.png"
    plt.savefig(filename)
    print(f"Saved module magnitude plot to {filename}")
    plt.close()

def plot_top_token_magnitudes_by_module(module_top_tokens, model_name):
    """Plots the top K tokens with the highest average max magnitude for multiple modules."""
    if not module_top_tokens:
        print("No token magnitude data to plot.")
        return

    num_modules = len(module_top_tokens)
    ncols = 2
    nrows = math.ceil(num_modules / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 7 * nrows), squeeze=False)
    #fig.suptitle(f'Top Tokens by Average Max Activation Magnitude ({model_name})', fontsize=20, y=0.96)
    fig.suptitle(f'Top Tokens by Max Activation Magnitude ({model_name})', fontsize=20, y=0.96)
    
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
    
    filename = f"top_token_magnitudes_by_module_{model_name.replace('/', '_')}.png"
    plt.savefig(filename)
    print(f"Saved top token magnitude plot to {filename}")
    plt.close()

def plot_module_magnitudes_per_layer(module_mags_data, model_name):
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

    plt.title(f'Module-wise Average Max Activation Magnitude Across Layers ({model_name})', fontsize=16)
    plt.xlabel('Layer Index')
    plt.ylabel('Average Max Magnitude')
    plt.xticks(sorted(layers))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    filename = f"module_magnitudes_per_layer_{model_name.replace('/', '_')}.png"
    plt.savefig(filename)
    print(f"Saved per-layer module magnitude plot to {filename}")
    plt.close()

def plot_activation_kurtosis(kurtosis_data, model_name):
    """
    Plots the average activation kurtosis for each layer and module type.

    Args:
        kurtosis_data (dict): A dict containing 'per_layer' and 'per_module' kurtosis results.
        model_name (str): The name of the model for plot titles.
    """
    per_layer_kurtosis = kurtosis_data.get('per_layer', {})
    per_module_kurtosis = kurtosis_data.get('per_module', {})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 14))
    fig.suptitle(f'Activation Kurtosis Analysis ({model_name})', fontsize=20, y=0.97)

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
    
    filename = f"activation_kurtosis_{model_name.replace('/', '_')}.png"
    plt.savefig(filename)
    print(f"Saved activation kurtosis plot to {filename}")
    plt.close()

def plot_top_token_kurtosis(data, title_prefix, filename_prefix, model_name):
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
    fig.suptitle(f'Top Tokens with Highest Activation Kurtosis ({model_name})', fontsize=20, y=0.96)
    
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
    
    filename = f"{filename_prefix}_{model_name.replace('/', '_')}.png"
    plt.savefig(filename)
    print(f"Saved {title_prefix} kurtosis plot to {filename}")
    plt.close()