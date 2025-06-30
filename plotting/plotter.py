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


# import matplotlib.pyplot as plt
# import numpy as np
# from collections import defaultdict
# import math

# def plot_layer_errors(layer_errors, model_name: str, bits: int, granularity: str):
#     """
#     Plots the average quantization error for each layer.

#     Args:
#         layer_errors (dict): A dictionary mapping layer index to a list of errors.
#         model_name (str): The name of the model for the plot title.
#         bits (int): The number of bits used for quantization.
#         granularity (str): The quantization granularity.
#     """
#     int_keys = sorted([k for k in layer_errors.keys() if isinstance(k, int)])
    
#     if not int_keys:
#         print("No layer data to plot.")
#         return

#     avg_errors = [np.mean(layer_errors[k]) for k in int_keys]

#     plt.figure(figsize=(15, 7))
#     plt.plot(int_keys, avg_errors, marker='o', linestyle='-')
#     plt.title(f'Layer-wise Average Activation Quantization Error\n({model_name} - {bits}-bit, {granularity})')
#     plt.xlabel('Layer Index')
#     plt.ylabel('Mean Squared Error (MSE)')
#     plt.xticks(int_keys)
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.tight_layout()
    
#     filename = f"layer_errors_{model_name.replace('/', '_')}_{bits}bit_{granularity}.png"
#     plt.savefig(filename)
#     print(f"Saved layer-wise error plot to {filename}")
#     plt.close()

# def plot_module_errors(module_errors, model_name: str, bits: int, granularity: str):
#     """
#     Plots the quantization error for each module type across all layers.

#     Args:
#         module_errors (dict): A dictionary mapping full module name to its error.
#         model_name (str): The name of the model for the plot title.
#         bits (int): The number of bits used for quantization.
#         granularity (str): The quantization granularity.
#     """
#     # Restructure data: {module_type: {layer_index: error}}
#     grouped_by_module = defaultdict(dict)
#     for name, error in module_errors.items():
#         parts = name.split('.')
#         if 'layers' in parts and len(parts) > 3:
#             try:
#                 layer_index = int(parts[2])
#                 module_type = ".".join(parts[3:]) # e.g., 'mlp.down_proj', 'self_attn.q_proj'
#                 grouped_by_module[module_type][layer_index] = error
#             except (ValueError, IndexError):
#                 continue
    
#     if not grouped_by_module:
#         print("No module data to plot.")
#         return

#     plt.figure(figsize=(15, 7))
#     for module_type, layer_data in sorted(grouped_by_module.items()):
#         if layer_data:
#             layers = sorted(layer_data.keys())
#             errors = [layer_data[l] for l in layers]
#             plt.plot(layers, errors, marker='o', linestyle='--', label=module_type)

#     plt.title(f'Module-wise Activation Quantization Error Across Layers\n({model_name} - {bits}-bit, {granularity})')
#     plt.xlabel('Layer Index')
#     plt.ylabel('Mean Squared Error (MSE)')
#     plt.grid(True, which='both', linestyle='--', linewidth=0.5)
#     plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#     plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for legend

#     filename = f"module_errors_{model_name.replace('/', '_')}_{bits}bit_{granularity}.png"
#     plt.savefig(filename)
#     print(f"Saved module-wise error plot to {filename}")
#     plt.close()


# def plot_top_token_errors_by_module(module_top_tokens, model_name, bits, granularity):
#     """
#     Plots the top K tokens with the highest average quantization error for multiple modules in subplots.

#     Args:
#         module_top_tokens (dict): A dict mapping module type to a list of tuples (avg_error, token_text).
#         model_name (str): The name of the model for the plot title.
#         bits (int): The number of bits used for quantization.
#         granularity (str): The quantization granularity.
#     """
#     if not module_top_tokens:
#         print("No token error data to plot.")
#         return

#     num_modules = len(module_top_tokens)
#     ncols = 2
#     nrows = math.ceil(num_modules / ncols)
    
#     fig, axes = plt.subplots(nrows, ncols, figsize=(12 * ncols, 6 * nrows), squeeze=False)
#     fig.suptitle(f'Top Tokens with Highest Avg. Activation Quantization Error ({model_name} - {bits}-bit, {granularity})', fontsize=18, y=0.95)

#     sorted_module_names = sorted(module_top_tokens.keys())

#     for i, module_name in enumerate(sorted_module_names):
#         ax = axes[i // ncols, i % ncols]
#         top_tokens = module_top_tokens[module_name]
        
#         if not top_tokens:
#             ax.text(0.5, 0.5, "No data", ha='center', va='center')
#             ax.set_title(module_name)
#             continue

#         top_tokens.reverse()  # Reverse for horizontal bar chart
        
#         avg_errors = [item[0] for item in top_tokens]
#         token_texts = [repr(item[1]) for item in top_tokens]

#         bars = ax.barh(token_texts, avg_errors, color='coral')
#         ax.set_xlabel('Average MSE')
#         ax.set_title(f'Module: {module_name}')
#         ax.tick_params(axis='x', rotation=45)

#     # Hide any unused subplots
#     for i in range(num_modules, nrows * ncols):
#         fig.delaxes(axes.flatten()[i])

#     plt.tight_layout(rect=[0, 0, 1, 0.93]) # Adjust layout for suptitle
    
#     filename = f"top_token_errors_by_module_{model_name.replace('/', '_')}_{bits}bit_{granularity}.png"
#     plt.savefig(filename)
#     print(f"Saved top token errors by module plot to {filename}")
#     plt.close()