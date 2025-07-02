import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from scipy.stats import kurtosis
from data.data_loader import get_calibration_data
from analysis.quantization import fake_quantize_activation, calculate_quantization_error
from plotting.plotter import plot_layer_errors, plot_module_errors, plot_top_token_errors_by_module, plot_layer_magnitudes, plot_module_magnitudes, plot_top_token_magnitudes_by_module, plot_module_magnitudes_per_layer, plot_activation_kurtosis, plot_top_token_kurtosis

class LLMAnalyser:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.captured_activations = defaultdict(list)
        self.hooks = []

    def _get_hook(self, name):
        def hook(model, input, output):
            # We are interested in the input to the linear layers, which are activations.
            # input is a tuple, we take the first element.
            self.captured_activations[name].append(input[0].detach())
        return hook

    def _register_hooks(self, module_names=None):
        """
        Registers forward hooks on specified or all linear layers of the model.

        Args:
            module_names (list, optional): A list of specific module name suffixes to hook. 
                                           If None, hooks all linear layers.
        """
        print("Registering hooks to capture activations...")
        target_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # If no specific module names are given, hook all linear layers.
                # If module_names are given, hook if the module's name ends with one of the suffixes.
                if module_names is None or any(name.endswith(suffix) for suffix in module_names):
                    target_modules.append((name, module))

        for name, module in target_modules:
            self.hooks.append(module.register_forward_hook(self._get_hook(name)))
        print(f"Registered {len(self.hooks)} hooks.")

    def _remove_hooks(self):
        """Removes all registered hooks and clears captured data."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.captured_activations.clear()
    
    @torch.no_grad()
    def run_activation_magnitude_analysis(self, calib_dataset, num_samples: int = 16, plot: bool = False):
        """
        Performs a unified analysis of activation magnitudes:
        1. Per-Layer: Average max magnitude for each layer.
        2. Per-Module (Total): Average max magnitude for each module type.
        3. Per-Module (Per-Layer): Average max magnitude for each module type across layers.
        4. Per-Token: Top tokens with the highest average max magnitude for key module types.
        """
        print("\nStarting unified activation magnitude analysis...")
        # Hook all linear layers to gather data for all analyses
        self._register_hooks()
        
        calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
        if calib_data is None: self._remove_hooks(); return

        # --- Data aggregation structures ---
        layer_mags = defaultdict(list)
        module_mags = defaultdict(list)
        token_mags_by_module = defaultdict(lambda: defaultdict(list))
        module_mags_per_layer = defaultdict(lambda: defaultdict(list))
        
        # We'll analyze top tokens for a representative set of modules
        token_target_suffixes = ['self_attn.q_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj']

        for i in tqdm(range(num_samples), desc="Processing calibration samples"):
            input_ids_gpu = calib_data[i]["input_ids"].to(self.device)
            self.model(input_ids_gpu)
            input_ids_cpu = calib_data[i]["input_ids"].view(-1)

            for name, activations in self.captured_activations.items():
                act_abs = activations[0].abs() # Shape: [1, seq_len, features]
                max_mag = act_abs.max().item()
                
                try:
                    layer_index = int(name.split('.')[2])
                    module_type = ".".join(name.split('.')[3:])
                    layer_mags[layer_index].append(max_mag)
                    module_mags[module_type].append(max_mag)
                    module_mags_per_layer[module_type][layer_index].append(max_mag)
                except (ValueError, IndexError):
                    continue
                
                module_suffix = next((s for s in token_target_suffixes if name.endswith(s)), None)
                if module_suffix:
                    token_max_mag_per_pos = act_abs.max(dim=-1).values.view(-1)
                    for token_idx, mag in enumerate(token_max_mag_per_pos.tolist()):
                        token_mags_by_module[module_suffix][input_ids_cpu[token_idx].item()].append(mag)

            self.captured_activations.clear()
        
        torch.cuda.empty_cache()
        self._remove_hooks()

        # --- Process, Report, and Plot Results ---
        model_name = self.model.config._name_or_path
        
        # 1. Per-Layer Analysis
        avg_layer_mags = {layer: np.mean(mags) for layer, mags in layer_mags.items()}
        print("\n--- Layer-wise Average Max Activation Magnitude ---")
        for layer, avg_mag in sorted(avg_layer_mags.items()):
            print(f"  Layer {layer}: {avg_mag:.4f}")
        if plot:
            plot_layer_magnitudes(avg_layer_mags, model_name)

        # 2. Per-Module (Total) Analysis
        avg_module_mags = {mtype: np.mean(mags) for mtype, mags in module_mags.items()}
        print("\n--- Module-wise Average Max Activation Magnitude (All Layers)---")
        for mtype, avg_mag in sorted(avg_module_mags.items()):
            print(f"  Module Type: {mtype:<20} | Avg. Max Magnitude: {avg_mag:.4f}")
        if plot:
            plot_module_magnitudes(avg_module_mags, model_name)
        
        # 3. Per-Module (Per-Layer) Analysis
        avg_module_mags_per_layer = defaultdict(dict)
        for mtype, ldata in module_mags_per_layer.items():
            for lidx, mags in ldata.items():
                avg_module_mags_per_layer[mtype][lidx] = np.mean(mags)
        if plot:
            plot_module_magnitudes_per_layer(avg_module_mags_per_layer, model_name)

        # 4. Per-Token Analysis
        module_top_tokens = {}
        print("\n--- Top Tokens by Average Max Magnitude ---")
        for module_suffix, token_mags_dict in token_mags_by_module.items():
            avg_token_mags = []
            for token_id, mags in token_mags_dict.items():
                avg_token_mags.append((np.max(mags), len(mags), self.tokenizer.decode(token_id)))
            avg_token_mags.sort(key=lambda x: x[0], reverse=True)
            module_top_tokens[module_suffix] = avg_token_mags[:10]
            
            print(f"\n  --- Module: {module_suffix} ---")
            for avg_mag, count, token_text in module_top_tokens[module_suffix]:
                print(f"    Token: {repr(token_text):<15} (n={count}) | Avg. Max Magnitude: {avg_mag:.4f}")
        if plot:
            plot_top_token_magnitudes_by_module(module_top_tokens, model_name)

    @torch.no_grad()
    def run_quantization_error_analysis(self, calib_dataset, bits: int, granularity: str, num_samples: int = 16, plot: bool = False):
        """
        Runs a forward pass on calibration data to capture activations,
        then calculates and prints the quantization error for all linear layers.
        This version is memory-efficient by processing one sample at a time.
        """
        self._register_hooks() # Hook all linear layers

        calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
        if calib_data is None:
            self._remove_hooks()
            return
            
        print("\nCalculating activation quantization error...")
        # Structure: {module_name: [error_sample_1, error_sample_2, ...]}
        module_errors_list = defaultdict(list)
        
        for i in tqdm(range(num_samples)):
            input_ids = calib_data[i]["input_ids"].to(self.device)
            self.model(input_ids)
            
            for name, activations in self.captured_activations.items():
                activation_tensor = activations[0] # We process one sample at a time
                quantized_tensor = fake_quantize_activation(activation_tensor, bits, granularity)
                error = calculate_quantization_error(activation_tensor, quantized_tensor)
                module_errors_list[name].append(error)
            
            self.captured_activations.clear() # Free memory after each sample

        # Average the errors for each module across all samples
        module_errors_avg = {name: np.mean(errors) for name, errors in module_errors_list.items()}
        
        self._remove_hooks()
        self._report_module_and_layer_errors(module_errors_avg, plot, bits, granularity)

    @torch.no_grad()
    def run_per_token_error_analysis(self, calib_dataset, bits: int, granularity: str, plot: bool = False, num_samples: int = 16):
        """
        Analyzes per-token quantization error for major module types across all layers
        to find the unique tokens that are most sensitive to quantization on average.
        This version is memory-efficient by processing one sample at a time.
        """
        # Define a representative set of module types for Llama-like architectures
        target_module_suffixes = [
            'self_attn.q_proj', # Represents q, k, v inputs
            'self_attn.o_proj',
            'mlp.gate_proj',    # Represents gate and up inputs
            'mlp.down_proj'
        ]
        
        self._register_hooks(module_names=target_module_suffixes)

        calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
        if calib_data is None:
            self._remove_hooks()
            return

        print(f"\nAnalyzing per-token error with {granularity} quantization...")
        
        # Structure: {module_suffix: {token_id: [error1, error2, ...]}}
        module_token_errors = defaultdict(lambda: defaultdict(list))
        
        for i in tqdm(range(num_samples)):
            input_ids = calib_data[i]["input_ids"].to(self.device)
            
            self.model(input_ids) # This populates self.captured_activations for this sample
            
            input_ids_cpu = calib_data[i]["input_ids"].view(-1)

            # Process activations for this single sample
            for name, activations in self.captured_activations.items():
                activation_tensor = activations[0] # Shape: [1, seq_len, features]
                
                # Determine module type from full name
                module_suffix = next((s for s in target_module_suffixes if name.endswith(s)), None)
                if not module_suffix: continue

                quantized_tensor = fake_quantize_activation(activation_tensor, bits, granularity)
                per_token_mse = (activation_tensor - quantized_tensor).pow(2).mean(dim=-1).view(-1)
                
                # Aggregate errors for this module type
                for token_idx in range(per_token_mse.size(0)):
                    token_id = input_ids_cpu[token_idx].item()
                    error = per_token_mse[token_idx].item()
                    module_token_errors[module_suffix][token_id].append(error)
            
            self.captured_activations.clear() # CRITICAL: Free memory after processing each sample

        torch.cuda.empty_cache()
        self._remove_hooks()

        # --- Calculate and Report Final Averages ---
        module_top_tokens = {}
        for module_suffix, token_errors_dict in module_token_errors.items():
            avg_token_errors = []
            for token_id, errors in token_errors_dict.items():
                if len(errors) > 0:
                    avg_error = np.mean(errors)
                    std_dev = np.std(errors) if len(errors) > 1 else 0.0
                    count = len(errors)
                    avg_token_errors.append((avg_error, std_dev, count, token_id))
            
            avg_token_errors.sort(key=lambda x: x[0], reverse=True)
            
            # Calculate median of all average token errors for this module
            all_avg_errors = [item[0] for item in avg_token_errors]
            median_mse = np.median(all_avg_errors) if all_avg_errors else 0.0

            top_10 = [(err, std, count, self.tokenizer.decode(tid)) for err, std, count, tid in avg_token_errors[:10]]
            module_top_tokens[module_suffix] = {
                "top_tokens": top_10,
                "median_mse": median_mse
            }

            print(f"\n    --- Top 10 Tokens for {module_suffix} ---")
            for avg_error, std_dev, count, token_text in top_10:
                print(f"      Token: {repr(token_text):<15} (n={count}) | Avg MSE: {avg_error:.8f}, Std: {std_dev:.8f}")

        if plot:
            print("\nGenerating plots...")
            model_name = self.model.config._name_or_path
            plot_top_token_errors_by_module(module_top_tokens, model_name, bits, granularity)

    @torch.no_grad()
    def run_activation_kurtosis_analysis(self, calib_dataset, num_samples: int = 16, plot: bool = False):
        """
        Analyzes the kurtosis of activation distributions to identify outlier-prone layers/modules.
        This version is memory-efficient and handles bfloat16 correctly.
        """
        print("\nStarting activation kurtosis analysis...")
        self._register_hooks()
        
        calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
        if calib_data is None: self._remove_hooks(); return
        
        layer_kurtosis = defaultdict(list)
        module_kurtosis = defaultdict(list)

        for i in tqdm(range(num_samples), desc="Processing samples for kurtosis"):
            self.model(calib_data[i]["input_ids"].to(self.device))
            
            for name, activations in self.captured_activations.items():
                # Cast to float32 before converting to numpy to avoid bfloat16 error
                act_tensor = activations[0].to(torch.float32).view(-1).cpu().numpy()
                kurt_val = kurtosis(act_tensor, fisher=True) # Fisher's definition (normal=0)

                try:
                    layer_index = int(name.split('.')[2])
                    module_type = ".".join(name.split('.')[3:])
                    layer_kurtosis[layer_index].append(kurt_val)
                    module_kurtosis[module_type].append(kurt_val)
                except (ValueError, IndexError):
                    continue
            
            # CRITICAL: Clear captured activations after each sample to save memory
            self.captured_activations.clear()

        torch.cuda.empty_cache()
        self._remove_hooks()

        # --- Process and Report ---
        avg_layer_kurtosis = {layer: np.mean(vals) for layer, vals in layer_kurtosis.items()}
        avg_module_kurtosis = {mtype: np.mean(vals) for mtype, vals in module_kurtosis.items()}
        
        print("\n--- Layer-wise Average Activation Kurtosis ---")
        for layer, avg_kurt in sorted(avg_layer_kurtosis.items()):
            print(f"  Layer {layer}: {avg_kurt:.4f}")

        print("\n--- Module-wise Average Activation Kurtosis ---")
        for mtype, avg_kurt in sorted(avg_module_kurtosis.items()):
            print(f"  Module Type: {mtype:<20} | Avg. Kurtosis: {avg_kurt:.4f}")
            
        if plot:
            kurtosis_data = {
                "per_layer": avg_layer_kurtosis,
                "per_module": avg_module_kurtosis
            }
            plot_activation_kurtosis(kurtosis_data, self.model.config._name_or_path)

    @torch.no_grad()
    def run_per_token_kurtosis_analysis(self, calib_dataset, num_samples: int = 16, plot: bool = False, layers_to_plot: list = [0, 15, 31]):
        """
        Analyzes per-token activation kurtosis for major module types and specific layers.
        """
        print("\nStarting per-token activation kurtosis analysis...")
        target_module_suffixes = ['self_attn.q_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj']
        self._register_hooks()

        calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
        if calib_data is None: self._remove_hooks(); return

        # --- Data aggregation structures ---
        # For module-wise analysis
        module_token_kurtosis = defaultdict(lambda: defaultdict(list))
        # For layer-wise analysis
        layer_token_kurtosis = defaultdict(lambda: defaultdict(list))
        
        for i in tqdm(range(num_samples), desc="Processing samples for per-token kurtosis"):
            self.model(calib_data[i]["input_ids"].to(self.device))
            input_ids_cpu = calib_data[i]["input_ids"].view(-1)

            for name, activations in self.captured_activations.items():
                act_tensor = activations[0] # Shape [1, seq_len, features]
                try:
                    layer_index = int(name.split('.')[2])
                    module_type = ".".join(name.split('.')[3:])
                except (ValueError, IndexError):
                    continue

                # Calculate kurtosis for each token's feature vector
                for token_idx in range(act_tensor.shape[1]):
                    token_activation_vector = act_tensor[0, token_idx, :].to(torch.float32).cpu().numpy()
                    kurt_val = kurtosis(token_activation_vector, fisher=True)
                    token_id = input_ids_cpu[token_idx].item()
                    
                    # Aggregate for layer-wise plot
                    layer_token_kurtosis[layer_index][token_id].append(kurt_val)
                    
                    # Aggregate for module-wise plot
                    if any(name.endswith(s) for s in target_module_suffixes):
                        suffix = next(s for s in target_module_suffixes if name.endswith(s))
                        module_token_kurtosis[suffix][token_id].append(kurt_val)

            self.captured_activations.clear()

        torch.cuda.empty_cache()
        self._remove_hooks()

        # --- Process and Report Module-wise Top Tokens ---
        module_top_tokens = {}
        print("\n--- Top Tokens by Average Kurtosis (Per Module Type) ---")
        for module_suffix, data in module_token_kurtosis.items():
            avg_kurt = [(np.mean(vals), len(vals), self.tokenizer.decode(tid)) for tid, vals in data.items()]
            avg_kurt.sort(key=lambda x: x[0], reverse=True)
            module_top_tokens[module_suffix] = avg_kurt[:10]
            print(f"\n  --- Module: {module_suffix} ---")
            for avg, count, text in module_top_tokens[module_suffix]:
                print(f"    Token: {repr(text):<15} (n={count}) | Avg. Kurtosis: {avg:.4f}")

        # --- Process and Report Layer-wise Top Tokens ---
        layer_top_tokens = {}
        print("\n--- Top Tokens by Average Kurtosis (Per Layer) ---")
        for layer_idx, data in sorted(layer_token_kurtosis.items()):
            if layer_idx not in layers_to_plot: continue # Only process layers we want to plot
            avg_kurt = [(np.mean(vals), len(vals), self.tokenizer.decode(tid)) for tid, vals in data.items()]
            avg_kurt.sort(key=lambda x: x[0], reverse=True)
            layer_top_tokens[layer_idx] = avg_kurt[:10]
            print(f"\n  --- Layer: {layer_idx} ---")
            for avg, count, text in layer_top_tokens[layer_idx]:
                print(f"    Token: {repr(text):<15} (n={count}) | Avg. Kurtosis: {avg:.4f}")
        
        if plot:
            model_name = self.model.config._name_or_path
            if module_top_tokens:
                plot_top_token_kurtosis(module_top_tokens, "Module", "top_token_kurtosis_by_module", model_name)
            if layer_top_tokens:
                plot_top_token_kurtosis(layer_top_tokens, "Layer", "top_token_kurtosis_by_layer", model_name)

    def _report_module_and_layer_errors(self, module_errors, plot, bits, granularity):
        """Prints the module/layer/model quantization errors and generates plots if requested."""
        print("\n--- Activation Quantization Error Report (MSE) ---")
        
        layer_errors = defaultdict(list)
        for name, error in module_errors.items():
            print(f"  Module: {name:<50} MSE: {error:.8f}")
            
            try:
                layer_index = int(name.split('.')[2])
                layer_errors[layer_index].append(error)
            except (ValueError, IndexError):
                layer_errors['other'].append(error)
        
        print("\n--- Layer-wise Average Activation Error ---")
        all_errors = []
        
        int_keys = sorted([k for k in layer_errors.keys() if isinstance(k, int)])
        str_keys = sorted([k for k in layer_errors.keys() if isinstance(k, str)])
        sorted_keys = int_keys + str_keys
        
        for layer_idx in sorted_keys:
            errors = layer_errors[layer_idx]
            if errors:
                avg_error = np.mean(errors)
                all_errors.extend(errors)
                print(f"  Layer {layer_idx}: {avg_error:.8f}")

        print("\n--- Model-wide Average Activation Error ---")
        if all_errors:
            model_avg_error = np.mean(all_errors)
            print(f"  Overall Model MSE: {model_avg_error:.8f}")

        if plot:
            print("\nGenerating plots...")
            model_name = self.model.config._name_or_path
            plot_layer_errors(layer_errors, model_name, bits, granularity)
            plot_module_errors(module_errors, model_name, bits, granularity)