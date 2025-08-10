import torch
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
from scipy.stats import kurtosis
from scipy.special import logsumexp
from data.data_loader import get_calibration_data
from analysis.quantization import fake_quantize_activation, calculate_quantization_error
from plotting.plotter import plot_layer_errors, plot_module_errors, plot_top_token_errors_by_module, plot_layer_magnitudes, plot_module_magnitudes, plot_top_token_magnitudes_by_module, plot_module_magnitudes_per_layer, plot_activation_kurtosis, plot_top_token_kurtosis, plot_down_proj_spikes, plot_token_occurrence_magnitudes, plot_prompt_spikes, plot_bops_analysis, plot_fisher_information, plot_max_median_ratio, plot_fgmp_sensitivity

from hadamard_utils import apply_structured_hadamard

class LLMAnalyser:
    def __init__(self, model, tokenizer, use_hadamard_transform=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.captured_activations = defaultdict(list)
        self.hooks = []
        self.use_hadamard_transform = use_hadamard_transform
        self.is_fisher_analysis = False
        if self.use_hadamard_transform:
            print("Hadamard transform is ENABLED for all activation analyses.")

    def _get_hook(self, name):
        def hook(model, input, output):
            activation_tensor = input[0]
            # Conditionally apply Hadamard transform
            if self.use_hadamard_transform:
                # The transform works on the last dimension (features/channels)
                activation_tensor = apply_structured_hadamard(activation_tensor)
            # For Fisher Info, we need to retain the grad of the intermediate tensor
            if self.is_fisher_analysis:
                activation_tensor.retain_grad()
                self.captured_activations[name].append(activation_tensor)
            else:
                # For all other analyses, detach to save memory
                self.captured_activations[name].append(activation_tensor.detach())
        return hook

    def _register_hooks(self, module_names=None):
        """
        Registers forward hooks on specified or all linear layers of the model.

        Args:
            module_names (list, optional): A list of specific module name suffixes to hook. 
                                           If None, hooks all linear layers.
        """
        #print("Registering hooks to capture activations...")
        target_modules = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # If no specific module names are given, hook all linear layers.
                # If module_names are given, hook if the module's name ends with one of the suffixes.
                if module_names is None or any(name.endswith(suffix) for suffix in module_names):
                    target_modules.append((name, module))

        for name, module in target_modules:
            self.hooks.append(module.register_forward_hook(self._get_hook(name)))
        #print(f"Registered {len(self.hooks)} hooks.")

    def _remove_hooks(self):
        """Removes all registered hooks and clears captured data."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.captured_activations.clear()

    def _get_excluded_token_ids(self, exclude_tokens):
        """Helper to convert a list of token strings to a set of token IDs."""
        if not exclude_tokens:
            return set()
            
        print(f"Excluding tokens from analysis: {exclude_tokens}")
        excluded_token_ids = set()
        for token_str in exclude_tokens:
            processed_str = bytes(token_str, "utf-8").decode("unicode_escape")
            # This more robust method handles various tokenization cases
            ids = self.tokenizer.encode(processed_str, add_special_tokens=False)
            if len(ids) == 1: excluded_token_ids.add(ids[0])
            elif len(ids) == 2:
                excluded_token_ids.add(ids[0])
                excluded_token_ids.add(ids[1])
            else:
                print(f"Warning: Token '{processed_str}' is tokenized into multiple IDs and will be ignored.")
            # Also handle the case with a leading space, which often has a different ID
            ids_with_space = self.tokenizer.encode("a" + processed_str, add_special_tokens=False)
            if len(ids_with_space) > 1: excluded_token_ids.add(ids_with_space[-1])
        print(f"Excluded token IDs: {excluded_token_ids}")
        return excluded_token_ids
    
    def _save_to_json(self, data, filename):
        """Helper to save dictionary data to a JSON file using a custom encoder."""
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.generic):
                    return obj.item()
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().tolist()
                if isinstance(obj, defaultdict):
                    return dict(obj)
                return json.JSONEncoder.default(self, obj)

        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4, cls=CustomEncoder)
            print(f"Saved results to {filename}")
        except Exception as e:
            print(f"Error saving data to {filename}: {e}")
    
    @torch.no_grad()
    def run_activation_magnitude_analysis(self, calib_dataset, num_samples: int = 16, plot: bool = False, exclude_tokens: list = None):
        """
        Performs a unified analysis of activation magnitudes:
        1. Per-Layer: Average max magnitude for each layer.
        2. Per-Module (Total): Average max magnitude for each module type.
        3. Per-Module (Per-Layer): Average max magnitude for each module type across layers.
        4. Per-Token: Top tokens with the highest average max magnitude for key module types.
        """
        print("\nStarting unified activation magnitude analysis...")

        if exclude_tokens:
            print(f"Excluding tokens from analysis: {exclude_tokens}")
     
        excluded_token_ids = self._get_excluded_token_ids(exclude_tokens)
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
                # Create a mask to filter out excluded tokens
                mask = torch.tensor([tid.item() not in excluded_token_ids for tid in input_ids_cpu], device=act_abs.device)
                

                filtered_act_abs = act_abs.squeeze(0)[mask]
                if filtered_act_abs.numel() == 0: continue # Skip if all tokens were excluded

                # max_mag = act_abs.max().item()
                max_mag = filtered_act_abs.max().item()

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
                        if input_ids_cpu[token_idx].item() not in excluded_token_ids:
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
            plot_layer_magnitudes(avg_layer_mags, model_name, self.use_hadamard_transform, exclude_tokens)

        # 2. Per-Module (Total) Analysis
        avg_module_mags = {mtype: np.mean(mags) for mtype, mags in module_mags.items()}
        print("\n--- Module-wise Average Max Activation Magnitude (All Layers)---")
        for mtype, avg_mag in sorted(avg_module_mags.items()):
            print(f"  Module Type: {mtype:<20} | Avg. Max Magnitude: {avg_mag:.4f}")
        if plot:
            plot_module_magnitudes(avg_module_mags, model_name, self.use_hadamard_transform, exclude_tokens)
        
        # 3. Per-Module (Per-Layer) Analysis
        avg_module_mags_per_layer = defaultdict(dict)
        for mtype, ldata in module_mags_per_layer.items():
            for lidx, mags in ldata.items():
                avg_module_mags_per_layer[mtype][lidx] = np.mean(mags)
        if plot:
            plot_module_magnitudes_per_layer(avg_module_mags_per_layer, model_name, self.use_hadamard_transform, exclude_tokens)

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
            plot_top_token_magnitudes_by_module(module_top_tokens, model_name, self.use_hadamard_transform, exclude_tokens)

    @torch.no_grad()
    def run_quantization_error_analysis(self, calib_dataset, bits: int, granularity: str, num_samples: int = 16, plot: bool = False, exclude_tokens: list = None):
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
        
        excluded_token_ids = self._get_excluded_token_ids(exclude_tokens)
            
        print("\nCalculating activation quantization error...")
        # Structure: {module_name: [error_sample_1, error_sample_2, ...]}
        module_errors_list = defaultdict(list)
        
        for i in tqdm(range(num_samples)):
            input_ids = calib_data[i]["input_ids"].to(self.device)
            self.model(input_ids)

            input_ids_cpu = calib_data[i]["input_ids"].view(-1)
            mask = torch.tensor([tid.item() not in excluded_token_ids for tid in input_ids_cpu], device=self.device)
            
            for name, activations in self.captured_activations.items():
                activation_tensor = activations[0] # We process one sample at a time
                quantized_tensor = fake_quantize_activation(activation_tensor, bits, granularity)

                # Filter both tensors before calculating error
                filtered_act = activation_tensor[:, mask, :]
                filtered_quant = quantized_tensor[:, mask, :]

                if filtered_act.numel() > 0:
                    error = calculate_quantization_error(filtered_act, filtered_quant)
                    module_errors_list[name].append(error)
            
            self.captured_activations.clear() # Free memory after each sample

        # Average the errors for each module across all samples
        module_errors_avg = {name: np.mean(errors) for name, errors in module_errors_list.items()}
        
        self._remove_hooks()
        self._report_module_and_layer_errors(module_errors_avg, plot, bits, granularity, exclude_tokens)

    @torch.no_grad()
    def run_per_token_error_analysis(self, calib_dataset, bits: int, granularity: str, plot: bool = False, num_samples: int = 16, exclude_tokens: list = None):
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
        
        excluded_token_ids = self._get_excluded_token_ids(exclude_tokens)

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
                # for token_idx in range(per_token_mse.size(0)):
                #     token_id = input_ids_cpu[token_idx].item()
                #     error = per_token_mse[token_idx].item()
                #     module_token_errors[module_suffix][token_id].append(error)
                for token_idx, error in enumerate(per_token_mse.tolist()):
                    token_id = input_ids_cpu[token_idx].item()
                    if token_id not in excluded_token_ids:
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
            plot_top_token_errors_by_module(module_top_tokens, model_name, bits, granularity, self.use_hadamard_transform, exclude_tokens)

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
            plot_activation_kurtosis(kurtosis_data, self.model.config._name_or_path, self.use_hadamard_transform)

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
                plot_top_token_kurtosis(module_top_tokens, "Module", "top_token_kurtosis_by_module", model_name, self.use_hadamard_transform)
            if layer_top_tokens:
                plot_top_token_kurtosis(layer_top_tokens, "Layer", "top_token_kurtosis_by_layer", model_name, self.use_hadamard_transform)

    @torch.no_grad()
    def run_down_proj_spike_analysis(self, calib_dataset: str, num_samples: int = 16, plot: bool = False):
        """
        Analyzes the top 3 tokens with the highest activation magnitude
        for each down_proj module in every layer.
        """
        print("\nAnalyzing top token activation spikes in down_proj modules...")
        self._register_hooks(module_names=['mlp.down_proj'])
        
        calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
        if calib_data is None: self._remove_hooks(); return

        # Structure: {layer_index: {token_id: [max_mag1, max_mag2, ...]}}
        layer_data = defaultdict(lambda: defaultdict(list))

        for i in tqdm(range(len(calib_data)), desc="Processing samples for down_proj spikes"):
            self.model(calib_data[i]["input_ids"].to(self.device))
            input_ids_cpu = calib_data[i]["input_ids"].view(-1)

            for name, activations in self.captured_activations.items():
                if not name.endswith('mlp.down_proj'): continue
                
                act_abs = activations[0].abs()
                token_max_mag = act_abs.max(dim=-1).values.view(-1)
                
                try:
                    layer_index = int(name.split('.')[2])
                    for token_idx, mag in enumerate(token_max_mag.tolist()):
                        token_id = input_ids_cpu[token_idx].item()
                        layer_data[layer_index][token_id].append(mag)
                except (ValueError, IndexError):
                    continue
            
            self.captured_activations.clear()

        torch.cuda.empty_cache()
        self._remove_hooks()

        # --- Process and Report ---
        plot_data = {}
        print("\n--- Top 5 Tokens with Highest Avg. Max Magnitude in down_proj Modules ---")
        for layer_idx, token_mags_dict in sorted(layer_data.items()):
            avg_token_mags = []
            for token_id, mags in token_mags_dict.items():
                avg_token_mags.append((np.mean(mags), self.tokenizer.decode(token_id)))
            
            avg_token_mags.sort(key=lambda x: x[0], reverse=True)
            plot_data[layer_idx] = avg_token_mags[:5]
            
            print(f"\n  --- Layer {layer_idx} ---")
            for avg_mag, token_text in plot_data[layer_idx]:
                print(f"    Token: {repr(token_text):<15} | Avg. Max Magnitude: {avg_mag:.4f}")

        if plot:
            plot_down_proj_spikes(plot_data, self.model.config._name_or_path, self.use_hadamard_transform)
    
    @torch.no_grad()
    def run_token_occurrence_analysis(self, calib_dataset: str, target_token_str: str, num_samples: int = 128, plot: bool = False):
        """
        Analyzes how the activation magnitude of a specific token changes based on its
        occurrence number within a sequence.
        """
        print(f"\nAnalyzing activation magnitude vs. occurrence for token: {repr(target_token_str)}")
        target_module_suffixes = ['self_attn.q_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj']
        self._register_hooks(module_names=target_module_suffixes)
        
        calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
        if calib_data is None: self._remove_hooks(); return

        try:
            target_token_id = self.tokenizer.encode(target_token_str, add_special_tokens=False)[0]
        except Exception as e:
            print(f"Could not encode target token '{target_token_str}'. Error: {e}")
            self._remove_hooks()
            return
            
        print(f"Target token '{target_token_str}' has ID: {target_token_id}")

        # Structure: {module_suffix: {occurrence_num: [mag1, mag2, ...]}}
        results = defaultdict(lambda: defaultdict(list))

        for i in tqdm(range(len(calib_data)), desc=f"Processing samples for token '{target_token_str}'"):
            input_ids_gpu = calib_data[i]["input_ids"].to(self.device)
            self.model(input_ids_gpu)
            input_ids_cpu = calib_data[i]["input_ids"].view(-1)
            
            # Find indices of the target token in this sample
            occurrence_indices = (input_ids_cpu == target_token_id).nonzero(as_tuple=True)[0]

            print(f"Sample {i+1}/{num_samples}: Found {len(occurrence_indices)} occurrences of target token '{target_token_str}'")
            
            if occurrence_indices.numel() > 0:
                for name, activations in self.captured_activations.items():
                    act_abs = activations[0].abs() # Shape: [1, seq_len, features]
                    module_suffix = next((s for s in target_module_suffixes if name.endswith(s)), None)
                    if not module_suffix: continue

                    # Get max magnitude for each token position
                    token_max_mag = act_abs.max(dim=-1).values.view(-1)
                    
                    # For each occurrence, record its magnitude
                    for i, token_pos in enumerate(occurrence_indices):
                        occurrence_num = i + 1 # 1-indexed
                        magnitude = token_max_mag[token_pos].item()
                        results[module_suffix][occurrence_num].append(magnitude)

            self.captured_activations.clear()

        torch.cuda.empty_cache()
        self._remove_hooks()

        # --- Process and Report ---
        plot_data = defaultdict(dict)
        print("\n--- Activation Magnitude vs. Token Occurrence ---")
        for module_suffix, data in sorted(results.items()):
            print(f"\n  --- Module: {module_suffix} ---")
            for occ_num, mags in sorted(data.items()):
                avg_mag = np.mean(mags)
                count = len(mags)
                plot_data[module_suffix][occ_num] = {'avg_mag': avg_mag, 'count': count}
                print(f"    Occurrence #{occ_num}: Avg. Max Magnitude = {avg_mag:.4f} (from {count} instances)")

        if plot:
            plot_token_occurrence_magnitudes(plot_data, target_token_str, self.model.config._name_or_path, self.use_hadamard_transform)

    @torch.no_grad()
    def run_prompt_spike_analysis(self, prompt_text: str, plot: bool = False, layers_to_plot: list = None):
        """
        Analyzes and plots the activation magnitudes for each token in a user-provided prompt.
        """
        print(f"\nAnalyzing activation magnitudes for prompt: \"{prompt_text}\"")
        target_module_suffixes = ['self_attn.q_proj', 'self_attn.o_proj', 'mlp.gate_proj', 'mlp.down_proj']
        self._register_hooks(module_names=target_module_suffixes)

        # Tokenize the user's prompt
        input_ids = self.tokenizer.encode(prompt_text, return_tensors="pt").to(self.device)
        token_labels = [self.tokenizer.decode(token_id) for token_id in input_ids[0]]
        
        # Run a single forward pass
        self.model(input_ids)

        # Structure: {module_suffix: {layer_idx: [mag_token_0, mag_token_1, ...]}}
        prompt_analysis_data = defaultdict(dict)

        for name, activations in self.captured_activations.items():
            act_abs = activations[0].abs() # Shape: [1, seq_len, features]
            token_max_mag = act_abs.max(dim=-1).values.view(-1).cpu().tolist()
            
            try:
                layer_index = int(name.split('.')[2])
                module_suffix = next((s for s in target_module_suffixes if name.endswith(s)), None)
                if module_suffix:
                    prompt_analysis_data[module_suffix][layer_index] = token_max_mag
            except (ValueError, IndexError):
                continue
        
        self._remove_hooks()

        # Report the data
        for module_name, layer_data in sorted(prompt_analysis_data.items()):
            print(f"\n--- Magnitudes for Module Type: {module_name} ---")
            for layer_idx, mags in sorted(layer_data.items()):
                print(f"  Layer {layer_idx}:")
                for i, mag in enumerate(mags):
                    print(f"    Token '{repr(token_labels[i])}': {mag:.4f}")

        if plot:
            print("\nGenerating prompt spike plot...")
            plot_prompt_spikes(prompt_analysis_data, token_labels, self.model.config._name_or_path, layers_to_plot, self.use_hadamard_transform)


    def _report_module_and_layer_errors(self, module_errors, plot, bits, granularity, exclude_tokens=None):
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
            plot_layer_errors(layer_errors, model_name, bits, granularity, self.use_hadamard_transform, exclude_tokens)
            plot_module_errors(module_errors, model_name, bits, granularity, self.use_hadamard_transform, exclude_tokens)
    
    @torch.no_grad()
    def run_bops_analysis(self, bits: int, plot: bool = False, block_size: int = 128):
        """
        Calculates the theoretical Bit-Operations (BOPs) for each layer, module, and block
        in the model, assuming a uniform weight and activation precision.
        """
        print(f"\nStarting BOPs analysis for {bits}-bit quantization with block size {block_size}...")
        
        bops_per_module = defaultdict(dict)
        bops_per_layer = defaultdict(float)
        bops_per_block = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                try:
                    layer_index = int(name.split('.')[2])
                    module_type = ".".join(name.split('.')[3:])
                except (ValueError, IndexError):
                    layer_index = 'head'
                    module_type = 'lm_head'

                # Per-module BOPs
                bops = (module.in_features * module.out_features * bits * bits) / 1e9
                bops_per_module[layer_index][module_type] = bops
                bops_per_layer[layer_index] += bops

                # Per-block BOPs
                if module.in_features % block_size != 0:
                    print(f"Warning: Module {name} in_features ({module.in_features}) not divisible by block_size ({block_size}). Skipping per-block BOPs calculation for this module.")
                    bops_per_block[name] = []
                else:
                    num_blocks = module.in_features // block_size
                    bops_block = (block_size * module.out_features * bits * bits) / 1e9
                    bops_per_block[name] = [bops_block] * num_blocks

        # --- Report Results ---
        print("\n--- BOPs Analysis Results (Giga-BOPs per token) ---")
        total_bops = sum(bops_per_layer.values())
        for layer_idx in sorted(bops_per_layer.keys(), key=lambda x: (isinstance(x, str), x)):
            print(f"  Layer {layer_idx}: Total G-BOPs = {bops_per_layer[layer_idx]:.4f}")
        print(f"\nTotal Model G-BOPs (Linear Layers): {total_bops:.4f}")

        # --- Save to JSON and Plot ---
        model_name_str = self.model.config._name_or_path.replace('/', '_')
        self._save_to_json(bops_per_layer, f"bops_per_layer_{model_name_str}_{bits}bit.json")
        self._save_to_json(bops_per_module, f"bops_per_module_{model_name_str}_{bits}bit.json")
        self._save_to_json(bops_per_block, f"bops_per_block_{model_name_str}_{bits}bit_bs{block_size}.json")

        if plot:
            print("Generating BOPs plot...")
            plot_bops_analysis(bops_per_module, self.model.config._name_or_path, bits)

    def run_fisher_information_analysis(self, calib_dataset: str, num_samples: int = 128, plot: bool = False):
        """
        Calculates the diagonal Fisher Information for activations.
        This is approximated by the mean of the squared gradients of the loss w.r.t the activations.
        """
        print("\nStarting Activation Fisher Information analysis...")
        self.is_fisher_analysis = True  # Enable gradient retention in hooks
        self._register_hooks()
        
        calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
        if calib_data is None: self._remove_hooks(); return

        # {module_name: [sum_of_sq_grads_sample1, ...]}
        fisher_info_list = defaultdict(list)

        for i in tqdm(range(len(calib_data)), desc="Processing samples for Fisher Info"):
            self.model.zero_grad()
            
            # --- Forward pass with hooks to capture activations ---
            input_ids = calib_data[i]["input_ids"].to(self.device)
            # We need to compute the loss, so we provide labels
            outputs = self.model(input_ids, labels=input_ids)
    
            loss = outputs.loss
            
            # --- Backward pass to compute gradients ---
            # The gradients will be captured by the backward hooks registered below
            loss.backward()

            # --- Process captured gradients ---
            for name, activations in self.captured_activations.items():
                # The gradient is stored in the .grad attribute of the captured tensor
                grad = activations[0].grad
                #print(f"Processing gradients for module: {name} ; Gradient = {grad}")
                if grad is not None:
                    # Fisher Info = E[ (d(log p)/da)^2 ] ~= mean( (dL/da)^2 )
                    fisher_val = torch.sum(grad**2).item()
                    fisher_info_list[name].append(fisher_val)

            self.captured_activations.clear()
            self.model.zero_grad()
            torch.cuda.empty_cache()

        self.is_fisher_analysis = False # Disable gradient retention
        self._remove_hooks()

        # --- Aggregate and Report Results ---
        # Average the Fisher Info across all samples
        avg_fisher_info = {name: np.mean(vals) for name, vals in fisher_info_list.items()}

        # Alternative using LogSumExp to approximate the max:
        # This focuses on the worst-case sensitivity observed.
        lse_fisher_info = {name: logsumexp(np.float64(vals)) for name, vals in fisher_info_list.items()}
        
        per_layer_fisher = defaultdict(float)
        per_module_fisher = defaultdict(lambda: defaultdict(float))

        per_layer_fisher_lse = defaultdict(float)
        per_module_fisher_lse = defaultdict(lambda: defaultdict(float))

        print("\n--- Activation Fisher Information (Average) Results ---")
        for name, fisher_val in sorted(avg_fisher_info.items()):
            print(f"  Module: {name:<50} Fisher Info: {fisher_val:.4f}")
            try:
                layer_index = int(name.split('.')[2])
                module_type = ".".join(name.split('.')[3:])
                per_layer_fisher[layer_index] += fisher_val
                per_module_fisher[module_type][layer_index] = fisher_val
            except (ValueError, IndexError): continue

        print("\n--- Layer-wise Total Fisher Information (Average) ---")
        for layer, val in sorted(per_layer_fisher.items()):
            print(f"  Layer {layer}: {val:.4f}")

        for name, fisher_val in sorted(lse_fisher_info.items()):
            print(f"  Module: {name:<50} Fisher Info (LSE): {fisher_val:.4f}")
            try:
                layer_index = int(name.split('.')[2])
                module_type = ".".join(name.split('.')[3:])
                per_layer_fisher_lse[layer_index] += fisher_val
                per_module_fisher_lse[module_type][layer_index] = fisher_val
            except (ValueError, IndexError): continue
        
        print("\n--- Layer-wise Total Fisher Information (LSE) ---")
        for layer, val in sorted(per_layer_fisher_lse.items()):
            print(f"  Layer {layer}: {val:.4f}")

        # Save results to JSON files
        model_name_str = self.model.config._name_or_path.replace('/', '_')
        self._save_to_json(per_layer_fisher, f"fisher_info_per_layer_{model_name_str}.json")
        self._save_to_json(per_module_fisher, f"fisher_info_per_module_{model_name_str}.json")

        self._save_to_json(per_layer_fisher_lse, f"fisher_info_lse_per_layer_{model_name_str}.json")
        self._save_to_json(per_module_fisher_lse, f"fisher_info_lse_per_module_{model_name_str}.json")

        if plot:
            fisher_data = {
                "per_layer": per_layer_fisher,
                "per_module": per_module_fisher
            }
            plot_fisher_information(fisher_data, self.model.config._name_or_path, agg="average")

            fisher_lse_data = {
                "per_layer": per_layer_fisher_lse,
                "per_module": per_module_fisher_lse
            }
            plot_fisher_information(fisher_lse_data, self.model.config._name_or_path, agg="lse")

    @torch.no_grad()
    def run_max_median_ratio_analysis(self, calib_dataset: str, num_samples: int = 16, plot: bool = False):
        """
        Calculates the max-to-median ratio of token-wise activation scales for each layer and module.
        """
        print("\nStarting Max-to-Median Ratio analysis...")
        self._register_hooks()
        
        calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
        if calib_data is None: self._remove_hooks(); return

        # {module_name: [all_token_scales_from_all_samples]}
        module_scales = defaultdict(list)

        for i in tqdm(range(len(calib_data)), desc="Processing samples for Max-Median Ratio"):
            self.model(calib_data[i]["input_ids"].to(self.device))
            
            for name, activations in self.captured_activations.items():
                act_abs = activations[0].abs()
                # S(m) is the set of token-wise activation scales
                # We define the scale of a token as its max absolute value along the feature dimension
                token_scales = act_abs.max(dim=-1).values.view(-1)
                module_scales[name].append(token_scales.cpu())

            self.captured_activations.clear()

        self._remove_hooks()

        # --- Aggregate and Report Results ---
        per_layer_ratio = {}
        per_module_ratio = defaultdict(dict)

        print("\n--- Max-to-Median Ratio Results ---")
        for name, scales_list in sorted(module_scales.items()):
            all_scales = torch.cat(scales_list).to(torch.float32).numpy()
            if all_scales.size == 0: continue
            
            max_val = np.max(all_scales)
            median_val = np.median(all_scales)
            ratio = float(max_val / median_val) if median_val > 1e-6 else 0.0

            print(f"  Module: {name:<50} Ratio: {ratio:.4f}")
            try:
                layer_index = int(name.split('.')[2])
                module_type = ".".join(name.split('.')[3:])
                per_module_ratio[module_type][layer_index] = ratio
            except (ValueError, IndexError): continue
        
        # Calculate per-layer ratio by aggregating all scales within a layer
        layer_scales = defaultdict(list)
        for name, scales_list in module_scales.items():
            try:
                layer_index = int(name.split('.')[2])
                layer_scales[layer_index].append(torch.cat(scales_list))
            except (ValueError, IndexError): continue
            
        print("\n--- Layer-wise Max-to-Median Ratio ---")
        for layer_idx, scales_list in sorted(layer_scales.items()):
            all_layer_scales = torch.cat(scales_list).to(torch.float32).numpy()
            if all_layer_scales.size == 0: continue
            max_val = np.max(all_layer_scales)
            median_val = np.median(all_layer_scales)
            ratio = float(max_val / median_val) if median_val > 1e-6 else 0.0
            per_layer_ratio[layer_idx] = ratio
            print(f"  Layer {layer_idx}: {ratio:.4f}")
        
        # Save results to JSON files
        model_name_str = self.model.config._name_or_path.replace('/', '_')
        self._save_to_json(per_layer_ratio, f"max_median_ratio_per_layer_{model_name_str}.json")
        self._save_to_json(per_module_ratio, f"max_median_ratio_per_module_{model_name_str}.json")

        if plot:
            ratio_data = {
                "per_layer": per_layer_ratio,
                "per_module": per_module_ratio
            }
            plot_max_median_ratio(ratio_data, self.model.config._name_or_path)
    
    
    def run_fgmp_sensitivity_analysis(self, calib_dataset: str, num_samples: int = 16, plot: bool = False, high_prec_bits: int = 8, low_prec_bits: int = 4, block_size: int = 128):
        """
        Calculates the FGMP sensitivity metric (Impact Score) for each activation block.
        This version is extremely memory-efficient by performing a separate forward/backward
        pass for each module, ensuring only one gradient tensor is in memory at a time.
        """
        print(f"\nStarting FGMP sensitivity analysis (FP{high_prec_bits} vs FP{low_prec_bits}) with block size {block_size}...")
        self.is_fisher_analysis = True
        
        calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
        if calib_data is None: return

        # Get a list of all linear module names first
        all_module_names = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                all_module_names.append(name)
        
        # Structure: {module_name: {'sum': tensor_of_sums, 'count': total_tokens_processed}}
        block_sensitivity_accumulator = {}

        for i in tqdm(range(len(calib_data)), desc="Processing samples for FGMP Sensitivity"):
            input_ids = calib_data[i]["input_ids"].to(self.device)

            # Now, loop through each module for this sample
            for name in tqdm(all_module_names, desc=f"Sample {i+1}/{len(calib_data)} - Modules", leave=False):
                self.model.zero_grad()
                self._remove_hooks()

                # Hook only the current module
                self._register_hooks(module_names=[name])
                
                # Perform a forward pass to capture the activation and build the graph
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                
                # Get the activation for the hooked module
                if name not in self.captured_activations or not self.captured_activations[name]:
                    self._remove_hooks()
                    continue
                act = self.captured_activations[name][0]
                
                # Perform backward pass. Graph is NOT retained.
                (grad,) = torch.autograd.grad(loss, act)

                if grad is not None:
                    # The rest of the calculation does not require gradients, so we wrap it in no_grad
                    with torch.no_grad():
                        original_features = act.shape[-1]
                        
                        padded_act = act
                        padded_grad = grad
                        if original_features % block_size != 0:
                            padding_size = block_size - (original_features % block_size)
                            padded_act = torch.nn.functional.pad(act, (0, padding_size))
                            padded_grad = torch.nn.functional.pad(grad, (0, padding_size))
                        
                        features = padded_act.shape[-1]
                        num_blocks = features // block_size

                        if name not in block_sensitivity_accumulator:
                            block_sensitivity_accumulator[name] = {
                                'sum': torch.zeros(num_blocks, device=self.device, dtype=torch.float32),
                                'count': 0
                            }

                        act_q_high = fake_quantize_activation(padded_act, high_prec_bits, 'per-block', block_size=block_size)
                        act_q_low = fake_quantize_activation(padded_act, low_prec_bits, 'per-block', block_size=block_size)
                        
                        delta_q_error_sq = act_q_low.sub_(act_q_high).pow_(2)
                        
                        fisher_info = padded_grad.pow_(2)
                        impact_score_tensor = fisher_info.mul_(delta_q_error_sq)
                        
                        batch_size, seq_len, _ = impact_score_tensor.shape
                        blocked_scores = impact_score_tensor.view(batch_size, seq_len, num_blocks, block_size)
                        
                        impact_scores_per_block = torch.sum(blocked_scores, dim=-1)

                        sum_over_seq = torch.sum(impact_scores_per_block.squeeze(0), dim=0)

                        block_sensitivity_accumulator[name]['sum'] += sum_over_seq.to(torch.float32)
                        block_sensitivity_accumulator[name]['count'] += seq_len
                
                self._remove_hooks()
                del outputs, loss, act, grad
                torch.cuda.empty_cache()

        self.is_fisher_analysis = False

        # --- Finalize Averages and Report Results ---
        final_block_scores = {}
        for name, data in block_sensitivity_accumulator.items():
            count = data['count']
            if count > 0:
                avg_scores = data['sum'] / count
                final_block_scores[name] = avg_scores.cpu().numpy()

        print("\n--- FGMP Per-Block Sensitivity Analysis Complete ---")
        print("Detailed block-wise sensitivity scores calculated for each module.")
        
        model_name_str = self.model.config._name_or_path.replace('/', '_')
        
        # Save the detailed per-block scores to a JSON file
        filename = f"fgmp_per_block_sensitivity_{model_name_str}_{high_prec_bits}_{low_prec_bits}_bs{block_size}.json"
        self._save_to_json(final_block_scores, filename)

        # --- Generate and Save Summary Data ---
        avg_module_sensitivity = {name: np.mean(scores) for name, scores in final_block_scores.items()}
        per_layer_sensitivity = defaultdict(list)
        per_module_sensitivity = defaultdict(lambda: defaultdict(float))

        for name, score in avg_module_sensitivity.items():
            try:
                layer_index = int(name.split('.')[2])
                module_type = ".".join(name.split('.')[3:])
                per_layer_sensitivity[layer_index].append(score)
                per_module_sensitivity[module_type][layer_index] = score
            except (ValueError, IndexError): continue

        avg_per_layer_sensitivity = {layer: np.mean(vals) for layer, vals in per_layer_sensitivity.items()}
        
        # Save summary files
        self._save_to_json(avg_per_layer_sensitivity, f"fgmp_sensitivity_per_layer_{model_name_str}_{high_prec_bits}_{low_prec_bits}_bs{block_size}.json")
        self._save_to_json(per_module_sensitivity, f"fgmp_sensitivity_per_module_{model_name_str}_{high_prec_bits}_{low_prec_bits}_bs{block_size}.json")

        if plot:
            print("\nGenerating summary plots...")
            sensitivity_data = {
                "per_layer": avg_per_layer_sensitivity,
                "per_module": per_module_sensitivity
            }
            plot_fgmp_sensitivity(sensitivity_data, self.model.config._name_or_path, high_prec_bits, low_prec_bits, block_size)

    # def run_fgmp_sensitivity_analysis(self, calib_dataset: str, num_samples: int = 16, plot: bool = False, high_prec_bits: int = 8, low_prec_bits: int = 4, block_size: int = 128):
    #     """
    #     Calculates the FGMP sensitivity metric (Impact Score) for activations.
    #     Impact Score = E[ grad^2 * (Q_low(act) - Q_high(act))^2 ]
    #     """
    #     print(f"\nStarting FGMP sensitivity analysis (FP{high_prec_bits} vs FP{low_prec_bits})...")
    #     self.is_fisher_analysis = True
    #     self._register_hooks()

    #     block_sensitivity_accumulator = {}
        
    #     calib_data = get_calibration_data(calib_dataset, self.tokenizer, n_samples=num_samples)
    #     if calib_data is None: self._remove_hooks(); return

    #     sensitivity_scores = defaultdict(list)

    #     for i in tqdm(range(len(calib_data)), desc="Processing samples for FGMP Sensitivity"):
    #         self.model.zero_grad()
    #         input_ids = calib_data[i]["input_ids"].to(self.device)
    #         outputs = self.model(input_ids, labels=input_ids)
    #         loss = outputs.loss
    #         loss.backward()

    #         for name, activations in self.captured_activations.items():
    #             act = activations[0]
    #             grad = act.grad
    #             if grad is not None:
    #                 features = act.shape[-1]
    #                 if features % block_size != 0:
    #                     print(f"Warning: Skipping module {name} for FGMP analysis. Feature dimension {features} is not divisible by block_size {block_size}.")
    #                     continue

    #                 # Initialize accumulator for this module if not seen before
    #                 if name not in block_sensitivity_accumulator:
    #                     num_blocks = features // block_size
    #                     block_sensitivity_accumulator[name] = {
    #                         'sum': torch.zeros(num_blocks, device=self.device, dtype=torch.float32),
    #                         'count': 0
    #                     }

    #                 # Calculate Fisher Info part
    #                 fisher_info = grad**2
                    
    #                 # Calculate Change in Quantization Error part using 'per-block' granularity
    #                 act_q_high = fake_quantize_activation(act, high_prec_bits, 'per-block', block_size=block_size)
    #                 act_q_low = fake_quantize_activation(act, low_prec_bits, 'per-block', block_size=block_size)
    #                 delta_q_error_sq = (act_q_low - act_q_high)**2
                    
    #                 # Calculate impact score per element
    #                 impact_score_tensor = fisher_info * delta_q_error_sq

    #                 # Reshape to calculate sum per block
    #                 batch_size, seq_len, _ = impact_score_tensor.shape
    #                 num_blocks = features // block_size
    #                 blocked_scores = impact_score_tensor.view(batch_size, seq_len, num_blocks, block_size)
                    
    #                 # Sum over the block dimension to get the total impact score for each block
    #                 impact_scores_per_block = torch.sum(blocked_scores, dim=-1) # Shape: [batch, seq_len, num_blocks]
                    
    #                 # Sum the block scores over the sequence dimension for this sample
    #                 sum_over_seq = torch.sum(impact_scores_per_block.squeeze(0), dim=0) # shape [num_blocks]

    #                 # Accumulate sums and counts (number of tokens)
    #                 block_sensitivity_accumulator[name]['sum'] += sum_over_seq.to(torch.float32)
    #                 block_sensitivity_accumulator[name]['count'] += seq_len

    #         self.captured_activations.clear()
    #         self.model.zero_grad()
    #         torch.cuda.empty_cache()

    #     self.is_fisher_analysis = False
    #     self._remove_hooks()

    #     # --- Finalize Averages and Report Results ---
    #     # This will have the structure: {module_name: [avg_score_block_0, avg_score_block_1, ...]}
    #     final_block_scores = {}
    #     for name, data in block_sensitivity_accumulator.items():
    #         count = data['count']
    #         if count > 0:
    #             # Average score per block over all tokens and samples
    #             avg_scores = data['sum'] / count
    #             final_block_scores[name] = avg_scores.cpu().numpy()

    #     print("\n--- FGMP Per-Block Sensitivity Analysis Complete ---")
    #     print("Detailed block-wise sensitivity scores calculated for each module.")
        
    #     # Save the detailed per-block scores to a JSON file
    #     model_name_str = self.model.config._name_or_path.replace('/', '_')
    #     filename = f"fgmp_per_block_sensitivity_{model_name_str}_{high_prec_bits}_{low_prec_bits}_bs{block_size}.json"
    #     self._save_to_json(final_block_scores, filename)

    #     if plot:
    #         print("\nGenerating summary plots...")
    #         # For plotting, we compute summary statistics from the detailed scores
            
    #         # Average sensitivity for each module instance (average of its block scores)
    #         avg_module_sensitivity = {name: np.mean(scores) for name, scores in final_block_scores.items()}

    #         per_layer_sensitivity = defaultdict(list)
    #         per_module_sensitivity = defaultdict(lambda: defaultdict(float))

    #         for name, score in avg_module_sensitivity.items():
    #             try:
    #                 layer_index = int(name.split('.')[2])
    #                 module_type = ".".join(name.split('.')[3:])
    #                 per_layer_sensitivity[layer_index].append(score)
    #                 per_module_sensitivity[module_type][layer_index] = score
    #             except (ValueError, IndexError): continue

    #         avg_per_layer_sensitivity = {layer: np.mean(vals) for layer, vals in per_layer_sensitivity.items()}

    #         sensitivity_data = {
    #             "per_layer": avg_per_layer_sensitivity,
    #             "per_module": per_module_sensitivity
    #         }
    #         plot_fgmp_sensitivity(sensitivity_data, self.model.config._name_or_path, high_prec_bits, low_prec_bits, block_size)