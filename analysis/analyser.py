import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from data.data_loader import get_c4_calibration_data
from analysis.quantization import fake_quantize_activation, calculate_quantization_error
from plotting.plotter import plot_layer_errors, plot_module_errors, plot_top_token_errors_by_module

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
    def run_quantization_error_analysis(self, bits: int, granularity: str, num_samples: int = 16, plot: bool = False):
        """
        Runs a forward pass on calibration data to capture activations,
        then calculates and prints the quantization error for all linear layers.
        This version is memory-efficient by processing one sample at a time.
        """
        self._register_hooks() # Hook all linear layers

        calib_data = get_c4_calibration_data(self.tokenizer, n_samples=num_samples)
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
    def run_per_token_error_analysis(self, bits: int, granularity: str, plot: bool = False, num_samples: int = 16):
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

        calib_data = get_c4_calibration_data(self.tokenizer, n_samples=num_samples)
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


# import torch
# import numpy as np
# from tqdm import tqdm
# from collections import defaultdict
# from data.data_loader import get_c4_calibration_data
# from analysis.quantization import fake_quantize_activation, calculate_quantization_error
# from plotting.plotter import plot_layer_errors, plot_module_errors, plot_top_token_errors_by_module

# class LLMAnalyser:
#     def __init__(self, model, tokenizer):
#         self.model = model
#         self.tokenizer = tokenizer
#         self.device = model.device
#         self.captured_activations = defaultdict(list)
#         self.hooks = []

#     def _get_hook(self, name):
#         def hook(model, input, output):
#             # We are interested in the input to the linear layers, which are activations.
#             # input is a tuple, we take the first element.
#             self.captured_activations[name].append(input[0].detach())
#         return hook

#     def _register_hooks(self, module_names=None):
#         """
#         Registers forward hooks on specified or all linear layers of the model.

#         Args:
#             module_names (list, optional): A list of specific module name suffixes to hook. 
#                                            If None, hooks all linear layers.
#         """
#         print("Registering hooks to capture activations...")
#         target_modules = []
#         for name, module in self.model.named_modules():
#             if isinstance(module, torch.nn.Linear):
#                 # If no specific module names are given, hook all linear layers.
#                 # If module_names are given, hook if the module's name ends with one of the suffixes.
#                 if module_names is None or any(name.endswith(suffix) for suffix in module_names):
#                     target_modules.append((name, module))

#         for name, module in target_modules:
#             self.hooks.append(module.register_forward_hook(self._get_hook(name)))
#         print(f"Registered {len(self.hooks)} hooks.")

#     def _remove_hooks(self):
#         """Removes all registered hooks and clears captured data."""
#         for hook in self.hooks:
#             hook.remove()
#         self.hooks = []
#         self.captured_activations.clear()

#     @torch.no_grad()
#     def run_quantization_error_analysis(self, bits: int, granularity: str, num_samples: int = 16, plot: bool = False):
#         """
#         Runs a forward pass on calibration data to capture activations,
#         then calculates and prints the quantization error for all linear layers.
#         This version is memory-efficient by processing one sample at a time.
#         """
#         self._register_hooks() # Hook all linear layers

#         calib_data = get_c4_calibration_data(self.tokenizer, n_samples=num_samples)
#         if calib_data is None:
#             self._remove_hooks()
#             return
            
#         print("\nCalculating activation quantization error...")
#         # Structure: {module_name: [error_sample_1, error_sample_2, ...]}
#         module_errors_list = defaultdict(list)
        
#         for i in tqdm(range(num_samples)):
#             input_ids = calib_data[i]["input_ids"].to(self.device)
#             self.model(input_ids)
            
#             for name, activations in self.captured_activations.items():
#                 activation_tensor = activations[0] # We process one sample at a time
#                 quantized_tensor = fake_quantize_activation(activation_tensor, bits, granularity)
#                 error = calculate_quantization_error(activation_tensor, quantized_tensor)
#                 module_errors_list[name].append(error)
            
#             self.captured_activations.clear() # Free memory after each sample

#         # Average the errors for each module across all samples
#         module_errors_avg = {name: np.mean(errors) for name, errors in module_errors_list.items()}
        
#         self._remove_hooks()
#         self._report_module_and_layer_errors(module_errors_avg, plot, bits, granularity)

#     @torch.no_grad()
#     def run_per_token_error_analysis(self, bits: int, granularity: str, plot: bool = False, num_samples: int = 16):
#         """
#         Analyzes per-token quantization error for major module types across all layers
#         to find the unique tokens that are most sensitive to quantization on average.
#         This version is memory-efficient by processing one sample at a time.
#         """
#         # Define a representative set of module types for Llama-like architectures
#         target_module_suffixes = [
#             'self_attn.q_proj', # Represents q, k, v inputs
#             'self_attn.o_proj',
#             'mlp.gate_proj',    # Represents gate and up inputs
#             'mlp.down_proj'
#         ]
        
#         self._register_hooks(module_names=target_module_suffixes)

#         calib_data = get_c4_calibration_data(self.tokenizer, n_samples=num_samples)
#         if calib_data is None:
#             self._remove_hooks()
#             return

#         print(f"\nAnalyzing per-token error with {granularity} quantization...")
        
#         # Structure: {module_suffix: {token_id: [error1, error2, ...]}}
#         module_token_errors = defaultdict(lambda: defaultdict(list))
        
#         for i in tqdm(range(num_samples)):
#             input_ids = calib_data[i]["input_ids"].to(self.device)
#             self.model(input_ids) # This populates self.captured_activations for this sample
            
#             input_ids_cpu = calib_data[i]["input_ids"].view(-1)

#             # Process activations for this single sample
#             for name, activations in self.captured_activations.items():
#                 activation_tensor = activations[0] # Shape: [1, seq_len, features]
                
#                 # Determine module type from full name
#                 module_suffix = next((s for s in target_module_suffixes if name.endswith(s)), None)
#                 if not module_suffix: continue

#                 quantized_tensor = fake_quantize_activation(activation_tensor, bits, granularity)
#                 per_token_mse = (activation_tensor - quantized_tensor).pow(2).mean(dim=-1).view(-1)
                
#                 # Aggregate errors for this module type
#                 for token_idx in range(per_token_mse.size(0)):
#                     token_id = input_ids_cpu[token_idx].item()
#                     error = per_token_mse[token_idx].item()
#                     module_token_errors[module_suffix][token_id].append(error)
            
#             self.captured_activations.clear() # CRITICAL: Free memory after processing each sample

#         torch.cuda.empty_cache()
#         self._remove_hooks()

#         # --- Calculate and Report Final Averages ---
#         module_top_tokens = {}
#         for module_suffix, token_errors_dict in module_token_errors.items():
#             avg_token_errors = []
#             for token_id, errors in token_errors_dict.items():
#                 avg_token_errors.append((np.mean(errors), token_id))
            
#             avg_token_errors.sort(key=lambda x: x[0], reverse=True)
#             top_10 = [(err, self.tokenizer.decode(tid)) for err, tid in avg_token_errors[:10]]
#             module_top_tokens[module_suffix] = top_10

#             print(f"\n    --- Top 10 Tokens for {module_suffix} ---")
#             for avg_error, token_text in top_10:
#                 print(f"      Token: {repr(token_text):<15} | Average MSE: {avg_error:.8f}")

#         if plot:
#             print("\nGenerating plots...")
#             model_name = self.model.config._name_or_path
#             plot_top_token_errors_by_module(module_top_tokens, model_name, bits, granularity)

#     def _report_module_and_layer_errors(self, module_errors, plot, bits, granularity):
#         """Prints the module/layer/model quantization errors and generates plots if requested."""
#         print("\n--- Activation Quantization Error Report (MSE) ---")
        
#         layer_errors = defaultdict(list)
#         for name, error in module_errors.items():
#             print(f"  Module: {name:<50} MSE: {error:.8f}")
            
#             try:
#                 layer_index = int(name.split('.')[2])
#                 layer_errors[layer_index].append(error)
#             except (ValueError, IndexError):
#                 layer_errors['other'].append(error)
        
#         print("\n--- Layer-wise Average Activation Error ---")
#         all_errors = []
        
#         int_keys = sorted([k for k in layer_errors.keys() if isinstance(k, int)])
#         str_keys = sorted([k for k in layer_errors.keys() if isinstance(k, str)])
#         sorted_keys = int_keys + str_keys
        
#         for layer_idx in sorted_keys:
#             errors = layer_errors[layer_idx]
#             if errors:
#                 avg_error = np.mean(errors)
#                 all_errors.extend(errors)
#                 print(f"  Layer {layer_idx}: {avg_error:.8f}")

#         print("\n--- Model-wide Average Activation Error ---")
#         if all_errors:
#             model_avg_error = np.mean(all_errors)
#             print(f"  Overall Model MSE: {model_avg_error:.8f}")

#         if plot:
#             print("\nGenerating plots...")
#             model_name = self.model.config._name_or_path
#             plot_layer_errors(layer_errors, model_name, bits, granularity)
#             plot_module_errors(module_errors, model_name, bits, granularity)