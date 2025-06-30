import torch

def get_q_params_activation(tensor, bits, granularity):
    """
    Calculates quantization scale and zero-point for an activation tensor.

    Args:
        tensor (torch.Tensor): The activation tensor, shape [batch, seq_len, features].
        bits (int): The number of bits for quantization.
        granularity (str): 'per-tensor' or 'per-token'.

    Returns:
        tuple: A tuple containing the scale and zero-point.
    """
    q_max = 2**bits - 1

    if granularity == "per-tensor":
        # Calculate one scale and zero-point for the entire tensor
        t_max = tensor.max()
        t_min = tensor.min()
        scale = (t_max - t_min).clamp(min=1e-5) / q_max
        zero_point = (-t_min / scale).round()

    elif granularity == "per-token":
        # Calculate a scale and zero-point for each token vector
        # This means we find the min/max across the 'features' dimension
        t_max = tensor.max(dim=-1, keepdim=True)[0] # Shape: [batch, seq_len, 1]
        t_min = tensor.min(dim=-1, keepdim=True)[0] # Shape: [batch, seq_len, 1]
        
        scale = (t_max - t_min).clamp(min=1e-5) / q_max
        zero_point = (-t_min / scale).round()
        
    else:
        raise ValueError(f"Unknown granularity for activations: {granularity}")
        
    return scale, zero_point

def fake_quantize_activation(tensor, bits, granularity):
    """
    Performs fake quantization (quantize and then dequantize) on an activation tensor.
    This simulates the precision loss of quantization.
    """
    scale, zero_point = get_q_params_activation(tensor, bits, granularity)
    
    # Quantize and dequantize
    q_tensor = (tensor / scale + zero_point).round().clamp(0, 2**bits - 1)
    dq_tensor = (q_tensor - zero_point) * scale
    
    return dq_tensor

def calculate_quantization_error(original_tensor, quantized_tensor):
    """
    Calculates the Mean Squared Error (MSE) between the original and quantized tensors.
    """
    # Ensure tensors are float for mse_loss
    original_tensor = original_tensor.to(torch.float32)
    quantized_tensor = quantized_tensor.to(torch.float32)
    return torch.nn.functional.mse_loss(original_tensor, quantized_tensor).item()

