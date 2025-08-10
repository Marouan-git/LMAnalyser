import torch

def get_q_params_activation(tensor, bits, granularity, block_size=128):
    """
    Calculates quantization scale and zero-point for an activation tensor.

    Args:
        tensor (torch.Tensor): The activation tensor, shape [batch, seq_len, features].
        bits (int): The number of bits for quantization.
        granularity (str): 'per-tensor', 'per-token', or 'per-block'.
        block_size (int): The size of blocks for 'per-block' granularity.

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
    elif granularity == "per-block":
        # Reshape for block-wise quantization
        batch_size, seq_len, features = tensor.shape
        if features % block_size != 0:
            # To handle this gracefully, we could pad, but for analysis, raising an error is clearer.
            raise ValueError(f"Feature dimension ({features}) must be divisible by block_size ({block_size})")
        num_blocks = features // block_size
        
        # Reshape to [batch, seq_len, num_blocks, block_size]
        blocked_tensor = tensor.view(batch_size, seq_len, num_blocks, block_size)
        
        # Calculate min/max over the last dimension (the block)
        t_max = blocked_tensor.max(dim=-1, keepdim=True)[0] # Shape: [batch, seq_len, num_blocks, 1]
        t_min = blocked_tensor.min(dim=-1, keepdim=True)[0] # Shape: [batch, seq_len, num_blocks, 1]

        scale = (t_max - t_min).clamp(min=1e-5) / q_max
        zero_point = (-t_min / scale).round()
        
    else:
        raise ValueError(f"Unknown granularity for activations: {granularity}")
        
    return scale, zero_point

def fake_quantize_activation(tensor, bits, granularity, block_size=128):
    """
    Performs fake quantization (quantize and then dequantize) on an activation tensor.
    This simulates the precision loss of quantization.
    """
    padded_tensor = tensor
    original_features = tensor.shape[-1]
    padding_size = 0

    if granularity == "per-block" and original_features % block_size != 0:
        padding_size = block_size - (original_features % block_size)
        padded_tensor = torch.nn.functional.pad(tensor, (0, padding_size))

    scale, zero_point = get_q_params_activation(tensor, bits, granularity, block_size)

    if granularity == "per-block":
        batch_size, seq_len, features = padded_tensor.shape
        num_blocks = features // block_size
        blocked_tensor = padded_tensor.view(batch_size, seq_len, num_blocks, block_size)
        
        q_tensor_blocked = (blocked_tensor / scale + zero_point).round().clamp(0, 2**bits - 1)
        dq_tensor_blocked = (q_tensor_blocked - zero_point) * scale
        
        dq_tensor = dq_tensor_blocked.view(batch_size, seq_len, features)
    else:
        # Quantize and dequantize
        q_tensor = (tensor / scale + zero_point).round().clamp(0, 2**bits - 1)
        dq_tensor = (q_tensor - zero_point) * scale
    
    if padding_size > 0:
        dq_tensor = dq_tensor[..., :original_features]
    
    return dq_tensor

def calculate_quantization_error(original_tensor, quantized_tensor):
    """
    Calculates the Mean Squared Error (MSE) between the original and quantized tensors.
    """
    # Ensure tensors are float for mse_loss
    original_tensor = original_tensor.to(torch.float32)
    quantized_tensor = quantized_tensor.to(torch.float32)
    return torch.nn.functional.mse_loss(original_tensor, quantized_tensor).item()

