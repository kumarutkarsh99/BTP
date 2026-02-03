# quantizer.py
import torch
from functools import partial

@torch.no_grad()
def quantize_to_adaptivfloat(fp32_tensor, total_bits, exponent_bits):
    """
    Implements the AdaptivFloat algorithm (simulated quantization).
    Based on Section 3 of the paper.
    """
    # --- 1. Calculate Format Parameters ---
    sign_bits = 1
    if total_bits <= (exponent_bits + sign_bits):
        raise ValueError("total_bits must be > exponent_bits + 1 (for sign bit)")
    mantissa_bits = total_bits - sign_bits - exponent_bits
    
    # --- 2. Find Exponent Bias (The "Shift") ---
    max_abs_val = torch.max(torch.abs(fp32_tensor))
    
    if max_abs_val == 0:
        return fp32_tensor # Tensor is all zeros

    max_exponent_needed = torch.floor(torch.log2(max_abs_val))
    max_exponent_storable = (2**exponent_bits) - 1
    exponent_bias = max_exponent_needed - max_exponent_storable
    
    # --- 3. Define the "Adaptiv" Quantization Range ---
    min_normal_exponent = 0 + exponent_bias
    max_exponent = max_exponent_storable + exponent_bias
    AF_min_normal = 2.0 ** min_normal_exponent
    max_mantissa = 2.0 - (2.0**(-mantissa_bits))
    AF_max = max_mantissa * (2.0**max_exponent)
    
    # --- 4. Quantize the Tensor (The "Snap-to-Grid") ---
    quantized_tensor = fp32_tensor.clone()
    sign = torch.sign(quantized_tensor)
    abs_tensor = torch.abs(quantized_tensor)
    
    # 4a. Handle Zeros (The "Sacrifice" from Sec 3.1 & 3.2)
    zero_threshold = AF_min_normal / 2.0
    zero_mask = abs_tensor < zero_threshold
    
    # 4b. Handle Clamping
    clamp_mask = abs_tensor > AF_max
    abs_tensor[clamp_mask] = AF_max
    
    # 4c. Quantize "Normal" Values
    normal_mask = ~zero_mask & ~clamp_mask
    
    if normal_mask.any():
        vals_to_quantize = abs_tensor[normal_mask]
        exponent_of_values = torch.floor(torch.log2(vals_to_quantize))
        mantissa = vals_to_quantize / (2.0**exponent_of_values)
        mantissa_fraction = mantissa - 1.0
        num_quant_steps = 2.0**mantissa_bits
        quantized_fraction = torch.round(mantissa_fraction * num_quant_steps) / num_quant_steps
        quantized_mantissa = 1.0 + quantized_fraction
        quantized_abs_val = quantized_mantissa * (2.0**exponent_of_values)
        abs_tensor[normal_mask] = quantized_abs_val
        
    # 4d. Put Zeros back
    abs_tensor[zero_mask] = 0.0
    
    # 4e. Re-apply the sign
    final_tensor = sign * abs_tensor
    
    return final_tensor

# --- Define our standard quantizer functions ---
quant_af4_func = partial(quantize_to_adaptivfloat, total_bits=4, exponent_bits=2)
quant_af4_func.__name__ = "AdaptivFloat_4bit"

quant_af8_func = partial(quantize_to_adaptivfloat, total_bits=8, exponent_bits=3)
quant_af8_func.__name__ = "AdaptivFloat_8bit"