"""
Advanced Quantization Module
Supports: Per-channel, Per-tensor, QAT, Mixed-granularity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np


# ============================================================================
# PER-CHANNEL ASYMMETRIC QUANTIZATION (Industry Standard)
# ============================================================================

@torch.no_grad()
def quantize_per_channel_asymmetric(tensor, bits=8, axis=0):
    """
    Per-channel asymmetric quantization.
    This is what TFLite, ONNX, PyTorch Mobile use.
    
    Args:
        tensor: Weight tensor [out_channels, in_channels, ...]
        bits: Quantization bits (4, 6, 8, 16)
        axis: Channel axis (0 for output channels)
    
    Returns:
        Quantized tensor (dequantized to FP32 for simulation)
    """
    if tensor.dim() < 2:
        # Fallback to per-tensor for 1D (biases)
        return quantize_per_tensor_asymmetric(tensor, bits)
    
    # Move channel axis to front
    tensor = tensor.transpose(0, axis) if axis != 0 else tensor
    original_shape = tensor.shape
    num_channels = tensor.shape[0]
    
    # Reshape to [num_channels, -1]
    tensor_2d = tensor.reshape(num_channels, -1)
    
    # Calculate per-channel min/max
    channel_min = tensor_2d.min(dim=1, keepdim=True)[0]
    channel_max = tensor_2d.max(dim=1, keepdim=True)[0]
    
    # Quantization range
    qmin = 0
    qmax = 2**bits - 1
    
    # Calculate scale and zero_point per channel
    scale = (channel_max - channel_min) / (qmax - qmin)
    scale = torch.clamp(scale, min=1e-8)  # Avoid division by zero
    
    zero_point = qmin - torch.round(channel_min / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)
    
    # Quantize
    quantized = torch.round(tensor_2d / scale + zero_point)
    quantized = torch.clamp(quantized, qmin, qmax)
    
    # Dequantize (for PTQ simulation)
    dequantized = (quantized - zero_point) * scale
    
    # Reshape back
    dequantized = dequantized.reshape(original_shape)
    
    # Transpose back if needed
    if axis != 0:
        dequantized = dequantized.transpose(0, axis)
    
    return dequantized


@torch.no_grad()
def quantize_per_tensor_asymmetric(tensor, bits=4):
    """Per-tensor asymmetric quantization (fallback)."""
    qmin = 0
    qmax = 2**bits - 1
    
    min_val = tensor.min()
    max_val = tensor.max()
    
    scale = (max_val - min_val) / (qmax - qmin)
    scale = max(scale, 1e-8)  # ← Check this isn't too small
    
    zero_point = qmin - torch.round(min_val / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)
    
    quantized = torch.round(tensor / scale + zero_point)
    quantized = torch.clamp(quantized, qmin, qmax)
    
    dequantized = (quantized - zero_point) * scale
    
    return dequantized


# ============================================================================
# MIXED-GRANULARITY QUANTIZATION
# ============================================================================

@torch.no_grad()
def quantize_attention_per_head(attention_weights, bits=4, num_heads=12):
    """
    Per-head quantization for attention layers.
    Each attention head quantized independently.
    
    Args:
        attention_weights: [hidden_dim, hidden_dim]
        bits: Quantization bits
        num_heads: Number of attention heads
    
    Returns:
        Quantized weights
    """
    hidden_dim = attention_weights.shape[0]
    head_dim = hidden_dim // num_heads
    
    # Reshape to [num_heads, head_dim, hidden_dim]
    heads = attention_weights.reshape(num_heads, head_dim, -1)
    
    quantized_heads = []
    for i in range(num_heads):
        head = heads[i]
        # Quantize this head independently
        quantized_head = quantize_per_channel_asymmetric(head, bits=bits, axis=0)
        quantized_heads.append(quantized_head)
    
    # Reassemble
    quantized = torch.stack(quantized_heads).reshape(hidden_dim, -1)
    
    return quantized


def is_attention_layer(layer_name):
    """Check if layer is attention."""
    attention_keywords = ['attention', 'attn', 'self', 'query', 'key', 'value']
    return any(kw in layer_name.lower() for kw in attention_keywords)


# ============================================================================
# QUANTIZATION-AWARE TRAINING (QAT) SUPPORT
# ============================================================================

class FakeQuantize(nn.Module):
    """
    Fake quantization layer for QAT.
    Simulates quantization during training.
    """
    
    def __init__(self, bits=8, per_channel=True, num_channels=None):
        super().__init__()
        self.bits = bits
        self.per_channel = per_channel
        
        # Learnable quantization parameters
        if per_channel and num_channels:
            self.register_buffer('scale', torch.ones(num_channels, 1))
            self.register_buffer('zero_point', torch.zeros(num_channels, 1))
        else:
            self.register_buffer('scale', torch.tensor(1.0))
            self.register_buffer('zero_point', torch.tensor(0.0))
        
        self.qmin = 0
        self.qmax = 2**bits - 1
    
    def forward(self, x):
        """Apply fake quantization."""
        # Calculate scale and zero_point if not initialized
        if self.training:
            with torch.no_grad():
                if self.per_channel and x.dim() >= 2:
                    # Per-channel
                    channel_min = x.reshape(x.shape[0], -1).min(dim=1, keepdim=True)[0].unsqueeze(1)
                    channel_max = x.reshape(x.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(1)
                else:
                    # Per-tensor
                    channel_min = x.min()
                    channel_max = x.max()
                
                self.scale = (channel_max - channel_min) / (self.qmax - self.qmin)
                self.scale = torch.clamp(self.scale, min=1e-8)
                self.zero_point = self.qmin - channel_min / self.scale
                self.zero_point = torch.clamp(self.zero_point, self.qmin, self.qmax)
        
        # Quantize
        x_quant = torch.round(x / self.scale + self.zero_point)
        x_quant = torch.clamp(x_quant, self.qmin, self.qmax)
        
        # Dequantize (with gradient flow)
        x_dequant = (x_quant - self.zero_point) * self.scale
        
        # Straight-through estimator for gradients
        if self.training:
            return x + (x_dequant - x).detach()
        else:
            return x_dequant


def prepare_model_for_qat(model, bits=8):
    """
    Prepare model for Quantization-Aware Training.
    Adds fake quantization nodes.
    
    Args:
        model: PyTorch model
        bits: Quantization bits
    
    Returns:
        Model with fake quantization layers
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Add fake quantization to weight
            num_channels = module.weight.shape[0]
            module.weight_fake_quant = FakeQuantize(bits=bits, per_channel=True, num_channels=num_channels)
    
    return model


# ============================================================================
# STANDARD QUANTIZERS (Pre-defined)
# ============================================================================

# Per-channel quantizers (recommended)
quant_per_channel_int4 = partial(quantize_per_channel_asymmetric, bits=4)
quant_per_channel_int4.__name__ = "PerChannel_INT4"

quant_per_channel_int6 = partial(quantize_per_channel_asymmetric, bits=6)
quant_per_channel_int6.__name__ = "PerChannel_INT6"

quant_per_channel_int8 = partial(quantize_per_channel_asymmetric, bits=8)
quant_per_channel_int8.__name__ = "PerChannel_INT8"

quant_per_channel_int16 = partial(quantize_per_channel_asymmetric, bits=16)
quant_per_channel_int16.__name__ = "PerChannel_INT16"

# Per-tensor quantizers (fallback)
quant_per_tensor_int4 = partial(quantize_per_tensor_asymmetric, bits=4)
quant_per_tensor_int4.__name__ = "PerTensor_INT4"

quant_per_tensor_int8 = partial(quantize_per_tensor_asymmetric, bits=8)
quant_per_tensor_int8.__name__ = "PerTensor_INT8"


# ============================================================================
# QUANTIZER REGISTRY
# ============================================================================

def get_quantizer(precision_str):
    """
    Get quantizer function by precision string.
    
    Args:
        precision_str: 'INT4', 'INT6', 'INT8', 'INT16', 'FP32'
    
    Returns:
        Quantization function
    """
    quantizers = {
        'INT4': quant_per_channel_int4,
        'INT6': quant_per_channel_int6,
        'INT8': quant_per_channel_int8,
        'INT16': quant_per_channel_int16,
        'FP32': lambda x: x,  # No quantization
        'FP16': lambda x: x.half().float(),  # FP16 simulation
    }
    
    return quantizers.get(precision_str, lambda x: x)


# ============================================================================
# APPLY QUANTIZATION TO MODEL
# ============================================================================

@torch.no_grad()
def apply_quantization_to_model(model, profile, use_mixed_granularity=True):
    """
    Apply quantization to model based on precision profile.
    
    Args:
        model: PyTorch model
        profile: Dict mapping layer names to precision strings
        use_mixed_granularity: Use per-head for attention, per-channel for FFN
    
    Returns:
        None (modifies model in-place)
    """
    for name, param in model.named_parameters():
        if 'weight' not in name or param.dim() < 2:
            continue  # Skip biases and 1D params
        
        # Get precision for this layer
        precision = profile.get(name, 'FP32')
        
        if precision == 'FP32':
            continue  # No quantization
        
        # Determine quantization method
        if use_mixed_granularity and is_attention_layer(name):
            # Per-head for attention
            bits = int(precision.replace('INT', '')) if 'INT' in precision else 8
            
            # Infer number of heads (common: 12 for BERT-base, 16 for BERT-large)
            hidden_dim = param.shape[0]
            num_heads = 12 if hidden_dim == 768 else (16 if hidden_dim == 1024 else 12)
            
            try:
                param.data = quantize_attention_per_head(param.data, bits=bits, num_heads=num_heads)
            except:
                # Fallback to per-channel if per-head fails
                quantizer = get_quantizer(precision)
                param.data = quantizer(param.data)
        else:
            # Per-channel for FFN and others
            quantizer = get_quantizer(precision)
            param.data = quantizer(param.data)


# ============================================================================
# SENSITIVITY ANALYSIS UTILITIES
# ============================================================================

def analyze_layer_sensitivity(model, val_loader, device, layer_name, precision='INT4'):
    """
    Measure sensitivity of a single layer to quantization.
    
    Args:
        model: PyTorch model
        val_loader: Validation dataloader
        device: Device
        layer_name: Name of layer to test
        precision: Precision to test
    
    Returns:
        float: Accuracy drop when this layer is quantized
    """
    from training.train import get_accuracy
    
    # Baseline accuracy (no quantization)
    baseline_acc = get_accuracy(model, val_loader, device)
    
    # Quantize only this layer
    original_weight = None
    for name, param in model.named_parameters():
        if name == layer_name:
            original_weight = param.data.clone()
            quantizer = get_quantizer(precision)
            param.data = quantizer(param.data)
            break
    
    # Measure accuracy with this layer quantized
    quantized_acc = get_accuracy(model, val_loader, device)
    
    # Restore original weight
    for name, param in model.named_parameters():
        if name == layer_name:
            param.data = original_weight
            break
    
    # Return sensitivity (accuracy drop)
    return baseline_acc - quantized_acc
