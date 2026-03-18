"""
Baseline Quantization Methods
For fair comparison with industry standards
"""

import torch
import copy
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# BASELINE EVALUATION FUNCTIONS
# ============================================================================

def evaluate_uniform_quantization(model, val_loader, device, precision='INT8'):
    """
    Uniform quantization baseline (all layers same precision).
    
    Args:
        model: Model to quantize
        val_loader: Validation dataloader
        device: Device
        precision: 'INT4', 'INT6', 'INT8', 'INT16'
    
    Returns:
        float: Accuracy
    """
    from training.train import get_accuracy
    from compression.quantizer import apply_quantization_to_model
    
    model_test = copy.deepcopy(model)
    
    layer_names = [n for n, p in model_test.named_parameters() 
                   if 'weight' in n and p.dim() > 1]
    
    profile = {n: precision for n in layer_names}
    
    apply_quantization_to_model(model_test, profile)
    acc = get_accuracy(model_test, val_loader, device)
    
    del model_test
    return acc


def evaluate_mixed_precision_16x8(model, val_loader, device):
    """
    Mixed INT16×INT8 (TFLite strategy).
    Embeddings + first layer: INT16 (simulated with FP32)
    Rest: INT8
    
    Args:
        model: Model to quantize
        val_loader: Validation dataloader
        device: Device
    
    Returns:
        float: Accuracy
    """
    from training.train import get_accuracy
    from compression.quantizer import apply_quantization_to_model
    
    model_test = copy.deepcopy(model)
    
    profile = {}
    
    for name, param in model_test.named_parameters():
        if 'weight' not in name or param.dim() < 2:
            continue
        
        # First layer and embeddings get higher precision
        if 'embed' in name.lower() or 'layer.0' in name or 'encoder.layer.0' in name:
            profile[name] = 'FP32'  # Simulate INT16
        else:
            profile[name] = 'INT8'
    
    apply_quantization_to_model(model_test, profile)
    acc = get_accuracy(model_test, val_loader, device)
    
    del model_test
    return acc


def evaluate_conservative_hybrid(model, val_loader, device, int4_ratio=0.25):
    """
    Conservative hybrid: Only quantize last N% of layers to INT4.
    
    Args:
        model: Model to quantize
        val_loader: Validation dataloader
        device: Device
        int4_ratio: Fraction of layers to use INT4 (0.25 = 25%)
    
    Returns:
        float: Accuracy
    """
    from training.train import get_accuracy
    from compression.quantizer import apply_quantization_to_model
    
    model_test = copy.deepcopy(model)
    
    layer_names = [n for n, p in model_test.named_parameters() 
                   if 'weight' in n and p.dim() > 1]
    
    # Start with all INT8
    profile = {n: 'INT8' for n in layer_names}
    
    # Quantize last 25% to INT4 (later layers typically less sensitive)
    num_int4 = int(len(layer_names) * int4_ratio)
    int4_layers = layer_names[-num_int4:]
    
    for layer in int4_layers:
        profile[layer] = 'INT4'
    
    apply_quantization_to_model(model_test, profile)
    acc = get_accuracy(model_test, val_loader, device)
    
    del model_test
    return acc


def evaluate_dynamic_quantization(model, val_loader, device):
    """
    Dynamic quantization (weights only, activations in FP32).
    Simulates runtime quantization.
    
    Args:
        model: Model to quantize
        val_loader: Validation dataloader
        device: Device
    
    Returns:
        float: Accuracy
    """
    # For PTQ, this is same as uniform INT8
    # In real deployment, activations would be quantized at runtime
    return evaluate_uniform_quantization(model, val_loader, device, 'INT8')


def evaluate_per_tensor_baseline(model, val_loader, device, bits=8):
    """
    Per-tensor quantization (worse than per-channel).
    For comparison to show per-channel improvement.
    
    Args:
        model: Model to quantize
        val_loader: Validation dataloader
        device: Device
        bits: Quantization bits
    
    Returns:
        float: Accuracy
    """
    from training.train import get_accuracy
    from compression.quantizer import quantize_per_tensor_asymmetric
    
    model_test = copy.deepcopy(model)
    
    # Apply per-tensor quantization
    for name, param in model_test.named_parameters():
        if 'weight' in name and param.dim() > 1:
            param.data = quantize_per_tensor_asymmetric(param.data, bits=bits)
    
    acc = get_accuracy(model_test, val_loader, device)
    
    del model_test
    return acc


# ============================================================================
# COMPREHENSIVE BASELINE SUITE
# ============================================================================

def evaluate_all_baselines(model, val_loader, device, include_per_tensor=False):
    """
    Evaluate all baseline methods for comprehensive comparison.
    
    Args:
        model: Model to evaluate
        val_loader: Validation dataloader
        device: Device
        include_per_tensor: Include per-tensor baselines
    
    Returns:
        dict: Results for all baselines
    """
    logger.info("\n" + "="*60)
    logger.info("BASELINE EVALUATIONS")
    logger.info("="*60)
    
    results = {}
    
    # 1. FP32 (no quantization)
    from training.train import get_accuracy
    results['FP32'] = get_accuracy(model, val_loader, device)
    logger.info(f"  FP32: {results['FP32']:.4f}")
    
    # 2. Uniform INT8
    results['Uniform_INT8'] = evaluate_uniform_quantization(
        model, val_loader, device, 'INT8'
    )
    logger.info(f"  Uniform INT8: {results['Uniform_INT8']:.4f}")
    
    # 3. Uniform INT6
    results['Uniform_INT6'] = evaluate_uniform_quantization(
        model, val_loader, device, 'INT6'
    )
    logger.info(f"  Uniform INT6: {results['Uniform_INT6']:.4f}")
    
    # 4. Uniform INT4
    results['Uniform_INT4'] = evaluate_uniform_quantization(
        model, val_loader, device, 'INT4'
    )
    logger.info(f"  Uniform INT4: {results['Uniform_INT4']:.4f}")
    
    # 5. Mixed INT16×8 (TFLite-style)
    results['Mixed_INT16x8'] = evaluate_mixed_precision_16x8(
        model, val_loader, device
    )
    logger.info(f"  Mixed INT16×8: {results['Mixed_INT16x8']:.4f}")
    
    # 6. Conservative Hybrid (25% INT4)
    results['Conservative_Hybrid_25'] = evaluate_conservative_hybrid(
        model, val_loader, device, int4_ratio=0.25
    )
    logger.info(f"  Conservative Hybrid (25% INT4): {results['Conservative_Hybrid_25']:.4f}")
    
    # 7. Conservative Hybrid (50% INT4)
    results['Conservative_Hybrid_50'] = evaluate_conservative_hybrid(
        model, val_loader, device, int4_ratio=0.50
    )
    logger.info(f"  Conservative Hybrid (50% INT4): {results['Conservative_Hybrid_50']:.4f}")
    
    # 8. Dynamic Quantization
    results['Dynamic_INT8'] = evaluate_dynamic_quantization(
        model, val_loader, device
    )
    logger.info(f"  Dynamic INT8: {results['Dynamic_INT8']:.4f}")
    
    # Optional: Per-tensor baselines (show per-channel improvement)
    if include_per_tensor:
        results['PerTensor_INT8'] = evaluate_per_tensor_baseline(
            model, val_loader, device, bits=8
        )
        logger.info(f"  Per-Tensor INT8: {results['PerTensor_INT8']:.4f}")
        
        results['PerTensor_INT4'] = evaluate_per_tensor_baseline(
            model, val_loader, device, bits=4
        )
        logger.info(f"  Per-Tensor INT4: {results['PerTensor_INT4']:.4f}")
    
    logger.info("="*60 + "\n")
    
    return results


# ============================================================================
# ABLATION BASELINES
# ============================================================================

def evaluate_ablations(model, train_loader, val_loader, device):
    """
    Ablation study: Test contribution of each component.
    
    Args:
        model: FP32 baseline model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device
    
    Returns:
        dict: Ablation results
    """
    from compression.pruning import prune_and_recover
    from training.train import get_accuracy
    from compression.quantizer import apply_quantization_to_model
    
    logger.info("\n" + "="*60)
    logger.info("ABLATION STUDIES")
    logger.info("="*60)
    
    results = {}
    
    # 1. FP32 Baseline
    results['FP32_Baseline'] = get_accuracy(model, val_loader, device)
    logger.info(f"  FP32 Baseline: {results['FP32_Baseline']:.4f}")
    
    # 2. Pruning Only (no quantization)
    model_prune_only = prune_and_recover(
        model, train_loader, val_loader, device,
        use_distillation=True
    )
    results['Pruning_Only'] = get_accuracy(model_prune_only, val_loader, device)
    logger.info(f"  Pruning Only: {results['Pruning_Only']:.4f}")
    
    # 3. Quantization Only (no pruning)
    model_quant_only = copy.deepcopy(model)
    layer_names = [n for n, p in model_quant_only.named_parameters() 
                   if 'weight' in n and p.dim() > 1]
    profile = {n: 'INT8' for n in layer_names}
    apply_quantization_to_model(model_quant_only, profile)
    results['Quantization_Only'] = get_accuracy(model_quant_only, val_loader, device)
    logger.info(f"  Quantization Only (INT8): {results['Quantization_Only']:.4f}")
    
    # 4. Pruning without recovery training
    model_no_recovery = copy.deepcopy(model)
    from compression.pruning import apply_structured_pruning, make_pruning_permanent
    apply_structured_pruning(model_no_recovery, amount=0.15)
    make_pruning_permanent(model_no_recovery)
    results['Pruning_No_Recovery'] = get_accuracy(model_no_recovery, val_loader, device)
    logger.info(f"  Pruning (No Recovery): {results['Pruning_No_Recovery']:.4f}")
    
    # 5. Pruning without distillation
    model_no_distill = prune_and_recover(
        model, train_loader, val_loader, device,
        use_distillation=False
    )
    results['Pruning_No_Distillation'] = get_accuracy(model_no_distill, val_loader, device)
    logger.info(f"  Pruning (No Distillation): {results['Pruning_No_Distillation']:.4f}")
    
    # Calculate contributions
    logger.info("\n  Component Contributions:")
    logger.info(f"    Recovery Training: +{(results['Pruning_Only'] - results['Pruning_No_Recovery'])*100:.2f}%")
    logger.info(f"    Distillation: +{(results['Pruning_Only'] - results['Pruning_No_Distillation'])*100:.2f}%")
    
    logger.info("="*60 + "\n")
    
    # Cleanup
    del model_prune_only, model_quant_only, model_no_recovery, model_no_distill
    
    return results
