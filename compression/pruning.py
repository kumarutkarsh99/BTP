"""
Advanced Pruning Module
- Structured L2-norm pruning
- Progressive pruning
- Knowledge distillation recovery
- QAT-aware pruning
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# STRUCTURED PRUNING
# ============================================================================

def apply_structured_pruning(model, amount=0.15, method='l2'):
    """
    Apply structured pruning to linear layers.
    
    Args:
        model: PyTorch model
        amount: Fraction of neurons to prune (0.15 = 15%)
        method: 'l2', 'l1', or 'random'
    
    Returns:
        None (modifies model in-place)
    """
    logger.info(f"Applying {amount*100:.1f}% structured pruning ({method})")
    
    parameters_to_prune = []
    
    # Collect all linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply structured pruning
    n = 2 if method == 'l2' else 1  # L2 or L1 norm
    
    for module, param_name in parameters_to_prune:
        prune.ln_structured(
            module,
            name=param_name,
            amount=amount,
            n=n,
            dim=0  # Prune output channels (rows)
        )
    
    logger.info(f"✓ Pruned {len(parameters_to_prune)} layers")


def make_pruning_permanent(model):
    """
    Remove pruning masks and make zeros permanent.
    
    Args:
        model: Pruned model with masks
    
    Returns:
        None (modifies model in-place)
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    
    logger.info("✓ Pruning masks removed (zeros permanent)")


def get_sparsity(model):
    """
    Calculate sparsity percentage.
    
    Args:
        model: PyTorch model
    
    Returns:
        float: Sparsity percentage
    """
    total = 0
    zeros = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            total += param.numel()
            zeros += (param == 0).sum().item()
    
    return (zeros / total * 100) if total > 0 else 0


# ============================================================================
# PROGRESSIVE PRUNING
# ============================================================================

def progressive_pruning(model, train_loader, val_loader, device,
                       target_sparsity=0.30, steps=3, recovery_epochs=2):
    """
    Gradually increase pruning in steps with recovery training.
    
    Args:
        model: Model to prune
        train_loader: Training data
        val_loader: Validation data
        device: Device
        target_sparsity: Final sparsity (0.30 = 30%)
        steps: Number of progressive steps
        recovery_epochs: Epochs per step
    
    Returns:
        Pruned model
    """
    from training.train import train_with_distillation, get_accuracy
    
    logger.info(f"Progressive pruning: {steps} steps to {target_sparsity*100}% sparsity")
    
    teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    
    current_sparsity = 0
    sparsity_increment = target_sparsity / steps
    
    for step in range(steps):
        current_sparsity += sparsity_increment
        
        logger.info(f"\nStep {step+1}/{steps}: Pruning to {current_sparsity*100:.1f}%")
        
        # Apply pruning
        apply_structured_pruning(model, amount=current_sparsity)
        
        # Recovery training with distillation
        model = train_with_distillation(
            model, teacher_model,
            train_loader, val_loader, device,
            epochs=recovery_epochs
        )
        
        # Evaluate
        acc = get_accuracy(model, val_loader, device)
        sparsity = get_sparsity(model)
        
        logger.info(f"Step {step+1} complete: Sparsity={sparsity:.1f}%, Acc={acc:.4f}")
    
    make_pruning_permanent(model)
    
    return model


# ============================================================================
# PRUNING WITH DISTILLATION
# ============================================================================

def prune_and_recover(model, train_loader, val_loader, device,
                     pruning_amount=0.15, use_distillation=True,
                     use_progressive=False, recovery_epochs=3,
                     temperature=3.0, alpha=0.7):
    """
    Complete pruning pipeline with recovery.
    
    Args:
        model: Model to prune
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: Device
        pruning_amount: Amount to prune (0.15 = 15%)
        use_distillation: Use knowledge distillation for recovery
        use_progressive: Use progressive pruning
        recovery_epochs: Epochs for recovery training
        temperature: Distillation temperature
        alpha: Distillation weight
    
    Returns:
        Pruned and recovered model
    """
    from training.train import train_with_distillation, get_accuracy
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PRUNING PIPELINE")
    logger.info(f"{'='*60}")
    
    # Progressive pruning
    if use_progressive:
        model_pruned = progressive_pruning(
            model, train_loader, val_loader, device,
            target_sparsity=pruning_amount,
            steps=3,
            recovery_epochs=2
        )
        return model_pruned
    
    # Standard pruning
    teacher_model = copy.deepcopy(model) if use_distillation else None
    model_pruned = copy.deepcopy(model)
    
    # Apply pruning
    apply_structured_pruning(model_pruned, amount=pruning_amount)
    
    # Recovery training
    if use_distillation and teacher_model:
        logger.info("Recovery with knowledge distillation")
        model_pruned = train_with_distillation(
            model_pruned, teacher_model,
            train_loader, val_loader, device,
            epochs=recovery_epochs,
            temperature=temperature,
            alpha=alpha
        )
    else:
        logger.info("Recovery with standard fine-tuning")
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
        
        optimizer = AdamW(model_pruned.parameters(), lr=2e-5)
        total_steps = len(train_loader) * recovery_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)
        
        model_pruned.train()
        for epoch in range(recovery_epochs):
            for batch in train_loader:
                optimizer.zero_grad()
                batch = {k: v.to(device) for k, v in batch.items()}
                if "label" in batch:
                    batch["labels"] = batch.pop("label")
                
                outputs = model_pruned(**batch)
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model_pruned.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
    
    # Make permanent
    make_pruning_permanent(model_pruned)
    
    # Final evaluation
    final_acc = get_accuracy(model_pruned, val_loader, device)
    final_sparsity = get_sparsity(model_pruned)
    
    logger.info(f"\n✅ Pruning complete:")
    logger.info(f"   Sparsity: {final_sparsity:.1f}%")
    logger.info(f"   Accuracy: {final_acc:.4f}")
    
    return model_pruned


# ============================================================================
# SENSITIVITY-AWARE PRUNING (Advanced)
# ============================================================================

def sensitivity_aware_pruning(model, val_loader, device, target_sparsity=0.30):
    """
    Prune layers based on sensitivity analysis.
    Sensitive layers pruned less, insensitive layers pruned more.
    
    Args:
        model: Model to prune
        val_loader: Validation data
        device: Device
        target_sparsity: Target overall sparsity
    
    Returns:
        Pruned model
    """
    from training.train import get_accuracy
    
    logger.info("Sensitivity-aware pruning")
    
    # Get baseline accuracy
    baseline_acc = get_accuracy(model, val_loader, device)
    
    # Analyze sensitivity per layer
    layer_sensitivity = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Save original weight
            original_weight = module.weight.data.clone()
            
            # Apply aggressive pruning to this layer only
            prune.ln_structured(module, 'weight', amount=0.50, n=2, dim=0)
            
            # Measure accuracy drop
            acc = get_accuracy(model, val_loader, device)
            sensitivity = baseline_acc - acc
            
            layer_sensitivity[name] = sensitivity
            
            # Restore weight
            module.weight.data = original_weight
            if hasattr(module, 'weight_mask'):
                delattr(module, 'weight_mask')
    
    # Sort layers by sensitivity (least sensitive first)
    sorted_layers = sorted(layer_sensitivity.items(), key=lambda x: x[1])
    
    # Assign pruning amounts (insensitive = more pruning)
    total_params = sum(module.weight.numel() 
                      for name, module in model.named_modules() 
                      if isinstance(module, nn.Linear))
    
    params_to_prune = int(total_params * target_sparsity)
    
    pruning_plan = {}
    pruned_so_far = 0
    
    for name, sensitivity in sorted_layers:
        module = dict(model.named_modules())[name]
        layer_params = module.weight.numel()
        
        # Less sensitive → more aggressive pruning
        if sensitivity < 0.01:  # Very insensitive
            amount = min(0.50, (params_to_prune - pruned_so_far) / layer_params)
        elif sensitivity < 0.05:  # Moderately insensitive
            amount = min(0.30, (params_to_prune - pruned_so_far) / layer_params)
        else:  # Sensitive
            amount = min(0.10, (params_to_prune - pruned_so_far) / layer_params)
        
        pruning_plan[name] = amount
        pruned_so_far += int(layer_params * amount)
        
        if pruned_so_far >= params_to_prune:
            break
    
    # Apply pruning plan
    for name, module in model.named_modules():
        if name in pruning_plan:
            amount = pruning_plan[name]
            prune.ln_structured(module, 'weight', amount=amount, n=2, dim=0)
            logger.info(f"  {name}: {amount*100:.1f}% pruned (sensitivity={layer_sensitivity[name]:.4f})")
    
    return model
