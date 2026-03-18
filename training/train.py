"""
Training Module
Supports: FP32 fine-tuning, QAT, Knowledge Distillation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import logging
import copy

logger = logging.getLogger(__name__)


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Get all predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        # Handle dict batches (HuggingFace)
        if isinstance(batch, dict):
            labels = batch.pop('label', batch.pop('labels', None))
            if labels is not None:
                labels = labels.to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
        else:
            # Handle tuple batches
            data, labels = batch
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
        
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        if labels is not None:
            all_labels.extend(labels.cpu().numpy())
    
    return all_labels, all_preds


@torch.no_grad()
def get_accuracy(model, dataloader, device):
    """Calculate accuracy."""
    from sklearn.metrics import accuracy_score
    
    y_true, y_pred = get_predictions(model, dataloader, device)
    if not y_true:
        return 0.0
    return accuracy_score(y_true, y_pred)


# ============================================================================
# DISTILLATION LOSS
# ============================================================================

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7):
    """
    Knowledge Distillation loss.
    
    Args:
        student_logits: Logits from student (compressed) model
        teacher_logits: Logits from teacher (full) model
        labels: Ground truth labels
        temperature: Softening temperature (higher = softer)
        alpha: Weight for distillation loss (1-alpha for hard labels)
    
    Returns:
        Combined loss
    """
    # Soft targets (KL divergence)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_predictions = F.log_softmax(student_logits / temperature, dim=1)
    
    distill_loss = F.kl_div(
        soft_predictions,
        soft_targets,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard labels (cross-entropy)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined
    return alpha * distill_loss + (1 - alpha) * hard_loss


# ============================================================================
# FP32 BASELINE TRAINING
# ============================================================================

def train_fp32(model_name, train_loader, val_loader, num_labels, device,
               epochs=3, lr=2e-5, warmup_ratio=0.1, use_amp=True,
               use_early_stopping=True, use_checkpointing=True,
               checkpoint_dir="checkpoints"):
    """
    Train FP32 baseline model.
    
    Args:
        model_name: HuggingFace model identifier
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_labels: Number of output labels
        device: torch.device
        epochs: Number of training epochs
        lr: Learning rate
        warmup_ratio: Warmup ratio for scheduler
        use_amp: Use automatic mixed precision
        use_early_stopping: Enable early stopping
        use_checkpointing: Save best checkpoint
    
    Returns:
        tuple: (trained_model, final_accuracy)
    """
    from utils.utils import EarlyStopping, CheckpointManager, AMPScaler
    
    logger.info(f"Training FP32 baseline: {model_name}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    ).to(device)
    
    # Handle missing pad_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    
    # Gradient checkpointing (memory optimization)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # AMP scaler
    scaler = AMPScaler(enabled=use_amp)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=3) if use_early_stopping else None
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        save_best_only=True
    ) if use_checkpointing else None
    
    # Training loop
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Prepare batch
            batch = {k: v.to(device) for k, v in batch.items()}
            if "label" in batch:
                batch["labels"] = batch.pop("label")
            
            # Forward pass with AMP
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Validation
        val_acc = get_accuracy(model, val_loader, device)
        avg_loss = total_loss / num_batches
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
        
        # Checkpointing
        if checkpoint_manager:
            checkpoint_manager.save(
                model, optimizer, epoch,
                {'accuracy': val_acc, 'loss': avg_loss},
                filename=f"fp32_{model_name.replace('/', '_')}_best.pt"
            )
        
        # Early stopping
        if early_stopping and early_stopping(val_acc):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    final_acc = get_accuracy(model, val_loader, device)
    
    logger.info(f"✅ FP32 training complete - Final accuracy: {final_acc:.4f}")
    
    return model, final_acc


# ============================================================================
# QUANTIZATION-AWARE TRAINING (QAT)
# ============================================================================

def train_qat(model, train_loader, val_loader, device, bits=8,
              epochs=2, lr=1e-5, warmup_ratio=0.1):
    """
    Quantization-Aware Training.
    
    Args:
        model: Pre-trained model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: torch.device
        bits: Quantization bits
        epochs: Number of QAT epochs
        lr: Learning rate (lower than initial training)
        warmup_ratio: Warmup ratio
    
    Returns:
        model: QAT-trained model
    """
    from compression.quantizer import prepare_model_for_qat
    
    logger.info(f"Starting Quantization-Aware Training (INT{bits})")
    
    # Prepare model for QAT (add fake quantization layers)
    model = prepare_model_for_qat(model, bits=bits)
    model.train()
    
    # Optimizer (lower LR for fine-tuning)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # QAT training loop
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            batch = {k: v.to(device) for k, v in batch.items()}
            if "label" in batch:
                batch["labels"] = batch.pop("label")
            
            # Forward with fake quantization
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        val_acc = get_accuracy(model, val_loader, device)
        avg_loss = total_loss / num_batches
        
        logger.info(f"QAT Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    model.eval()
    logger.info("✅ QAT complete")
    
    return model


# ============================================================================
# PRUNING WITH KNOWLEDGE DISTILLATION
# ============================================================================

def train_with_distillation(student_model, teacher_model, train_loader, val_loader, device,
                           epochs=3, lr=2e-5, temperature=3.0, alpha=0.7):
    """
    Train student model with knowledge distillation from teacher.
    
    Args:
        student_model: Compressed/pruned model (student)
        teacher_model: Full FP32 model (teacher)
        train_loader: Training dataloader
        val_loader: Validation dataloader
        device: torch.device
        epochs: Number of epochs
        lr: Learning rate
        temperature: Distillation temperature
        alpha: Distillation weight
    
    Returns:
        student_model: Trained student model
    """
    logger.info("Training with knowledge distillation")
    
    # Freeze teacher
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    # Student in training mode
    student_model.train()
    
    # Optimizer
    optimizer = AdamW(student_model.parameters(), lr=lr)
    
    # Scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            batch = {k: v.to(device) for k, v in batch.items()}
            if "label" in batch:
                labels = batch.pop("label")
            elif "labels" in batch:
                labels = batch["labels"]
            else:
                labels = None
            
            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                teacher_logits = teacher_outputs.logits
            
            # Student forward
            student_outputs = student_model(**batch)
            student_logits = student_outputs.logits
            
            # Distillation loss
            if labels is not None:
                loss = distillation_loss(
                    student_logits,
                    teacher_logits,
                    labels,
                    temperature=temperature,
                    alpha=alpha
                )
            else:
                loss = student_outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        val_acc = get_accuracy(student_model, val_loader, device)
        avg_loss = total_loss / num_batches
        
        logger.info(f"Distillation Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    student_model.eval()
    logger.info("✅ Knowledge distillation complete")
    
    return student_model
