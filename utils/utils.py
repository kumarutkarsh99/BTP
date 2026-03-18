import torch
import torch.nn as nn
import numpy as np
import random
import os
import logging
from pathlib import Path


# ============================================================================
# SEED AND REPRODUCIBILITY
# ============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_file="research.log", level="INFO"):
    """Setup logging configuration."""
    
    # Create logs directory
    log_dir = Path(log_file).parent
    if log_dir != Path('.'):
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# ============================================================================
# DEVICE AND MEMORY UTILITIES
# ============================================================================

def get_optimal_batch_size(model, device, default_batch_size=16):
    """
    Automatically determine optimal batch size based on GPU memory.
    
    Args:
        model: PyTorch model
        device: torch.device
        default_batch_size: Fallback batch size
    
    Returns:
        Optimal batch size (int)
    """
    if device.type == 'cpu':
        return min(default_batch_size, 8)  # Conservative for CPU
    
    try:
        # Get model size
        param_count = sum(p.numel() for p in model.parameters())
        
        # Get available GPU memory
        if torch.cuda.is_available():
            device_idx = device.index if device.index is not None else 0
            props = torch.cuda.get_device_properties(device_idx)
            total_memory = props.total_memory
            allocated = torch.cuda.memory_allocated(device_idx)
            free_memory = total_memory - allocated
            free_memory_mb = free_memory / 1024**2
        else:
            free_memory_mb = 4000
        
        # Heuristic: batch_size based on params and memory
        if param_count < 20e6:  # <20M params
            optimal = min(64, int(free_memory_mb / 100))
        elif param_count < 100e6:  # 20-100M params
            optimal = min(32, int(free_memory_mb / 200))
        else:  # >100M params
            optimal = min(16, int(free_memory_mb / 400))
        
        # Ensure within bounds
        optimal = max(4, min(optimal, 64))
        
        logging.info(f"Auto-selected batch size: {optimal} (Free GPU mem: {free_memory_mb:.0f}MB)")
        return optimal
        
    except Exception as e:
        logging.warning(f"Could not auto-determine batch size: {e}. Using default: {default_batch_size}")
        return default_batch_size


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=3, min_delta=0.001, mode='max'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, current_value):
        """
        Check if should stop.
        
        Returns:
            bool: True if should stop training
        """
        if self.best_value is None:
            self.best_value = current_value
            return False
        
        if self.mode == 'max':
            improved = current_value > self.best_value + self.min_delta
        else:
            improved = current_value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
        
        return False


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    """Manage model checkpoints during training."""
    
    def __init__(self, checkpoint_dir, save_best_only=True, metric='accuracy', mode='max'):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_best_only: Only save when metric improves
            metric: Metric name to track
            mode: 'max' to maximize metric, 'min' to minimize
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.metric = metric
        self.mode = mode
        self.best_value = None
    
    def save(self, model, optimizer, epoch, metrics, filename=None):
        """
        Save checkpoint if it's the best or if save_best_only is False.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dict of metrics
            filename: Optional custom filename
        
        Returns:
            bool: True if saved
        """
        current_value = metrics.get(self.metric)
        
        if current_value is None:
            logging.warning(f"Metric '{self.metric}' not found in metrics. Saving anyway.")
            should_save = True
        elif self.save_best_only:
            if self.best_value is None:
                should_save = True
            elif self.mode == 'max':
                should_save = current_value > self.best_value
            else:
                should_save = current_value < self.best_value
        else:
            should_save = True
        
        if should_save:
            if self.best_value is None or \
               (self.mode == 'max' and current_value > self.best_value) or \
               (self.mode == 'min' and current_value < self.best_value):
                self.best_value = current_value
            
            if filename is None:
                filename = f"checkpoint_epoch_{epoch}.pt"
            
            checkpoint_path = self.checkpoint_dir / filename
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics,
                'best_value': self.best_value,
            }, checkpoint_path)
            
            logging.info(f"Saved checkpoint: {checkpoint_path} ({self.metric}={current_value:.4f})")
            return True
        
        return False
    
    def load_best(self, model, optimizer=None):
        """
        Load the best checkpoint.
        
        Args:
            model: PyTorch model to load into
            optimizer: Optional optimizer to restore
        
        Returns:
            dict: Metrics from best checkpoint
        """
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        if not checkpoint_files:
            logging.warning("No checkpoints found")
            return None
        
        # Find checkpoint with best metric
        best_checkpoint = None
        best_value = None
        
        for checkpoint_file in checkpoint_files:
            checkpoint = torch.load(checkpoint_file)
            value = checkpoint['metrics'].get(self.metric)
            
            if value is not None:
                if best_value is None or \
                   (self.mode == 'max' and value > best_value) or \
                   (self.mode == 'min' and value < best_value):
                    best_value = value
                    best_checkpoint = checkpoint_file
        
        if best_checkpoint is None:
            logging.warning("No valid checkpoints found")
            return None
        
        # Load best checkpoint
        checkpoint = torch.load(best_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logging.info(f"Loaded best checkpoint: {best_checkpoint} ({self.metric}={best_value:.4f})")
        
        return checkpoint['metrics']


# ============================================================================
# MIXED PRECISION TRAINING
# ============================================================================

class AMPScaler:
    """Wrapper for automatic mixed precision training."""
    
    def __init__(self, enabled=True):
        self.enabled = enabled and torch.cuda.is_available()
        
        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def scale(self, loss):
        """Scale loss for mixed precision."""
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        """Optimizer step with gradient scaling."""
        if self.enabled:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def unscale_(self, optimizer):
        """Unscale gradients before clipping."""
        if self.enabled:
            self.scaler.unscale_(optimizer)


# ============================================================================
# LAYER FEATURE EXTRACTION (for NAS)
# ============================================================================

def extract_layer_features(model, layer_names):
    """
    Extract features for each layer for NAS-based search.
    
    Features:
        - Layer depth (position in network)
        - Input/output dimensions
        - Number of parameters
        - Layer type encoding
        - Normalized position
    
    Args:
        model: PyTorch model
        layer_names: List of layer names
    
    Returns:
        torch.Tensor: [num_layers, num_features]
    """
    features = []
    total_layers = len(layer_names)
    
    layer_dict = dict(model.named_modules())
    
    for i, layer_name in enumerate(layer_names):
        # Get the layer
        parts = layer_name.split('.')
        module = model
        for part in parts[:-1]:  # Navigate to parent
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        
        # Extract features
        layer_features = []
        
        # 1. Normalized position in network
        layer_features.append(i / total_layers)
        
        # 2. Layer depth (number of dots in name)
        layer_features.append(layer_name.count('.') / 10.0)
        
        # 3. Parameter count (normalized)
        param = None
        for name, p in model.named_parameters():
            if name == layer_name:
                param = p
                break
        
        if param is not None:
            num_params = param.numel()
            layer_features.append(np.log10(num_params + 1) / 10.0)
            
            # 4. Input/output dimensions
            if param.dim() >= 2:
                layer_features.append(param.shape[0] / 1000.0)  # Output dim
                layer_features.append(param.shape[1] / 1000.0)  # Input dim
            else:
                layer_features.append(0.0)
                layer_features.append(0.0)
        else:
            layer_features.extend([0.0, 0.0, 0.0])
        
        # 5. Layer type encoding
        if 'attention' in layer_name:
            layer_features.append(1.0)
        elif 'ffn' in layer_name or 'intermediate' in layer_name:
            layer_features.append(0.5)
        else:
            layer_features.append(0.0)
        
        # 6. Is first/last layer
        layer_features.append(1.0 if i == 0 else 0.0)
        layer_features.append(1.0 if i == total_layers - 1 else 0.0)
        
        # Pad to fixed size (10 features)
        while len(layer_features) < 10:
            layer_features.append(0.0)
        
        features.append(layer_features[:10])
    
    return torch.tensor(features, dtype=torch.float32)


# ============================================================================
# MODEL SIZE ESTIMATION
# ============================================================================

def estimate_compressed_size(model, profile):
    """
    Estimate model size after compression.
    
    Args:
        model: PyTorch model
        profile: Dict mapping layer names to precision ('INT4', 'INT8', etc.)
    
    Returns:
        float: Estimated size in MB
    """
    bits_per_precision = {
        'FP32': 32,
        'FP16': 16,
        'INT16': 16,
        'INT8': 8,
        'INT4': 4,
    }
    
    total_bits = 0
    
    for name, param in model.named_parameters():
        num_elements = param.numel()
        
        # Get precision for this layer
        precision = profile.get(name, 'FP32')
        bits = bits_per_precision.get(precision, 32)
        
        total_bits += num_elements * bits
    
    # Convert to MB
    size_mb = total_bits / (8 * 1024 * 1024)
    
    return size_mb


# ============================================================================
# PROGRESS TRACKING
# ============================================================================

class ProgressTracker:
    """Track and display progress with ETA."""
    
    def __init__(self, total_steps, description="Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = None
    
    def start(self):
        """Start tracking."""
        import time
        self.start_time = time.time()
        self.current_step = 0
    
    def update(self, step=None):
        """Update progress."""
        import time
        
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        if self.start_time is None:
            return
        
        elapsed = time.time() - self.start_time
        
        if self.current_step > 0:
            rate = elapsed / self.current_step
            eta = rate * (self.total_steps - self.current_step)
            
            percent = 100 * self.current_step / self.total_steps
            
            print(f"\r{self.description}: {self.current_step}/{self.total_steps} "
                  f"({percent:.1f}%) | ETA: {eta:.0f}s", end='', flush=True)
    
    def finish(self):
        """Finish tracking."""
        print()  # New line


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
    }


def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return None
    
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    
    return {
        'allocated_mb': allocated,
        'reserved_mb': reserved,
    }


def format_time(seconds):
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def create_results_summary(results_list):
    """Create summary statistics from list of results."""
    import pandas as pd
    
    df = pd.DataFrame(results_list)
    
    summary = {}
    for col in df.columns:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            summary[f"{col}_mean"] = df[col].mean()
            summary[f"{col}_std"] = df[col].std()
            summary[f"{col}_min"] = df[col].min()
            summary[f"{col}_max"] = df[col].max()
    
    return summary
