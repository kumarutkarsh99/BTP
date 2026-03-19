from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import pickle
import hashlib
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# TASK CONFIGURATIONS
# ============================================================================

def get_task_config(task):
    """
    Get configuration for GLUE tasks.
    
    Returns:
        tuple: (num_labels, keys, split, metric, is_regression)
    """
    configs = {
        "mnli": (3, ("premise", "hypothesis"), "validation_matched", "accuracy", False),
        "qnli": (2, ("question", "sentence"), "validation", "accuracy", False),
        "sst2": (2, ("sentence", None), "validation", "accuracy", False),
        "mrpc": (2, ("sentence1", "sentence2"), "validation", "f1", False),
        "qqp": (2, ("question1", "question2"), "validation", "f1", False),
        "rte": (2, ("sentence1", "sentence2"), "validation", "accuracy", False),
        "wnli": (2, ("sentence1", "sentence2"), "validation", "accuracy", False),
        "cola": (2, ("sentence", None), "validation", "matthews_corrcoef", False),
        "stsb": (1, ("sentence1", "sentence2"), "validation", "pearson", True),
    }
    
    if task not in configs:
        raise ValueError(f"Unsupported task: {task}. Supported: {list(configs.keys())}")
    
    return configs[task]


# ============================================================================
# CACHING UTILITIES
# ============================================================================

def get_cache_path(model_name, task, max_length, cache_dir="data/cache"):
    """Generate unique cache path with hash."""
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_key = f"{model_name}_{task}_{max_length}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
    
    clean_model = model_name.replace('/', '_').replace('-', '_')
    filename = f"{clean_model}_{task}_{cache_hash}.pkl"
    
    return os.path.join(cache_dir, filename)


# ============================================================================
# ADAPTIVE TRAINING PARAMETERS
# ============================================================================

def get_adaptive_epochs(train_size, base_epochs=3):
    """
    Adjust epochs based on dataset size.
    Small datasets need more epochs to converge.
    """
    if train_size < 5000:  # Very small (WNLI, etc.)
        return min(base_epochs * 4, 20)
    elif train_size < 10000:  # Small (RTE, MRPC, etc.)
        return min(base_epochs * 2, 10)
    else:  # Normal/Large
        return base_epochs


def estimate_training_time(train_size, batch_size, epochs, seconds_per_batch=0.5):
    """Estimate total training time."""
    num_batches = (train_size + batch_size - 1) // batch_size
    total_batches = num_batches * epochs
    estimated_seconds = total_batches * seconds_per_batch
    
    return estimated_seconds


# ============================================================================
# MAIN DATA PREPARATION FUNCTION
# ============================================================================

def prepare_data(model_name, task, batch_size=16, max_length=128, 
                num_workers=4, pin_memory=True, use_cache=True):
    """
    Prepare dataloaders for a model and task.
    
    Args:
        model_name: HuggingFace model identifier
        task: GLUE task name
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        num_workers: Number of workers for data loading
        pin_memory: Pin memory for faster GPU transfer
        use_cache: Whether to use cached preprocessed data
    
    Returns:
        tuple: (train_loader, val_loader, num_labels, task_info)
    """
    
    # Get task configuration
    num_labels, keys, split, metric, is_regression = get_task_config(task)
    
    logger.info(f"Preparing data for {task.upper()} with {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Preprocessing function
    def preprocess(batch):
        if keys[1] is None:
            # Single sentence task
            return tokenizer(
                batch[keys[0]],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
        else:
            # Sentence pair task
            return tokenizer(
                batch[keys[0]],
                batch[keys[1]],
                truncation=True,
                padding="max_length",
                max_length=max_length
            )
    
    # Check cache
    cache_path = get_cache_path(model_name, task, max_length)
    dataset = None
    
    if use_cache and os.path.exists(cache_path):
        try:
            logger.info(f"Loading cached dataset from {cache_path}")
            with open(cache_path, "rb") as f:
                dataset = pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache load failed: {e}. Reprocessing...")
            dataset = None
    
    # Load and preprocess if not cached
    if dataset is None:
        logger.info(f"Downloading and preprocessing {task.upper()} dataset...")
        
        dataset = load_dataset("glue", task)
        dataset = dataset.map(preprocess, batched=True)
        
        # Save cache
        if use_cache:
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(dataset, f)
                logger.info(f"Dataset cached to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    # Set format for PyTorch
    dataset.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    
    # Create dataloaders with improved settings
    train_loader = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,  # Keep all data
        persistent_workers=num_workers > 0,  # Faster with persistent workers
    )
    
    val_loader = DataLoader(
        dataset[split],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    
    # Task metadata
    train_size = len(dataset["train"])
    val_size = len(dataset[split])
    
    task_info = {
        'num_labels': num_labels,
        'metric': metric,
        'is_regression': is_regression,
        'train_size': train_size,
        'val_size': val_size,
        'is_small_dataset': train_size < 10000,
        'adaptive_epochs': get_adaptive_epochs(train_size),
        'estimated_time_min': estimate_training_time(train_size, batch_size, 3) / 60,
    }
    
    logger.info(f"✅ Dataset ready - Train: {train_size:,}, Val: {val_size:,}, "
                f"Adaptive epochs: {task_info['adaptive_epochs']}")
    
    return train_loader, val_loader, num_labels, task_info


# ============================================================================
# DATASET STATISTICS
# ============================================================================

GLUE_DATASET_INFO = {
    'mnli': {'size': 392_702, 'difficulty': 'hard', 'type': 'nli'},
    'qqp': {'size': 363_846, 'difficulty': 'medium', 'type': 'similarity'},
    'qnli': {'size': 104_743, 'difficulty': 'medium', 'type': 'nli'},
    'sst2': {'size': 67_349, 'difficulty': 'easy', 'type': 'sentiment'},
    'stsb': {'size': 5_749, 'difficulty': 'medium', 'type': 'regression'},
    'mrpc': {'size': 3_668, 'difficulty': 'hard', 'type': 'similarity'},
    'rte': {'size': 2_490, 'difficulty': 'hard', 'type': 'nli'},
    'cola': {'size': 8_551, 'difficulty': 'hard', 'type': 'acceptability'},
    'wnli': {'size': 635, 'difficulty': 'very_hard', 'type': 'nli'},
}


def clear_cache(cache_dir="data/cache"):
    """Clear all cached datasets."""
    import shutil
    
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info(f"Cache cleared: {cache_dir}")
    else:
        logger.info("No cache to clear")
