import torch
import os

# ============================================================================
# DEVICE AND HARDWARE CONFIGURATION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_MULTI_GPU = torch.cuda.device_count() > 1
GPU_IDS = list(range(torch.cuda.device_count())) if USE_MULTI_GPU else [0]
USE_AMP = True  # Automatic Mixed Precision (FP16 training for speed)

# ============================================================================
# MODEL SELECTION - Edge-Optimized Architecture Categories
# ============================================================================

MODELS_TINY = [
    # Ultra-lightweight (<50M params) - Microcontroller/IoT
    "prajjwal1/bert-tiny",              # 4.4M params
    "google/mobilebert-uncased",        # 25M params - Mobile-optimized
    "microsoft/xtremedistil-l6-h256-uncased",  # 13M params
]

MODELS_SMALL = [
    # Small (50-150M params) - Mobile/Raspberry Pi
    "distilbert-base-uncased",          # 66M params
    "google/electra-small-discriminator",  # 14M params
    "albert-base-v2",                   # 12M params
    "squeezebert/squeezebert-uncased",  # 51M params
]

MODELS_MEDIUM = [
    # Medium (150-500M params) - Edge servers/Jetson
    "bert-base-uncased",                # 110M params
    "roberta-base",                     # 125M params
    "microsoft/deberta-v3-base",        # 86M params
]

MODELS_LARGE = [
    # Large (>500M params) - Scalability testing
    "bert-large-uncased",               # 335M params
    "roberta-large",                    # 355M params
]

# Default configuration
MODELS = MODELS_TINY + MODELS_SMALL + MODELS_MEDIUM

# Quick test override (uncomment for fast testing)
# MODELS = ["prajjwal1/bert-tiny", "distilbert-base-uncased"]

# ============================================================================
# TASK SELECTION - Comprehensive GLUE Coverage
# ============================================================================

TASKS_CLASSIFICATION = [
    "sst2",      # Sentiment (2-class, single) - EASY
    "qnli",      # Question NLI (2-class, pair) - MEDIUM  
    "mnli",      # Multi-NLI (3-class, pair) - HARD
]

TASKS_SIMILARITY = [
    "mrpc",      # Paraphrase (2-class, pair) - SMALL
    "qqp",       # Question pairs (2-class) - LARGE
    "stsb",      # Similarity (regression)
]

TASKS_INFERENCE = [
    "rte",       # Entailment (2-class) - SMALL, HARD
    "wnli",      # Winograd (2-class) - VERY SMALL
]

# Default: Balanced mix
TASKS = TASKS_CLASSIFICATION + ["mrpc", "rte"]

# Quick test override
# TASKS = ["sst2"]

# ============================================================================
# EXPERIMENTAL DESIGN
# ============================================================================

SEEDS = [42, 123, 456, 789, 2024]

# Quick test override
# SEEDS = [42]

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

# Batch size (will be auto-adjusted based on GPU memory)
BATCH_SIZE = 16
MAX_BATCH_SIZE = 64  # Upper limit for auto-adjustment
MIN_BATCH_SIZE = 4   # Lower limit

# Learning rates
LR = 2e-5  # Standard for BERT fine-tuning
LR_WARMUP_RATIO = 0.1  # 10% of steps for warmup

# Epochs
EPOCHS_FINETUNE = 3
EPOCHS_RECOVERY = 3
EPOCHS_QAT = 2  # Quantization-aware training

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 0.001

# Gradient clipping
GRADIENT_CLIP_NORM = 1.0

# ============================================================================
# COMPRESSION HYPERPARAMETERS
# ============================================================================

# Pruning
PRUNING_AMOUNT = 0.15  # 15% structured pruning
PRUNING_AMOUNTS_ABLATION = [0.10, 0.15, 0.20, 0.30]
USE_PROGRESSIVE_PRUNING = False  # Gradual pruning
PROGRESSIVE_PRUNING_STEPS = 3

# Quantization
SENSITIVITY_THRESHOLD = 0.01  # 1.5% accuracy drop tolerance

QUANTIZATION_BITS = {
    'ultra_low': 4,
    'low': 8,
    'high': 16,
}

# Quantization mode
USE_QAT = True  # Quantization-Aware Training (better accuracy)
USE_MIXED_GRANULARITY = True  # Per-head in attention, per-channel in FFN

# ============================================================================
# SEARCH ALGORITHM CONFIGURATION
# ============================================================================

SEARCH_METHOD = 'simulated_annealing'  # Options: 'greedy', 'simulated_annealing', 'nas'

# Simulated Annealing parameters
SA_MAX_ITERATIONS = 100
SA_INITIAL_TEMP = 1.0
SA_COOLING_RATE = 0.95
USE_SENSITIVITY_CACHE = True  # Speed up search

# NAS parameters (if using RL-based search)
NAS_EPISODES = 50
NAS_LR = 1e-3
NAS_EMBEDDING_DIM = 64

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

MAX_LENGTH = 128

# Task-specific max lengths (optional override)
MAX_LENGTH_PER_TASK = {
    'sst2': 64,
    'mrpc': 128,
    'qnli': 256,
    'mnli': 128,
    'qqp': 128,
    'rte': 256,
    'stsb': 128,
    'wnli': 128,
    'cola': 128,
}

# Data loading
NUM_WORKERS = 4  # Parallel data loading
PIN_MEMORY = True  # Faster GPU transfer

# ============================================================================
# MEMORY OPTIMIZATION
# ============================================================================

USE_GRADIENT_CHECKPOINTING = True  # Trade compute for memory
CLEAR_CACHE_EVERY_N_STEPS = 100

# ============================================================================
# LOGGING AND CHECKPOINTING
# ============================================================================

# Directories
CACHE_DIR = "data/cache"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
COMPRESSED_MODELS_DIR = "compressed_models"

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "research.log"
LOG_INTERVAL = 100  # Log every N batches

# Checkpointing
SAVE_CHECKPOINTS = True
SAVE_BEST_ONLY = True  # Only save best checkpoint per run
CHECKPOINT_METRIC = "accuracy"  # Metric to track for best model

# ============================================================================
# ABLATION STUDY SETTINGS
# ============================================================================

RUN_ABLATIONS = True

ABLATION_CONFIGS = {
    'pruning_only': {
        'pruning': True,
        'quantization': False,
        'distillation': False,
        'qat': False,
    },
    'quantization_only': {
        'pruning': False,
        'quantization': True,
        'distillation': False,
        'qat': False,
    },
    'no_distillation': {
        'pruning': True,
        'quantization': True,
        'distillation': False,
        'qat': False,
    },
    'no_qat': {
        'pruning': True,
        'quantization': True,
        'distillation': True,
        'qat': False,
    },
    'full_method': {
        'pruning': True,
        'quantization': True,
        'distillation': True,
        'qat': True,
    },
}

# ============================================================================
# BASELINE COMPARISONS
# ============================================================================

COMPARE_BASELINES = [
    'fp32',
    'uniform_int8',
    'uniform_int4',
    'mixed_16x8',
    'conservative_hybrid',
]

# ============================================================================
# HARDWARE BENCHMARKING (Optional)
# ============================================================================

ENABLE_HARDWARE_TESTING = False  # Set True if you have hardware
HARDWARE_DEVICES = {
    'raspberry_pi': {
        'ip': '192.168.1.100',
        'enabled': False,
    },
    'jetson_nano': {
        'ip': '192.168.1.101',
        'enabled': False,
    },
}

# ============================================================================
# EXPORT SETTINGS
# ============================================================================

EXPORT_TO_ONNX = True
EXPORT_TO_TFLITE = False  # Requires TensorFlow
ONNX_OPSET_VERSION = 14

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

DETERMINISTIC = True
CUDNN_BENCHMARK = not DETERMINISTIC  # Faster but non-deterministic

# ============================================================================
# OUTPUT FILES
# ============================================================================

RAW_RESULTS_FILE = "raw_seed_results.csv"
SUMMARY_RESULTS_FILE = "final_research_results.csv"
ABLATION_RESULTS_FILE = "ablation_study_results.csv"
BASELINE_RESULTS_FILE = "baseline_comparison.csv"

# ============================================================================
# VISUALIZATION
# ============================================================================

PLOT_DPI = 300
PLOT_FORMAT = ['png', 'pdf']  # Save in both formats
GENERATE_LATEX_TABLES = True

# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

# Find the line with CONFIG_PRESET and change it to:
def get_quick_test_config():
    """Ultra-fast config for testing (5-10 min)."""
    return {
        'MODELS': ["distilbert-base-uncased"],  # Changed from bert-tiny
        'TASKS': ["sst2"],
        'SEEDS': [42, 123],
        'EPOCHS_FINETUNE': 1,
        'EPOCHS_RECOVERY': 1,
        'EPOCHS_QAT': 1,
        'SA_MAX_ITERATIONS': 100,
        'RUN_ABLATIONS': False,
        'USE_QAT': True,
        'EXPORT_TO_ONNX': False,
    }

def get_development_config():
    """Medium config for development (1-2 hours)."""
    return {
        'MODELS': ["prajjwal1/bert-tiny", "distilbert-base-uncased"],
        'TASKS': ["sst2", "qnli"],
        'SEEDS': [42, 123],
        'EPOCHS_FINETUNE': 2,
        'EPOCHS_RECOVERY': 2,
        'EPOCHS_QAT': 1,
        'SA_MAX_ITERATIONS': 50,
        'RUN_ABLATIONS': True,
        'USE_QAT': True,
    }

def get_paper_config():
    """Full config for paper (24-48 hours)."""
    return {
        'MODELS': MODELS_TINY + MODELS_SMALL + MODELS_MEDIUM,
        'TASKS': TASKS_CLASSIFICATION + TASKS_SIMILARITY + ["rte"],
        'SEEDS': [42, 123, 456, 789, 2024, 1111, 2222, 3333, 4444, 5555],
        'EPOCHS_FINETUNE': 3,
        'EPOCHS_RECOVERY': 3,
        'EPOCHS_QAT': 2,
        'SA_MAX_ITERATIONS': 100,
        'RUN_ABLATIONS': True,
        'USE_QAT': True,
        'EXPORT_TO_ONNX': True,
    }

# ============================================================================
# APPLY PRESET (Uncomment ONE to override defaults)
# ============================================================================

# Quick test (5-10 minutes):
CONFIG_PRESET = get_quick_test_config()

# Development (1-2 hours):
# CONFIG_PRESET = get_development_config()

# Full paper (24-48 hours):
# CONFIG_PRESET = get_paper_config()

# Apply preset if defined
if 'CONFIG_PRESET' in locals():
    for key, value in CONFIG_PRESET.items():
        globals()[key] = value
    print(f"✅ Applied preset configuration")

# ============================================================================
# AUTO-CREATE DIRECTORIES
# ============================================================================

for directory in [CACHE_DIR, CHECKPOINT_DIR, RESULTS_DIR, PLOTS_DIR, COMPRESSED_MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)


