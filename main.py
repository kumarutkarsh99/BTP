"""
ULTIMATE POST-TRAINING QUANTIZATION PIPELINE
Complete implementation with all advanced features
"""

import torch
import pandas as pd
import numpy as np
import logging
import traceback
import time
from pathlib import Path
from scipy.stats import ttest_rel

# Configuration
from config.config import *

# Data loading
from data.data_loader import prepare_data

# Training
from training.train import (
    train_fp32, train_qat, train_with_distillation, get_accuracy
)

# Compression
from compression.pruning import prune_and_recover
from compression.quantizer import apply_quantization_to_model

# Search
from search.hybrid_search import run_hybrid_search

# Evaluation
from evaluation.baselines import evaluate_all_baselines, evaluate_ablations

# Utils
from utils.utils import (
    set_seed, setup_logging, get_optimal_batch_size,
    clear_gpu_cache, get_gpu_memory_usage, count_parameters
)


# ============================================================================
# SETUP
# ============================================================================

logger = setup_logging(LOG_FILE, LOG_LEVEL)

logger.info("="*80)
logger.info(" ULTIMATE POST-TRAINING QUANTIZATION PIPELINE ".center(80))
logger.info("="*80)
logger.info(f"Device: {DEVICE}")
logger.info(f"Models: {len(MODELS)}")
logger.info(f"Tasks: {len(TASKS)}")
logger.info(f"Seeds: {len(SEEDS)}")
logger.info(f"Search Method: {SEARCH_METHOD}")
logger.info(f"Use QAT: {USE_QAT}")
logger.info(f"Use Progressive Pruning: {USE_PROGRESSIVE_PRUNING}")
logger.info("="*80 + "\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_compression_metrics(fp32_size, compressed_size, fp32_acc, compressed_acc):
    """Calculate standard compression metrics."""
    return {
        'compression_ratio': fp32_size / compressed_size if compressed_size > 0 else 0,
        'size_reduction_pct': (1 - compressed_size / fp32_size) * 100 if fp32_size > 0 else 0,
        'accuracy_drop': fp32_acc - compressed_acc,
        'accuracy_retention_pct': (compressed_acc / fp32_acc) * 100 if fp32_acc > 0 else 0,
    }


def estimate_model_size(profile):
    """Estimate compressed model size."""
    bits_map = {'FP32': 32, 'INT16': 16, 'INT8': 8, 'INT6': 6, 'INT4': 4}
    
    total_params = sum(1 for _ in profile.keys())
    if total_params == 0:
        return 0
    
    total_bits = sum(bits_map.get(v, 32) for v in profile.values())
    avg_bits = total_bits / total_params
    
    # Rough estimate (assumes uniform param distribution)
    return avg_bits / 8  # Convert to bytes per parameter


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_single_experiment(model_name, task, seed, all_results, raw_results):
    """Run single experiment (one model + task + seed)."""
    
    try:
        logger.info("\n" + "="*80)
        logger.info(f"  {model_name} | {task} | Seed {seed}  ".center(80))
        logger.info("="*80 + "\n")
        
        set_seed(seed)
        
        start_time = time.time()
        
        # ----------------------------------------------------------------
        # DATA PREPARATION
        # ----------------------------------------------------------------
        
        logger.info("📦 Preparing data...")
        train_loader, val_loader, num_labels, task_info = prepare_data(
            model_name, task,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        
        # Adaptive batch size
        if hasattr(train_loader.dataset, '__len__'):
            optimal_batch = get_optimal_batch_size(None, DEVICE, BATCH_SIZE)
            if optimal_batch != BATCH_SIZE:
                logger.info(f"  Adjusting batch size: {BATCH_SIZE} → {optimal_batch}")
        
        # ----------------------------------------------------------------
        # FP32 BASELINE TRAINING
        # ----------------------------------------------------------------
        
        logger.info("\n🎯 Training FP32 baseline...")
        model_fp32, fp32_acc = train_fp32(
            model_name, train_loader, val_loader, num_labels, DEVICE,
            epochs=EPOCHS_FINETUNE,
            lr=LR,
            warmup_ratio=LR_WARMUP_RATIO,
            use_amp=USE_AMP,
            use_early_stopping=USE_EARLY_STOPPING,
            use_checkpointing=SAVE_CHECKPOINTS,
            checkpoint_dir=CHECKPOINT_DIR
        )
        
        # Get model stats
        param_stats = count_parameters(model_fp32)
        fp32_size_mb = param_stats['total'] * 4 / (1024**2)
        
        logger.info(f"✅ FP32: Acc={fp32_acc:.4f}, Size={fp32_size_mb:.2f}MB, Params={param_stats['total']:,}")
        
        # ----------------------------------------------------------------
        # BASELINE EVALUATIONS
        # ----------------------------------------------------------------
        
        logger.info("\n📊 Evaluating baselines...")
        baseline_results = evaluate_all_baselines(
            model_fp32, val_loader, DEVICE,
            include_per_tensor=True
        )
        
        # ----------------------------------------------------------------
        # ABLATION STUDIES (once per model-task)
        # ----------------------------------------------------------------
        
        ablation_results = {}
        if RUN_ABLATIONS and seed == SEEDS[0]:
            logger.info("\n🔬 Running ablation studies...")
            ablation_results = evaluate_ablations(
                model_fp32, train_loader, val_loader, DEVICE
            )
        
        # ----------------------------------------------------------------
        # PRUNING + DISTILLATION
        # ----------------------------------------------------------------
        
        logger.info("\n✂️  Pruning with distillation recovery...")
        model_pruned = prune_and_recover(
            model_fp32, train_loader, val_loader, DEVICE,
            pruning_amount=PRUNING_AMOUNT,
            use_distillation=True,
            use_progressive=USE_PROGRESSIVE_PRUNING,
            recovery_epochs=EPOCHS_RECOVERY
        )
        
        prune_acc = get_accuracy(model_pruned, val_loader, DEVICE)
        
        from compression.pruning import get_sparsity
        sparsity = get_sparsity(model_pruned)
        
        logger.info(f"✅ Pruned: Acc={prune_acc:.4f}, Sparsity={sparsity:.1f}%")
        
        # ----------------------------------------------------------------
        # QUANTIZATION-AWARE TRAINING (Optional)
        # ----------------------------------------------------------------
        
        if USE_QAT:
            logger.info("\n🎓 Quantization-Aware Training...")
            model_qat = train_qat(
                model_pruned, train_loader, val_loader, DEVICE,
                bits=8, epochs=EPOCHS_QAT
            )
            qat_acc = get_accuracy(model_qat, val_loader, DEVICE)
            logger.info(f"✅ QAT: Acc={qat_acc:.4f}")
            model_for_search = model_qat
        else:
            qat_acc = None
            model_for_search = model_pruned
        
        # ----------------------------------------------------------------
        # HYBRID MIXED-PRECISION SEARCH
        # ----------------------------------------------------------------
        
        logger.info(f"\n🔍 Hybrid search ({SEARCH_METHOD})...")
        
        search_kwargs = {
            'sensitivity_threshold': SENSITIVITY_THRESHOLD,
        }
        
        if SEARCH_METHOD == 'simulated_annealing':
            search_kwargs.update({
                'max_iterations': SA_MAX_ITERATIONS,
                'initial_temp': SA_INITIAL_TEMP,
                'cooling_rate': SA_COOLING_RATE,
                'use_cache': USE_SENSITIVITY_CACHE,
            })
        elif SEARCH_METHOD == 'nas':
            search_kwargs.update({
                'train_loader': train_loader,
                'episodes': NAS_EPISODES,
                'lr': NAS_LR,
            })
        
        hybrid_acc, hybrid_profile = run_hybrid_search(
            model_for_search, val_loader, DEVICE,
            method=SEARCH_METHOD,
            **search_kwargs
        )
        
        # Analyze profile
        num_int4 = sum(1 for v in hybrid_profile.values() if v == 'INT4')
        num_int8 = sum(1 for v in hybrid_profile.values() if v == 'INT8')
        int4_ratio = num_int4 / (num_int4 + num_int8) if (num_int4 + num_int8) > 0 else 0
        
        # Estimate compressed size
        avg_bits_per_param = estimate_model_size(hybrid_profile)
        hybrid_size_mb = param_stats['total'] * avg_bits_per_param / (1024**2)
        
        logger.info(f"✅ Hybrid: Acc={hybrid_acc:.4f}, INT4={num_int4}/{num_int4+num_int8} ({int4_ratio*100:.1f}%), Size≈{hybrid_size_mb:.2f}MB")
        
        # ----------------------------------------------------------------
        # COMPRESSION METRICS
        # ----------------------------------------------------------------
        
        compression_metrics = calculate_compression_metrics(
            fp32_size_mb, hybrid_size_mb, fp32_acc, hybrid_acc
        )
        
        logger.info(f"\n📈 Compression: {compression_metrics['compression_ratio']:.2f}x, "
                   f"Size reduction: {compression_metrics['size_reduction_pct']:.1f}%, "
                   f"Acc drop: {compression_metrics['accuracy_drop']*100:.2f}%")
        
        # ----------------------------------------------------------------
        # HARDWARE METRICS
        # ----------------------------------------------------------------
        
        gpu_stats = get_gpu_memory_usage()
        
        # ----------------------------------------------------------------
        # SAVE RESULTS
        # ----------------------------------------------------------------
        
        elapsed_time = time.time() - start_time
        
        result = {
            'Model': model_name,
            'Task': task,
            'Seed': seed,
            
            # FP32 Baseline
            'FP32_Acc': fp32_acc,
            'FP32_Size_MB': fp32_size_mb,
            'Total_Params': param_stats['total'],
            
            # Baselines
            **baseline_results,
            
            # Pruning
            'Pruned_Acc': prune_acc,
            'Sparsity_Pct': sparsity,
            
            # QAT
            'QAT_Acc': qat_acc if qat_acc else np.nan,
            
            # Hybrid
            'Hybrid_Acc': hybrid_acc,
            'Hybrid_INT4_Layers': num_int4,
            'Hybrid_INT8_Layers': num_int8,
            'Hybrid_INT4_Ratio': int4_ratio,
            'Hybrid_Size_MB': hybrid_size_mb,
            
            # Compression
            **compression_metrics,
            
            # Ablations (if run)
            **ablation_results,
            
            # Hardware
            'Elapsed_Time_Sec': elapsed_time,
        }
        
        if gpu_stats:
            result.update(gpu_stats)
        
        raw_results.append(result)
        
        # Save incrementally
        pd.DataFrame(raw_results).to_csv(
            f"{RESULTS_DIR}/{RAW_RESULTS_FILE}",
            index=False
        )
        
        logger.info(f"\n✅ Experiment complete in {elapsed_time/60:.1f} minutes")
        logger.info("="*80 + "\n")
        
        # Cleanup
        del model_fp32, model_pruned
        if USE_QAT:
            del model_qat
        clear_gpu_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ERROR in {model_name}|{task}|{seed}: {e}")
        logger.error(traceback.format_exc())
        return False


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistics(raw_results):
    """Compute summary statistics across seeds."""
    
    df = pd.DataFrame(raw_results)
    
    # Group by model and task
    grouped = df.groupby(['Model', 'Task'])
    
    summary_results = []
    
    for (model, task), group in grouped:
        summary = {
            'Model': model,
            'Task': task,
        }
        
        # Calculate mean and std for all numeric columns
        for col in group.columns:
            if group[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                summary[f"{col}_Mean"] = group[col].mean()
                summary[f"{col}_Std"] = group[col].std()
        
        # Statistical tests (if multiple seeds)
        if len(group) > 1:
            try:
                # Hybrid vs baselines
                _, p_int4 = ttest_rel(group['Hybrid_Acc'], group['Uniform_INT4'])
                _, p_int8 = ttest_rel(group['Hybrid_Acc'], group['Uniform_INT8'])
                
                summary['Hybrid_vs_INT4_pvalue'] = p_int4
                summary['Hybrid_vs_INT8_pvalue'] = p_int8
                summary['Hybrid_vs_INT4_significant'] = p_int4 < 0.05
                summary['Hybrid_vs_INT8_significant'] = p_int8 < 0.05
            except:
                pass
        
        summary_results.append(summary)
    
    return pd.DataFrame(summary_results)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    raw_results = []
    all_results = []
    
    total_experiments = len(MODELS) * len(TASKS) * len(SEEDS)
    completed = 0
    failed = 0
    
    logger.info(f"Starting {total_experiments} experiments\n")
    
    # Run all experiments
    for model_name in MODELS:
        for task in TASKS:
            for seed in SEEDS:
                
                success = run_single_experiment(
                    model_name, task, seed,
                    all_results, raw_results
                )
                
                if success:
                    completed += 1
                else:
                    failed += 1
                
                logger.info(f"Progress: {completed}/{total_experiments} completed, {failed} failed\n")
    
    # ----------------------------------------------------------------
    # STATISTICAL ANALYSIS
    # ----------------------------------------------------------------
    
    if raw_results:
        logger.info("\n" + "="*80)
        logger.info(" STATISTICAL ANALYSIS ".center(80))
        logger.info("="*80 + "\n")
        
        summary_df = compute_statistics(raw_results)
        
        # Save summary
        summary_df.to_csv(
            f"{RESULTS_DIR}/{SUMMARY_RESULTS_FILE}",
            index=False
        )
        
        logger.info(f"✅ Summary statistics saved to {SUMMARY_RESULTS_FILE}")
        
        # Print summary
        print("\n" + "="*80)
        print(" RESULTS SUMMARY ".center(80))
        print("="*80 + "\n")
        print(summary_df[[
            'Model', 'Task',
            'FP32_Acc_Mean', 'Hybrid_Acc_Mean',
            'compression_ratio_Mean', 'Hybrid_INT4_Ratio_Mean'
        ]].to_string(index=False))
        print("\n" + "="*80 + "\n")
    
    # ----------------------------------------------------------------
    # VISUALIZATION (if enabled)
    # ----------------------------------------------------------------
    
    if GENERATE_LATEX_TABLES:
        try:
            from visualization.plots import generate_all_plots
            logger.info("\n📊 Generating visualizations...")
            generate_all_plots(f"{RESULTS_DIR}/{RAW_RESULTS_FILE}", PLOTS_DIR)
            logger.info("✅ Plots saved to {PLOTS_DIR}/")
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")
    
    # ----------------------------------------------------------------
    # FINAL SUMMARY
    # ----------------------------------------------------------------
    
    logger.info("\n" + "="*80)
    logger.info(" PIPELINE COMPLETE ".center(80))
    logger.info("="*80)
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Completed: {completed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"\nResults saved to:")
    logger.info(f"  - {RESULTS_DIR}/{RAW_RESULTS_FILE}")
    logger.info(f"  - {RESULTS_DIR}/{SUMMARY_RESULTS_FILE}")
    if GENERATE_LATEX_TABLES:
        logger.info(f"  - {PLOTS_DIR}/ (visualizations)")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
