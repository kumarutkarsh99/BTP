"""
Pruning Amount Study
Systematically evaluate different pruning amounts to find optimal compression-accuracy tradeoff
"""

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import time

# Import from your pipeline
from config.config import *
from data.data_loader import prepare_data
from training.train import train_fp32, get_accuracy
from compression.pruning import prune_and_recover, get_sparsity
from compression.quantizer import apply_quantization_to_model
from search.hybrid_search import run_hybrid_search
from utils.utils import set_seed, count_parameters

logger = logging.getLogger(__name__)


# ============================================================================
# PRUNING STUDY CONFIGURATION
# ============================================================================

PRUNING_AMOUNTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
# 0% (baseline), 5%, 10%, 15%, 20%, 25%, 30%, 40%, 50%

STUDY_CONFIG = {
    'models': ["distilbert-base-uncased", "bert-base-uncased"],  # Can add more
    'tasks': ["sst2", "qnli"],  # Can add: "qnli", "mnli"
    'seeds': [42, 123, 999],  # Multiple seeds for statistical validity
    'use_distillation': True,  # Compare with/without
    'use_progressive': False,  # Compare one-shot vs progressive
    'recovery_epochs': 3,
    'run_quantization': True,  # Test pruning + quantization
}


# ============================================================================
# SINGLE PRUNING EXPERIMENT
# ============================================================================

def run_pruning_experiment(model_name, task, seed, pruning_amount, 
                          use_distillation=True, use_progressive=False):
    """
    Run single pruning experiment.
    
    Returns:
        dict: Results with metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Pruning={pruning_amount*100:.0f}% | {model_name} | {task} | Seed={seed}")
    logger.info(f"{'='*80}")
    
    set_seed(seed)
    
    try:
        # Data preparation
        train_loader, val_loader, num_labels, task_info = prepare_data(
            model_name, task,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH
        )
        
        # FP32 baseline (only once per model-task-seed)
        model_fp32, fp32_acc = train_fp32(
            model_name, train_loader, val_loader, num_labels, DEVICE,
            epochs=EPOCHS_FINETUNE
        )
        
        param_stats = count_parameters(model_fp32)
        fp32_size_mb = param_stats['total'] * 4 / (1024**2)
        
        # Special case: 0% pruning (baseline)
        if pruning_amount == 0.0:
            result = {
                'Model': model_name,
                'Task': task,
                'Seed': seed,
                'Pruning_Amount': 0.0,
                'Use_Distillation': use_distillation,
                'Use_Progressive': use_progressive,
                
                'FP32_Acc': fp32_acc,
                'Pruned_Acc': fp32_acc,
                'Accuracy_Drop': 0.0,
                'Accuracy_Retention': 100.0,
                
                'FP32_Size_MB': fp32_size_mb,
                'Pruned_Size_MB': fp32_size_mb,
                'Sparsity_Actual': 0.0,
                'Size_Reduction': 0.0,
                
                'Params_Total': param_stats['total'],
                'Params_Trainable': param_stats['trainable'],
            }
            
            return result
        
        # Apply pruning
        model_pruned = prune_and_recover(
            model_fp32, train_loader, val_loader, DEVICE,
            pruning_amount=pruning_amount,
            use_distillation=use_distillation,
            use_progressive=use_progressive,
            recovery_epochs=STUDY_CONFIG['recovery_epochs']
        )
        
        # Evaluate pruned model
        pruned_acc = get_accuracy(model_pruned, val_loader, DEVICE)
        actual_sparsity = get_sparsity(model_pruned)
        
        # Calculate pruned size (accounting for sparsity)
        # Sparse storage: only non-zero values + indices
        pruned_size_mb = fp32_size_mb * (1 - actual_sparsity/100)
        
        # Optional: Test with quantization
        quant_int8_acc = None
        quant_int4_acc = None
        hybrid_acc = None
        hybrid_size_mb = None
        
        if STUDY_CONFIG['run_quantization']:
            # Import copy module
            import copy
            
            # Uniform INT8
            logger.info("Testing INT8 quantization on pruned model...")
            model_q8 = copy.deepcopy(model_pruned)
            layer_names = [n for n, p in model_q8.named_parameters() 
                          if 'weight' in n and p.dim() > 1]
            profile_int8 = {n: 'INT8' for n in layer_names}
            apply_quantization_to_model(model_q8, profile_int8)
            quant_int8_acc = get_accuracy(model_q8, val_loader, DEVICE)
            logger.info(f"  INT8 Acc: {quant_int8_acc:.4f}")
            
            # Uniform INT4
            logger.info("Testing INT4 quantization on pruned model...")
            model_q4 = copy.deepcopy(model_pruned)
            profile_int4 = {n: 'INT4' for n in layer_names}
            apply_quantization_to_model(model_q4, profile_int4)
            quant_int4_acc = get_accuracy(model_q4, val_loader, DEVICE)
            logger.info(f"  INT4 Acc: {quant_int4_acc:.4f}")
            
            # Hybrid search (on pruned model)
            try:
                logger.info("Running hybrid search on pruned model...")
                model_hybrid = copy.deepcopy(model_pruned)
                hybrid_acc, hybrid_profile = run_hybrid_search(
                    model_hybrid, val_loader, DEVICE,
                    method='greedy',  # Use greedy for speed
                    sensitivity_threshold=0.01
                )
                
                # Estimate hybrid size
                num_int4 = sum(1 for v in hybrid_profile.values() if v == 'INT4')
                num_int8 = sum(1 for v in hybrid_profile.values() if v == 'INT8')
                if num_int4 + num_int8 > 0:
                    avg_bits = (num_int4 * 4 + num_int8 * 8) / (num_int4 + num_int8)
                    hybrid_size_mb = pruned_size_mb * (avg_bits / 32)
                else:
                    hybrid_size_mb = pruned_size_mb
                
                logger.info(f"  Hybrid Acc: {hybrid_acc:.4f} ({num_int4}/{num_int4+num_int8} INT4)")
                
                del model_hybrid
            except Exception as e:
                logger.warning(f"Hybrid search failed: {e}")
            
            # Cleanup
            del model_q8, model_q4
        
        # Compile results
        result = {
            'Model': model_name,
            'Task': task,
            'Seed': seed,
            'Pruning_Amount': pruning_amount,
            'Use_Distillation': use_distillation,
            'Use_Progressive': use_progressive,
            
            # Accuracy metrics
            'FP32_Acc': fp32_acc,
            'Pruned_Acc': pruned_acc,
            'Accuracy_Drop': fp32_acc - pruned_acc,
            'Accuracy_Retention': (pruned_acc / fp32_acc * 100) if fp32_acc > 0 else 0,
            
            # Size metrics
            'FP32_Size_MB': fp32_size_mb,
            'Pruned_Size_MB': pruned_size_mb,
            'Sparsity_Target': pruning_amount * 100,
            'Sparsity_Actual': actual_sparsity,
            'Size_Reduction': (1 - pruned_size_mb / fp32_size_mb) * 100,
            
            # Quantization metrics
            'Quant_INT8_Acc': quant_int8_acc,
            'Quant_INT4_Acc': quant_int4_acc,
            'Hybrid_Acc': hybrid_acc,
            'Hybrid_Size_MB': hybrid_size_mb,
            
            # Model info
            'Params_Total': param_stats['total'],
            'Params_Trainable': param_stats['trainable'],
        }
        
        logger.info(f"✅ Results: Acc={pruned_acc:.4f} (-{result['Accuracy_Drop']*100:.2f}%), "
                   f"Sparsity={actual_sparsity:.1f}%, Size={pruned_size_mb:.1f}MB")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# MAIN STUDY RUNNER
# ============================================================================

def run_pruning_study(output_dir="results/pruning_study"):
    """
    Run complete pruning amount study.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*80)
    logger.info(" PRUNING AMOUNT STUDY ".center(80))
    logger.info("="*80)
    logger.info(f"Pruning amounts: {[f'{p*100:.0f}%' for p in PRUNING_AMOUNTS]}")
    logger.info(f"Models: {STUDY_CONFIG['models']}")
    logger.info(f"Tasks: {STUDY_CONFIG['tasks']}")
    logger.info(f"Seeds: {STUDY_CONFIG['seeds']}")
    logger.info("="*80 + "\n")
    
    all_results = []
    
    total_experiments = (
        len(STUDY_CONFIG['models']) * 
        len(STUDY_CONFIG['tasks']) * 
        len(STUDY_CONFIG['seeds']) * 
        len(PRUNING_AMOUNTS)
    )
    
    completed = 0
    
    for model_name in STUDY_CONFIG['models']:
        for task in STUDY_CONFIG['tasks']:
            for seed in STUDY_CONFIG['seeds']:
                for pruning_amount in PRUNING_AMOUNTS:
                    
                    result = run_pruning_experiment(
                        model_name, task, seed, pruning_amount,
                        use_distillation=STUDY_CONFIG['use_distillation'],
                        use_progressive=STUDY_CONFIG['use_progressive']
                    )
                    
                    if result:
                        all_results.append(result)
                        
                        # Save incrementally
                        df = pd.DataFrame(all_results)
                        df.to_csv(f"{output_dir}/pruning_study_raw.csv", index=False)
                    
                    completed += 1
                    logger.info(f"\nProgress: {completed}/{total_experiments} experiments\n")
    
    # Save final results
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(f"{output_dir}/pruning_study_raw.csv", index=False)
    
    # Generate summary statistics
    summary = generate_summary(df_final)
    summary.to_csv(f"{output_dir}/pruning_study_summary.csv", index=False)
    
    # Generate visualizations
    generate_visualizations(df_final, output_dir)
    
    logger.info("\n" + "="*80)
    logger.info(" PRUNING STUDY COMPLETE ".center(80))
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}/")
    logger.info("="*80 + "\n")
    
    return df_final, summary


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def generate_summary(df):
    """Generate summary statistics across seeds."""
    
    summary_data = []
    
    for pruning_amount in df['Pruning_Amount'].unique():
        subset = df[df['Pruning_Amount'] == pruning_amount]
        
        summary = {
            'Pruning_Amount': pruning_amount,
            'Pruning_Pct': pruning_amount * 100,
            
            # Accuracy
            'Pruned_Acc_Mean': subset['Pruned_Acc'].mean(),
            'Pruned_Acc_Std': subset['Pruned_Acc'].std(),
            'Accuracy_Drop_Mean': subset['Accuracy_Drop'].mean(),
            'Accuracy_Drop_Std': subset['Accuracy_Drop'].std(),
            
            # Sparsity
            'Sparsity_Actual_Mean': subset['Sparsity_Actual'].mean(),
            'Sparsity_Actual_Std': subset['Sparsity_Actual'].std(),
            
            # Size
            'Size_Reduction_Mean': subset['Size_Reduction'].mean(),
            'Size_Reduction_Std': subset['Size_Reduction'].std(),
            
            # Quantization
            'Quant_INT8_Acc_Mean': subset['Quant_INT8_Acc'].mean(),
            'Quant_INT4_Acc_Mean': subset['Quant_INT4_Acc'].mean(),
            'Hybrid_Acc_Mean': subset['Hybrid_Acc'].mean(),
        }
        
        summary_data.append(summary)
    
    return pd.DataFrame(summary_data).sort_values('Pruning_Amount')


# ============================================================================
# VISUALIZATIONS
# ============================================================================

def generate_visualizations(df, output_dir):
    """Generate publication-quality plots."""
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # Calculate means across seeds
    summary = df.groupby('Pruning_Amount').agg({
        'Pruned_Acc': ['mean', 'std'],
        'Accuracy_Drop': ['mean', 'std'],
        'Sparsity_Actual': ['mean', 'std'],
        'Size_Reduction': ['mean', 'std'],
        'Quant_INT8_Acc': 'mean',
        'Quant_INT4_Acc': 'mean',
        'Hybrid_Acc': 'mean',
    }).reset_index()
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary['Pruning_Pct'] = summary['Pruning_Amount'] * 100
    
    # ========================================================================
    # PLOT 1: Accuracy vs Pruning Amount
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(summary['Pruning_Pct'], summary['Pruned_Acc_mean'], 
                yerr=summary['Pruned_Acc_std'], 
                marker='o', linewidth=2, capsize=5, 
                label='Pruned (w/ Distillation)', color='blue')
    
    if 'Quant_INT8_Acc_mean' in summary.columns:
        ax.plot(summary['Pruning_Pct'], summary['Quant_INT8_Acc_mean'], 
                marker='s', linewidth=2, label='Pruned + INT8', color='green')
    
    if 'Hybrid_Acc_mean' in summary.columns:
        ax.plot(summary['Pruning_Pct'], summary['Hybrid_Acc_mean'], 
                marker='^', linewidth=2, label='Pruned + Hybrid', color='red')
    
    ax.set_xlabel('Pruning Amount (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title('Accuracy vs Pruning Amount', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_vs_pruning.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/accuracy_vs_pruning.pdf", bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 2: Accuracy Drop vs Pruning Amount
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.errorbar(summary['Pruning_Pct'], summary['Accuracy_Drop_mean'] * 100, 
                yerr=summary['Accuracy_Drop_std'] * 100,
                marker='o', linewidth=2, capsize=5, color='red')
    
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='<1% drop (excellent)')
    ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='<2% drop (good)')
    ax.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='<5% drop (acceptable)')
    
    ax.set_xlabel('Pruning Amount (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontweight='bold', fontsize=12)
    ax.set_title('Accuracy Degradation vs Pruning Amount', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_drop_vs_pruning.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/accuracy_drop_vs_pruning.pdf", bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 3: Pareto Frontier (Accuracy vs Size Reduction)
    # ========================================================================
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(summary['Size_Reduction_mean'], 
                        summary['Pruned_Acc_mean'],
                        c=summary['Pruning_Pct'],
                        s=200, alpha=0.7, cmap='viridis',
                        edgecolors='black', linewidths=2)
    
    # Annotate points
    for idx, row in summary.iterrows():
        ax.annotate(f"{row['Pruning_Pct']:.0f}%",
                   (row['Size_Reduction_mean'], row['Pruned_Acc_mean']),
                   fontsize=9, fontweight='bold',
                   xytext=(5, 5), textcoords='offset points')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Pruning Amount (%)', fontweight='bold')
    
    ax.set_xlabel('Size Reduction (%)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title('Pareto Frontier: Accuracy vs Compression', fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pareto_frontier.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/pareto_frontier.pdf", bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # PLOT 4: Multi-metric Comparison
    # ========================================================================
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Accuracy retention
    ax1.plot(summary['Pruning_Pct'], 
            (summary['Pruned_Acc_mean'] / summary['Pruned_Acc_mean'].iloc[0] * 100),
            marker='o', linewidth=2, color='blue')
    ax1.axhline(y=99, color='green', linestyle='--', alpha=0.5, label='99% retention')
    ax1.axhline(y=95, color='orange', linestyle='--', alpha=0.5, label='95% retention')
    ax1.set_xlabel('Pruning Amount (%)', fontweight='bold')
    ax1.set_ylabel('Accuracy Retention (%)', fontweight='bold')
    ax1.set_title('Accuracy Retention', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Right: Actual vs Target Sparsity
    ax2.plot(summary['Pruning_Pct'], summary['Sparsity_Actual_mean'], 
            marker='o', linewidth=2, label='Actual', color='blue')
    ax2.plot(summary['Pruning_Pct'], summary['Pruning_Pct'], 
            linestyle='--', linewidth=2, label='Target', color='red')
    ax2.set_xlabel('Target Pruning Amount (%)', fontweight='bold')
    ax2.set_ylabel('Sparsity (%)', fontweight='bold')
    ax2.set_title('Target vs Actual Sparsity', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_metric.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_dir}/multi_metric.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"✅ Generated 4 plots in {output_dir}/")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    from utils.utils import setup_logging
    setup_logging("pruning_study.log", "INFO")
    
    # Run study
    results_df, summary_df = run_pruning_study()
    
    # Print summary
    print("\n" + "="*80)
    print(" PRUNING STUDY SUMMARY ".center(80))
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Find optimal pruning amount
    optimal = summary_df.loc[summary_df['Accuracy_Drop_Mean'].idxmin()]
    print(f"\n🏆 OPTIMAL PRUNING AMOUNT: {optimal['Pruning_Pct']:.0f}%")
    print(f"   Accuracy Drop: {optimal['Accuracy_Drop_Mean']*100:.2f}%")
    print(f"   Size Reduction: {optimal['Size_Reduction_Mean']:.1f}%")
    print(f"   Actual Sparsity: {optimal['Sparsity_Actual_Mean']:.1f}%")