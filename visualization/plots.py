"""
Simple visualization module.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
sns.set_palette("husl")


def generate_all_plots(results_csv, output_dir="plots"):
    """Generate publication plots from results."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load results
    if isinstance(results_csv, str):
        df = pd.read_csv(results_csv)
    else:
        df = pd.DataFrame(results_csv)
    
    print(f"📊 Generating plots from {len(df)} results...")
    
    # Plot 1: Method Comparison
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        methods = ['FP32', 'Uniform_INT8', 'Uniform_INT4', 'Hybrid_Acc']
        labels = ['FP32', 'Uniform INT8', 'Uniform INT4', 'Hybrid (Ours)']
        
        if all(col in df.columns for col in methods):
            means = [df[col].mean() * 100 for col in methods]
            stds = [df[col].std() * 100 for col in methods]
            
            bars = ax.bar(labels, means, yerr=stds, capsize=5, 
                         color=['red', 'blue', 'green', 'purple'], 
                         alpha=0.7, edgecolor='black')
            
            # Add value labels
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{mean:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title('Method Comparison', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/method_comparison.png", bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: method_comparison.png")
    except Exception as e:
        print(f"  ✗ Could not create method comparison: {e}")
    
    # Plot 2: Compression vs Accuracy
    try:
        if 'compression_ratio' in df.columns and 'accuracy_drop' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            scatter = ax.scatter(
                df['compression_ratio'],
                df['accuracy_drop'] * 100,
                c=df.index,
                s=100,
                alpha=0.6,
                cmap='viridis',
                edgecolors='black'
            )
            
            ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='<1% drop (excellent)')
            ax.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='<2% drop (good)')
            
            ax.set_xlabel('Compression Ratio (×)', fontweight='bold')
            ax.set_ylabel('Accuracy Drop (%)', fontweight='bold')
            ax.set_title('Compression Efficiency', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            ax.invert_yaxis()
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/compression_vs_accuracy.png", bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: compression_vs_accuracy.png")
    except Exception as e:
        print(f"  ✗ Could not create compression plot: {e}")
    
    # Plot 3: Ablation Study (if data available)
    try:
        ablation_cols = ['Pruning_Only', 'Quantization_Only', 'Pruning_No_Distillation', 'Hybrid_Acc']
        
        if all(col in df.columns for col in ablation_cols):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ablation_data = {
                'FP32 Baseline': df['FP32'].mean() * 100,
                'Pruning Only': df['Pruning_Only'].mean() * 100,
                'Quantization Only': df['Quantization_Only'].mean() * 100,
                'Pruning (No Distill)': df['Pruning_No_Distillation'].mean() * 100,
                'Full Method': df['Hybrid_Acc'].mean() * 100,
            }
            
            bars = ax.bar(ablation_data.keys(), ablation_data.values(),
                         color=['red', 'lightblue', 'lightgreen', 'yellow', 'darkgreen'],
                         alpha=0.8, edgecolor='black')
            
            for bar, val in zip(bars, ablation_data.values()):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Accuracy (%)', fontweight='bold')
            ax.set_title('Ablation Study', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.xticks(rotation=15, ha='right')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/ablation_study.png", bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved: ablation_study.png")
    except Exception as e:
        print(f"  ✗ Could not create ablation plot: {e}")
    
    print(f"\n✅ Plots saved to {output_dir}/")


if __name__ == "__main__":
    # Test
    import sys
    if len(sys.argv) > 1:
        generate_all_plots(sys.argv[1], "plots")
    else:
        print("Usage: python plots.py <results.csv>")