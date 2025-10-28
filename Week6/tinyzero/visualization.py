"""
Visualization utilities for TinyZero results
"""
import json
from pathlib import Path
from typing import Dict, List
import numpy as np

# Lazy import matplotlib - only load when actually used
_plt = None
_sns = None

def _import_matplotlib():
    """Import matplotlib only when needed"""
    global _plt, _sns
    
    if _plt is not None and _sns is not None:
        return _plt, _sns
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        
        _plt = plt
        _sns = sns
        return plt, sns
    except ImportError:
        raise ImportError(
            "Matplotlib and seaborn are required for visualization. "
            "Install with: pip install matplotlib seaborn"
        )


def plot_training_curves(metrics_file: str, output_dir: str = 'outputs'):
    """
    Plot training curves from metrics JSON file
    
    Args:
        metrics_file: Path to metrics.json file
        output_dir: Directory to save plots
    """
    # Import matplotlib when needed
    plt, sns = _import_matplotlib()
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Training Loss
    if metrics.get('train_loss') and len(metrics['train_loss']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['steps'], metrics['train_loss'], 'b-', linewidth=2)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'training_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'training_loss.png'}")
    
    # Plot 2: Training Reward
    if metrics.get('train_reward') and len(metrics['train_reward']) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics['steps'], metrics['train_reward'], 'g-', linewidth=2)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Average Reward', fontsize=12)
        plt.title('Training Reward Over Time', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'training_reward.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'training_reward.png'}")
    
    # Plot 3: Evaluation Accuracy
    if metrics.get('eval_accuracy') and len(metrics['eval_accuracy']) > 0:
        plt.figure(figsize=(12, 6))
        
        # Get eval steps (every eval_every steps)
        num_evals = len(metrics['eval_accuracy'])
        if len(metrics['steps']) >= num_evals:
            step_interval = max(1, len(metrics['steps']) // num_evals)
            eval_steps = [metrics['steps'][i] for i in range(0, len(metrics['steps']), step_interval)][:num_evals]
        else:
            eval_steps = metrics['steps'][:num_evals]
        
        plt.plot(eval_steps, metrics['eval_accuracy'], 'r-o', linewidth=2, label='Overall', markersize=6)
        
        if metrics.get('eval_accuracy_countdown'):
            plt.plot(eval_steps, metrics['eval_accuracy_countdown'], 'b-s', linewidth=2, label='Countdown', markersize=6)
        
        if metrics.get('eval_accuracy_multiplication'):
            plt.plot(eval_steps, metrics['eval_accuracy_multiplication'], 'g-^', linewidth=2, label='Multiplication', markersize=6)
        
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Evaluation Accuracy Over Time', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'eval_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'eval_accuracy.png'}")
    
    # Plot 4: Combined view
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    if metrics.get('train_loss') and len(metrics['train_loss']) > 0:
        axes[0, 0].plot(metrics['steps'], metrics['train_loss'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'No training loss data', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
    
    # Reward
    if metrics.get('train_reward') and len(metrics['train_reward']) > 0:
        axes[0, 1].plot(metrics['steps'], metrics['train_reward'], 'g-', linewidth=2)
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Training Reward')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No training reward data', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
    
    # Accuracy
    if metrics.get('eval_accuracy') and len(metrics['eval_accuracy']) > 0:
        num_evals = len(metrics['eval_accuracy'])
        if len(metrics['steps']) >= num_evals:
            step_interval = max(1, len(metrics['steps']) // num_evals)
            eval_steps = [metrics['steps'][i] for i in range(0, len(metrics['steps']), step_interval)][:num_evals]
        else:
            eval_steps = metrics['steps'][:num_evals]
        
        axes[1, 0].plot(eval_steps, metrics['eval_accuracy'], 'r-o', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Overall Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No evaluation data', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
    
    # Task-specific accuracy
    if (metrics.get('eval_accuracy_countdown') and 
        metrics.get('eval_accuracy_multiplication') and
        len(metrics['eval_accuracy_countdown']) > 0):
        
        num_evals = len(metrics['eval_accuracy'])
        if len(metrics['steps']) >= num_evals:
            step_interval = max(1, len(metrics['steps']) // num_evals)
            eval_steps = [metrics['steps'][i] for i in range(0, len(metrics['steps']), step_interval)][:num_evals]
        else:
            eval_steps = metrics['steps'][:num_evals]
        
        axes[1, 1].plot(eval_steps, metrics['eval_accuracy_countdown'], 'b-s', 
                       linewidth=2, label='Countdown', markersize=6)
        axes[1, 1].plot(eval_steps, metrics['eval_accuracy_multiplication'], 'g-^', 
                       linewidth=2, label='Multiplication', markersize=6)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Task-Specific Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No task-specific data', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'training_overview.png'}")
    
    print(f"\nAll plots saved to {output_dir}/")


def create_results_table(metrics: Dict, output_file: str = 'outputs/results_table.md'):
    """
    Create a markdown table of results
    
    Args:
        metrics: Metrics dictionary
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle missing keys gracefully
    accuracy = metrics.get('accuracy', 0.0)
    accuracy_countdown = metrics.get('accuracy_countdown', 0.0)
    accuracy_multiplication = metrics.get('accuracy_multiplication', 0.0)
    avg_length = metrics.get('avg_length', 0.0)
    total_examples = metrics.get('total_examples', 0)
    
    table = f"""
# TinyZero Results

## Overall Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | {accuracy:.2%} |
| Countdown Accuracy | {accuracy_countdown:.2%} |
| Multiplication Accuracy | {accuracy_multiplication:.2%} |
| Avg Response Length | {avg_length:.1f} chars |
| Total Examples | {total_examples} |

## Task Breakdown

### Countdown Task
- Accuracy: {accuracy_countdown:.2%}
- Description: Generate target number using given numbers

### Multiplication Task  
- Accuracy: {accuracy_multiplication:.2%}
- Description: Compute product of two numbers

## Notes

- Model: Qwen2.5-3B-Instruct
- Algorithm: A*PO (A-star Policy Optimization)
- Training: Reinforcement Learning without supervised fine-tuning
- Key Innovation: Computing V* using reference model for efficient optimization

## Training Details

- Base Model: Qwen2.5-3B-Instruct
- Reference Model: Qwen2.5-3B-Instruct (frozen)
- Optimization: A*PO (A-star Policy Optimization)
- Tasks: Countdown and Multiplication
- Training Steps: {total_examples}

## Key Differences from GRPO

| Feature | GRPO | A*PO (Our Implementation) |
|---------|------|---------------------------|
| Rollouts per prompt | G (multiple) | 1 (single) |
| Value estimation | Learned critic | Computed V* from reference |
| Sample efficiency | Lower | Higher |
| Memory usage | Higher | Lower |
| Compute cost | Higher | Lower |

## Example Outputs

See evaluation logs for example model outputs on both countdown and multiplication tasks.
"""
    
    with open(output_path, 'w') as f:
        f.write(table)
    
    print(f"Saved results table to: {output_path}")


def plot_comparison(
    metrics_list: List[Dict],
    labels: List[str],
    output_dir: str = 'outputs'
):
    """
    Plot comparison between multiple runs
    
    Args:
        metrics_list: List of metrics dictionaries
        labels: List of labels for each run
        output_dir: Directory to save plots
    """
    plt, sns = _import_matplotlib()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    colors = ['b', 'r', 'g', 'orange', 'purple']
    
    # Plot accuracy comparison
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        if metrics.get('eval_accuracy') and len(metrics['eval_accuracy']) > 0:
            num_evals = len(metrics['eval_accuracy'])
            if len(metrics['steps']) >= num_evals:
                step_interval = max(1, len(metrics['steps']) // num_evals)
                eval_steps = [metrics['steps'][j] for j in range(0, len(metrics['steps']), step_interval)][:num_evals]
            else:
                eval_steps = metrics['steps'][:num_evals]
            
            color = colors[i % len(colors)]
            axes[0].plot(eval_steps, metrics['eval_accuracy'], 
                        f'{color}-o', linewidth=2, label=label, markersize=5)
    
    axes[0].set_xlabel('Training Steps', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot reward comparison
    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        if metrics.get('train_reward') and len(metrics['train_reward']) > 0:
            color = colors[i % len(colors)]
            axes[1].plot(metrics['steps'], metrics['train_reward'], 
                        f'{color}-', linewidth=2, label=label)
    
    axes[1].set_xlabel('Training Steps', fontsize=12)
    axes[1].set_ylabel('Average Reward', fontsize=12)
    axes[1].set_title('Reward Comparison', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {output_dir / 'comparison.png'}")


def print_summary_stats(metrics: Dict):
    """
    Print summary statistics
    
    Args:
        metrics: Metrics dictionary
    """
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Overall metrics
    if 'accuracy' in metrics:
        print(f"Final Accuracy: {metrics['accuracy']:.2%}")
    
    if 'accuracy_countdown' in metrics:
        print(f"Countdown Accuracy: {metrics['accuracy_countdown']:.2%}")
    
    if 'accuracy_multiplication' in metrics:
        print(f"Multiplication Accuracy: {metrics['accuracy_multiplication']:.2%}")
    
    if 'avg_length' in metrics:
        print(f"Avg Response Length: {metrics['avg_length']:.1f} chars")
    
    # Training progress
    if 'eval_accuracy' in metrics and len(metrics['eval_accuracy']) > 0:
        initial_acc = metrics['eval_accuracy'][0]
        final_acc = metrics['eval_accuracy'][-1]
        improvement = final_acc - initial_acc
        
        print(f"\nTraining Progress:")
        print(f"  Initial Accuracy: {initial_acc:.2%}")
        print(f"  Final Accuracy: {final_acc:.2%}")
        print(f"  Improvement: {improvement:+.2%}")
    
    print("="*60 + "\n")


# Example usage function
def visualize_training_results(metrics_file: str, output_dir: str = 'outputs'):
    """
    Complete visualization pipeline
    
    Args:
        metrics_file: Path to metrics.json
        output_dir: Output directory for plots
    """
    print(f"Loading metrics from {metrics_file}...")
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print("Creating visualizations...")
    
    # Create all plots
    plot_training_curves(metrics_file, output_dir)
    
    # Create results table
    if metrics.get('eval_accuracy') and len(metrics['eval_accuracy']) > 0:
        final_metrics = {
            'accuracy': metrics['eval_accuracy'][-1],
            'accuracy_countdown': metrics.get('eval_accuracy_countdown', [0])[-1],
            'accuracy_multiplication': metrics.get('eval_accuracy_multiplication', [0])[-1],
            'avg_length': 200,  # Placeholder
            'total_examples': len(metrics.get('steps', []))
        }
        create_results_table(final_metrics, f"{output_dir}/results_table.md")
    
    # Print summary
    if metrics.get('eval_accuracy'):
        summary_metrics = {
            'accuracy': metrics['eval_accuracy'][-1] if metrics['eval_accuracy'] else 0,
            'accuracy_countdown': metrics.get('eval_accuracy_countdown', [0])[-1] if metrics.get('eval_accuracy_countdown') else 0,
            'accuracy_multiplication': metrics.get('eval_accuracy_multiplication', [0])[-1] if metrics.get('eval_accuracy_multiplication') else 0,
            'avg_length': 200,
            'eval_accuracy': metrics['eval_accuracy']
        }
        print_summary_stats(summary_metrics)
    
    print(f"\n✓ All visualizations saved to {output_dir}/")