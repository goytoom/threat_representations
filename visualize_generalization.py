"""
Visualize Step 3 Generalization Results
Creates comprehensive analysis plots for steering test results

Usage:
python visualize_generalization.py \
    --json_path steering_test_results/detailed_results/step3_generalization_20251111_132226.json \
    --output_dir steering_test_results/visualizations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_generalization_data(json_path: str) -> pd.DataFrame:
    """Load and parse generalization results"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Convert ratings to numeric
    df['gpt4_rating'] = pd.to_numeric(df['gpt4_rating'], errors='coerce')
    df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
    df['layer'] = pd.to_numeric(df['layer'], errors='coerce')
    
    # Filter out invalid data
    df = df[df['gpt4_rating'].notna()].copy()
    
    print(f"✅ Loaded {len(df)} evaluations")
    print(f"   Layers: {sorted(df['layer'].unique())}")
    print(f"   Alphas: {sorted(df['alpha'].unique())}")
    print(f"   Scenarios: {len(df['scenario'].unique())}")
    
    return df


def create_comprehensive_visualization(df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualization with 6 subplots"""
    
    layers = sorted(df['layer'].unique())
    alphas = sorted(df['alpha'].unique())
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    fig.suptitle(
        'Generalization Test Results: Hostile Vector Steering\n'
        f'Testing {len(layers)} Layers on {len(df["scenario"].unique())} Novel Scenarios',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # ========== PLOT 1: Overall Distribution (Violin) ==========
    ax1 = fig.add_subplot(gs[0, :2])
    
    df_plot = df.copy()
    df_plot['alpha_str'] = df_plot['alpha'].apply(lambda x: f'α={x:+.1f}')
    
    sns.violinplot(data=df_plot, x='alpha_str', y='gpt4_rating', ax=ax1,
                   palette='RdYlGn_r', inner='box', cut=0)
    
    ax1.set_xlabel('Steering Strength (α)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Hostility Rating', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Distribution Across All Layers', fontsize=14, fontweight='bold')
    ax1.axhline(y=2.0, color='orange', linestyle='--', linewidth=1.5, 
               label='Hostile threshold', alpha=0.7)
    ax1.set_ylim(-0.5, 4.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== PLOT 2: Mean ± Std by Layer ==========
    ax2 = fig.add_subplot(gs[0, 2])
    
    summary = df.groupby(['layer', 'alpha'])['gpt4_rating'].agg(['mean', 'std']).reset_index()
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(layers)))
    
    for i, layer in enumerate(layers):
        layer_data = summary[summary['layer'] == layer].sort_values('alpha')
        ax2.errorbar(layer_data['alpha'], layer_data['mean'], 
                    yerr=layer_data['std'], marker='o', linewidth=2,
                    markersize=8, label=f'Layer {int(layer)}',
                    color=colors[i], capsize=5, capthick=2)
    
    ax2.set_xlabel('Steering α', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Mean Rating ± Std', fontsize=11, fontweight='bold')
    ax2.set_title('Mean Response by Layer', fontsize=13, fontweight='bold')
    ax2.axhline(y=2.0, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 4.5)
    
    # ========== PLOT 3: Steering Effectiveness (Δ) ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Calculate delta for each layer
    deltas = []
    for layer in layers:
        layer_df = df[df['layer'] == layer]
        
        neg_ratings = layer_df[layer_df['alpha'] == alphas[0]]['gpt4_rating']
        pos_ratings = layer_df[layer_df['alpha'] == alphas[-1]]['gpt4_rating']
        
        if len(neg_ratings) > 0 and len(pos_ratings) > 0:
            delta = pos_ratings.mean() - neg_ratings.mean()
            deltas.append({'layer': int(layer), 'delta': delta})
    
    delta_df = pd.DataFrame(deltas).sort_values('delta', ascending=True)
    
    colors_delta = ['red' if d > 2.0 else 'steelblue' for d in delta_df['delta']]
    
    ax3.barh(range(len(delta_df)), delta_df['delta'], color=colors_delta, alpha=0.8)
    ax3.set_yticks(range(len(delta_df)))
    ax3.set_yticklabels([f"L{l}" for l in delta_df['layer']])
    ax3.set_xlabel('Δ = Rating(α_max) - Rating(α_min)', fontsize=11, fontweight='bold')
    ax3.set_title('Steering Effectiveness by Layer', fontsize=13, fontweight='bold')
    ax3.axvline(x=2.0, color='green', linestyle='--', linewidth=1.5, 
               label='Strong effect (Δ>2.0)', alpha=0.7)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Annotate best
    best_idx = delta_df['delta'].argmax()
    best_layer = delta_df.iloc[best_idx]['layer']
    best_delta = delta_df.iloc[best_idx]['delta']
    ax3.annotate(f'Best: L{best_layer}\nΔ={best_delta:.2f}',
                xy=(best_delta, best_idx), xytext=(10, 5),
                textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # ========== PLOT 4: Hostile Rate by Layer ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    hostile_rate = df.groupby(['layer', 'alpha'])['gpt4_is_hostile'].mean().reset_index()
    hostile_rate.columns = ['layer', 'alpha', 'hostile_rate']
    
    pivot = hostile_rate.pivot(index='layer', columns='alpha', values='hostile_rate')
    
    # Create grouped bar chart
    x = np.arange(len(layers))
    width = 0.25
    
    for i, alpha in enumerate(alphas):
        if alpha in pivot.columns:
            values = [pivot.loc[l, alpha] if l in pivot.index else 0 for l in layers]
            ax4.bar(x + i*width, values, width, label=f'α={alpha:+.1f}',
                   alpha=0.8)
    
    ax4.set_xlabel('Layer', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Hostile Rate', fontsize=11, fontweight='bold')
    ax4.set_title('Hostile Response Rate', fontsize=13, fontweight='bold')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels([f'L{int(l)}' for l in layers])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1.0)
    
    # ========== PLOT 5: Scenario-wise Heatmap ==========
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Get top scenarios by variance across alphas
    scenario_variance = df.groupby('scenario')['gpt4_rating'].var().sort_values(ascending=False)
    top_scenarios = scenario_variance.head(10).index
    
    df_heatmap = df[df['scenario'].isin(top_scenarios)].copy()
    df_heatmap['scenario_short'] = df_heatmap['scenario'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
    
    # Average across layers for each scenario
    heatmap_data = df_heatmap.groupby(['scenario_short', 'alpha'])['gpt4_rating'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='scenario_short', columns='alpha', values='gpt4_rating')
    
    sns.heatmap(heatmap_pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
               vmin=0, vmax=4, ax=ax5, cbar_kws={'label': 'Rating'})
    
    ax5.set_xlabel('Steering α', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Scenario (Top 10 by variance)', fontsize=11, fontweight='bold')
    ax5.set_title('Scenario-wise Response', fontsize=13, fontweight='bold')
    
    # ========== PLOT 6: Box Plot Comparison ==========
    ax6 = fig.add_subplot(gs[2, :])
    
    df_box = df.copy()
    df_box['layer_alpha'] = df_box.apply(lambda row: f"L{int(row['layer'])} α={row['alpha']:+.1f}", axis=1)
    
    # Sort by layer and alpha
    layer_alpha_order = []
    for layer in sorted(layers):
        for alpha in sorted(alphas):
            layer_alpha_order.append(f"L{int(layer)} α={alpha:+.1f}")
    
    df_box['layer_alpha'] = pd.Categorical(df_box['layer_alpha'], categories=layer_alpha_order, ordered=True)
    df_box = df_box.sort_values('layer_alpha')
    
    sns.boxplot(data=df_box, x='layer_alpha', y='gpt4_rating', ax=ax6,
               palette='Set3', showmeans=True, meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markersize': 5})
    
    ax6.set_xlabel('Layer × Steering Strength', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Hostility Rating', fontsize=12, fontweight='bold')
    ax6.set_title('Distribution Comparison Across All Conditions', fontsize=14, fontweight='bold')
    ax6.axhline(y=2.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    return fig


def create_layer_specific_plots(df: pd.DataFrame, output_dir: str):
    """Create individual plots for each layer"""
    
    layers = sorted(df['layer'].unique())
    alphas = sorted(df['alpha'].unique())
    
    for layer in layers:
        layer_df = df[df['layer'] == layer].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Layer {int(layer)} Detailed Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Rating distribution by alpha
        ax1 = axes[0, 0]
        layer_df['alpha_str'] = layer_df['alpha'].apply(lambda x: f'α={x:+.1f}')
        sns.violinplot(data=layer_df, x='alpha_str', y='gpt4_rating', ax=ax1,
                      palette='RdYlGn_r', inner='quartile')
        ax1.set_title('Rating Distribution by α', fontweight='bold')
        ax1.set_ylabel('Hostility Rating')
        ax1.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7)
        
        # Plot 2: Scenario-wise response
        ax2 = axes[0, 1]
        
        # Show top 5 most variable scenarios
        scenario_var = layer_df.groupby('scenario')['gpt4_rating'].var().sort_values(ascending=False).head(5)
        top_scenarios = scenario_var.index
        
        for scenario in top_scenarios:
            scenario_data = layer_df[layer_df['scenario'] == scenario].sort_values('alpha')
            ax2.plot(scenario_data['alpha'], scenario_data['gpt4_rating'],
                    marker='o', label=scenario[:30] + '...', linewidth=2, markersize=6)
        
        ax2.set_xlabel('Steering α')
        ax2.set_ylabel('Hostility Rating')
        ax2.set_title('Top 5 Most Variable Scenarios', fontweight='bold')
        ax2.legend(fontsize=7, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mean with confidence interval
        ax3 = axes[1, 0]
        
        summary = layer_df.groupby('alpha')['gpt4_rating'].agg(['mean', 'std', 'count']).reset_index()
        summary['se'] = summary['std'] / np.sqrt(summary['count'])
        summary['ci'] = 1.96 * summary['se']  # 95% CI
        
        ax3.plot(summary['alpha'], summary['mean'], 'o-', linewidth=2.5, 
                markersize=10, color='darkred', label='Mean')
        ax3.fill_between(summary['alpha'], 
                        summary['mean'] - summary['ci'],
                        summary['mean'] + summary['ci'],
                        alpha=0.3, color='red', label='95% CI')
        
        ax3.set_xlabel('Steering α', fontweight='bold')
        ax3.set_ylabel('Mean Hostility Rating', fontweight='bold')
        ax3.set_title('Mean Response with 95% Confidence Interval', fontweight='bold')
        ax3.axhline(y=2.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"Layer {int(layer)} Statistics\n\n"
        
        for alpha in alphas:
            alpha_df = layer_df[layer_df['alpha'] == alpha]
            ratings = alpha_df['gpt4_rating']
            hostile_rate = alpha_df['gpt4_is_hostile'].mean()
            
            stats_text += f"α = {alpha:+.1f}:\n"
            stats_text += f"  Mean: {ratings.mean():.2f} ± {ratings.std():.2f}\n"
            stats_text += f"  Median: {ratings.median():.2f}\n"
            stats_text += f"  Range: [{ratings.min():.1f}, {ratings.max():.1f}]\n"
            stats_text += f"  Hostile rate: {hostile_rate:.1%}\n"
            stats_text += f"  N: {len(ratings)}\n\n"
        
        # Calculate steering effectiveness
        neg_mean = layer_df[layer_df['alpha'] == alphas[0]]['gpt4_rating'].mean()
        pos_mean = layer_df[layer_df['alpha'] == alphas[-1]]['gpt4_rating'].mean()
        delta = pos_mean - neg_mean
        
        stats_text += f"Steering Effectiveness:\n"
        stats_text += f"  Δ = {delta:.2f}\n"
        stats_text += f"  ({pos_mean:.2f} - {neg_mean:.2f})"
        
        ax4.text(0.1, 0.9, stats_text, fontsize=10, verticalalignment='top',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        output_path = Path(output_dir) / f'layer_{int(layer)}_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved: {output_path}")


def generate_summary_report(df: pd.DataFrame, output_dir: str):
    """Generate text summary report"""
    
    layers = sorted(df['layer'].unique())
    alphas = sorted(df['alpha'].unique())
    
    report = []
    report.append("=" * 80)
    report.append("GENERALIZATION TEST SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("📊 OVERALL STATISTICS")
    report.append("-" * 80)
    report.append(f"Total evaluations: {len(df)}")
    report.append(f"Layers tested: {len(layers)} → {layers}")
    report.append(f"Alpha range: {alphas}")
    report.append(f"Novel scenarios: {len(df['scenario'].unique())}")
    report.append("")
    
    # Global statistics by alpha
    report.append("🌐 GLOBAL AVERAGE BY STEERING STRENGTH")
    report.append("-" * 80)
    for alpha in alphas:
        alpha_df = df[df['alpha'] == alpha]
        ratings = alpha_df['gpt4_rating']
        hostile_rate = alpha_df['gpt4_is_hostile'].mean()
        
        report.append(f"α = {alpha:+.1f}:")
        report.append(f"  Mean rating: {ratings.mean():.2f} ± {ratings.std():.2f}")
        report.append(f"  Hostile rate: {hostile_rate:.1%}")
        report.append("")
    
    # Layer ranking by effectiveness
    report.append("🏆 LAYER RANKING BY STEERING EFFECTIVENESS")
    report.append("-" * 80)
    
    layer_effectiveness = []
    for layer in layers:
        layer_df = df[df['layer'] == layer]
        
        neg_ratings = layer_df[layer_df['alpha'] == alphas[0]]['gpt4_rating']
        pos_ratings = layer_df[layer_df['alpha'] == alphas[-1]]['gpt4_rating']
        
        if len(neg_ratings) > 0 and len(pos_ratings) > 0:
            delta = pos_ratings.mean() - neg_ratings.mean()
            layer_effectiveness.append({
                'layer': int(layer),
                'delta': delta,
                'baseline': neg_ratings.mean(),
                'max_steering': pos_ratings.mean()
            })
    
    layer_effectiveness.sort(key=lambda x: x['delta'], reverse=True)
    
    for rank, item in enumerate(layer_effectiveness, 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"#{rank}"
        report.append(f"{medal} Layer {item['layer']}:")
        report.append(f"     Δ = {item['delta']:+.2f}")
        report.append(f"     Baseline (α={alphas[0]}): {item['baseline']:.2f}")
        report.append(f"     Max steering (α={alphas[-1]}): {item['max_steering']:.2f}")
        report.append("")
    
    # Best layer detailed analysis
    if layer_effectiveness:
        best = layer_effectiveness[0]
        report.append("⭐ BEST LAYER DETAILED ANALYSIS")
        report.append("-" * 80)
        report.append(f"Layer: {best['layer']}")
        report.append(f"Steering effectiveness: Δ = {best['delta']:.2f}")
        report.append("")
        
        best_layer_df = df[df['layer'] == best['layer']]
        
        report.append("Performance by alpha:")
        for alpha in alphas:
            alpha_df = best_layer_df[best_layer_df['alpha'] == alpha]
            ratings = alpha_df['gpt4_rating']
            hostile_rate = alpha_df['gpt4_is_hostile'].mean()
            
            report.append(f"  α = {alpha:+.1f}: mean={ratings.mean():.2f}, "
                         f"hostile={hostile_rate:.0%}, n={len(ratings)}")
        
        report.append("")
    
    # Scenario difficulty analysis
    report.append("📋 SCENARIO DIFFICULTY ANALYSIS")
    report.append("-" * 80)
    
    scenario_stats = []
    for scenario in df['scenario'].unique():
        scenario_df = df[df['scenario'] == scenario]
        mean_rating = scenario_df['gpt4_rating'].mean()
        std_rating = scenario_df['gpt4_rating'].std()
        
        scenario_stats.append({
            'scenario': scenario,
            'mean': mean_rating,
            'std': std_rating
        })
    
    scenario_stats.sort(key=lambda x: x['mean'], reverse=True)
    
    report.append("Top 5 most hostile scenarios:")
    for i, item in enumerate(scenario_stats[:5], 1):
        report.append(f"{i}. {item['scenario'][:60]}...")
        report.append(f"   Mean: {item['mean']:.2f} ± {item['std']:.2f}")
    
    report.append("")
    report.append("Top 5 least hostile scenarios:")
    for i, item in enumerate(scenario_stats[-5:], 1):
        report.append(f"{i}. {item['scenario'][:60]}...")
        report.append(f"   Mean: {item['mean']:.2f} ± {item['std']:.2f}")
    
    report.append("")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    output_path = Path(output_dir) / 'generalization_summary_report.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ Summary report saved: {output_path}")
    
    # Also print to console
    print("\n" + report_text)


def main():
    parser = argparse.ArgumentParser(description='Visualize Generalization Results')
    parser.add_argument('--json_path', type=str, required=True,
                       help='Path to step3_generalization JSON file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualizations (default: same as JSON dir)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(Path(args.json_path).parent.parent / 'visualizations')
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("GENERALIZATION RESULTS VISUALIZATION")
    print(f"{'='*70}")
    print(f"Input: {args.json_path}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Load data
    df = load_generalization_data(args.json_path)
    
    # Create comprehensive visualization
    print("\nCreating comprehensive visualization...")
    fig = create_comprehensive_visualization(df, args.output_dir)
    output_path = Path(args.output_dir) / 'generalization_comprehensive.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_path}")
    
    # Create layer-specific plots
    print("\nCreating layer-specific analyses...")
    create_layer_specific_plots(df, args.output_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, args.output_dir)
    
    print(f"\n{'='*70}")
    print("✅ VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"All outputs saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()