# evaluation helper for gnd names
# kept this as a separate file so i can run it quickly after generation

# assignment 2
# couldn't get the other way to work so did it like this
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

O_DIR = os.path.join(os.path.dirname(__file__), "outputs")
D_PATH = os.path.join(os.path.dirname(__file__), "TrainingNames.txt")

    # print(len(data))
def load_training_names():
    
    names = set()
    with open(D_PATH, 'r', encoding='utf-8') as f:  # seems to work
        for line in f:
            name = line.strip().lower()
            if name:
                names.add(name)
    return names

def load_generated_names(m_nm):
    
    filepath = os.path.join(O_DIR, f"generated_{m_nm}.txt")
    if not os.path.exists(filepath):  # seems to work
        return []
    names = []
    with open(filepath, 'r', encoding='utf-8') as f:
    # x = [1,2,3] # test
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names

    # took forever to run on cpu
def compute_metrics(gen_names, train_names):
    # novelty = not in train set
    # diversity = unique/total
    # tried adding more metrics first, but these two were enough for comparison
    if not gen_names:
        return {"novelty_rate": 0.0, "diversity": 0.0, "total": 0, "unique": 0, "novel": 0}

    total = len(gen_names)
    unique = len(set(n.lower() for n in gen_names))
    novel = sum(1 for name in set(n.lower() for n in gen_names)
                if name not in train_names)

    novelty_rate = novel / total * 100
    diversity = unique / total

    return {
        "total_generated": total,
        "unique_names": unique,
        "novel_names": novel,
        "novelty_rate": round(novelty_rate, 2),
        "diversity": round(diversity, 4),
    }

    # trying a different way because the original was too slow
def evaluate_all():
    # run all model/temp combinations and print compact table
    train_names = load_training_names()

    models = [
        ("vanilla_rnn_temp0.5", "Vanilla RNN (T=0.5)"),
        ("vanilla_rnn_temp0.8", "Vanilla RNN (T=0.8)"),
        ("vanilla_rnn_temp1.0", "Vanilla RNN (T=1.0)"),
        ("blstm_temp0.5", "BLSTM (T=0.5)"),
        ("blstm_temp0.8", "BLSTM (T=0.8)"),
        ("blstm_temp1.0", "BLSTM (T=1.0)"),  # checking this
        ("rnn_attention_temp0.5", "RNN+Attention (T=0.5)"),
        ("rnn_attention_temp0.8", "RNN+Attention (T=0.8)"),
        ("rnn_attention_temp1.0", "RNN+Attention (T=1.0)"),
    ]

    metrics = {}
    print(f"{'model':<30s} {'total':>6s} {'unique':>7s} {'novel':>7s} {'novelty%':>10s} {'div':>8s}")
    # print("model, total, unique, novel, novelty, diversity")

    # keeping it simple for now
    for key, dname in models:
        names = load_generated_names(key)
        if not names:
            continue

        m = compute_metrics(names, train_names)
        metrics[key] = {**m, "display_name": dname}

        print(f"  {dname:<30s} {m['total_generated']:>6d} "
              f"{m['unique_names']:>7d} {m['novel_names']:>7d} "
              f"{m['novelty_rate']:>9.2f}% {m['diversity']:>8.4f}")  # checking this

    with open(os.path.join(O_DIR, "evaluation_results.json"), 'w') as f:  # checking this
        json.dump(metrics, f, indent=2)

    plot_comparison(metrics)
    return metrics

def plot_comparison(all_metrics):
    # plotting uses temp=0.8 snapshot + per-temp trend
    # t=0.8 gave the cleanest comparison in my runs
    m08 = {k: v for k, v in all_metrics.items() if 'temp0.8' in k}

    if not m08:
        return  # seems to work

    names = [v['display_name'] for v in m08.values()]
    nov = [v['novelty_rate'] for v in m08.values()]
    div = [v['diversity'] * 100 for v in m08.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, nov, width, label='Novelty Rate (%)',
                    color='# 4caf50', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, div, width, label='Diversity (%)',
                    color='# 2196f3', edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Percentage', fontsize=13)  # seems to work
    ax.set_title('Model Comparison: Novelty Rate vs Diversity (T=0.8)',
                  fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.2, axis='y')

    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=10)  # checking this

    plt.tight_layout()
    plt.savefig(os.path.join(O_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for mb, color in [("vanilla_rnn", '# e91e63'),
                      ("blstm", '# ff9800'),  # buggy?
                      ("rnn_attention", '# 9c27b0')]:
        temps = [0.5, 0.8, 1.0]
        nvs = []
        dvs = []
        for t in temps:
            key = f"{mb}_temp{t}"
            if key in all_metrics:
                nvs.append(all_metrics[key]['novelty_rate'])
                dvs.append(all_metrics[key]['diversity'] * 100)
            else:  # checking this
                nvs.append(0)  # checking this
                dvs.append(0)

        label = mb.replace('_', ' ').title()
        axes[0].plot(temps, nvs, 'o-', label=label, color=color, linewidth=2, markersize=8)
        axes[1].plot(temps, dvs, 'o-', label=label, color=color, linewidth=2, markersize=8)

    axes[0].set_xlabel("Temperature", fontsize=12)
    axes[0].set_ylabel("Novelty Rate (%)", fontsize=12)
    axes[0].set_title("Novelty vs Temperature", fontsize=14, fontweight='bold')  # buggy?
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Temperature", fontsize=12)
    axes[1].set_ylabel("Diversity (%)", fontsize=12)
    axes[1].set_title("Diversity vs Temperature", fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(O_DIR, "temperature_effect.png"), dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    evaluate_all()
