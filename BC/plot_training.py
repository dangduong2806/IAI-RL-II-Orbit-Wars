import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def parse_train_log(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by architecture blocks
    blocks = re.split(r'={50,}', text)
    
    architectures = {}
    current_name = None
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # Check for architecture name
        name_match = re.search(r'Training Architecture:\s*(.+)', block)
        if name_match:
            current_name = name_match.group(1).strip()
            architectures[current_name] = {
                'train_loss': [], 'val_loss': [],
                'train_acc': [], 'val_acc': [],
                'train_f1': [], 'val_f1': [],
                'test_acc': None, 'test_f1': None
            }
            continue
        
        if current_name is None:
            continue
        
        # Parse epoch lines
        epoch_pattern = r'Epoch\s+\d+\s*\|\s*Train\s*\[L:\s*([\d.]+),\s*Acc:\s*([\d.]+),\s*F1:\s*([\d.]+)\]\s*\|\s*Val\s*\[L:\s*([\d.]+),\s*Acc:\s*([\d.]+),\s*F1:\s*([\d.]+)\]'
        for match in re.finditer(epoch_pattern, block):
            tl, ta, tf, vl, va, vf = [float(x) for x in match.groups()]
            architectures[current_name]['train_loss'].append(tl)
            architectures[current_name]['train_acc'].append(ta)
            architectures[current_name]['train_f1'].append(tf)
            architectures[current_name]['val_loss'].append(vl)
            architectures[current_name]['val_acc'].append(va)
            architectures[current_name]['val_f1'].append(vf)
        
        # Parse test results
        test_match = re.search(r"FINAL TEST RESULTS FOR '(\w+)'\s*-\s*Accuracy:\s*([\d.]+)\s*\|\s*F1-Score:\s*([\d.]+)", block)
        if test_match:
            architectures[current_name]['test_acc'] = float(test_match.group(2))
            architectures[current_name]['test_f1'] = float(test_match.group(3))
    
    return architectures

def plot_all_architectures(architectures, save_path):
    colors = {
        'Shallow [64, 32]': ('#FF6B6B', '#FF6B6B'),      # Red
        'Medium [128, 64]': ('#4ECDC4', '#4ECDC4'),       # Teal
        'Deep [256, 128, 64]': ('#45B7D1', '#45B7D1')     # Blue
    }
    
    labels = {
        'Shallow [64, 32]': 'Shallow [64→32]',
        'Medium [128, 64]': 'Medium [128→64]',
        'Deep [256, 128, 64]': 'Deep [256→128→64]'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('So sánh Kiến trúc MLP - Behavior Cloning Training', fontsize=16, fontweight='bold', y=1.02)
    
    metrics = [
        ('Loss (BCE)', 'train_loss', 'val_loss'),
        ('Accuracy', 'train_acc', 'val_acc'),
        ('F1-Score', 'train_f1', 'val_f1')
    ]
    
    for ax_idx, (ylabel, train_key, val_key) in enumerate(metrics):
        ax = axes[ax_idx]
        
        for arch_name, data in architectures.items():
            color = colors.get(arch_name, ('#999', '#999'))[0]
            label = labels.get(arch_name, arch_name)
            epochs = range(1, len(data[train_key]) + 1)
            
            # Train: nét liền (solid), không marker
            ax.plot(epochs, data[train_key], linestyle='-', color=color, alpha=0.6, linewidth=1.5, label=f'{label} (Train)')
            # Val: nét đứt (dashed), có marker
            ax.plot(epochs, data[val_key], linestyle='--', marker='o', color=color, linewidth=2, markersize=4, label=f'{label} (Val)')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=14, fontweight='bold')
        
        # Chỉ hiển thị legend cho subplot đầu tiên hoặc chia thành 2 cột để gọn hơn
        if ax_idx == 1: # Để legend ở giữa cho cân hoặc ở mỗi subplot tùy thích. Ở đây để mỗi subplot
            ax.legend(fontsize=8, loc='best', ncol=2)
        else:
            ax.legend(fontsize=8, loc='best')
            
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
    
    # Add test results annotation
    test_text = "Final Test Results:\n"
    for arch_name, data in architectures.items():
        label = labels.get(arch_name, arch_name)
        if data['test_acc'] is not None:
            test_text += f"  {label}: Acc={data['test_acc']:.4f}, F1={data['test_f1']:.4f}\n"
    
    fig.text(0.5, -0.06, test_text.strip(), ha='center', fontsize=11,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")

if __name__ == "__main__":
    data = parse_train_log(r"d:\Kaggle_final\train.txt")
    print(f"Parsed {len(data)} architectures: {list(data.keys())}")
    for name, d in data.items():
        print(f"  {name}: {len(d['train_loss'])} epochs, Test Acc={d['test_acc']}, Test F1={d['test_f1']}")
    plot_all_architectures(data, r"d:\Kaggle_final\all_architectures_comparison.png")
