import os
import json
import tarfile
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ================================
# Load experiment data from files
# ================================
def load_all_experiment_data(data_dir="result/data", file_suffix=""):
    all_data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.json') and file_suffix in filename:
            with open(os.path.join(data_dir, filename), 'r') as f:
                all_data.append(json.load(f))
    return all_data

# ================================
# Plot attention flows for multiple examples
# ================================
def plot_overlayed_graphs(
    all_experiments, flow_code, flow_name,
    output_dir="result/overlayed", p_name="overlay",
    layer_interval=1, curve_kind='cubic', point_stride=1
):
    import matplotlib as mpl
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab10.colors  # 10 default color palette

    for i, exp in enumerate(all_experiments):
        prompt_name = exp["prompt_name"]
        x_full = np.array(exp["selected_layers"])
        y_full = np.array(exp["flows"][flow_code])

        x = x_full[::point_stride]
        y = y_full[::point_stride]

        if len(x) < 2:
            print(f"Skipping {prompt_name} for flow '{flow_code}' due to insufficient data points.")
            continue

        x_smooth = np.linspace(x.min(), x.max(), 300)

        try:
            spl = interp1d(x, y, kind=curve_kind)
        except Exception as e:
            print(f"Warning: Failed {curve_kind} interpolation for {prompt_name} ({flow_code}): {e}")
            spl = interp1d(x, y, kind='linear')

        y_smooth = spl(x_smooth)

        def normalize_and_scale(data):
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val == min_val:
                return np.full_like(data, 0.475)
            norm = (data - min_val) / (max_val - min_val)
            return 0.1 + norm * 0.75  # scale to [0.1, 0.85] for better visual separation

        y_scaled = normalize_and_scale(y_smooth)

        plt.plot(
            x_smooth,
            y_scaled,
            label=prompt_name,
            color=colors[i % len(colors)],
            linewidth=2.5
        )

    plt.xlabel("Layer", fontsize=24)
    plt.ylabel(f"{flow_name} Information Flow", fontsize=24)
    plt.yticks(np.linspace(0.0, 1.0, 6))
    plt.title("")
    plt.ylim(0.0, 1.0)

    plt.legend(
        loc='upper right',
        ncol=1,
        fontsize=30,
        frameon=True,
        handlelength=1.5,
        columnspacing=1.2,
        borderpad=0.6
    )

    filename = f"{output_dir}/IF_{flow_code}_{layer_interval}_{p_name}_full.pdf"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {filename}")

# ================================
# Main execution logic
# ================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--f_name', type=str, required=True, help="Suffix used during experiment output naming")
    parser.add_argument('--layer_interval', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default="result/data")
    parser.add_argument('--output_dir', type=str, default="result/overlayed")
    parser.add_argument('--flows', type=str, nargs='*', default=None)
    parser.add_argument('--curve_kind', type=str, default='cubic', choices=['linear', 'cubic', 'quadratic'])
    parser.add_argument('--point_stride', type=int, default=5)
    args = parser.parse_args()

    all_data = load_all_experiment_data(data_dir=args.data_dir, file_suffix=args.f_name)
    if not all_data:
        print("No matching data found.")
        return

    # Define mapping from flow codes to human-readable labels
    flow_names = {
        "ca": "Context→Answer", "ac": "Answer→Context",
        "qc": "Question→Context", "cq": "Context→Question",
        "cp": "Context→Prompt Template", "pc": "Prompt Template→Context",
        "aq": "Answer→Question", "qa":
