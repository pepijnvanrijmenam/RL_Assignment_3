import os
import numpy as np
import pandas as pd
from scipy import stats
from shared_functions import SEEDS

ALPHA = 0.05

def load_strategy_data(strategy):
    rows = []
    sorted_avg_cols = None

    for seed in SEEDS:
        path = os.path.join("results", f"experiment_results_{strategy}_{seed}.xlsx")
        df = pd.read_excel(path)
        
        avg_columns = []
        for col in df.columns:
            if col.startswith("Avg_"):
                avg_columns.append(col)
                
        sorted_avg_cols = sorted(avg_columns, key=lambda x: float(x.split("_")[1]))
        
        flattened_row_values = df[sorted_avg_cols].to_numpy().flatten()
        rows.append(flattened_row_values)

    lengths_list = []
    for col in sorted_avg_cols:
        length_val = float(col.split("_")[1])
        lengths_list.append(length_val)

    lengths = np.array(lengths_list)
    return np.array(rows), lengths

def main():
    baseline, lengths = load_strategy_data("baseline")
    uniform, _ = load_strategy_data("uniform")
    curriculum, _ = load_strategy_data("curriculum")
    curriculum_g_080, _ = load_strategy_data("curriculum_g_080")

    results = {}

    # H1: Uniform vs Baseline
    # Paired one-sided t-test
    u_mean = uniform.mean(axis=0)
    b_mean = baseline.mean(axis=0)
    t_h1, p_h1 = stats.ttest_rel(u_mean, b_mean, alternative="greater")
    
    longer_then_1 = lengths > 1.0
    t_h1_long, p_h1_long = stats.ttest_rel(u_mean[longer_then_1], b_mean[longer_then_1], alternative="greater")
    
    results["H1"] = {
        "mean_uniform": float(u_mean.mean()),
        "mean_baseline": float(b_mean.mean()),
        "t_stat": float(t_h1),
        "p_val": float(p_h1),
        "mean_uniform_long": float(u_mean[longer_then_1].mean()),
        "poles_mean_baseline_long": float(b_mean[longer_then_1].mean()),
        "t_stat_long": float(t_h1_long),
        "p_val_long": float(p_h1_long),
        "reject_null": bool(p_h1 < ALPHA ),
        "reject_null_long": bool(p_h1_long < ALPHA)
    }

    # H2: Curriculum vs Uniform
    # Paired one-sided t-test
    c_mean = curriculum.mean(axis=0)

    extreems = (lengths < 0.6) | (lengths > 1.6)
    t_h2_ext, p_h2_ext = stats.ttest_rel(c_mean[extreems], u_mean[extreems], alternative="greater")

    results["H2"] = {
        "mean_curriculum_extremes": float(c_mean[extreems].mean()),
        "mean_uniform_extremes": float(u_mean[extreems].mean()),
        "t_stat_extremes": float(t_h2_ext),
        "p_val_extremes": float(p_h2_ext),
        "reject_null": bool(p_h2_ext < ALPHA)
    }

    # H3: curriculum_g_080 vs Curriculumin
    # One-sided paired Wilcoxon
    g_080_mean = curriculum_g_080.mean(axis=0)

    p2_threshold = 1000  # survived wind
    p3_threshold = 1500  # survived increasing force fore 500 steps

    p2_c80 = (curriculum_g_080 > p2_threshold).mean(axis=0)
    p2_c = (curriculum > p2_threshold).mean(axis=0)
    p3_c80 = (curriculum_g_080 > p3_threshold).mean(axis=0)
    p3_c = (curriculum > p3_threshold).mean(axis=0)

    w_p2, pw_p2 = stats.wilcoxon(p2_c80, p2_c, alternative="greater", zero_method="zsplit")
    w_p3, pw_p3 = stats.wilcoxon(p3_c80, p3_c, alternative="greater", zero_method="zsplit")
    
    results["H3"] = {
        "overall_mean_gamma_080": float(g_080_mean.mean()),
        "overall_mean_gamma_099": float(c_mean.mean()),
        "phase2_survival_rate_080": float(p2_c80.mean()),
        "phase2_survival_rate_099": float(p2_c.mean()),
        "wilcoxon_phase2": float(w_p2),
        "p_val_wilcoxon_phase2": float(pw_p2),
        "phase3_survival_rate_080": float(p3_c80.mean()),
        "phase3_survival_rate_099": float(p3_c.mean()),
        "wilcoxon_phase3": float(w_p3),
        "p_val_wilcoxon_phase3": float(pw_p3),
        "reject_null": bool(pw_p2 < ALPHA and pw_p3 < ALPHA)
    }

    for h, data in results.items():
        print(h)
        for i, j in data.items():
            print(f"{i}: {j}")
        print()

    return results

if __name__ == "__main__":
    main()