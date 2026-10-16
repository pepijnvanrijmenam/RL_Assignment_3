from test_script import test_script, bar_plot
from shared_functions import SEEDS
import pandas as pd
import os

os.makedirs("results", exist_ok=True)

STRATEGIES = ["baseline", "uniform", "curriculum", "curriculum_g_080"]

def main():
    for strategy in STRATEGIES:
        dfs = []
        for seed in SEEDS:
            path = f"{strategy}_seed{seed}.pth"
            test_script(path, strategy, seed)
            
            # Load the Excel file just created by test_script
            excel_path = os.path.join("results", f"experiment_results_{strategy}_{seed}.xlsx")
            dfs.append(pd.read_excel(excel_path))
        
        # Average the results over all seeds
        avg_df = pd.concat(dfs).mean().to_frame().T
        bar_plot(avg_df.to_dict(orient="records"), name=strategy, seed="all_seeds_avg")
        avg_df.to_excel(os.path.join("results", f"experiment_results_{strategy}_avg.xlsx"), index=False)

    print("Evaluation finished")

if __name__ == "__main__":
    main()
