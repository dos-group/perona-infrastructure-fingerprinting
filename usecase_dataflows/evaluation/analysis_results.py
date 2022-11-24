import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

from evaluation import load_and_prepare

warnings.filterwarnings("ignore", module="matplotlib\..*")

root_dir = Path(__file__).absolute().parents[1]

cherrypick = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ0_cherrypick", "multiple_soo_cherrypick.csv"))
arrow = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ0_arrow", "multiple_soo_arrow.csv"))
cherrypick_ext = load_and_prepare(
    os.path.join(root_dir, "artifacts", "RQ0_cherrypickext", "multiple_soo_cherrypickext.csv"))
arrow_ext = load_and_prepare(os.path.join(root_dir, "artifacts", "RQ0_arrowext", "multiple_soo_arrowext.csv"))


def cost_and_time_at_stopping_condition(df, id_, n):
    for _, row in df.iterrows():
        if row.acqf_value < 0.1 and (n is None or row.profiling_counter >= n):
            break
    return {
        id_: row[id_],
        "timeout": row.profiling_counter_not_completed_not_abandon / row.profiling_counter,
        "total_search_cost": row.total_search_cost,
        "total_search_time": row.total_search_time,
        "best_cost_found": row.best_cost_found,
    }


def cost_and_time_df(g, id_, n=None):
    result = []
    for name, group in g:
        result.append(cost_and_time_at_stopping_condition(group, id_, n))
    return pd.DataFrame(result)

### Use Case Dataflows: Figure 1

df = pd.concat((
    cherrypick,
    arrow,
    cherrypick_ext,
    arrow_ext,
), axis=0, ignore_index=True)

df.loc[(df["identifier"] == "ArrowOptimizer"), "approach"] = "Arrow"
df.loc[(df["identifier"] == "CherryPickOptimizer"), "approach"] = "CherryPick"
df.loc[(df["identifier"] == "ArrowExtOptimizer"), "approach"] = "Arrow w/ Perona"
df.loc[(df["identifier"] == "CherryPickExtOptimizer"), "approach"] = "CherryPick w/ Perona"

sns.set_style("whitegrid")

f, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(5, 5))
f.tight_layout(h_pad=2.5, w_pad=0)

sns.boxplot(data=df[df["optimizer_strategy_sub"] == "NaiveBO"], x="profiling_counter", y="best_cost_found",
            hue="approach", ax=ax1,
            showfliers=False, palette=["#968D88", *sns.color_palette("flare")[:1]])

sns.boxplot(data=df[df["optimizer_strategy_sub"] == "AugmentedBO"], x="profiling_counter", y="best_cost_found",
            hue="approach", ax=ax2,
            showfliers=False, palette=["#968D88", *sns.color_palette("flare")[:1]])

lim = (1, 2.5)

ax1.set_ylim(lim)
ax1.set_ylabel("Difference to optimal cost")
ax1.set_yticks([1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
ax1.set_yticklabels(["0%", "25%", "50%", "75%", "100%", "125%", "150%"])
ax1.set_xlabel("")
ax1.set_title("NaiveBO (CherryPick)")
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles=handles, labels=labels)

ax2.set_ylim(lim)
ax2.set_ylabel("Difference to optimal cost")
ax2.set_xlabel("Number of profiling runs")
ax2.set_title("AugmentedBO (Arrow)")
handles, labels = ax2.get_legend_handles_labels()
ax2.legend(handles=handles, labels=labels)

plt.savefig(os.path.join(root_dir, "artifacts", "usecase_dataflows1.pdf"), dpi=300, bbox_inches='tight')
