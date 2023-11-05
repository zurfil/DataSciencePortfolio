import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D


# Strip the standard deviation from the "Normalized Mutual Information" column
def strip_std(df):
    df["Normalized Mutual Information"] = df["Normalized Mutual Information"].astype(str)
    df["Normalized Mutual Information"] = df["Normalized Mutual Information"].str.split("(", expand=True)[0].astype(float)
    return df


# Load data
balanced = pd.read_csv("results_balanced_total.csv")
quality = pd.read_csv("results_quality_total.csv")
diversity = pd.read_csv("results_diversity_total.csv")
baselines = pd.read_csv("baselines.csv")

# Strip std
balanced_df = strip_std(balanced)
quality_df = strip_std(quality)
diversity_df = strip_std(diversity)
baselines_df = strip_std(baselines)

# Extract unique values for datasets, noise types, and algorithms
unique_datasets = balanced_df["Dataset"].unique()
unique_noise_types = balanced_df["Noise Type"].unique()
unique_algorithms = balanced_df["Algorithm"].unique()
linkage_types = ["Single", "Average", "Complete"]

# Create unique color combinations for each Algorithm + Linkage pair
unique_colors = list(mcolors.TABLEAU_COLORS.values())
color_map_unique = {f"{algo}_{linkage}": color for (algo, linkage), color
                    in zip([(a, l) for a in unique_algorithms for l in linkage_types], unique_colors)}


# Create plots for each dataset
for dataset in unique_datasets:
    fig, axarr = plt.subplots(5, 2, figsize=(12, 20))
    # Add an overall title for the entire figure
    fig.suptitle(f"{dataset}", fontsize=26)

    # Combined list of Noise Types and Member Selection Methods
    combinations = [(nt, msm) for nt in unique_noise_types for msm in ["Diversity", "Quality", "Balanced"]]

    # Loop over each combination of Noise Type and Member Selection Method
    for (noise_type, result_type), ax in zip(combinations, axarr.flat[:-1]):
        # Determine which dataframe to use based on the result type
        if result_type == "Balanced":
            result_df = balanced_df
        elif result_type == "Quality":
            result_df = quality_df
        else:
            result_df = diversity_df

        # Extract baseline for the given dataset and noise type
        baseline_data = baselines_df[(baselines_df["Dataset"] == dataset) & (baselines_df["Noise Type"] == noise_type)]

        # Extract data for the given dataset and noise type
        plot_data = result_df[(result_df["Dataset"] == dataset) & (result_df["Noise Type"] == noise_type)]

        # Plot each algorithm's data with different linkages
        for algo in unique_algorithms:
            for linkage in linkage_types:
                algo_data = plot_data[(plot_data["Algorithm"] == algo) & (plot_data["Linkage"] == linkage)]
                label = f"{algo} ({linkage})"
                ax.plot(algo_data["Noise Level"], algo_data["Normalized Mutual Information"], label=label,
                        color=color_map_unique[f"{algo}_{linkage}"], marker="o")

        # Plot baseline last to ensure it is on top
        ax.plot(baseline_data["Noise Level"], baseline_data["Normalized Mutual Information"], color="black",
                linewidth=4, label="Baseline")

        ax.set_title(f"{noise_type}: {result_type}")
        ax.set_xlabel("Noise Level")
        ax.set_ylabel("NMI Score")
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.02, 0.05, 0.10, 0.15, 0.25])
        ax.grid(True)

    # Set the legend in the last subplot
    legend_lines = [Line2D([0], [0], color=color, label=f"{algo} ({linkage})") for (algo, linkage), color in
                    zip([(a, l) for a in unique_algorithms for l in linkage_types], unique_colors)]
    legend_lines.append(Line2D([0], [0], color="k", linestyle="-",
                               linewidth=4, label="Baseline"))
    axarr[-1, -1].legend(handles=legend_lines, loc="center", fontsize="medium")
    axarr[-1, -1].axis("off")

    # Adjust layout to avoid overlap and add title to the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f"{dataset}.png")

# I am embarrassed to admit how much time it took me to make these graphs but at least they look nice now :)
# Possibly the most annoying part of this dissertation project
