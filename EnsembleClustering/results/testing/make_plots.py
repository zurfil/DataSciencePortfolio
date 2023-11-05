import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from time import sleep


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

# Add titles to the saved plots
for dataset in unique_datasets:
    for noise_type in unique_noise_types:
        sleep(6)  # Sleep to prevent matplotlib.pyplot request limit
        fig, axarr = plt.subplots(2, 2, figsize=(12, 10))
        plt.suptitle(f"{dataset} with {noise_type} Noise")

        # Extract baseline for the given dataset and noise type
        baseline_data = baselines_df[(baselines_df["Dataset"] == dataset) & (baselines_df["Noise Type"] == noise_type)]

        # Loop over each type of result
        for ax, result_df, result_type in zip(axarr.flat[:-1], [balanced_df, quality_df, diversity_df],
                                              ["Balanced", "Quality", "Diversity"]):
            # Extract data for the given dataset and noise type
            plot_data = result_df[(result_df["Dataset"] == dataset) & (result_df["Noise Type"] == noise_type)]

            # Plot each algorithm's data with different linkages
            for algo in unique_algorithms:
                for linkage in linkage_types:
                    algo_data = plot_data[(plot_data["Algorithm"] == algo) & (plot_data["Linkage"] == linkage)]
                    label = f"{algo} ({linkage})"
                    ax.plot(algo_data["Noise Level"], algo_data["Normalized Mutual Information"], label=label,
                            color=color_map_unique[f"{algo}_{linkage}"], marker="o")

            # Plot baseline last to ensure it is on top, and make it thicker
            ax.plot(baseline_data["Noise Level"], baseline_data["Normalized Mutual Information"], color="black",
                    linewidth=4, label="Baseline")

            ax.set_title(f"{result_type}")
            ax.set_xlabel("Noise Level")
            ax.set_ylabel("NMI Score")
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.02, 0.05, 0.10])
            ax.grid(True)

        # Set the legend in the last subplot
        legend_lines = [Line2D([0], [0], color=color, label=f"{algo} ({linkage})") for (algo, linkage), color
                        in zip([(a, l) for a in unique_algorithms for l in linkage_types], unique_colors)]
        legend_lines.append(Line2D([0], [0], color="k", linestyle="-",
                                   linewidth=4, label="Baseline"))
        axarr[-1, -1].legend(handles=legend_lines, loc="center", fontsize="x-large")
        axarr[-1, -1].axis("off")

        # Adjust layout to avoid overlap and add title to the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.savefig(f"{dataset} with {noise_type} Noise.png")

