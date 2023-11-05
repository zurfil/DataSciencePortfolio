
import matplotlib.pyplot as plt
import pandas as pd

DATASETS = [
    "2blobs_spray",
    "3blobs",
    "3lines",
    "3rings",
    "5blobs",
    "image",
    "spiral",
    "two_bananas"
]

fig, axs = plt.subplots(4, 2,  figsize=(8, 16))


for ax, DATASET in zip(axs.flatten(), DATASETS):
    data = pd.read_csv(f"../data/{DATASET}.csv")
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data.iloc[:, 2], cmap='Paired')
    ax.set_title(f"{DATASET} True Labels")


plt.tight_layout()
plt.savefig("true_labels.png")
plt.show()


