from functions_diversity import make_test_diversity
from functions_quality import make_test_quality
from functions_balanced import make_test_balanced
from get_baselines import get_baselines


# MAIN CONFIG
# Store datasets in /data directory, write names here without the extension type
DATASETS = [
    "2blobs_spray",
    "3blobs",
    "3lines",
    "3rings",
    "5blobs",
    "four_gaussian",
    "image",
    "iris",
    "spiral",
    "two_bananas",
    "dermatology",
    "glass",
    "ionosphere",
    "lymphography",
    "voice",
    "wine",
    "zoo"
]
NOISE_LEVELS = [0.00, 0.02, 0.05, 0.1, 0.15, 0.25]

# TESTING CONFIG
# DATASETS = ["3rings", "5blobs", "glass"]
# NOISE_LEVELS = [0.00, 0.02, 0.05, 0.1]


def main():
    get_baselines(DATASETS, NOISE_LEVELS)
    make_test_quality(DATASETS, NOISE_LEVELS)
    make_test_diversity(DATASETS, NOISE_LEVELS)
    make_test_balanced(DATASETS, NOISE_LEVELS)


if __name__ == "__main__":
    main()
