import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import confusion_matrix


def drop_personal_data(df, dropage=True):
    df = df.drop(["Photo", "Flag", "Club Logo", "Special", "International Reputation",
                  "Body Type", "Real Face", "Joined", "Loaned From", "Contract Valid Until",
                  "Release Clause", "ID", "Name", "Nationality", "Club", "Value", "Wage", "Jersey Number",
                  "Height", "Weight", "Marking", "Potential", "Best Position", "Best Overall Rating"],
                 axis=1, inplace=False)
    if dropage:
        df = df.drop(["Age"], axis=1, inplace=False)
    return df


def drop_categorical_data(df, dropposition=True):
    df = df.drop(["Preferred Foot", "Weak Foot", "Skill Moves", "Work Rate", "DefensiveAwareness"],
                 axis=1, inplace=False)
    if dropposition:
        df = df.drop(["Position"], axis=1, inplace=False)
    return df


def drop_goalkeeper_data(df):
    df = df.drop(["GKDiving", "GKHandling", "GKKicking", "GKPositioning", "GKReflexes"], axis=1, inplace=False)
    df.drop(df[df["Position"] == "GK"].index, inplace=True)
    return df


def extract_positions(df):
    positions = df["Position"].str.extract(r">\s*([^<]+)")
    return positions


def import_train_data():
    fifa17_raw_data = pd.read_csv("data/FIFA17_official_data.csv")
    fifa18_raw_data = pd.read_csv("data/FIFA18_official_data.csv")
    fifa19_raw_data = pd.read_csv("data/FIFA19_official_data.csv")
    fifa20_raw_data = pd.read_csv("data/FIFA20_official_data.csv")
    fifa21_raw_data = pd.read_csv("data/FIFA21_official_data.csv")

    df = pd.concat([fifa17_raw_data, fifa18_raw_data,
                    fifa19_raw_data, fifa20_raw_data,
                    fifa21_raw_data], ignore_index=True)
    return df


def map_positions(df):
    mapping = {
        'Defender': ['LB', 'LCB', 'LWB', 'RB', 'RCB', 'RWB', 'CB'],
        'Midfielder': ['CAM', 'CDM', 'CM', 'LAM', 'LCM', 'LDM', 'LM',
                       'RAM', 'RCM', 'RDM', 'RM'],
        'Striker': ['CF', 'LF', 'LS', 'RF', 'RS', 'ST', 'RW', 'LW']
    }
    df = df.map(
        {pos: category for category, positions
         in mapping.items() for pos in positions})
    return df


def make_conf_matrix(labels, predictions):
    plt.figure(figsize=(10, 8))
    conf_matrix = confusion_matrix(labels, predictions)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")
    plt.show()
    return


def drop_positions(df, positions):
    for position in positions:
        df = (df.drop(df[df["Position"] == position].index))
    return df


def bin_overall_to_quality(df):
    bins = [-1, 64, 74, 100]
    labels = ["Bronze", "Silver", "Gold"]
    df["Rank"] = pd.cut(df["Overall"], bins=bins, labels=labels, include_lowest=True)
    df = df.drop(["Overall"], axis=1, inplace=False)
    return df
