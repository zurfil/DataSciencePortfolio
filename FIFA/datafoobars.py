import pandas as pd


def drop_personal_data(df, dropage=True):
    df = df.drop(["Photo", "Flag", "Club Logo", "Special", "International Reputation",
                  "Body Type", "Real Face", "Joined", "Loaned From", "Contract Valid Until",
                  "Release Clause", "ID", "Name", "Nationality", "Club", "Value", "Wage", "Jersey Number",
                  "Height", "Weight", "Marking", "Potential", "Best Position", "Best Overall Rating"],
                 axis=1, inplace=True)
    if dropage:
        df = df.drop(["Age"], axis=1, inplace=True)
    return df


def drop_categorical_data(df, dropposition=True):
    df = df.drop(["Preferred Foot", "Weak Foot", "Skill Moves", "Work Rate", "DefensiveAwareness"],
                 axis=1, inplace=True)
    if dropposition:
        df = df.drop(["Position"], axis=1, inplace=True)
    return df


def drop_goalkeeper_data(df):
    df = df.drop(["GKDiving", "GKHandling","GKKicking", "GKPositioning", "GKReflexes"], axis=1, inplace=True)
    df.drop(df[df["Position"] == "GK"].index, inplace=True)
    return df
