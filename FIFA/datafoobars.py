import pandas as pd


def drop_personal_data(df, dropage=True):
    # Personal data can be considered anything outside on-field abilities, for example player's photo.
    if dropage:
        df = df.drop(["Photo", "Flag", "Club Logo", "Special", "International Reputation",
                      "Body Type", "Real Face", "Joined", "Loaned From", "Contract Valid Until",
                      "Release Clause", "ID", "Name", "Nationality", "Club", "Value", "Wage", "Jersey Number",
                      "Height", "Weight", "Marking", "Age"], axis=1, inplace=False)
    elif not dropage:
        df = df.drop(["Photo", "Flag", "Club Logo", "Special", "International Reputation",
                      "Body Type", "Real Face", "Joined", "Loaned From", "Contract Valid Until",
                      "Release Clause", "ID", "Name", "Nationality", "Club", "Value", "Wage", "Jersey Number",
                      "Height", "Weight", "Marking"], axis=1, inplace=False)
    return df


def drop_categorical_data(df):
    df = df.drop(["Preferred Foot", "Weak Foot", "Skill Moves", "Work Rate"], axis=1, inplace=False)
    return df


def drop_goalkeeper_data(df):
    df = df.drop(["GKDiving", "GKHandling","GKKicking", "GKPositioning", "GKReflexes"], axis=1, inplace=False)
    return df
