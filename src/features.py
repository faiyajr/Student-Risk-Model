import pandas as pd

def prepare_features(df: pd.DataFrame):
    """
    Converts raw student data into model-ready features.
    """

    features = df[[
        "attendance_rate",
        "avg_assignment_score",
        "avg_exam_score",
        "late_submissions"
    ]]

    labels = df["risk_status"]

    return features, labels
