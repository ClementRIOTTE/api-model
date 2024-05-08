import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib


def ingest_data(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["survived", "pclass", "sex", "age"]]
    df = df.dropna()
    df["sex"] = df["sex"].replace({"male": 0, "female": 1})
    return df


def train_model(de: pd.DataFrame) -> ClassifierMixin:
    model = KNeighborsClassifier(4)
    y = df["survived"]
    x = df.drop("survived", axis=1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    print(f"model score: {score}")
    return model


if __name__ == "__main__":
    df = ingest_data("titanic.xls")
    df = clean_data(df)
    model = train_model(df)
    joblib.dump(model, "model_titanic.joblib")
