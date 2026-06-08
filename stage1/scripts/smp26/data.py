from pathlib import Path
import pandas as pd


def load_train_test(data_dir="video-data"):
    data_dir = Path(data_dir)

    posts_train = pd.read_parquet(data_dir / "posts_train.parquet")
    posts_test = pd.read_parquet(data_dir / "posts_test.parquet")
    users_train = pd.read_parquet(data_dir / "users_train.parquet")
    users_test = pd.read_parquet(data_dir / "users_test.parquet")
    videos_train = pd.read_parquet(data_dir / "videos_train.parquet")
    videos_test = pd.read_parquet(data_dir / "videos_test.parquet")
    labels_train = pd.read_parquet(data_dir / "labels_train.parquet")

    train = posts_train.merge(users_train, on="uid", how="left")
    train = train.merge(videos_train, on=["pid", "uid"], how="left")
    train = train.merge(labels_train[["pid", "popularity"]], on="pid", how="left")

    test = posts_test.merge(users_test, on="uid", how="left")
    test = test.merge(videos_test, on=["pid", "uid"], how="left")

    return train, test
