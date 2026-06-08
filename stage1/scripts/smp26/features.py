import numpy as np
import pandas as pd


def prepare_features(df, train_min_time=None):
    df = df.copy()
    df["post_time"] = pd.to_datetime(df["post_time"], errors="coerce")
    df["dt"] = df["post_time"]

    if train_min_time is None:
        train_min_time = df["post_time"].min()

    df["hour"] = df["dt"].dt.hour.fillna(0).astype(float)
    df["dow"] = df["dt"].dt.dayofweek.fillna(0).astype(float)
    df["day"] = df["dt"].dt.day.fillna(1).astype(float)
    df["month"] = df["dt"].dt.month.fillna(1).astype(float)
    df["is_weekend"] = (df["dow"] >= 5).astype(float)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Text fields
    df["content_text"] = df["post_content"].fillna("").astype(str)
    df["suggested_text"] = df["post_suggested_words"].apply(
        lambda x: " ".join(x) if isinstance(x, (list, np.ndarray))
        else str(x) if pd.notna(x) else ""
    )
    df["music_text"] = df["music_title"].fillna("").astype(str)
    df["full_text"] = df["content_text"] + " " + df["suggested_text"] + " " + df["music_text"]

    df["post_content_len"] = df["content_text"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) else 0
    )
    df["post_content_tokens"] = df["content_text"].apply(
        lambda x: len(str(x).replace(",", " ").split())
    )
    df["post_content_unique_tokens"] = df["content_text"].apply(
        lambda x: len(set(str(x).replace(",", " ").split()))
    )
    df["suggested_text_len"] = df["suggested_text"].apply(
        lambda x: len(x.split()) if isinstance(x, str) and x else 0
    )
    df["suggested_text_tokens"] = df["suggested_text_len"]
    df["suggested_text_unique_tokens"] = df["suggested_text"].apply(
        lambda x: len(set(x.split())) if isinstance(x, str) and x else 0
    )
    df["full_text_len"] = df["full_text"].apply(
        lambda x: len(x.split()) if isinstance(x, str) and x else 0
    )
    df["full_text_tokens"] = df["full_text_len"]
    df["full_text_unique_tokens"] = df["full_text"].apply(
        lambda x: len(set(x.split())) if isinstance(x, str) and x else 0
    )

    def count_suggested(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        if isinstance(x, (list, np.ndarray)):
            return len(x)
        return 0
    df["suggested_count"] = df["post_suggested_words"].apply(count_suggested)
    df["is_original_sound"] = df["suggested_count"].apply(lambda x: 1.0 if x == 0 else 0.0)

    df["content_text_len"] = df["content_text"].str.len()
    df["content_text_len"] = pd.to_numeric(df["content_text_len"], errors="coerce").fillna(0)

    # User stats
    for col in ["user_following_count", "user_follower_count", "user_likes_count",
                "user_video_count", "user_digg_count", "user_heart_count", "user_friend_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[f"log1p_{col}"] = np.log1p(df[col].fillna(0))

    # Fix column names that might differ from expected
    if "user_likes_count" in df.columns:
        df["user_likes_count_fixed"] = df["user_likes_count"]
        df["log1p_user_likes_count_fixed"] = np.log1p(df["user_likes_count"].fillna(0))
    if "user_heart_count" in df.columns:
        df["user_heart_count_fixed"] = df["user_heart_count"]
        df["log1p_user_heart_count_fixed"] = np.log1p(df["user_heart_count"].fillna(0))

    # Video stats
    for col in ["video_height", "video_width", "video_duration", "music_duration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "video_height" in df.columns and "video_width" in df.columns:
        df["aspect_ratio"] = df["video_width"] / df["video_height"].replace(0, np.nan)
        df["log1p_pixel_count"] = np.log1p(
            df["video_height"].fillna(0) * df["video_width"].fillna(0)
        )

    if "video_duration" in df.columns and "music_duration" in df.columns:
        df["music_to_video_duration"] = (
            df["music_duration"] / df["video_duration"].replace(0, np.nan)
        )
        df["duration_gap"] = df["video_duration"] - df["music_duration"]

    # Ratio features
    follower = df.get("user_follower_count", pd.Series(1, index=df.index)).fillna(1).replace(0, 1)
    likes = df.get("user_likes_count", pd.Series(0, index=df.index)).fillna(0)
    heart = df.get("user_heart_count", pd.Series(0, index=df.index)).fillna(0)
    digg = df.get("user_digg_count", pd.Series(0, index=df.index)).fillna(0)
    following = df.get("user_following_count", pd.Series(1, index=df.index)).fillna(1).replace(0, 1)
    video_count = df.get("user_video_count", pd.Series(1, index=df.index)).fillna(1).replace(0, 1)

    df["likes_per_video"] = likes / video_count
    df["heart_per_video"] = heart / video_count
    df["followers_per_video"] = follower / video_count
    df["digg_per_video"] = digg / video_count
    df["heart_per_follower"] = heart / follower
    df["likes_per_follower"] = likes / follower
    df["following_follower_ratio"] = following / follower
    df["follower_per_following"] = follower / following
    df["likes_per_digg"] = likes / digg.replace(0, 1)
    df["followers_per_digg"] = follower / digg.replace(0, 1)

    for c in ["likes_per_video", "heart_per_video", "followers_per_video",
              "digg_per_video", "heart_per_follower", "likes_per_follower",
              "following_follower_ratio", "follower_per_following",
              "likes_per_digg", "followers_per_digg"]:
        df[f"log1p_{c}"] = np.log1p(df[c].fillna(0))

    return df


def numeric_feature_columns(df):
    """Return numeric feature columns matching the expected schema."""
    preferred = {
        "user_following_count", "user_follower_count", "user_likes_count",
        "user_video_count", "user_digg_count", "user_heart_count",
        "user_friend_count", "video_height", "video_width",
        "video_duration", "music_duration",
        "log1p_user_follower_count", "log1p_user_likes_count",
        "log1p_user_likes_count_fixed", "log1p_user_video_count",
        "log1p_user_digg_count", "log1p_user_heart_count",
        "log1p_user_heart_count_fixed", "log1p_user_friend_count",
        "likes_per_video", "heart_per_video", "followers_per_video",
        "digg_per_video", "heart_per_follower", "likes_per_follower",
        "following_follower_ratio", "follower_per_following",
        "likes_per_digg", "followers_per_digg",
        "log1p_likes_per_video", "log1p_heart_per_video",
        "log1p_followers_per_video", "log1p_digg_per_video",
        "log1p_heart_per_follower", "log1p_likes_per_follower",
        "log1p_following_follower_ratio", "log1p_follower_per_following",
        "log1p_likes_per_digg", "log1p_followers_per_digg",
        "content_text_len", "full_text_len", "suggested_count",
        "is_original_sound",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos",
    }
    skip = {"pid", "uid", "vid", "post_time", "video_path",
            "post_content", "post_suggested_words", "dt", "split",
            "popularity", "oof_pred", "month_key",
            "content_text", "suggested_text", "music_text", "full_text",
            "asr_text", "ocr_text", "blip_caption"}
    cols = [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]
    # Return in expected order
    result = [c for c in cols if c in preferred]
    result += [c for c in cols if c not in preferred]
    return result
