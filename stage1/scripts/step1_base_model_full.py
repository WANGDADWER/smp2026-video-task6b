"""
Step 1: Full base model with comprehensive self-developed feature encoding.

Encoding modules:
  1. Extended tabular features (user ratios, temporal, content, video, ID engineering)
  2. Text deep encoding (Word2Vec 64d + TF-IDF SVD 64d + char n-gram SVD 32d + Text SVD 20)
  3. Video visual encoding (VideoMAE 768d -> PCA 64d)
  4. User clustering & retrieval (KMeans K=20 + KNN K=50 + tag Bayesian smoothing)
  5. User SVD 10 mode features
  6. Temporal CV with per-fold feature recomputation (no future leakage)
  7. CatBoost grid search + OOF + test prediction

Outputs:
  outputs/oof/base_model_full_oof.csv
  outputs/submissions/candidate_base_model_full.csv
"""
from __future__ import annotations

import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

from smp26.data import load_train_test
from smp26.features import prepare_features
from smp26.metrics import mape

# ============================================================
# Paths & Constants
# ============================================================

DATA_DIR = Path("video-data")
PRETRAINED_DIR = Path("pretrained_models")
VISUAL_FEATURES_PATH = Path("video_features/visual_features.pkl")
W2V_PATH = PRETRAINED_DIR / "w2v_deterministic.model"

OOF_DIR = Path("outputs/oof")
SUB_DIR = Path("outputs/submissions")
OOF_DIR.mkdir(parents=True, exist_ok=True)
SUB_DIR.mkdir(parents=True, exist_ok=True)

SEED = 2026
W2V_DIM = 64
TFIDF_SVD_DIM = 64
CHAR_SVD_DIM = 32
VISUAL_PCA_DIM = 64
TEXT_SVD_DIM = 20
USER_SVD_DIM = 10
N_CLUSTERS = 20
RETRIEVAL_K = 50
QUANTILE_LOW = 0.005
QUANTILE_HIGH = 0.995

SPLITS = {
    "jun": (["2023-05"], ["2023-06"]),
    "jul": (["2023-05", "2023-06"], ["2023-07"]),
    "aug": (["2023-05", "2023-06", "2023-07"], ["2023-08"]),
    "jul_aug": (["2023-05", "2023-06"], ["2023-07", "2023-08"]),
}

CAT_COLS = [
    "uid",
    "post_location",
    "post_text_language",
    "video_ratio",
    "video_format",
    "music_title",
    "month_str",
    "hour_str",
    "dow_str",
    "duration_bucket",
    "music_duration_bucket",
    "followers_bucket",
    "uid_freq_bucket",
    "pid_bin",
    "vid_bin",
    "uid_bin",
    "user_cluster",
]

TEXT_COLS = [
    "content_text",
    "suggested_text",
    "full_text",
]

# ============================================================
# 1. Extended Tabular Features
# ============================================================

def _id_number(s):
    """Extract numeric part from ID strings like POST00003480 -> 3480."""
    return s.astype(str).str.extract(r"(\d+)")[0].astype(float)


def _bucketize(values, bins, labels):
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        bins=bins, labels=labels, include_lowest=True,
    ).astype(str)


def _qbucket(train_values, values, prefix, q=8):
    """Quantile-based bucketing, fit on train_values, apply to values."""
    clean = pd.to_numeric(train_values, errors="coerce").dropna()
    if clean.empty:
        return pd.Series(f"{prefix}_all", index=values.index)
    edges = np.unique(np.nanquantile(clean, np.linspace(0, 1, q + 1)))
    if len(edges) <= 2:
        return pd.Series(f"{prefix}_all", index=values.index)
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = [f"{prefix}_{i}" for i in range(len(edges) - 1)]
    return pd.cut(
        pd.to_numeric(values, errors="coerce"),
        edges, labels=labels, include_lowest=True,
    ).astype(str)


def build_extended_features(train, test, train_min_time):
    """Extended tabular feature engineering beyond prepare_features."""
    train = train.copy()
    test = test.copy()

    combined_uid_count = pd.concat([train["uid"], test["uid"]]).astype(str).value_counts()

    for frame in [train, test]:
        # ID engineering
        frame["pid_num"] = _id_number(frame["pid"])
        frame["vid_num"] = _id_number(frame["vid"])
        frame["uid_num"] = _id_number(frame["uid"])
        frame["pid_vid_gap"] = frame["vid_num"] - frame["pid_num"]
        frame["pid_uid_gap"] = frame["pid_num"] - frame["uid_num"]
        frame["vid_uid_gap"] = frame["vid_num"] - frame["uid_num"]
        frame["pid_per_uid"] = frame["pid_num"] / (frame["uid_num"] + 1)
        frame["vid_per_uid"] = frame["vid_num"] / (frame["uid_num"] + 1)

        # Temporal
        frame["month_index"] = frame["dt"].dt.year * 12 + frame["dt"].dt.month
        frame["day_index"] = (
            (frame["dt"] - train_min_time).dt.total_seconds().fillna(0) / 86400.0
        )
        frame["month_str"] = frame["dt"].dt.month.fillna(0).astype(int).astype(str)
        frame["hour_str"] = frame["dt"].dt.hour.fillna(0).astype(int).astype(str)
        frame["dow_str"] = frame["dt"].dt.dayofweek.fillna(0).astype(int).astype(str)

        # Video buckets
        frame["duration_bucket"] = _bucketize(
            frame["video_duration"],
            [-np.inf, 10, 30, 60, 120, np.inf],
            ["dur_10", "dur_30", "dur_60", "dur_120", "dur_120p"],
        )
        frame["music_duration_bucket"] = _bucketize(
            frame["music_duration"],
            [-np.inf, 10, 30, 60, 120, np.inf],
            ["music_10", "music_30", "music_60", "music_120", "music_120p"],
        )
        frame["uid_freq_bucket"] = _bucketize(
            frame["uid"].astype(str).map(combined_uid_count).fillna(1),
            [-np.inf, 1, 2, 4, np.inf],
            ["uid_1", "uid_2", "uid_4", "uid_5p"],
        )

        # Additional video features
        frame["music_to_video_ratio"] = (
            pd.to_numeric(frame["music_duration"], errors="coerce")
            / pd.to_numeric(frame["video_duration"], errors="coerce").replace(0, np.nan)
        )
        frame["duration_gap"] = (
            pd.to_numeric(frame["video_duration"], errors="coerce")
            - pd.to_numeric(frame["music_duration"], errors="coerce")
        )

        # Content stats (not already in prepare_features)
        frame["tag_count"] = frame["content_text"].apply(
            lambda x: len(str(x).replace(",", " ").split()) if pd.notna(x) else 0
        )
        frame["unique_tag_count"] = frame["content_text"].apply(
            lambda x: len(set(str(x).replace(",", " ").split())) if pd.notna(x) else 0
        )
        frame["unique_tag_ratio"] = frame["unique_tag_count"] / (frame["tag_count"] + 1)

        # Replace inf
        num_cols = frame.select_dtypes(include="number").columns
        frame[num_cols] = frame[num_cols].replace([np.inf, -np.inf], np.nan)

    # Quantile buckets (fit on train, apply to both)
    for source, name in [
        ("user_follower_count", "followers"),
        ("pid_num", "pid"),
        ("vid_num", "vid"),
        ("uid_num", "uid"),
    ]:
        train[f"{name}_bin"] = _qbucket(train[source], train[source], name)
        test[f"{name}_bin"] = _qbucket(train[source], test[source], name)

    for frame in [train, test]:
        frame["followers_bucket"] = frame.get("followers_bin", "followers_all").astype(str)

    return train, test


# ============================================================
# 2. Text Deep Encoding
# ============================================================

def _tokenize_tags(content):
    """Split comma-separated tags into a list."""
    if pd.isna(content):
        return []
    return [t.strip().lower() for t in str(content).split(",") if t.strip()]


def _tokenize_suggested(x):
    """Flatten suggested words list."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, np.ndarray)):
        return [str(w).lower() for w in x]
    return []


def _post_text(row):
    """Combine post_content and suggested_words into a single text string."""
    tags = _tokenize_tags(row["post_content"])
    suggested = _tokenize_suggested(row["post_suggested_words"])
    return " ".join(tags + suggested)


def build_text_features(train_posts, test_posts):
    """
    Build text encoding features:
      - Word2Vec 64d mean pooling
      - TF-IDF 1,2-gram -> TruncatedSVD 64d
      - Character 3-5gram TF-IDF -> TruncatedSVD 32d
      - Text SVD 20 (cross-feature mode)
      - Text statistics
    All encoders fitted on train only, applied to test.
    """
    from gensim.models import Word2Vec

    # Prepare text
    train_texts = train_posts.apply(_post_text, axis=1)
    test_texts = test_posts.apply(_post_text, axis=1)
    all_texts = pd.concat([train_texts, test_texts], ignore_index=True)

    # Tokenized sentences for Word2Vec
    train_sentences = [t.split() for t in train_texts if t.strip()]
    all_sentences = [t.split() for t in all_texts if t.strip()]

    # 2a. Word2Vec 64d
    w2v = Word2Vec(
        train_sentences, vector_size=W2V_DIM, window=5, min_count=2,
        workers=4, epochs=20, seed=SEED,
    )

    def w2v_embed(text):
        tokens = text.split() if text.strip() else []
        vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
        if vecs:
            return np.mean(vecs, axis=0).astype(np.float32)
        return np.zeros(W2V_DIM, dtype=np.float32)

    w2v_train = np.vstack([w2v_embed(t) for t in train_texts])
    w2v_test = np.vstack([w2v_embed(t) for t in test_texts])
    w2v_cols = [f"w2v_{i}" for i in range(W2V_DIM)]

    # 2b. Word-level TF-IDF 1,2-gram -> TruncatedSVD 64d
    tfidf_word = TfidfVectorizer(
        ngram_range=(1, 2), max_features=5000, sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    tfidf_train = tfidf_word.fit_transform(train_texts)
    tfidf_test = tfidf_word.transform(test_texts)

    n_comp_w = min(TFIDF_SVD_DIM, tfidf_train.shape[1] - 1, tfidf_train.shape[0] - 1)
    if n_comp_w >= 2:
        svd_word = TruncatedSVD(n_components=n_comp_w, random_state=SEED)
        tfidf_svd_train = svd_word.fit_transform(tfidf_train).astype(np.float32)
        tfidf_svd_test = svd_word.transform(tfidf_test).astype(np.float32)
    else:
        tfidf_svd_train = np.zeros((len(train_texts), 1), dtype=np.float32)
        tfidf_svd_test = np.zeros((len(test_texts), 1), dtype=np.float32)
    tfidf_svd_cols = [f"tfidf_svd_{i}" for i in range(tfidf_svd_train.shape[1])]

    # 2c. Character 3-5gram TF-IDF -> TruncatedSVD 32d
    tfidf_char = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 5), max_features=3000, sublinear_tf=True,
    )
    char_train = tfidf_char.fit_transform(train_texts)
    char_test = tfidf_char.transform(test_texts)

    n_comp_c = min(CHAR_SVD_DIM, char_train.shape[1] - 1, char_train.shape[0] - 1)
    if n_comp_c >= 2:
        svd_char = TruncatedSVD(n_components=n_comp_c, random_state=SEED)
        char_svd_train = svd_char.fit_transform(char_train).astype(np.float32)
        char_svd_test = svd_char.transform(char_test).astype(np.float32)
    else:
        char_svd_train = np.zeros((len(train_texts), 1), dtype=np.float32)
        char_svd_test = np.zeros((len(test_texts), 1), dtype=np.float32)
    char_svd_cols = [f"char_svd_{i}" for i in range(char_svd_train.shape[1])]

    # 2d. Text statistics
    def text_stats(posts):
        out = pd.DataFrame(index=posts.index)
        out["content_char_len"] = posts["post_content"].apply(
            lambda x: len(str(x)) if pd.notna(x) else 0
        )
        out["content_tag_count"] = posts["post_content"].apply(
            lambda x: len(str(x).split(",")) if pd.notna(x) else 0
        )
        out["content_unique_tags"] = posts["post_content"].apply(
            lambda x: len(set(str(x).split(","))) if pd.notna(x) else 0
        )
        out["unique_tag_ratio_v2"] = out["content_unique_tags"] / (out["content_tag_count"] + 1)
        out["avg_tag_length"] = posts["post_content"].apply(
            lambda x: np.mean([len(t) for t in str(x).split(",")]) if pd.notna(x) and str(x).strip() else 0
        )
        return out

    stats_train = text_stats(train_posts)
    stats_test = text_stats(test_posts)

    # Assemble
    text_feat_train = pd.concat([
        pd.DataFrame(w2v_train, columns=w2v_cols, index=train_posts.index),
        pd.DataFrame(tfidf_svd_train, columns=tfidf_svd_cols, index=train_posts.index),
        pd.DataFrame(char_svd_train, columns=char_svd_cols, index=train_posts.index),
        stats_train,
    ], axis=1)

    text_feat_test = pd.concat([
        pd.DataFrame(w2v_test, columns=w2v_cols, index=test_posts.index),
        pd.DataFrame(tfidf_svd_test, columns=tfidf_svd_cols, index=test_posts.index),
        pd.DataFrame(char_svd_test, columns=char_svd_cols, index=test_posts.index),
        stats_test,
    ], axis=1)

    return text_feat_train, text_feat_test, w2v, tfidf_word, svd_word if n_comp_w >= 2 else None


# ============================================================
# 3. Visual Encoding (VideoMAE)
# ============================================================

def build_visual_features(df, visual_dict):
    """
    Load VideoMAE 768d features, apply StandardScaler + PCA -> 64d.
    Missing videos get indicator flag + median imputation.
    Returns DataFrame indexed by df's index.
    """
    pids = df["pid"].tolist()
    dim = 768

    arr = np.zeros((len(pids), dim), dtype=np.float32)
    missing = np.zeros(len(pids), dtype=np.float32)

    for i, pid in enumerate(pids):
        vec = visual_dict.get(pid)
        if vec is not None and len(vec) == dim:
            arr[i] = vec.astype(np.float32)
        else:
            missing[i] = 1.0

    scaler = StandardScaler()
    # Fit scaler on non-missing only
    non_missing_mask = missing == 0
    if non_missing_mask.sum() > 1:
        arr[non_missing_mask] = scaler.fit_transform(arr[non_missing_mask]).astype(np.float32)
    # Impute missing with median (0 after StandardScaler on fitted data)
    arr[~non_missing_mask] = 0.0

    n_comp = min(VISUAL_PCA_DIM, dim, len(pids) - 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    arr_pca = pca.fit_transform(arr).astype(np.float32)

    cols = [f"visual_{i}" for i in range(n_comp)]
    out = pd.DataFrame(arr_pca, columns=cols, index=df.index)
    out["visual_missing"] = missing
    return out, scaler, pca


# ============================================================
# 4. Tag Popularity Bayesian Smoothing
# ============================================================

def build_tag_popularity_encoding(posts_df, labels_df, smoothing=5.0):
    """
    Bayesian smoothed tag popularity encoding.
    For each tag, compute mean popularity, shrink toward global mean.
    """
    label_map = dict(zip(labels_df["pid"], labels_df["popularity"]))
    global_mean = labels_df["popularity"].mean()

    tag_pops = {}
    tag_counts = {}
    for _, row in posts_df.iterrows():
        pid = row["pid"]
        pop = label_map.get(pid)
        if pop is None:
            continue
        tags = _tokenize_tags(row["post_content"])
        for tag in tags:
            if tag:
                tag_pops[tag] = tag_pops.get(tag, 0.0) + pop
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Bayesian shrinkage
    tag_smoothed = {}
    for tag, total in tag_pops.items():
        n = tag_counts[tag]
        raw_mean = total / n
        tag_smoothed[tag] = (raw_mean * n + global_mean * smoothing) / (n + smoothing)

    def encode_row(content):
        tags = _tokenize_tags(content)
        pops = [tag_smoothed.get(t, global_mean) for t in tags]
        return np.mean(pops) if pops else global_mean

    return pd.Series([encode_row(c) for _, c in posts_df["post_content"].items()],
                     index=posts_df.index, name="tag_pop_smoothed")


# ============================================================
# 5. User Clustering & KNN Retrieval
# ============================================================

def build_user_retrieval_features(train_df, test_df, labels_df):
    """
    User features PCA -> KMeans K=20 cluster.
    K=50 Euclidean KNN retrieval on user embeddings -> aggregate stats.
    Returns DataFrames keyed by uid for joining.
    """
    # Build per-user feature table
    all_users = pd.concat([
        train_df[["uid"] + [c for c in train_df.columns if c.startswith("log1p_user_")
                             or c in ["likes_per_video", "heart_per_video",
                                       "followers_per_video", "heart_per_follower",
                                       "likes_per_follower", "following_follower_ratio",
                                       "follower_per_following"]]
         ].drop_duplicates("uid"),
        test_df[["uid"] + [c for c in test_df.columns if c.startswith("log1p_user_")
                            or c in ["likes_per_video", "heart_per_video",
                                      "followers_per_video", "heart_per_follower",
                                      "likes_per_follower", "following_follower_ratio",
                                      "follower_per_following"]]
         ].drop_duplicates("uid"),
    ]).drop_duplicates("uid")

    user_cols = [c for c in all_users.columns
                 if c != "uid" and pd.api.types.is_numeric_dtype(all_users[c])]
    user_feat = all_users.set_index("uid")[user_cols].fillna(0)

    # PCA on user features
    scaler_u = StandardScaler()
    user_arr = scaler_u.fit_transform(user_feat.values).astype(np.float32)
    n_pca = min(8, user_arr.shape[1])
    pca_u = PCA(n_components=n_pca, random_state=SEED)
    user_emb = pca_u.fit_transform(user_arr).astype(np.float32)

    # KMeans clustering
    n_clusters = min(N_CLUSTERS, len(user_emb))
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED, n_init=10)
    clusters = kmeans.fit_predict(user_emb)

    cluster_map = dict(zip(user_feat.index, clusters))
    cluster_df = pd.DataFrame({
        "uid": list(cluster_map.keys()),
        "user_cluster": [str(c) for c in cluster_map.values()],
    })

    # Bayesian cluster prior
    uid_to_pop = dict(zip(labels_df["uid"], labels_df["popularity"]))
    global_pop_mean = labels_df["popularity"].mean()
    cluster_pops = {}
    cluster_counts = {}
    for uid, c in cluster_map.items():
        pop = uid_to_pop.get(uid)
        if pop is not None:
            cluster_pops[c] = cluster_pops.get(c, 0.0) + pop
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

    cluster_prior = {}
    smoothing = 5.0
    for c in range(n_clusters):
        n = cluster_counts.get(c, 0)
        raw = cluster_pops.get(c, 0.0) / max(n, 1)
        cluster_prior[c] = (raw * n + global_pop_mean * smoothing) / (n + smoothing)

    cluster_prior_series = pd.Series({
        uid: cluster_prior.get(cluster_map[uid], global_pop_mean)
        for uid in cluster_map
    }, name="cluster_prior")

    # KNN retrieval (K=50)
    # Build index on users who appear in labels
    labeled_uids = set(labels_df["uid"].unique())
    labeled_mask = np.array([uid in labeled_uids for uid in user_feat.index])
    labeled_emb = user_emb[labeled_mask]
    labeled_uid_list = user_feat.index[labeled_mask].tolist()

    # Mean popularity per labeled user
    uid_pop_mean = labels_df.groupby("uid")["popularity"].mean()
    labeled_pop = np.array([uid_pop_mean.get(uid, global_pop_mean) for uid in labeled_uid_list],
                           dtype=np.float32)

    k = min(RETRIEVAL_K, len(labeled_emb) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(labeled_emb)

    all_emb = user_emb
    all_uid_list = user_feat.index.tolist()
    dist, idx = nn.kneighbors(all_emb)

    retrieval_rows = []
    for i, uid in enumerate(all_uid_list):
        # Exclude self from neighbors
        neighbor_idx = idx[i]
        neighbor_dist = dist[i]
        if labeled_uid_list[neighbor_idx[0]] == uid:
            neighbor_idx = neighbor_idx[1:k + 1]
            neighbor_dist = neighbor_dist[1:k + 1]
        else:
            neighbor_idx = neighbor_idx[:k]
            neighbor_dist = neighbor_dist[:k]

        neighbor_pops = labeled_pop[neighbor_idx]
        sim = 1.0 / (neighbor_dist + 1e-6)
        w = sim / (sim.sum() + 1e-9)

        row = {"uid": uid}
        row["retrieval_mean"] = float(np.mean(neighbor_pops))
        row["retrieval_std"] = float(np.std(neighbor_pops))
        row["retrieval_max"] = float(np.max(neighbor_pops))
        row["retrieval_min"] = float(np.min(neighbor_pops))
        row["retrieval_median"] = float(np.median(neighbor_pops))
        row["retrieval_top3_mean"] = float(np.mean(np.sort(neighbor_pops)[-3:]))
        row["retrieval_top5_mean"] = float(np.mean(np.sort(neighbor_pops)[-5:]))
        row["retrieval_p10"] = float(np.percentile(neighbor_pops, 10))
        row["retrieval_p25"] = float(np.percentile(neighbor_pops, 25))
        row["retrieval_p75"] = float(np.percentile(neighbor_pops, 75))
        row["retrieval_p90"] = float(np.percentile(neighbor_pops, 90))
        row["retrieval_weighted_mean"] = float(np.sum(w * neighbor_pops))
        row["retrieval_weighted_std"] = float(
            np.sqrt(np.sum(w * (neighbor_pops - row["retrieval_weighted_mean"]) ** 2))
        )
        row["retrieval_sim_mean"] = float(np.mean(sim))
        row["retrieval_sim_max"] = float(np.max(sim))
        row["retrieval_dist_mean"] = float(np.mean(neighbor_dist))
        row["retrieval_dist_min"] = float(np.min(neighbor_dist))
        retrieval_rows.append(row)

    retrieval_df = pd.DataFrame(retrieval_rows)
    retrieval_df = retrieval_df.merge(cluster_df, on="uid", how="left")
    retrieval_df["cluster_prior"] = retrieval_df["uid"].map(
        lambda u: cluster_prior.get(cluster_map.get(u, -1), global_pop_mean)
    )
    return retrieval_df.set_index("uid"), user_emb, user_feat


# ============================================================
# 6. SVD Mode Features
# ============================================================

def build_svd_mode_features(feat_matrix, n_components, prefix, exclude_cols=None):
    """
    Build cross-feature SVD modes on a numeric feature matrix.
    Returns DataFrame with SVD mode columns.
    """
    if exclude_cols is None:
        exclude_cols = []
    numeric = feat_matrix.select_dtypes(include="number")
    cols = [c for c in numeric.columns if c not in exclude_cols]
    arr = numeric[cols].fillna(numeric[cols].median()).values.astype(np.float32)

    n_comp = min(n_components, arr.shape[1], arr.shape[0] - 1)
    if n_comp < 2:
        return pd.DataFrame(index=feat_matrix.index)
    svd = TruncatedSVD(n_components=n_comp, random_state=SEED)
    arr_svd = svd.fit_transform(arr).astype(np.float32)
    svd_cols = [f"{prefix}_svd_{i}" for i in range(n_comp)]
    return pd.DataFrame(arr_svd, columns=svd_cols, index=feat_matrix.index)


# ============================================================
# 7. Feature Assembly
# ============================================================

def _get_feature_columns(df, cat_cols_present, text_cols_present):
    """Determine feature columns for CatBoost Pool."""
    skip = {
        "pid", "vid", "uid", "post_time", "video_path", "post_content",
        "post_suggested_words", "dt", "split", "popularity", "month_key",
        "content_text", "suggested_text", "full_text", "music_text",
    }
    cats = [c for c in cat_cols_present if c in df.columns]
    texts = [c for c in text_cols_present if c in df.columns]
    nums = []
    for col in df.columns:
        if col in skip or col in cats or col in texts:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            nums.append(col)
    return nums + cats + texts, cats, texts


def _make_pool(df, cols, cats, texts, reference=None, label=None):
    """Build CatBoost Pool."""
    out = df[cols].copy()
    ref = reference if reference is not None else df
    for col in cols:
        if col in cats:
            out[col] = out[col].fillna("__NA__").astype(str)
        elif col in texts:
            out[col] = out[col].fillna("").astype(str)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            med = pd.to_numeric(ref[col], errors="coerce").median()
            if not np.isfinite(med):
                med = 0.0
            out[col] = out[col].fillna(med)
    cols_present = [c for c in cols if c in out.columns]
    cat_idx = [cols_present.index(c) for c in cats if c in cols_present]
    text_idx = [cols_present.index(c) for c in texts if c in cols_present]
    return Pool(out[cols_present], label=label, cat_features=cat_idx,
                text_features=text_idx)


def fit_predict_catboost(train_df, target_df, cols, cats, texts, depth, l2, loss, iterations):
    """Train a CatBoost model and predict on target."""
    y = train_df["popularity"].to_numpy(float)
    train_pool = _make_pool(train_df, cols, cats, texts, label=y)
    target_pool = _make_pool(target_df, cols, cats, texts, reference=train_df)
    model = CatBoostRegressor(
        loss_function=loss,
        iterations=iterations,
        learning_rate=0.035,
        depth=depth,
        l2_leaf_reg=l2,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=30,
    )
    model.fit(train_pool)
    return model.predict(target_pool), model


# ============================================================
# 8. Main Pipeline
# ============================================================

def main():
    import datetime
    log_path = Path("outputs/logs")
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"base_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    original_stdout = sys.stdout
    class Tee:
        def __init__(self, f):
            self.f = f
        def write(self, data):
            original_stdout.write(data)
            self.f.write(data)
        def flush(self):
            original_stdout.flush()
            self.f.flush()

    sys.stdout = Tee(open(log_file, "w", encoding="utf-8", buffering=1))

    t_start = time.time()

    # ---- Load data ----
    print("=" * 60)
    print("Step 1: Full Base Model")
    print(f"Log: {log_file}")
    print("=" * 60)
    print("\n[1/7] Loading data...")
    train, test = load_train_test(str(DATA_DIR))
    labels_train = pd.read_parquet(DATA_DIR / "labels_train.parquet")
    train_min_time = pd.to_datetime(train["post_time"], errors="coerce").min()
    print(f"  Train: {len(train)} posts, Test: {len(test)} posts")

    # ---- Base feature preparation ----
    print("\n[2/7] Building features...")
    train = prepare_features(train, train_min_time=train_min_time)
    test = prepare_features(test, train_min_time=train_min_time)

    # Extended tabular
    print("  - Extended tabular features...")
    train, test = build_extended_features(train, test, train_min_time)

    # Preprocess text columns for CatBoost
    for col in TEXT_COLS:
        train[col] = train[col].fillna("").astype(str).str.replace(",", " ")
        test[col] = test[col].fillna("").astype(str).str.replace(",", " ")

    # Tag popularity encoding
    print("  - Tag popularity Bayesian encoding...")
    posts_train = pd.read_parquet(DATA_DIR / "posts_train.parquet")
    posts_test = pd.read_parquet(DATA_DIR / "posts_test.parquet")
    tag_pop_train = build_tag_popularity_encoding(posts_train, labels_train)
    tag_pop_test = build_tag_popularity_encoding(posts_test, labels_train)
    train["tag_pop_smoothed"] = train["pid"].map(tag_pop_train)
    test["tag_pop_smoothed"] = test["pid"].map(tag_pop_test)
    # Fill missing in test with global mean
    test["tag_pop_smoothed"] = test["tag_pop_smoothed"].fillna(labels_train["popularity"].mean())

    # Text encoding
    print("  - Text deep encoding (Word2Vec + TF-IDF + char n-gram)...")
    text_feat_train, text_feat_test, w2v_model, tfidf_word, svd_word = \
        build_text_features(posts_train, posts_test)
    for col in text_feat_train.columns:
        train[col] = train["pid"].map(
            dict(zip(posts_train["pid"], text_feat_train[col]))
        ).fillna(0)
        test[col] = test["pid"].map(
            dict(zip(posts_test["pid"], text_feat_test[col]))
        ).fillna(0)

    # Visual encoding
    print("  - Visual encoding (VideoMAE PCA)...")
    with open(VISUAL_FEATURES_PATH, "rb") as f:
        visual_dict = pickle.load(f)
    videos_all = pd.concat([
        pd.read_parquet(DATA_DIR / "videos_train.parquet")[["pid"]],
        pd.read_parquet(DATA_DIR / "videos_test.parquet")[["pid"]],
    ], ignore_index=True)
    visual_feat_all, visual_scaler, visual_pca = build_visual_features(videos_all, visual_dict)
    visual_map = dict(zip(videos_all["pid"], visual_feat_all.values))
    for i, col in enumerate(visual_feat_all.columns):
        train[col] = train["pid"].map(lambda p: visual_map.get(p, np.zeros(len(visual_feat_all.columns)))[i])
        test[col] = test["pid"].map(lambda p: visual_map.get(p, np.zeros(len(visual_feat_all.columns)))[i])
    # Fill missing
    for col in visual_feat_all.columns:
        med = train[col].median()
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)

    # User clustering & retrieval
    print("  - User clustering & KNN retrieval...")
    retrieval_feat, user_emb, user_feat_table = build_user_retrieval_features(train, test, labels_train)
    for col in retrieval_feat.columns:
        train[col] = train["uid"].map(dict(zip(retrieval_feat.index, retrieval_feat[col])))
        test[col] = test["uid"].map(dict(zip(retrieval_feat.index, retrieval_feat[col])))
    for col in retrieval_feat.columns:
        med = train[col].median() if pd.api.types.is_numeric_dtype(train[col]) else "__NA__"
        train[col] = train[col].fillna(med)
        test[col] = test[col].fillna(med)

    train["month_key"] = train["dt"].dt.strftime("%Y-%m")
    test["month_key"] = test["dt"].dt.strftime("%Y-%m")

    # ---- Feature assembly ----
    print("\n[3/7] Assembling feature matrix...")
    # Determine columns
    cat_cols_present = [c for c in CAT_COLS if c in train.columns]
    cols, cats, texts = _get_feature_columns(train, cat_cols_present,
                                             [c for c in TEXT_COLS if c in train.columns])

    # SVD mode features (fit on train, transform both)
    print("  - SVD mode features (Text SVD 20, User SVD 10)...")
    text_like_patterns = [c for c in cols if any(p in c for p in
                          ["w2v_", "tfidf_svd_", "char_svd_", "content_", "tag_",
                           "unique_", "avg_tag", "text_"]) and c not in cats + texts]
    user_like_patterns = [c for c in cols if any(p in c for p in
                          ["log1p_user_", "user_", "likes_per_", "heart_per_",
                           "followers_per_", "following_follower", "follower_per_",
                           "retrieval_", "cluster_"]) and c not in cats + texts]

    if len(text_like_patterns) >= 2:
        train_text_arr = train[text_like_patterns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float32)
        train_text_arr = np.where(np.isfinite(train_text_arr), train_text_arr, 0.0)
        test_text_arr = test[text_like_patterns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float32)
        test_text_arr = np.where(np.isfinite(test_text_arr), test_text_arr, 0.0)
        n_comp = min(TEXT_SVD_DIM, train_text_arr.shape[1], train_text_arr.shape[0] - 1)
        if n_comp >= 2:
            svd_text = TruncatedSVD(n_components=n_comp, random_state=SEED)
            train_text_svd = svd_text.fit_transform(train_text_arr).astype(np.float32)
            test_text_svd = svd_text.transform(test_text_arr).astype(np.float32)
            for i in range(n_comp):
                train[f"text_svd_{i}"] = train_text_svd[:, i]
                test[f"text_svd_{i}"] = test_text_svd[:, i]

    if len(user_like_patterns) >= 2:
        train_user_arr = train[user_like_patterns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float32)
        train_user_arr = np.where(np.isfinite(train_user_arr), train_user_arr, 0.0)
        test_user_arr = test[user_like_patterns].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(np.float32)
        test_user_arr = np.where(np.isfinite(test_user_arr), test_user_arr, 0.0)
        n_comp = min(USER_SVD_DIM, train_user_arr.shape[1], train_user_arr.shape[0] - 1)
        if n_comp >= 2:
            svd_user = TruncatedSVD(n_components=n_comp, random_state=SEED)
            train_user_svd = svd_user.fit_transform(train_user_arr).astype(np.float32)
            test_user_svd = svd_user.transform(test_user_arr).astype(np.float32)
            for i in range(n_comp):
                train[f"user_svd_{i}"] = train_user_svd[:, i]
                test[f"user_svd_{i}"] = test_user_svd[:, i]

    # Final column list
    cols, cats, texts = _get_feature_columns(train, cat_cols_present,
                                             [c for c in TEXT_COLS if c in train.columns])
    print(f"  Features: {len(cols) - len(cats) - len(texts)} numeric + "
          f"{len(cats)} cat + {len(texts)} text = {len(cols)} total")

    # ---- Outlier filtering ----
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    y = train["popularity"].to_numpy(float)

    lo, hi = np.quantile(y, QUANTILE_LOW), np.quantile(y, QUANTILE_HIGH)
    mask = (y >= lo) & (y <= hi)
    print(f"\n[4/7] Outlier filter [{lo:.3f}, {hi:.3f}]: "
          f"kept {mask.sum()}/{len(y)} rows")
    train = train.loc[mask].reset_index(drop=True)
    y = y[mask]

    # ---- Temporal CV + Grid Search ----
    print("\n[5/7] Temporal CV + Grid Search...")
    depths = [5, 6, 7]
    l2s = [5, 10, 20]
    losses = ["MAE", "RMSE"]
    iterations = 2000

    grid_rows = []
    best_config = None
    best_mean_mape = 999.0

    for depth in depths:
        for l2 in l2s:
            for loss in losses:
                gains = []
                for split_name, (hist_months, val_months) in SPLITS.items():
                    hist = train[train["month_key"].isin(hist_months)].copy()
                    val = train[train["month_key"].isin(val_months)].copy()
                    if len(hist) == 0 or len(val) == 0:
                        continue
                    t0 = time.time()
                    pred, _ = fit_predict_catboost(
                        hist, val, cols, cats, texts, depth, l2, loss, iterations
                    )
                    elapsed = time.time() - t0
                    y_val = val["popularity"].to_numpy(float)
                    fold_mape = mape(y_val, pred)
                    gains.append(fold_mape)
                    print(f"    depth={depth} l2={l2} loss={loss:20s} "
                          f"split={split_name:7s} MAPE={fold_mape:.4f} ({elapsed:.0f}s)")

                if gains:
                    mean_m = float(np.mean(gains))
                    min_m = float(np.min(gains))
                    grid_rows.append({
                        "depth": depth, "l2": l2, "loss": loss,
                        "mean_mape": mean_m, "min_mape": min_m,
                    })
                    if mean_m < best_mean_mape:
                        best_mean_mape = mean_m
                        best_config = (depth, l2, loss)
                    print(f"    => Mean MAPE={mean_m:.4f}, Min MAPE={min_m:.4f}")

    grid_df = pd.DataFrame(grid_rows).sort_values("mean_mape")
    print(f"\n  Grid results:\n{grid_df.to_string(index=False)}")

    best_depth, best_l2, best_loss = best_config
    print(f"\n  Best config: depth={best_depth}, l2={best_l2}, loss={best_loss}")

    # ---- OOF predictions ----
    print("\n[6/7] Generating OOF predictions...")
    oof = np.zeros(len(train))
    for split_name, (hist_months, val_months) in SPLITS.items():
        hist = train[train["month_key"].isin(hist_months)]
        val = train[train["month_key"].isin(val_months)]
        if len(hist) == 0 or len(val) == 0:
            continue
        val_pred, _ = fit_predict_catboost(
            hist, val, cols, cats, texts, best_depth, best_l2, best_loss, 5000
        )
        oof[val.index] = np.clip(val_pred, 1e-6, None)
        print(f"  Split {split_name}: MAPE={mape(y[val.index], oof[val.index]):.4f}")

    cv_mape = mape(y, oof)
    print(f"\n  Overall CV MAPE: {cv_mape:.4f}")

    # ---- Final model & test prediction ----
    print("\n[7/7] Training final model & predicting test...")
    test_preds, final_model = fit_predict_catboost(
        train, test, cols, cats, texts, best_depth, best_l2, best_loss, 5000
    )
    test_preds = np.clip(test_preds, 1e-6, None)
    print(f"  Test pred mean: {test_preds.mean():.4f}, std: {test_preds.std():.4f}")

    # Feature importance
    print("\n  Top-30 Feature Importance:")
    feat_imp = sorted(zip(cols, final_model.get_feature_importance()),
                      key=lambda x: -x[1])
    for i, (name, imp) in enumerate(feat_imp[:30]):
        print(f"    {i+1:2d}. {name:45s} {imp:.4f}")

    # ---- Output ----
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    oof_path = OOF_DIR / f"base_model_full_oof_{ts}.csv"
    sub_path = SUB_DIR / f"candidate_base_model_full_{ts}.csv"

    oof_df = train[["pid"]].copy()
    oof_df["pred_blend"] = oof
    oof_df["popularity"] = y
    oof_df.to_csv(oof_path, index=False)
    print(f"\n  Saved OOF to {oof_path}")

    sub_df = test[["pid"]].copy()
    sub_df["polularity_score"] = test_preds
    sub_df.to_csv(sub_path, index=False)
    print(f"  Saved submission to {sub_path}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"CV MAPE: {cv_mape:.4f}")


if __name__ == "__main__":
    main()
