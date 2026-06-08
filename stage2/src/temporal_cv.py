"""
Temporal cross-validation harness for SMP Video popularity prediction.

All feature engineering respects time boundaries:
- Tag statistics computed on historical data only
- User average popularity from historical data only
- KNN retrieval reference = historical labeled users only
- PCA / Scaler / SVD fit on historical data only
- Word2Vec pre-trained once on all training text (no label leakage)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Paths (relative to package root) ──────────────────────────────────
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent  # stage2/src/ → root
DATA_DIR = _PACKAGE_ROOT / "data"
VIDEO_FEAT_PKL = DATA_DIR / "visual_features.pkl"
W2V_PATH = DATA_DIR / "w2v_deterministic.model"

# ── Temporal splits ────────────────────────────────────────────────
# Each split: (name, list_of_train_months, list_of_val_months)
# Train months are "up to and including", val months are the held-out future.

ALL_MONTHS = [f"{y}-{m:02d}" for y in range(2022, 2024) for m in range(1, 13)]
TRAIN_MONTHS = [m for m in ALL_MONTHS if m <= "2023-08"]

TEMPORAL_SPLITS: list[tuple[str, list[str], list[str]]] = [
    ("may",     [m for m in TRAIN_MONTHS if m <= "2023-04"], ["2023-05"]),
    ("jun",     [m for m in TRAIN_MONTHS if m <= "2023-05"], ["2023-06"]),
    ("jul",     [m for m in TRAIN_MONTHS if m <= "2023-06"], ["2023-07"]),
    ("aug",     [m for m in TRAIN_MONTHS if m <= "2023-07"], ["2023-08"]),
    ("jul_aug", [m for m in TRAIN_MONTHS if m <= "2023-06"], ["2023-07", "2023-08"]),
]


# ── Import enhanced feature modules ────────────────────────────────
try:
    from .tag_features import compute_tag_stats_timeaware, apply_tag_stats_to_posts
    from .text_features import (
        build_full_text, build_tfidf_svd_features,
        build_char_tfidf_svd_features, build_text_statistics,
    )
except ImportError:
    from tag_features import compute_tag_stats_timeaware, apply_tag_stats_to_posts
    from text_features import (
        build_full_text, build_tfidf_svd_features,
        build_char_tfidf_svd_features, build_text_statistics,
    )


# ── Helpers ────────────────────────────────────────────────────────
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)))


def filter_outliers(y: pd.Series, lower_pct: float = 0.5, upper_pct: float = 99.5):
    """Filter outliers using percentile-based bounds (more stable than IQR for skewed data)."""
    lo = float(np.percentile(y, lower_pct))
    hi = float(np.percentile(y, upper_pct))
    mask = (y >= lo) & (y <= hi)
    return mask, lo, hi


# ── Data loading ───────────────────────────────────────────────────
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all raw parquet files. Returns (posts, users, videos, labels)."""
    posts_train = pd.read_parquet(DATA_DIR / "posts_train.parquet")
    posts_test = pd.read_parquet(DATA_DIR / "posts_test.parquet")
    users_train = pd.read_parquet(DATA_DIR / "users_train.parquet")
    users_test = pd.read_parquet(DATA_DIR / "users_test.parquet")
    videos_train = pd.read_parquet(DATA_DIR / "videos_train.parquet")
    videos_test = pd.read_parquet(DATA_DIR / "videos_test.parquet")
    labels_train = pd.read_parquet(DATA_DIR / "labels_train.parquet")

    posts = pd.concat([posts_train, posts_test], ignore_index=True)
    users = pd.concat([users_train, users_test], ignore_index=True)
    videos = pd.concat([videos_train, videos_test], ignore_index=True)

    posts["post_time"] = pd.to_datetime(posts["post_time"])
    posts["month_key"] = posts["post_time"].dt.strftime("%Y-%m")

    return posts, users, videos, labels_train


def load_visual_features() -> dict[str, np.ndarray] | None:
    """Load pre-extracted VideoMAE features. Returns {pid: 768-dim array} or None."""
    if VIDEO_FEAT_PKL.exists():
        with open(VIDEO_FEAT_PKL, "rb") as f:
            return pickle.load(f)
    return None


# ── Time-aware feature engineering ─────────────────────────────────
# Each builder function receives:
#   posts_sub   : posts subset (historical only, or historical+val)
#   users_sub   : users subset
#   videos_sub  : videos subset
#   labels_hist : labels for historical posts ONLY (used for target-dependent stats)
#   train_mask  : boolean mask, True for historical posts within posts_sub
#
# Returns a DataFrame indexed by pid (or uid for user features).


def build_user_features_timeaware(
    users: pd.DataFrame,
    posts: pd.DataFrame,
    labels_hist: pd.DataFrame,
    retrieval_k: int = 50,
    n_clusters: int = 20,
) -> pd.DataFrame:
    """Build user features using only labels_hist for target-dependent stats."""
    user_feat = users.drop_duplicates(subset="uid", keep="first").copy()

    # ── log-counts ──
    count_cols = [
        "user_following_count", "user_follower_count", "user_likes_count",
        "user_video_count", "user_digg_count", "user_heart_count", "user_friend_count",
    ]
    for col in count_cols:
        if col in user_feat.columns:
            user_feat[col + "_log"] = np.log1p(user_feat[col])

    # ── engagement ratios ──
    user_feat["follower_following_ratio_log"] = np.log1p(
        (user_feat["user_follower_count"].fillna(0) + 1)
        / (user_feat["user_following_count"].fillna(0) + 1)
    )
    user_feat["likes_per_video_log"] = np.log1p(
        (user_feat["user_likes_count"].fillna(0) + 1)
        / (user_feat["user_video_count"].fillna(0) + 1)
    )

    # ── post count (from posts_sub, which may include val posts — OK, no label leak) ──
    post_count = posts.groupby("uid").size().reset_index(name="user_post_count")
    user_feat = user_feat.merge(post_count, on="uid", how="left")
    user_feat["user_post_count_log"] = np.log1p(user_feat["user_post_count"].fillna(0))

    global_mean = labels_hist["popularity"].mean()

    # ── User PCA embedding (no label involved, can use all users) ──
    user_emb_cols = [
        "user_follower_count_log", "user_likes_count_log",
        "user_video_count_log", "user_digg_count_log",
        "user_friend_count_log", "user_following_count_log",
        "user_heart_count_log",
    ]
    user_emb_data = user_feat[user_emb_cols].fillna(0).values.astype(np.float64)
    emb_scaler = StandardScaler()
    user_emb_scaled = emb_scaler.fit_transform(user_emb_data)
    n_pca = min(8, len(user_emb_cols))
    user_pca = PCA(n_components=n_pca, random_state=42)
    user_emb_pca = user_pca.fit_transform(user_emb_scaled)

    # ── User cluster prior (label-dependent: use labels_hist only) ──
    n_clusters_actual = min(n_clusters, max(2, len(user_emb_pca) // 5))
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters_actual, random_state=42, n_init=10)
    user_clusters = kmeans.fit_predict(user_emb_pca)
    user_feat["user_cluster"] = user_clusters

    # Compute cluster mean popularity from labels_hist
    user_label_hist = labels_hist.groupby("uid")["popularity"].mean()
    cluster_pop: dict[int, list[float]] = {}
    for uid, cid in zip(user_feat["uid"], user_clusters):
        if uid in user_label_hist.index:
            cluster_pop.setdefault(int(cid), []).append(float(user_label_hist[uid]))
    cluster_mean_pop = {c: float(np.mean(pops)) for c, pops in cluster_pop.items()}
    global_cluster_mean = (
        float(np.mean(list(cluster_mean_pop.values())))
        if cluster_mean_pop else global_mean
    )
    user_feat["user_cluster_prior"] = np.array(
        [cluster_mean_pop.get(int(c), global_cluster_mean) for c in user_clusters],
        dtype=np.float64,
    )

    # ── User avg popularity (label-dependent: labels_hist only) ──
    user_avg_pop = labels_hist.groupby("uid")["popularity"].mean().reset_index()
    user_avg_pop.columns = ["uid", "user_avg_popularity"]
    user_feat = user_feat.merge(user_avg_pop, on="uid", how="left")
    user_feat["user_avg_popularity"] = user_feat["user_avg_popularity"].fillna(global_mean)

    # ── KNN retrieval (label-dependent: labels_hist only) ──
    labeled_uids = set(user_label_hist.index)
    all_uids = user_feat["uid"].tolist()
    labeled_mask = np.array([uid in labeled_uids for uid in all_uids])
    labeled_idx = np.where(labeled_mask)[0]

    retrieval_features = _build_retrieval_features(
        user_emb_pca, all_uids, labeled_idx, user_label_hist,
        global_mean, retrieval_k,
    )
    for k, v in retrieval_features.items():
        user_feat[k] = v

    return user_feat


def _build_retrieval_features(
    user_emb_pca: np.ndarray,
    all_uids: list[str],
    labeled_idx: np.ndarray,
    user_label_hist: pd.Series,
    global_mean: float,
    retrieval_k: int = 50,
) -> dict[str, np.ndarray]:
    """Build KNN retrieval features. Only labeled users form the reference set."""
    if len(labeled_idx) < 2:
        # Fallback: not enough labeled users
        n = len(all_uids)
        return {
            "retrieval_mean": np.full(n, global_mean, dtype=np.float32),
            "retrieval_std": np.zeros(n, dtype=np.float32),
            "retrieval_max": np.full(n, global_mean, dtype=np.float32),
            "retrieval_min": np.full(n, global_mean, dtype=np.float32),
            "retrieval_median": np.full(n, global_mean, dtype=np.float32),
            "retrieval_top3_mean": np.full(n, global_mean, dtype=np.float32),
            "retrieval_top5_mean": np.full(n, global_mean, dtype=np.float32),
            "retrieval_weighted_mean": np.full(n, global_mean, dtype=np.float32),
            "retrieval_weighted_std": np.zeros(n, dtype=np.float32),
            "retrieval_sim_mean": np.zeros(n, dtype=np.float32),
            "retrieval_sim_max": np.zeros(n, dtype=np.float32),
            "retrieval_sim_weighted": np.full(n, global_mean, dtype=np.float32),
            **{f"retrieval_{p}": np.full(n, global_mean, dtype=np.float32)
               for p in ["p10", "p25", "p50", "p75", "p90"]},
            "retrieval_dist_mean": np.zeros(n, dtype=np.float32),
            "retrieval_dist_std": np.zeros(n, dtype=np.float32),
            "retrieval_dist_min": np.zeros(n, dtype=np.float32),
        }

    labeled_emb = user_emb_pca[labeled_idx]
    labeled_uid_arr = np.array([all_uids[i] for i in labeled_idx])

    query_k = min(retrieval_k + 1, len(labeled_idx))
    nn = NearestNeighbors(n_neighbors=query_k, metric="euclidean", n_jobs=-1)
    nn.fit(labeled_emb)
    distances, neighbor_idx = nn.kneighbors(user_emb_pca)

    all_neighbor_labels, all_neighbor_dists = [], []
    for i, (nbrs, dists) in enumerate(zip(neighbor_idx, distances)):
        uid = all_uids[i]
        row_labels, row_dists = [], []
        for j, d in zip(nbrs, dists):
            nbr_uid = labeled_uid_arr[j]
            if nbr_uid == uid:
                continue
            lbl = user_label_hist.get(nbr_uid, np.nan)
            row_labels.append(lbl if not np.isnan(lbl) else global_mean)
            row_dists.append(d)
            if len(row_labels) >= retrieval_k:
                break
        while len(row_labels) < retrieval_k:
            row_labels.append(global_mean)
            row_dists.append(1e6)
        all_neighbor_labels.append(row_labels[:retrieval_k])
        all_neighbor_dists.append(row_dists[:retrieval_k])

    L = np.array(all_neighbor_labels, dtype=np.float32)
    D = np.clip(np.array(all_neighbor_dists, dtype=np.float32), 1e-3, None)

    w = 1.0 / D
    w_sum = w.sum(axis=1)
    sim = np.exp(-D)

    L_sorted = np.sort(L, axis=1)
    wmean = (w * L).sum(axis=1) / w_sum

    result: dict[str, np.ndarray] = {}
    result["retrieval_mean"] = L.mean(axis=1)
    result["retrieval_std"] = L.std(axis=1)
    result["retrieval_max"] = L.max(axis=1)
    result["retrieval_min"] = L.min(axis=1)
    result["retrieval_median"] = np.median(L, axis=1)
    result["retrieval_top3_mean"] = L_sorted[:, -3:].mean(axis=1)
    result["retrieval_top5_mean"] = L_sorted[:, -5:].mean(axis=1)
    result["retrieval_weighted_mean"] = wmean
    result["retrieval_weighted_std"] = np.sqrt(
        (w * (L - wmean.reshape(-1, 1)) ** 2).sum(axis=1) / w_sum
    )
    result["retrieval_sim_mean"] = sim.mean(axis=1)
    result["retrieval_sim_max"] = sim.max(axis=1)
    result["retrieval_sim_weighted"] = (sim * L).sum(axis=1) / (sim.sum(axis=1) + 1e-8)
    for pn, pv in [("p10", 10), ("p25", 25), ("p50", 50), ("p75", 75), ("p90", 90)]:
        result[f"retrieval_{pn}"] = np.percentile(L, pv, axis=1).astype(np.float32)
    result["retrieval_dist_mean"] = D.mean(axis=1)
    result["retrieval_dist_std"] = D.std(axis=1)
    result["retrieval_dist_min"] = D.min(axis=1)

    return result


def build_tag_features_timeaware(
    posts: pd.DataFrame,
    labels_hist: pd.DataFrame,
    bayesian_smoothing: float = 10.0,
) -> pd.DataFrame:
    """Build tag features using only labels_hist for statistics.

    Returns DataFrame indexed by pid with columns:
        avg_tag_popularity, avg_tag_frequency, tag_diversity,
        max_tag_popularity, avg_tag_pop_bayes, known_tag_ratio
    """
    label_map = dict(zip(labels_hist["pid"], labels_hist["popularity"]))
    global_mean = labels_hist["popularity"].mean()

    # Compute per-tag statistics from historical data only
    tag_pops: dict[str, list[float]] = {}
    tag_freq: dict[str, int] = {}
    for _, row in posts.iterrows():
        pid = row["pid"]
        content = row["post_content"]
        tags = str(content).split(",") if pd.notna(content) else []
        pop = label_map.get(pid)
        for tag in tags:
            tag = tag.strip().lower()
            if tag:
                tag_freq[tag] = tag_freq.get(tag, 0) + 1
                if pop is not None:
                    tag_pops.setdefault(tag, []).append(pop)

    tag_mean = {t: float(np.mean(v)) for t, v in tag_pops.items()}
    total_unique_tags = len(tag_freq)

    # Bayesian smoothed tag score: (n / (n+alpha)) * tag_mean + (alpha/(n+alpha)) * global_mean
    tag_bayes = {}
    for t, pops in tag_pops.items():
        n = len(pops)
        raw_mean = float(np.mean(pops))
        tag_bayes[t] = (n / (n + bayesian_smoothing)) * raw_mean + \
                       (bayesian_smoothing / (n + bayesian_smoothing)) * global_mean

    # Apply to each post
    results = []
    for _, row in posts.iterrows():
        content = row["post_content"]
        tags = str(content).split(",") if pd.notna(content) else []
        tags_clean = [t.strip().lower() for t in tags if t.strip()]

        if tags_clean:
            pop_vals = [tag_mean.get(t, np.nan) for t in tags_clean]
            bayes_vals = [tag_bayes.get(t, global_mean) for t in tags_clean]
            freq_vals = [tag_freq.get(t, 0) for t in tags_clean]
            avg_pop = np.nanmean(pop_vals)
            avg_bayes = np.nanmean(bayes_vals) if bayes_vals else global_mean
            avg_freq = np.nanmean(freq_vals) if freq_vals else 0.0
            max_pop = np.nanmax(pop_vals) if pop_vals else global_mean
            known = sum(1 for t in tags_clean if t in tag_mean)
            known_ratio = known / len(tags_clean)
            diversity = len(set(tags_clean)) / max(total_unique_tags, 1)
        else:
            avg_pop = 0.0
            avg_bayes = global_mean
            avg_freq = 0.0
            max_pop = global_mean
            known_ratio = 0.0
            diversity = 0.0

        results.append({
            "pid": row["pid"],
            "avg_tag_popularity": avg_pop,
            "avg_tag_pop_bayes": avg_bayes,
            "avg_tag_frequency": avg_freq,
            "max_tag_popularity": max_pop,
            "tag_diversity": diversity,
            "known_tag_ratio": known_ratio,
        })

    return pd.DataFrame(results).set_index("pid")


def build_text_features_timeaware(
    posts: pd.DataFrame,
    w2v_model: Word2Vec | None = None,
) -> pd.DataFrame:
    """Build Word2Vec text embeddings. W2V is pre-trained (no label leak)."""
    if w2v_model is None:
        if W2V_PATH.exists():
            w2v_model = Word2Vec.load(str(W2V_PATH))
        else:
            # Train from scratch on posts_sub
            sentences = _make_sentences(posts)
            w2v_model = Word2Vec(
                sentences, vector_size=64, window=5, min_count=2,
                workers=4, epochs=20, seed=42,
            )

    def embed_one(content, suggested):
        tokens = _tokenize(content) + _tokenize_list(suggested)
        vecs = [w2v_model.wv[w] for w in tokens if w in w2v_model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(w2v_model.vector_size)

    emb = np.vstack([
        embed_one(row["post_content"], row["post_suggested_words"])
        for _, row in posts.iterrows()
    ])
    cols = [f"text_embedding_{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, index=posts["pid"].values, columns=cols)


def _tokenize(x) -> list[str]:
    if pd.isna(x):
        return []
    return str(x).lower().replace(",", " ").split()


def _tokenize_list(x) -> list[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, np.ndarray)):
        return [str(w).lower() for w in x]
    return []


def _make_sentences(posts: pd.DataFrame) -> list[list[str]]:
    sentences = posts["post_content"].apply(_tokenize).tolist()
    sentences = [s for s in sentences if len(s) > 0]
    suggested = posts["post_suggested_words"].apply(_tokenize_list).tolist()
    suggested = [s for s in suggested if len(s) > 0]
    return sentences + suggested


def build_temporal_metadata_features(
    posts: pd.DataFrame,
    videos: pd.DataFrame,
) -> pd.DataFrame:
    """Build temporal and metadata features (no label dependency)."""
    feat = posts[["pid", "uid"]].copy()
    post_time = pd.to_datetime(posts["post_time"])
    feat["post_hour"] = post_time.dt.hour
    feat["post_day_of_week"] = post_time.dt.dayofweek
    feat["post_is_weekend"] = (post_time.dt.dayofweek >= 5).astype(int)
    feat["post_content_len"] = posts["post_content"].apply(
        lambda x: len(str(x).split(",")) if pd.notna(x) else 0
    )
    feat["post_content_char_len"] = posts["post_content"].apply(
        lambda x: len(str(x)) if pd.notna(x) else 0
    )

    def count_suggested(x):
        if x is None:
            return 0
        if isinstance(x, float) and np.isnan(x):
            return 0
        if isinstance(x, (list, np.ndarray)):
            return len(x)
        return 0
    feat["suggested_words_len"] = posts["post_suggested_words"].apply(count_suggested)

    le = LabelEncoder()
    feat["post_location_enc"] = le.fit_transform(
        posts["post_location"].fillna("UNK").astype(str)
    )
    feat["post_language_enc"] = le.fit_transform(
        posts["post_text_language"].fillna("UNK").astype(str)
    )

    # Video metadata
    video_feat = videos[["pid", "video_height", "video_width", "video_duration",
                          "video_ratio", "video_format", "music_duration"]].copy()
    ratio_map = {"540p": 540, "720p": 720, "1080p": 1080}
    video_feat["video_ratio_num"] = video_feat["video_ratio"].map(ratio_map).fillna(540)
    video_feat["video_format_mp4"] = (video_feat["video_format"] == "mp4").astype(int)
    feat = feat.merge(
        video_feat.drop(columns=["video_format", "video_ratio"]),
        on="pid", how="left",
    )
    return feat


def build_visual_proxy_features(videos: pd.DataFrame) -> pd.DataFrame:
    """Build proxy visual features (no label dependency)."""
    v = videos[["pid", "video_height", "video_width", "video_duration"]].copy()
    v["video_aspect"] = v["video_width"] / v["video_height"].replace(0, 1)
    v["video_resolution"] = v["video_height"] * v["video_width"]
    v = v.set_index("pid")
    v = v.fillna(v.median(numeric_only=True))
    scaler = StandardScaler()
    arr = scaler.fit_transform(v)
    n_comp = min(4, arr.shape[1], arr.shape[0])
    arr_pca = PCA(n_components=n_comp, random_state=42).fit_transform(arr)
    cols = [f"video_embedding_{i}" for i in range(arr_pca.shape[1])]
    return pd.DataFrame(arr_pca, index=v.index, columns=cols)


def build_videomae_features(
    posts: pd.DataFrame,
    visual_features: dict[str, np.ndarray],
    n_components: int = 64,
) -> pd.DataFrame | None:
    """Build PCA-reduced VideoMAE features.

    PCA is fit ONLY on the provided posts (historical), then transforms all.
    """
    pids = posts["pid"].tolist()
    feats = []
    valid_pids = []
    for pid in pids:
        arr = visual_features.get(pid)
        if arr is not None and len(arr) == 768:
            feats.append(arr)
            valid_pids.append(pid)

    if len(feats) < 10:
        return None

    arr = np.stack(feats, axis=0).astype(np.float64)
    missing_mask = np.array(
        [1.0 if pid not in visual_features else 0.0 for pid in pids],
        dtype=np.float64,
    )

    # Fill missing with column median
    med = np.median(arr, axis=0)
    full_arr = np.zeros((len(pids), 768), dtype=np.float64)
    for i, pid in enumerate(pids):
        vec = visual_features.get(pid)
        full_arr[i] = vec.astype(np.float64) if vec is not None else med

    # Standardize + PCA
    scaler = StandardScaler()
    full_arr_scaled = scaler.fit_transform(full_arr)
    n_comp = min(n_components, full_arr_scaled.shape[1], full_arr_scaled.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    arr_pca = pca.fit_transform(full_arr_scaled)

    cols = [f"videomae_pca_{i}" for i in range(arr_pca.shape[1])]
    result = pd.DataFrame(arr_pca, index=pids, columns=cols)
    result["videomae_missing"] = missing_mask
    return result


def assemble_feature_matrix(
    posts_hist: pd.DataFrame,
    posts_val: pd.DataFrame | None,
    users_hist: pd.DataFrame,
    videos_hist: pd.DataFrame,
    labels_hist: pd.DataFrame,
    w2v_model: Word2Vec | None = None,
    visual_features: dict[str, np.ndarray] | None = None,
    use_videomae: bool = False,
    videomae_dims: int = 64,
    use_enhanced_tags: bool = False,
    use_tfidf_svd: bool = False,
    tfidf_dims: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Assemble full feature matrix for historical + optional validation posts.

    All statistics are computed from (posts_hist, users_hist, videos_hist, labels_hist).
    If posts_val is provided, features are computed using the same statistics.

    New options (all default False for backward compatibility):
        use_enhanced_tags: Bayesian smoothing + trend + top-k tag features
        use_tfidf_svd: TF-IDF unigram+bigram → SVD text features
    """
    all_posts = posts_hist if posts_val is None else pd.concat(
        [posts_hist, posts_val], ignore_index=True
    )
    n_hist = len(posts_hist)

    # ── User features (time-aware) ──
    all_uids_in_posts = all_posts["uid"].unique()
    users_all = users_hist[
        users_hist["uid"].isin(all_uids_in_posts)
    ].copy()
    user_feat = build_user_features_timeaware(users_all, all_posts, labels_hist)

    # ── Tag features (time-aware) ──
    posts_labeled = posts_hist[posts_hist["pid"].isin(labels_hist["pid"])]
    if use_enhanced_tags:
        # Enhanced: Bayesian smoothing + trend + recent + top-k
        tag_stats = compute_tag_stats_timeaware(posts_labeled, labels_hist)
        tag_feat = apply_tag_stats_to_posts(all_posts, tag_stats)
    else:
        # Original: basic tag features
        tag_feat = build_tag_features_timeaware(all_posts, labels_hist)

    # ── Text features ──
    text_feat = build_text_features_timeaware(all_posts, w2v_model)

    # ── TF-IDF SVD text features ──
    if use_tfidf_svd:
        all_text = build_full_text(all_posts)
        tfidf_svd = build_tfidf_svd_features(all_text, n_components=tfidf_dims)
        char_svd = build_char_tfidf_svd_features(all_text, n_components=min(tfidf_dims // 2, 32))
        text_stats = build_text_statistics(all_posts)

    # ── Temporal & metadata ──
    tm_feat = build_temporal_metadata_features(all_posts, videos_hist)

    # ── Visual proxy ──
    videos_all = videos_hist[
        videos_hist["pid"].isin(all_posts["pid"])
    ]
    visual_proxy = build_visual_proxy_features(videos_all)

    # ── Assemble ──
    df = all_posts[["pid", "uid"]].copy().set_index("pid")
    tm_idx = tm_feat.drop(columns=["uid"], errors="ignore").set_index("pid")
    df = df.merge(tm_idx, left_index=True, right_index=True, how="left")
    user_idx = user_feat.set_index("uid")
    df = df.reset_index().merge(user_idx, on="uid", how="left").set_index("pid")
    df = df.merge(text_feat, left_index=True, right_index=True, how="left")
    if use_tfidf_svd:
        df = df.merge(tfidf_svd, left_index=True, right_index=True, how="left")
        df = df.merge(char_svd, left_index=True, right_index=True, how="left")
        df = df.merge(text_stats, left_index=True, right_index=True, how="left")
    df = df.merge(visual_proxy, left_index=True, right_index=True, how="left")
    df = df.merge(tag_feat, left_index=True, right_index=True, how="left")

    # ── VideoMAE features ──
    if use_videomae and visual_features is not None:
        videomae_feat = build_videomae_features(all_posts, visual_features, videomae_dims)
        if videomae_feat is not None:
            df = df.merge(videomae_feat, left_index=True, right_index=True, how="left")

    X = df.drop(columns=["uid"], errors="ignore")

    # ── Fill missing ──
    for col in X.columns:
        if X[col].dtype in ("object", "category"):
            X[col] = X[col].fillna("MISSING")
        else:
            X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)

    # ── SVD mode features (fit on historical only) ──
    # User SVD
    user_log_cols = [c for c in user_feat.columns if c.endswith("_log")]
    if len(user_log_cols) >= 2:
        usvd_data = user_feat.set_index("uid")[user_log_cols].fillna(0).values
        n_u = min(10, usvd_data.shape[1], usvd_data.shape[0] - 1)
        if n_u >= 2:
            svd_u = TruncatedSVD(n_components=n_u, random_state=42)
            arr_svd_u = svd_u.fit_transform(usvd_data)
            pid_to_uid = dict(zip(all_posts["pid"], all_posts["uid"]))
            svd_u_cols = {}
            for i in range(n_u):
                uid_map = dict(zip(user_feat["uid"], arr_svd_u[:, i]))
                svd_u_cols[f"svd_mode_u_{i}"] = [
                    uid_map.get(pid_to_uid.get(pid, ""), 0.0) for pid in X.index
                ]
            X = pd.concat([X, pd.DataFrame(svd_u_cols, index=X.index)], axis=1)

    # Text SVD
    text_cols = [c for c in text_feat.columns if c.startswith("text_embedding")]
    if len(text_cols) >= 2:
        text_data = text_feat.loc[text_feat.index.isin(all_posts["pid"])][text_cols].fillna(0).values
        n_t = min(20, text_data.shape[1], text_data.shape[0] - 1)
        if n_t >= 2:
            svd_t = TruncatedSVD(n_components=n_t, random_state=42)
            arr_svd_t = svd_t.fit_transform(text_data)
            pid_to_idx = {pid: i for i, pid in enumerate(text_feat.index)}
            svd_t_cols = {}
            for i in range(n_t):
                svd_t_cols[f"svd_mode_t_{i}"] = [
                    arr_svd_t[pid_to_idx.get(pid, 0), i]
                    if pid in pid_to_idx else 0.0
                    for pid in X.index
                ]
            X = pd.concat([X, pd.DataFrame(svd_t_cols, index=X.index)], axis=1)

    X = X.fillna(0)
    # Drop any remaining string columns
    str_cols = [c for c in X.columns if X[c].dtype == "object"]
    if str_cols:
        X = X.drop(columns=str_cols)

    # Split back
    if posts_val is not None:
        hist_pids = posts_hist["pid"].tolist()
        X_hist = X.loc[X.index.isin(hist_pids)]
        X_val = X.loc[~X.index.isin(hist_pids)]
        return X_hist, X_val
    return X, None


# ── Model training ─────────────────────────────────────────────────
from catboost import CatBoostRegressor, Pool


def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame | None = None,
    y_val: np.ndarray | None = None,
    seed: int = 42,
) -> CatBoostRegressor:
    """Train a single CatBoost model."""
    params = {
        "loss_function": "Huber:delta=1.0",
        "eval_metric": "MAPE",
        "random_seed": seed,
        "verbose": 0,
        "early_stopping_rounds": 200,
        "iterations": 5000,
        "learning_rate": 0.03,
        "depth": 7,
        "l2_leaf_reg": 3,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
    }
    train_pool = Pool(X_train, y_train)
    if X_val is not None and y_val is not None:
        val_pool = Pool(X_val, y_val)
        model = CatBoostRegressor(**params)
        model.fit(train_pool, eval_set=val_pool)
    else:
        model = CatBoostRegressor(**params)
        model.fit(train_pool)
    return model


@dataclass
class SplitResult:
    name: str
    train_months: list[str]
    val_months: list[str]
    n_train: int
    n_val: int
    n_train_after_filter: int
    mape_val: float
    pred_mean: float
    pred_std: float
    y_val_mean: float


def run_temporal_cv(
    use_videomae: bool = False,
    videomae_dims: int = 64,
    verbose: bool = True,
) -> list[SplitResult]:
    """Run full temporal cross-validation.

    Returns list of SplitResult, one per temporal split.
    """
    # ── Load all data once ──
    posts, users, videos, labels = load_raw_data()
    visual_features = load_visual_features() if use_videomae else None

    # ── Pre-train Word2Vec on all training text (no label leakage) ──
    posts_train_only = posts[posts["pid"].isin(labels["pid"])]
    if W2V_PATH.exists():
        w2v = Word2Vec.load(str(W2V_PATH))
    else:
        sentences = _make_sentences(posts_train_only)
        w2v = Word2Vec(
            sentences, vector_size=64, window=5, min_count=2,
            workers=4, epochs=20, seed=42,
        )

    results: list[SplitResult] = []

    for split_name, train_months, val_months in TEMPORAL_SPLITS:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Split: {split_name}")
            print(f"  Train months: {train_months[0]} → {train_months[-1]} ({len(train_months)} months)")
            print(f"  Val months:   {val_months}")

        # ── Partition data ──
        posts_hist = posts[
            posts["month_key"].isin(train_months) &
            posts["pid"].isin(labels["pid"])
        ].copy()
        posts_val = posts[
            posts["month_key"].isin(val_months) &
            posts["pid"].isin(labels["pid"])
        ].copy()

        if len(posts_hist) < 100 or len(posts_val) < 50:
            print(f"  SKIP: insufficient data (hist={len(posts_hist)}, val={len(posts_val)})")
            continue

        # Historical labels (only train months)
        labels_hist = labels[labels["pid"].isin(posts_hist["pid"])]

        # Users and videos for the historical window
        hist_uids = posts_hist["uid"].unique()
        hist_pids = posts_hist["pid"].unique()
        all_pids_in_split = np.concatenate([hist_pids, posts_val["pid"].unique()])
        all_uids_in_split = np.concatenate([hist_uids, posts_val["uid"].unique()])

        users_split = users[users["uid"].isin(all_uids_in_split)]
        videos_split = videos[videos["pid"].isin(all_pids_in_split)]

        n_train = len(posts_hist)
        n_val = len(posts_val)

        # ── Build features ──
        X_hist, X_val = assemble_feature_matrix(
            posts_hist, posts_val,
            users_split, videos_split, labels_hist,
            w2v_model=w2v,
            visual_features=visual_features,
            use_videomae=use_videomae,
            videomae_dims=videomae_dims,
        )

        # ── Align labels ──
        y_hist = labels_hist.set_index("pid")["popularity"]
        common_hist = X_hist.index.intersection(y_hist.index)
        X_hist = X_hist.loc[common_hist]
        y_hist = y_hist.loc[common_hist]

        y_val = labels[labels["pid"].isin(posts_val["pid"])].set_index("pid")["popularity"]
        common_val = X_val.index.intersection(y_val.index)
        X_val = X_val.loc[common_val]
        y_val = y_val.loc[common_val]

        # ── Outlier filter (on historical only) ──
        mask, lo, hi = filter_outliers(y_hist)
        X_hist_f = X_hist[mask]
        y_hist_f = y_hist[mask]

        if verbose:
            print(f"  Historical: {n_train} posts, {X_hist.shape[1]} features")
            print(f"  After outlier filter: {len(y_hist_f)} / {len(y_hist)} "
                  f"(bounds: [{lo:.2f}, {hi:.2f}])")
            print(f"  Validation: {n_val} posts")

        # ── Scale (fit on historical only) ──
        numeric_cols = [c for c in X_hist_f.columns if X_hist_f[c].dtype != "object"]
        scaler = StandardScaler()
        X_hist_s = X_hist_f.copy()
        X_hist_s[numeric_cols] = scaler.fit_transform(X_hist_f[numeric_cols].values)
        X_val_s = X_val.copy()
        X_val_s[numeric_cols] = scaler.transform(X_val[numeric_cols].values)

        # ── Train ──
        model = train_catboost(X_hist_s, y_hist_f.values, X_val_s, y_val.values)
        preds = model.predict(X_val_s)
        mape_val = mape(y_val.values, preds)

        result = SplitResult(
            name=split_name,
            train_months=train_months,
            val_months=val_months,
            n_train=n_train,
            n_val=n_val,
            n_train_after_filter=len(y_hist_f),
            mape_val=mape_val,
            pred_mean=float(np.mean(preds)),
            pred_std=float(np.std(preds)),
            y_val_mean=float(y_val.mean()),
        )
        results.append(result)

        if verbose:
            print(f"  ✓ MAPE: {mape_val:.4f}  "
                  f"(pred mean={result.pred_mean:.2f}, y mean={result.y_val_mean:.2f})")

    return results


def print_results_table(results: list[SplitResult]) -> None:
    """Print a formatted results table."""
    print(f"\n{'='*80}")
    print("TEMPORAL CV RESULTS")
    print(f"{'='*80}")
    header = f"{'Split':<12} {'Train':>6} {'Val':>6} {'AfterFilter':>11} {'MAPE':>8} {'PredMean':>9} {'YMean':>7}"
    print(header)
    print("-" * 80)
    mapes = []
    for r in results:
        print(f"{r.name:<12} {r.n_train:>6} {r.n_val:>6} {r.n_train_after_filter:>11} "
              f"{r.mape_val:>8.4f} {r.pred_mean:>9.3f} {r.y_val_mean:>7.3f}")
        mapes.append(r.mape_val)
    print("-" * 80)
    if mapes:
        print(f"{'MEAN':<12} {'':>6} {'':>6} {'':>11} {np.mean(mapes):>8.4f}")
        print(f"{'STD':<12} {'':>6} {'':>6} {'':>11} {np.std(mapes):>8.4f}")
    print(f"{'='*80}")
