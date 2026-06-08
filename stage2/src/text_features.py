"""
TF-IDF + SVD text features for SMP Video task.
Combines multiple text sources (post_content, suggested_words, music_title,
ASR, OCR, BLIP captions) into a unified text representation.

Features:
- TF-IDF unigrams + bigrams → TruncatedSVD (64-128 dims)
- Character n-gram TF-IDF → SVD (32 dims)
- Text statistics (lengths, token counts, unique token counts)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def build_full_text(posts: pd.DataFrame, feature_csvs: dict[str, pd.DataFrame] | None = None) -> pd.Series:
    """Build unified text field for each post.

    Uses weighted concatenation of available text sources:
        post_content (×2.0) — hashtags are the primary text signal
        post_suggested_words (×1.5)
        music_title (×1.0)
        asr_text (×1.2)
        ocr_text (×1.3)
        blip_caption (×0.8)

    Returns pd.Series indexed by pid with the combined text.
    """
    texts = {}

    for pid in posts["pid"]:
        parts = []

        row = posts[posts["pid"] == pid].iloc[0]

        # Post content (hashtags) — highest weight, repeated for emphasis
        content = str(row["post_content"]) if pd.notna(row["post_content"]) else ""
        if content:
            # Replace commas with spaces so TF-IDF treats individual tags as tokens
            content_clean = content.replace(",", " ").lower()
            parts.append(f"{content_clean} {content_clean}")  # ×2 weight via repetition

        # Suggested words
        suggested = row["post_suggested_words"]
        if suggested is not None and not (isinstance(suggested, float) and np.isnan(suggested)):
            if isinstance(suggested, (list, np.ndarray)):
                sugg_text = " ".join(str(w).lower() for w in suggested)
                parts.append(f"{sugg_text} {sugg_text}")  # close to ×2 weight

        # Music title
        music_title = row.get("music_title", "")
        if pd.notna(music_title) and str(music_title).strip():
            parts.append(str(music_title).lower())

        texts[pid] = " ".join(parts)

    # Add external text sources if available
    if feature_csvs:
        for pid in texts:
            extra_parts = []
            for name, feat_df in feature_csvs.items():
                if pid in feat_df.index:
                    feat_row = feat_df.loc[pid]
                    for col in ["asr_text", "ocr_text", "blip_caption"]:
                        if col in feat_df.columns:
                            val = feat_row.get(col, "")
                            if pd.notna(val) and str(val).strip():
                                extra_parts.append(str(val).lower())
            if extra_parts:
                texts[pid] = texts[pid] + " " + " ".join(extra_parts)

    return pd.Series(texts, name="full_text")


def build_tfidf_svd_features(
    text_series: pd.Series,
    n_components: int = 64,
    ngram_range: tuple[int, int] = (1, 2),
    max_features: int = 20000,
    min_df: int = 2,
    max_df: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build TF-IDF → TruncatedSVD features.

    Args:
        text_series: pd.Series indexed by pid, values are combined text strings
        n_components: SVD output dimensions
        ngram_range: (1,2) for unigrams + bigrams
        max_features: TF-IDF vocabulary size cap
        min_df: minimum document frequency
        max_df: maximum document frequency (filters out overly common tokens)

    Returns:
        DataFrame indexed by pid with columns tfidf_svd_0 .. tfidf_svd_{n_components-1}
    """
    texts = text_series.fillna("").astype(str).tolist()
    pids = text_series.index.tolist()

    # TF-IDF
    tfidf = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z0-9_#\-]{2,}\b",
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,  # 1 + log(tf), reduces impact of very frequent terms
        norm="l2",
    )
    X_tfidf = tfidf.fit_transform(texts)

    # SVD
    n_comp = min(n_components, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    if n_comp < 2:
        n_comp = min(X_tfidf.shape[1], X_tfidf.shape[0])

    svd = TruncatedSVD(n_components=max(n_comp, 1), random_state=random_state)
    X_svd = svd.fit_transform(X_tfidf)

    cols = [f"tfidf_svd_{i}" for i in range(X_svd.shape[1])]
    result = pd.DataFrame(X_svd, index=pids, columns=cols)

    # Add explained variance ratio as metadata
    result.attrs["tfidf_vocab_size"] = len(tfidf.vocabulary_)
    result.attrs["svd_explained_variance"] = float(
        np.sum(svd.explained_variance_ratio_)
    )

    return result


def build_char_tfidf_svd_features(
    text_series: pd.Series,
    n_components: int = 32,
    max_features: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build character-level TF-IDF → SVD features.

    Character n-grams capture subword patterns useful for hashtags
    (e.g., 'woodworking' → 'woo', 'ood', 'odw', 'dwo', ...).
    """
    texts = text_series.fillna("").astype(str).tolist()
    pids = text_series.index.tolist()

    tfidf = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",  # character n-grams within word boundaries
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.9,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )
    X_tfidf = tfidf.fit_transform(texts)

    n_comp = min(n_components, X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    n_comp = max(n_comp, 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    X_svd = svd.fit_transform(X_tfidf)

    cols = [f"char_tfidf_svd_{i}" for i in range(X_svd.shape[1])]
    result = pd.DataFrame(X_svd, index=pids, columns=cols)
    result.attrs["char_vocab_size"] = len(tfidf.vocabulary_)
    result.attrs["char_svd_explained"] = float(np.sum(svd.explained_variance_ratio_))

    return result


def build_text_statistics(posts: pd.DataFrame) -> pd.DataFrame:
    """Build text statistics features (no label dependency)."""
    stats = []
    for _, row in posts.iterrows():
        content = str(row["post_content"]) if pd.notna(row["post_content"]) else ""
        tags = content.split(",") if content else []
        tags_clean = [t.strip().lower() for t in tags if t.strip()]

        suggested = row["post_suggested_words"]
        if suggested is None or (isinstance(suggested, float) and np.isnan(suggested)):
            suggested_list = []
        elif isinstance(suggested, (list, np.ndarray)):
            suggested_list = [str(w).lower() for w in suggested]
        else:
            suggested_list = []

        all_tokens = tags_clean + suggested_list

        stats.append({
            "pid": row["pid"],
            "num_hashtags": len(tags_clean),
            "num_suggested_words": len(suggested_list),
            "num_total_tokens": len(all_tokens),
            "num_unique_tokens": len(set(all_tokens)),
            "token_uniqueness": len(set(all_tokens)) / max(len(all_tokens), 1),
            "hashtag_avg_length": np.mean([len(t) for t in tags_clean]) if tags_clean else 0.0,
            "hashtag_max_length": max([len(t) for t in tags_clean]) if tags_clean else 0,
        })

    return pd.DataFrame(stats).set_index("pid")
