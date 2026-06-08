"""
Enhanced tag features with:
1. Time-aware OOF computation (per temporal split)
2. Bayesian smoothing (Empirical Bayes)
3. Tag trend features (recent vs all-time)
4. Per-post aggregation statistics (mean, max, min, top-k)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import defaultdict


def compute_tag_stats_timeaware(
    posts_hist: pd.DataFrame,
    labels_hist: pd.DataFrame,
    bayesian_smoothing: float = 10.0,
    recent_days: int = 60,
) -> dict:
    """Compute all tag statistics from historical data only.

    Returns a dict with keys:
        tag_mean: {tag: mean popularity}
        tag_bayes: {tag: Bayesian-smoothed popularity}
        tag_freq: {tag: document frequency}
        tag_mean_recent: {tag: mean popularity in recent window}
        tag_freq_recent: {tag: frequency in recent window}
        tag_trend: {tag: recent_mean - all_mean}
        global_mean: float
        total_unique_tags: int
    """
    label_map = dict(zip(labels_hist["pid"], labels_hist["popularity"]))
    global_mean = float(labels_hist["popularity"].mean())

    # Determine recent cutoff
    post_times = pd.to_datetime(posts_hist["post_time"])
    max_time = post_times.max()
    recent_cutoff = max_time - pd.Timedelta(days=recent_days)

    tag_pops: dict[str, list[float]] = defaultdict(list)
    tag_pops_recent: dict[str, list[float]] = defaultdict(list)
    tag_freq: dict[str, int] = defaultdict(int)
    tag_freq_recent: dict[str, int] = defaultdict(int)

    for _, row in posts_hist.iterrows():
        pid = row["pid"]
        content = row["post_content"]
        tags = str(content).split(",") if pd.notna(content) else []
        pop = label_map.get(pid)
        is_recent = post_times.loc[row.name] >= recent_cutoff if row.name in post_times.index else False

        for tag in tags:
            tag = tag.strip().lower()
            if tag:
                tag_freq[tag] += 1
                if is_recent:
                    tag_freq_recent[tag] += 1
                if pop is not None:
                    tag_pops[tag].append(pop)
                    if is_recent:
                        tag_pops_recent[tag].append(pop)

    # Mean statistics
    tag_mean = {t: float(np.mean(v)) for t, v in tag_pops.items()}
    tag_mean_recent = {
        t: float(np.mean(v)) if v else tag_mean.get(t, global_mean)
        for t, v in tag_pops_recent.items()
    }

    # Bayesian smoothed
    tag_bayes = {}
    for t, pops in tag_pops.items():
        n = len(pops)
        raw_mean = float(np.mean(pops))
        tag_bayes[t] = (n / (n + bayesian_smoothing)) * raw_mean + \
                       (bayesian_smoothing / (n + bayesian_smoothing)) * global_mean

    # Trend
    tag_trend = {}
    for t in tag_mean:
        tag_trend[t] = tag_mean_recent.get(t, tag_mean[t]) - tag_mean[t]

    # Frequency delta (log-space)
    tag_freq_delta = {}
    for t in tag_freq:
        all_f = tag_freq[t]
        recent_f = tag_freq_recent.get(t, 0)
        tag_freq_delta[t] = np.log1p(recent_f) - np.log1p(all_f)

    total_unique_tags = len(tag_freq)

    return {
        "tag_mean": tag_mean,
        "tag_bayes": tag_bayes,
        "tag_freq": dict(tag_freq),
        "tag_mean_recent": tag_mean_recent,
        "tag_freq_recent": dict(tag_freq_recent),
        "tag_trend": tag_trend,
        "tag_freq_delta": tag_freq_delta,
        "global_mean": global_mean,
        "total_unique_tags": total_unique_tags,
    }


def apply_tag_stats_to_posts(
    posts: pd.DataFrame,
    tag_stats: dict,
) -> pd.DataFrame:
    """Apply pre-computed tag statistics to each post.

    Returns DataFrame indexed by pid with enhanced tag features:
        avg_tag_popularity, avg_tag_pop_bayes, max_tag_popularity,
        avg_tag_pop_recent, avg_tag_trend, avg_tag_freq_delta,
        avg_tag_frequency, known_tag_ratio, tag_diversity,
        top3_tag_pop, top1_tag_pop
    """
    tag_mean = tag_stats["tag_mean"]
    tag_bayes = tag_stats["tag_bayes"]
    tag_freq = tag_stats["tag_freq"]
    tag_mean_recent = tag_stats["tag_mean_recent"]
    tag_trend = tag_stats["tag_trend"]
    tag_freq_delta = tag_stats["tag_freq_delta"]
    global_mean = tag_stats["global_mean"]
    total_unique_tags = max(tag_stats["total_unique_tags"], 1)

    results = []
    for _, row in posts.iterrows():
        content = row["post_content"]
        tags = str(content).split(",") if pd.notna(content) else []
        tags_clean = [t.strip().lower() for t in tags if t.strip()]

        if tags_clean:
            # Collect stats per tag
            pop_vals = [tag_mean.get(t, np.nan) for t in tags_clean]
            bayes_vals = [tag_bayes.get(t, global_mean) for t in tags_clean]
            freq_vals = [tag_freq.get(t, 0) for t in tags_clean]
            recent_vals = [tag_mean_recent.get(t, np.nan) for t in tags_clean]
            trend_vals = [tag_trend.get(t, 0.0) for t in tags_clean]
            freq_delta_vals = [tag_freq_delta.get(t, 0.0) for t in tags_clean]

            pop_arr = np.array([v for v in pop_vals if not np.isnan(v)])
            recent_arr = np.array([v for v in recent_vals if not np.isnan(v)])

            # Aggregations
            avg_pop = float(np.nanmean(pop_vals)) if pop_arr.size > 0 else 0.0
            avg_bayes = float(np.nanmean(bayes_vals))
            max_pop = float(np.nanmax(pop_vals)) if pop_arr.size > 0 else global_mean
            min_pop = float(np.nanmin(pop_vals)) if pop_arr.size > 0 else global_mean
            avg_recent = float(np.nanmean(recent_vals)) if recent_arr.size > 0 else avg_pop
            avg_trend = float(np.nanmean(trend_vals))
            avg_freq_delta = float(np.nanmean(freq_delta_vals))
            avg_freq = float(np.nanmean(freq_vals)) if freq_vals else 0.0

            # Top-k
            sorted_idx = np.argsort(pop_arr) if pop_arr.size > 0 else None
            if sorted_idx is not None and len(sorted_idx) >= 1:
                top1 = float(pop_arr[sorted_idx[-1]])
                top3 = float(np.mean(pop_arr[sorted_idx[-3:]])) if len(sorted_idx) >= 3 else top1
            else:
                top1 = global_mean
                top3 = global_mean

            known = sum(1 for t in tags_clean if t in tag_mean)
            known_ratio = known / len(tags_clean)
            diversity = len(set(tags_clean)) / total_unique_tags
        else:
            avg_pop = 0.0
            avg_bayes = global_mean
            max_pop = global_mean
            min_pop = global_mean
            avg_recent = global_mean
            avg_trend = 0.0
            avg_freq_delta = 0.0
            avg_freq = 0.0
            top1 = global_mean
            top3 = global_mean
            known_ratio = 0.0
            diversity = 0.0

        results.append({
            "pid": row["pid"],
            "avg_tag_popularity": avg_pop,
            "avg_tag_pop_bayes": avg_bayes,
            "max_tag_popularity": max_pop,
            "min_tag_popularity": min_pop,
            "avg_tag_pop_recent": avg_recent,
            "avg_tag_trend": avg_trend,
            "avg_tag_freq_delta": avg_freq_delta,
            "avg_tag_frequency": avg_freq,
            "top1_tag_popularity": top1,
            "top3_tag_popularity": top3,
            "known_tag_ratio": known_ratio,
            "tag_diversity": diversity,
        })

    return pd.DataFrame(results).set_index("pid")
