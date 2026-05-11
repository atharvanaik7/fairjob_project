"""
fairness_analysis.py
====================
Main analysis pipeline for "From Prediction to Control: Fairness in Job Ad
Exposure Using the FairJob Dataset" (DSCI 599, USC, 2025).

Atharva Naik  <asnaik@usc.edu>
Shreyas Pasumarthi  <srpasuma@usc.edu>

Requirements
------------
    pip install datasets lightgbm scikit-learn pandas numpy

Usage
-----
    python fairness_analysis.py

The script loads the FairJob dataset from HuggingFace, trains a LightGBM
model, computes baseline fairness metrics, runs re-ranking interventions
(score boost + FA*IR), evaluates a position-debiased model, and prints a
full results table.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datasets import load_dataset
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_fairjob() -> pd.DataFrame:
    """
    Download FairJob from HuggingFace and return as a DataFrame.
    Columns include: click, label (seniority), gender (protected attr),
    num16-num50 (numeric), cat0-cat12 (categorical).
    """
    print("Loading FairJob dataset from HuggingFace...")
    ds = load_dataset("criteo/FairJob", split="train")
    df = ds.to_pandas()
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    print(f"  Columns: {list(df.columns)}")

    # Auto-detect protected attribute column
    protected_candidates = ["gender", "sensitive", "protected_attr", "protected"]
    protected_col = next((c for c in protected_candidates if c in df.columns), None)
    if protected_col is None:
        exclude = {"user_id", "click", "label", "senior", "displayrandom"}
        candidates = [c for c in df.columns
                      if c not in exclude
                      and not c.startswith(("num", "cat"))
                      and df[c].nunique() == 2]
        if candidates:
            protected_col = candidates[0]
            print(f"  Auto-detected protected column: '{protected_col}'")
        else:
            raise ValueError(f"Could not find protected attribute. Columns: {list(df.columns)}")

    # Auto-detect seniority column
    senior_candidates = ["label", "senior", "seniority"]
    senior_col = next((c for c in senior_candidates if c in df.columns), None)
    if senior_col is None:
        raise ValueError("Could not find seniority/label column.")

    print(f"  Protected: '{protected_col}', Senior: '{senior_col}'")
    df.rename(columns={protected_col: "protected", senior_col: "senior"}, inplace=True)

    # Encode categoricals
    cat_cols = [c for c in df.columns if c.startswith("cat")]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Session filter: keep sessions with >= 10 candidates
    session_sizes = df.groupby("user_id").size()
    valid_users = session_sizes[session_sizes >= 10].index
    df = df[df["user_id"].isin(valid_users)].copy()
    print(f"  After session filter: {len(df):,} rows, "
          f"{df['user_id'].nunique():,} sessions")

    return df.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE SETS
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return all feature columns, excluding protected attribute and labels."""
    exclude = {"user_id", "click", "senior", "protected", "displayrandom"}
    return [c for c in df.columns if c not in exclude]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_lgbm(X_train, y_train, X_test, y_test) -> tuple:
    """Train LightGBM click predictor; return model and test AUC."""
    model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, auc


def train_logistic(X_train, y_train, X_test, y_test) -> tuple:
    """Train logistic regression baseline; return model and test AUC."""
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, auc


# ─────────────────────────────────────────────────────────────────────────────
# 4.  FAIRNESS METRICS
# ─────────────────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: np.ndarray, k: int = 10) -> float:
    """
    Compute DCG@k for a single ranked list.
    relevances: array of binary relevance values in rank order (best first).
    """
    relevances = np.asarray(relevances[:k], dtype=float)
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1)
    return float(np.sum(relevances / np.log2(positions + 1)))


def ideal_dcg_at_k(relevances: np.ndarray, k: int = 10) -> float:
    """Compute ideal DCG@k (sort by relevance descending first)."""
    sorted_rels = np.sort(relevances)[::-1]
    return dcg_at_k(sorted_rels, k)


def compute_session_metrics(
    session_df: pd.DataFrame,
    score_col: str = "score",
    k: int = 10,
) -> dict:
    """
    For one ranking session, sort by score, then compute per-group metrics:
      - senior exposure rate in top-k
      - DCG@k
      - NDCG@k
      - average rank of senior items
    Returns a dict with keys suffixed _g0 and _g1.
    """
    ranked = session_df.sort_values(score_col, ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1  # 1-indexed

    results = {}
    for g in [0, 1]:
        grp = ranked[ranked["protected"] == g]
        top_k = ranked.head(k)
        top_k_grp = top_k[top_k["protected"] == g]

        # Exposure rate: fraction of top-k that are (this group) AND senior
        n_senior_top_k = top_k["senior"].sum()
        grp_senior_top_k = top_k_grp["senior"].sum()
        exp_rate = grp_senior_top_k / k

        # DCG@k: relevance = seniority label for items of this group in top-k
        # We compute over the full top-k list, treating non-group items as 0
        rel_vector = np.where(
            (top_k["protected"] == g) & (top_k["senior"] == 1), 1, 0
        )
        dcg = dcg_at_k(rel_vector, k)

        # NDCG: ideal is all senior items of this group at the top
        ideal_rel = np.ones(min(int(grp["senior"].sum()), k))
        idcg = ideal_dcg_at_k(ideal_rel, k)
        ndcg = dcg / idcg if idcg > 0 else 0.0

        # Average rank of senior items for this group
        senior_ranks = grp[grp["senior"] == 1]["rank"]
        avg_rank = float(senior_ranks.mean()) if len(senior_ranks) > 0 else np.nan

        results[f"exp_g{g}"] = float(exp_rate)
        results[f"dcg_g{g}"] = float(dcg)
        results[f"ndcg_g{g}"] = float(ndcg)
        results[f"avg_rank_g{g}"] = float(avg_rank)

    return results


def compute_fairness_metrics(
    df: pd.DataFrame,
    score_col: str = "score",
    k: int = 10,
) -> pd.DataFrame:
    """
    Compute per-session fairness metrics and return a summary DataFrame.
    """
    records = []
    for uid, session in df.groupby("user_id"):
        m = compute_session_metrics(session, score_col=score_col, k=k)
        m["user_id"] = uid
        records.append(m)

    sess_df = pd.DataFrame(records)

    summary = {
        "exp_g0": sess_df["exp_g0"].mean(),
        "exp_g1": sess_df["exp_g1"].mean(),
        "exp_gap": sess_df["exp_g0"].mean() - sess_df["exp_g1"].mean(),
        "dcg_g0": sess_df["dcg_g0"].mean(),
        "dcg_g1": sess_df["dcg_g1"].mean(),
        "dcg_gap": sess_df["dcg_g0"].mean() - sess_df["dcg_g1"].mean(),
        "ndcg_g0": sess_df["ndcg_g0"].mean(),
        "ndcg_g1": sess_df["ndcg_g1"].mean(),
        "avg_rank_g0": sess_df["avg_rank_g0"].mean(),
        "avg_rank_g1": sess_df["avg_rank_g1"].mean(),
    }
    return summary, sess_df


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INTERVENTIONS
# ─────────────────────────────────────────────────────────────────────────────

def score_boost_rerank(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    """
    Add lambda to predicted score for Group 0 senior items.
    Returns a copy of df with 'score' updated.
    """
    df = df.copy()
    mask = (df["protected"] == 0) & (df["senior"] == 1)
    df.loc[mask, "score"] = df.loc[mask, "score"] + lam
    return df


def fair_rerank(session_df: pd.DataFrame, p: float, k: int = 10) -> pd.DataFrame:
    """
    Simple FA*IR-style re-ranking for one session.
    Enforces that at least ceil(p * k * base_rate) Group 0 items appear in top-k.
    Falls back to score ranking for remaining slots.

    p: minimum proportion of Group 0 in top-k (fairness parameter)
    k: cutoff
    """
    session_df = session_df.sort_values("score", ascending=False).reset_index(drop=True)

    # How many Group 0 items do we need in top-k?
    n_g0_required = int(np.ceil(p * k * 0.5))  # 0.5 = balanced groups
    n_g0_required = min(n_g0_required, k)

    g0_items = session_df[session_df["protected"] == 0].copy()
    g1_items = session_df[session_df["protected"] == 1].copy()

    top_k = []
    g0_count = 0

    for pos in range(k):
        slots_left = k - pos
        g0_still_needed = n_g0_required - g0_count

        if g0_still_needed >= slots_left and len(g0_items) > 0:
            # Must pick Group 0
            top_k.append(g0_items.iloc[0])
            g0_items = g0_items.iloc[1:]
            g0_count += 1
        elif len(g1_items) > 0 and (
            len(g0_items) == 0 or
            (g0_count >= n_g0_required and
             g1_items.iloc[0]["score"] >= (g0_items.iloc[0]["score"] if len(g0_items) > 0 else -np.inf))
        ):
            top_k.append(g1_items.iloc[0])
            g1_items = g1_items.iloc[1:]
        elif len(g0_items) > 0:
            top_k.append(g0_items.iloc[0])
            g0_items = g0_items.iloc[1:]
            g0_count += 1
        elif len(g1_items) > 0:
            top_k.append(g1_items.iloc[0])
            g1_items = g1_items.iloc[1:]

    # Remaining items (beyond top-k) appended in score order
    remaining = pd.concat([g0_items, g1_items]).sort_values("score", ascending=False)
    result = pd.concat([pd.DataFrame(top_k), remaining]).reset_index(drop=True)
    return result


def apply_fair_rerank(df: pd.DataFrame, p: float, k: int = 10) -> pd.DataFrame:
    """Apply FA*IR re-ranking across all sessions."""
    reranked = []
    for uid, session in df.groupby("user_id"):
        reranked.append(fair_rerank(session, p=p, k=k))
    return pd.concat(reranked).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TEMPORAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def temporal_analysis(df: pd.DataFrame, n_windows: int = 5) -> pd.DataFrame:
    """
    Split sessions into n_windows contiguous windows and compute fairness
    metrics per window. Returns a DataFrame with one row per window.
    """
    sessions = list(df["user_id"].unique())
    window_size = len(sessions) // n_windows
    rows = []

    for i in range(n_windows):
        start = i * window_size
        end = (i + 1) * window_size if i < n_windows - 1 else len(sessions)
        window_users = sessions[start:end]
        window_df = df[df["user_id"].isin(window_users)]

        summary, _ = compute_fairness_metrics(window_df, score_col="score")
        rows.append({
            "window": f"W{i+1} ({start}–{end-1})",
            "exp_gap": summary["exp_gap"],
            "dcg_gap": summary["dcg_gap"],
            "dcg_g0": summary["dcg_g0"],
            "dcg_g1": summary["dcg_g1"],
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(label: str, summary: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    print(f"  Exposure G0/G1:  {summary['exp_g0']:.3f} / {summary['exp_g1']:.3f}  "
          f"(gap {summary['exp_gap']:+.3f})")
    print(f"  DCG@10  G0/G1:  {summary['dcg_g0']:.3f} / {summary['dcg_g1']:.3f}  "
          f"(gap {summary['dcg_gap']:+.3f})")
    print(f"  NDCG@10 G0/G1:  {summary['ndcg_g0']:.3f} / {summary['ndcg_g1']:.3f}")
    print(f"  Avg Rank G0/G1: {summary['avg_rank_g0']:.1f} / {summary['avg_rank_g1']:.1f}")


def main():
    # ── Load & preprocess ──────────────────────────────────────────────────
    df_raw = load_fairjob()
    df = preprocess(df_raw)

    feature_cols = get_feature_cols(df)
    X = df[feature_cols]
    y = df["click"]

    # 80/20 train/test split (stratified by click)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train models ──────────────────────────────────────────────────────
    print("\nTraining LightGBM...")
    lgbm_model, lgbm_auc = train_lgbm(X_train, y_train, X_test, y_test)
    print(f"  LightGBM AUC = {lgbm_auc:.3f}")

    print("Training Logistic Regression...")
    lr_model, lr_auc = train_logistic(X_train, y_train, X_test, y_test)
    print(f"  Logistic Regression AUC = {lr_auc:.3f}")

    # ── Assign scores to full dataset ─────────────────────────────────────
    df = df.copy()
    df["score"] = lgbm_model.predict_proba(df[feature_cols])[:, 1]

    # ── Baseline fairness ─────────────────────────────────────────────────
    baseline_summary, baseline_sess = compute_fairness_metrics(df, score_col="score")
    print_summary("BASELINE (LightGBM, no intervention)", baseline_summary)

    # Click rate and TPR per group
    for g in [0, 1]:
        grp = df[df["protected"] == g]
        click_rate = grp["click"].mean()
        # TPR at median predicted score threshold
        threshold = df["score"].median()
        pred_pos = grp["score"] >= threshold
        tpr = (pred_pos & (grp["click"] == 1)).sum() / max(grp["click"].sum(), 1)
        print(f"  Click rate G{g}: {click_rate:.4f}  |  TPR G{g}: {tpr:.3f}")

    # ── Score boost re-ranking ────────────────────────────────────────────
    print("\n\nSCORE BOOST RE-RANKING (Group 0 boosted)")
    print(f"  {'Lambda':<10} {'Exp Gap':>10} {'DCG Gap':>10} {'DCG G0':>10} {'DCG G1':>10}")
    lambdas = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1]
    for lam in lambdas:
        df_reranked = score_boost_rerank(df, lam=lam)
        s, _ = compute_fairness_metrics(df_reranked, score_col="score")
        print(f"  {lam:<10.3f} {s['exp_gap']:>+10.3f} {s['dcg_gap']:>+10.3f} "
              f"{s['dcg_g0']:>10.3f} {s['dcg_g1']:>10.3f}")

    # ── FA*IR re-ranking ──────────────────────────────────────────────────
    print("\n\nFA*IR RE-RANKING")
    print(f"  {'p':<10} {'Exp Gap':>10} {'DCG Gap':>10} {'DCG G0':>10} {'DCG G1':>10}")
    for p in [0.5, 0.7, 0.9]:
        df_fair = apply_fair_rerank(df, p=p, k=10)
        s, _ = compute_fairness_metrics(df_fair, score_col="score")
        print(f"  {p:<10.1f} {s['exp_gap']:>+10.3f} {s['dcg_gap']:>+10.3f} "
              f"{s['dcg_g0']:>10.3f} {s['dcg_g1']:>10.3f}")

    # ── Position-debiased model ───────────────────────────────────────────
    if "displayrandom" in df.columns:
        print("\nTraining position-debiased LightGBM (displayrandom only)...")
        df_rand = df[df["displayrandom"] == 1]
        if len(df_rand) > 1000:
            Xr = df_rand[feature_cols]
            yr = df_rand["click"]
            Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
                Xr, yr, test_size=0.2, random_state=42, stratify=yr
            )
            debiased_model, debiased_auc = train_lgbm(Xr_tr, yr_tr, Xr_te, yr_te)
            print(f"  Debiased LightGBM AUC = {debiased_auc:.3f}")
            df_debiased = df.copy()
            df_debiased["score"] = debiased_model.predict_proba(df[feature_cols])[:, 1]
            s, _ = compute_fairness_metrics(df_debiased, score_col="score")
            print_summary("POSITION-DEBIASED MODEL", s)
        else:
            print("  Not enough displayrandom rows for training.")

    # ── Temporal analysis ─────────────────────────────────────────────────
    print("\n\nTEMPORAL ANALYSIS (5 windows)")
    temporal_df = temporal_analysis(df, n_windows=5)
    print(temporal_df.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
