"""
model.py — Feature Selection & Model Training for GitHub User Churn Prediction
===============================================================================
Step 5: Feature Selection (Four Methods)
Course: Introduction to Data Science — Informatics Engineering

PURPOSE
-------
This module does two distinct jobs that the documentation separates clearly:

  JOB 1 — FEATURE SELECTION (the analytical core of Step 5)
    Apply all four required selection methods to the 10-feature pool
    produced by features.py and synthesize a comparison table. This answers
    the question: "which of my 10 hypotheses actually hold up empirically?"

  JOB 2 — MODEL TRAINING & SERIALIZATION
    Train a Random Forest on the FINAL selected features and save it to
    model.pkl so that main.py (FastAPI) can load it at startup and serve
    predictions without re-training.

THE FOUR SELECTION METHODS (from IDS_May29th.pdf, Step 5)
----------------------------------------------------------
  Method 1 — Filter (model-agnostic, fast, first pass)
    • Variance Threshold : drop features that barely vary across users —
      a feature that is nearly constant can't separate churners from retained.
    • Correlation Matrix : drop one of any pair correlated > 0.9 —
      redundant features add noise without adding information.
    • ANOVA F-test (SelectKBest / f_classif) : rank features by how much
      their distribution DIFFERS between churned and retained users.
      A feature where both groups look identical is useless for classification.

  Method 2 — Wrapper: RFE with Logistic Regression
    Recursive Feature Elimination trains a LogisticRegression, ranks features
    by coefficient magnitude, removes the weakest, retrains, and repeats until
    n_features_to_select remain. Unlike filters it accounts for feature
    interactions, but it can only capture LINEAR relationships (LR is a linear
    model). That is why its results may diverge from tree-based methods.

  Method 3 — Decision Tree importance
    A single DecisionTreeClassifier (max_depth=5) is trained and its
    feature_importances_ attribute is read. Each importance score is the
    total reduction in Gini impurity attributable to that feature across all
    splits. Fast and interpretable, but UNSTABLE — a different random seed
    can produce meaningfully different rankings.

  Method 4 — Random Forest importance
    A RandomForestClassifier (n_estimators=100) averages feature_importances_
    across 100 independently trained trees. The averaging smooths out the
    instability of any single tree. This is the most TRUSTED ranking: if a
    feature is consistently important across 100 trees trained on different
    bootstrap samples, the signal is real.

FINAL FEATURE SELECTION RULE (from comparison table)
------------------------------------------------------
  ✅ KEEP   — feature ranks in the top half of ALL four methods
  ⚠️ OPTIONAL — ranks highly in 2–3 methods; test with/without
  ❌ DROP   — low importance across all methods; adds noise, no signal

  The FINAL_FEATURES constant at the bottom of this file captures the
  empirical result of running select_features() on your actual data.
  It must be updated to match your real comparison table output BEFORE
  running train_and_save() or importing this module into main.py.

DESIGN PRINCIPLE: SEPARATION OF CONCERNS
-----------------------------------------
  scraper.py  → fetches raw data, assigns churn label     (Step 3)
  features.py → transforms raw data into feature matrix   (Step 4)
  model.py    → selects features, trains model, saves pkl (Step 5 + bridge to Step 6)
  main.py     → loads pkl, serves /predict endpoint       (Step 6)

  model.py has NO knowledge of the GitHub API. It only sees DataFrames.
  This makes it testable in isolation and reusable with any data source.

USAGE
-----
  # Run standalone to execute all four selection methods and train the model:
  python model.py

  # Or import into the EDA notebook:
  from model import select_features, train_and_save, FINAL_FEATURES
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

# scikit-learn: feature selection tools
from sklearn.feature_selection import (
    VarianceThreshold,   # Method 1a — removes near-constant features
    SelectKBest,         # Method 1c — ranks by ANOVA F-score
    f_classif,           # scoring function for SelectKBest (ANOVA F-test)
    RFE,                 # Method 2  — Recursive Feature Elimination
)

# scikit-learn: models used inside selection methods
from sklearn.linear_model import LogisticRegression   # used inside RFE
from sklearn.tree import DecisionTreeClassifier        # Method 3
from sklearn.ensemble import RandomForestClassifier    # Method 4 + final model

# scikit-learn: evaluation tools
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.preprocessing import StandardScaler      # needed for RFE / LogReg

# Our own modules — features.py (Step 4) is the upstream dependency
from features import generate_features, FEATURE_COLUMNS

warnings.filterwarnings("ignore")  # suppress convergence warnings from LogisticRegression


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Path where the trained model is saved for main.py to load.
# Must match the path in the Dockerfile COPY and main.py's joblib.load() call.
MODEL_PATH = "model.pkl"

# Path where the processed feature CSV is stored (written by features.py __main__).
FEATURES_CSV_PATH = "data/processed/github_features.csv"

# Raw data path — used if the processed CSV doesn't exist yet.
RAW_CSV_PATH = "data/raw/github_users.csv"

# ── FINAL SELECTED FEATURES ───────────────────────────────────────────────────
# This list is the OUTPUT of running select_features() on your actual dataset.
# It must be updated after you run the selection pipeline for the first time.
#
# HOW TO UPDATE:
#   1. Run: python model.py
#   2. Look at the printed comparison table at the bottom.
#   3. Identify features marked ✅ KEEP (high rank across all four methods).
#   4. Replace the list below with those feature names.
#   5. main.py's UserFeatures Pydantic model must match this list exactly.
#
# DEFAULT (pre-run placeholder): uses the four features from the docs example.
# After your first run, replace with your actual empirical results.
FINAL_FEATURES = [
    "days_since_last_activity",  # Almost always top-ranked; primary recency signal
    "activity_frequency",        # Average days between activities — captures consistent engagement vs bursts
    "repos_per_year",            # Rate-normalized productivity
]
# NOTE: activity_frequency and repos_per_year are mathematically identical
# (same formula, see features.py F5 vs F6). After running selection, only ONE
# of them will survive the correlation filter. Update FINAL_FEATURES accordingly.


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LOADING HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def load_feature_matrix(
    features_csv: str = FEATURES_CSV_PATH,
    raw_csv: str = RAW_CSV_PATH,
) -> tuple:
    """
    Load the feature matrix (X) and churn label (y) from disk.

    Tries the pre-processed CSV first (faster). Falls back to re-generating
    features from the raw CSV if the processed version doesn't exist yet.

    Parameters
    ----------
    features_csv : Path to the processed feature CSV (output of features.py)
    raw_csv      : Path to the raw scraper CSV (output of scraper.py)

    Returns
    -------
    X : pd.DataFrame  — feature matrix, columns = FEATURE_COLUMNS
    y : pd.Series     — binary churn labels (0 / 1)
    """
    if os.path.exists(features_csv):
        # Fast path: load the already-processed feature matrix
        print(f"Loading feature matrix from: {features_csv}")
        df = pd.read_csv(features_csv)
    elif os.path.exists(raw_csv):
        # Slow path: re-generate features from raw data
        print(f"Processed CSV not found. Generating features from: {raw_csv}")
        df_raw = pd.read_csv(raw_csv)
        X_gen, y_gen = generate_features(df_raw)
        # Re-attach the churn label so we can split below
        df = X_gen.copy()
        if y_gen is not None:
            df["churn"] = y_gen.values
    else:
        raise FileNotFoundError(
            f"Neither '{features_csv}' nor '{raw_csv}' found.\n"
            f"Run scraper.py first to collect data, then features.py to process it."
        )

    # Separate features from label
    if "churn" not in df.columns:
        raise ValueError("Dataset must contain a 'churn' column. Run scraper.py first.")

    # Keep only the columns that features.py defined — no accidental extras
    available = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[available].copy()
    y = df["churn"].astype(int)

    print(f"  Loaded: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Churn rate: {y.mean():.1%}  ({y.sum()} churned / {len(y)} total)\n")
    return X, y


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — METHOD 1: FILTER METHODS
# ═══════════════════════════════════════════════════════════════════════════════

def filter_method(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Method 1 — Filter Methods (three sub-filters combined into one pass).

    Filters are MODEL-AGNOSTIC: they evaluate each feature independently,
    without ever training a classifier. This makes them very fast and a
    good first pass to eliminate obviously useless features before the
    more expensive Wrapper and tree-based methods run.

    The three sub-filters applied here:

    1a. Variance Threshold
        Removes features whose variance is below 0.01.
        A feature with near-zero variance is nearly constant across all users —
        it cannot distinguish churners from retained users by definition.
        Example: if 99% of users have has_complete_profile = 0, that feature
        is useless no matter how theoretically interesting it is.

    1b. Correlation Matrix
        Identifies pairs of features correlated above 0.9.
        Keeping both adds almost no new information but increases model
        complexity and can destabilize coefficient-based methods (like RFE).
        Convention: drop the feature with the LOWER ANOVA F-score when a
        correlated pair is found (keeps the more individually predictive one).
        Note: activity_frequency and repos_per_year are mathematically
        identical (same formula in features.py), so they will always show
        correlation = 1.0 here — one will always be dropped.

    1c. ANOVA F-test (SelectKBest with f_classif)
        For each feature, computes the F-statistic measuring how much that
        feature's distribution DIFFERS between the churned (1) and retained (0)
        groups. A high F-score means the two groups have clearly different
        distributions for that feature — it IS useful for classification.
        A low F-score means the distributions overlap — the feature is
        not discriminating churn from retention.

    Parameters
    ----------
    X : Feature matrix (all 10 features from features.py)
    y : Binary churn labels

    Returns
    -------
    dict with keys:
      'variance_survivors'  : feature names that passed the variance threshold
      'corr_survivors'      : feature names that survived correlation pruning
      'anova_scores'        : pd.Series of F-scores, sorted descending
      'anova_top5'          : top 5 feature names by ANOVA F-score
      'filter_final'        : final list of features surviving all three filters
    """
    print("─" * 60)
    print("  METHOD 1 — FILTER METHODS")
    print("─" * 60)

    # ── 1a: Variance Threshold ────────────────────────────────────────────────
    # threshold=0.01 is the value from the documentation's code example.
    # Features with variance < 0.01 are essentially constant — useless.
    var_selector = VarianceThreshold(threshold=0.01)
    var_selector.fit(X)

    # Boolean mask: True = feature survived the variance threshold
    var_mask = var_selector.get_support()
    variance_survivors = X.columns[var_mask].tolist()

    print(f"\n  1a. Variance Threshold (threshold=0.01)")
    print(f"      Survived : {variance_survivors}")
    dropped_by_var = [c for c in X.columns if c not in variance_survivors]
    if dropped_by_var:
        print(f"      Dropped  : {dropped_by_var}  ← near-constant, no discriminating power")
    else:
        print(f"      Dropped  : none")

    # ── 1b: Correlation Matrix ────────────────────────────────────────────────
    # Only check features that survived the variance threshold.
    X_var = X[variance_survivors]
    corr_matrix = X_var.corr().abs()  # absolute value — we care about magnitude, not sign

    # Build a set of features to drop due to multicollinearity.
    # Strategy: iterate through all unique pairs; if correlation > 0.9,
    # mark the feature with the LOWER individual ANOVA F-score for removal.
    # This ensures we always keep the more individually predictive of the two.

    # Pre-compute ANOVA F-scores for all variance survivors (needed for tie-breaking)
    f_scores_series, _ = f_classif(X_var, y)
    f_scores_dict = dict(zip(variance_survivors, f_scores_series))

    to_drop_corr = set()
    cols = variance_survivors
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if corr_matrix.loc[cols[i], cols[j]] > 0.9:
                # Drop whichever has the lower F-score
                if f_scores_dict[cols[i]] < f_scores_dict[cols[j]]:
                    to_drop_corr.add(cols[i])
                else:
                    to_drop_corr.add(cols[j])

    corr_survivors = [c for c in variance_survivors if c not in to_drop_corr]

    print(f"\n  1b. Correlation Matrix (threshold=0.9)")
    if to_drop_corr:
        print(f"      Dropped  : {sorted(to_drop_corr)}  ← correlated > 0.9 with another feature")
    else:
        print(f"      Dropped  : none")
    print(f"      Survived : {corr_survivors}")

    # ── 1c: ANOVA F-test ──────────────────────────────────────────────────────
    # Run SelectKBest on the full original X to get a complete ranking of all
    # features. This gives us the F-score for every feature, not just survivors.
    # We use k='all' to get scores for every feature without actually dropping any.
    selector_k = SelectKBest(score_func=f_classif, k="all")
    selector_k.fit(X, y)

    anova_scores = pd.Series(
        selector_k.scores_,
        index=X.columns
    ).sort_values(ascending=False)

    # Top 5 by ANOVA F-score (matches the docs' "run RFE asking for top 5")
    anova_top5 = anova_scores.head(5).index.tolist()

    print(f"\n  1c. ANOVA F-test (f_classif) — all feature scores:")
    for feat, score in anova_scores.items():
        marker = "✅" if feat in anova_top5 else "  "
        print(f"      {marker} {feat:<35} F={score:.2f}")

    # ── Final filter result: intersection of corr_survivors and ANOVA top-5 ──
    # A feature must pass BOTH the structural filters (variance + correlation)
    # AND show meaningful statistical separation (ANOVA top 5) to be kept.
    filter_final = [f for f in corr_survivors if f in anova_top5]

    print(f"\n  Filter final selection (corr_survivors ∩ ANOVA top-5): {filter_final}")

    return {
        "variance_survivors": variance_survivors,
        "corr_survivors": corr_survivors,
        "anova_scores": anova_scores,
        "anova_top5": anova_top5,
        "filter_final": filter_final,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — METHOD 2: WRAPPER (RFE)
# ═══════════════════════════════════════════════════════════════════════════════

def rfe_method(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int = 5,
) -> dict:
    """
    Method 2 — Wrapper: Recursive Feature Elimination (RFE).

    RFE is fundamentally different from filters: it uses an ACTUAL MODEL
    to evaluate features. The process (from the documentation):
      1. Train LogisticRegression on all features
      2. Rank features by their absolute coefficient magnitude
         (larger coefficient = more important to the linear model)
      3. Remove the feature with the smallest coefficient
      4. Retrain on the remaining features
      5. Repeat until n_features_to_select features remain

    WHY LOGISTIC REGRESSION as the estimator:
    LR is the standard choice for RFE in the docs because it is fast,
    its coefficients are directly interpretable as feature importance,
    and it handles binary classification naturally. The trade-off:
    LR is a LINEAR model — it cannot capture non-linear relationships
    between features and churn. This is why RFE results may diverge
    from the tree-based methods (DT/RF), which can capture non-linearity.
    That divergence is analytically valuable — it tells you which
    features have LINEAR vs NON-LINEAR relationships with churn.

    SCALING NOTE:
    Logistic Regression is sensitive to feature scale — a feature with
    values in the thousands (like account_age_days) would dominate a
    feature with values 0–4 (like profile_completeness) just because of
    its magnitude, not its predictive power. We StandardScale before RFE
    to put all features on the same scale (mean=0, std=1).

    Parameters
    ----------
    X_train      : Training feature matrix (training split only — never touch test)
    y_train      : Training churn labels
    n_features   : How many features RFE should select (docs say 5)

    Returns
    -------
    dict with keys:
      'selected'  : list of feature names RFE kept
      'ranking'   : pd.Series of RFE rankings (1 = selected, higher = eliminated earlier)
      'rfe_object': the fitted RFE object (useful for debugging)
    """
    print("─" * 60)
    print("  METHOD 2 — WRAPPER: RFE (Recursive Feature Elimination)")
    print("─" * 60)

    # Scale features: LogisticRegression requires this for fair coefficient comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)  # fit only on training data

    # LogisticRegression with max_iter=1000 as specified in the docs
    lr = LogisticRegression(max_iter=1000, random_state=42)

    # RFE: n_features_to_select=5 as specified in the docs
    rfe = RFE(estimator=lr, n_features_to_select=n_features)
    rfe.fit(X_scaled, y_train)

    # rfe.support_ is a boolean array: True = selected by RFE
    selected = X_train.columns[rfe.support_].tolist()

    # rfe.ranking_ gives each feature's elimination order:
    # 1 = selected (survived to the end)
    # 2 = eliminated in the last round (was second-to-last weakest)
    # higher = eliminated earlier (weaker feature)
    ranking = pd.Series(rfe.ranking_, index=X_train.columns).sort_values()

    print(f"\n  RFE ranking (1 = selected, higher = eliminated first):")
    for feat, rank in ranking.items():
        marker = "✅" if rank == 1 else "  "
        print(f"      {marker} {feat:<35} rank={rank}")

    print(f"\n  RFE selected ({n_features} features): {selected}")

    return {
        "selected": selected,
        "ranking": ranking,
        "rfe_object": rfe,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — METHOD 3: DECISION TREE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def decision_tree_method(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 5,
) -> dict:
    """
    Method 3 — Decision Tree Feature Importance.

    A single DecisionTreeClassifier is trained and its feature_importances_
    attribute is read. Each importance value is the fraction of total Gini
    impurity reduction attributable to that feature across ALL splits in
    the tree. Features that appear near the root (early splits) and split
    the data cleanly get higher importance.

    ADVANTAGE: Interpretability. You can literally visualize the tree and see
    exactly WHICH feature and WHICH threshold drives each decision. This is
    uniquely valuable for the written report — you can say "the model first
    asks whether days_since_last_activity > 400" and trace the logic.

    DISADVANTAGE: Instability. A single tree is sensitive to the random
    seed used in train_test_split and the tree's own random_state. Training
    on slightly different data can produce a very different tree with
    different importance rankings. This is why the docs say "trust RF over DT
    when they disagree."

    max_depth=5: capping depth prevents the tree from overfitting perfectly
    to the training data, which would produce importances that don't generalize.

    Parameters
    ----------
    X_train  : Training feature matrix
    y_train  : Training churn labels
    max_depth: Maximum tree depth (docs recommend 5)

    Returns
    -------
    dict with keys:
      'importances' : pd.Series of importance scores, sorted descending
      'top5'        : list of top 5 feature names
      'model'       : the fitted DecisionTreeClassifier object
    """
    print("─" * 60)
    print("  METHOD 3 — DECISION TREE IMPORTANCE")
    print("─" * 60)

    # Train a single decision tree with max_depth=5 as recommended by docs
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)

    # feature_importances_ sums to 1.0 across all features
    importances = pd.Series(
        dt.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    top5 = importances.head(5).index.tolist()

    print(f"\n  Decision Tree importance scores (max_depth={max_depth}):")
    for i, (feat, imp) in enumerate(importances.items(), 1):
        marker = "✅" if feat in top5 else "  "
        print(f"      {marker} #{i}  {feat:<35} importance={imp:.4f}")

    print(f"\n  DT top-5: {top5}")

    return {
        "importances": importances,
        "top5": top5,
        "model": dt,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — METHOD 4: RANDOM FOREST IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════

def random_forest_method(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
) -> dict:
    """
    Method 4 — Random Forest Feature Importance.

    A RandomForestClassifier trains n_estimators=100 decision trees, each on
    a BOOTSTRAP SAMPLE (random subset with replacement) of the training data
    and a random subset of features at each split. The final feature_importances_
    is the AVERAGE importance across all 100 trees.

    WHY THIS IS MORE RELIABLE THAN A SINGLE DT:
    The averaging across 100 independently trained trees smooths out the
    variance of any individual tree's random choices. If a feature consistently
    appears as important across 100 trees trained on different slices of the data,
    that signal is real — it's not an artifact of one lucky split. The docs are
    explicit: "if a feature ranks highly in both DT and RF, it is genuinely
    important." When they disagree, trust RF.

    WHY THIS IS ALSO THE FINAL MODEL:
    The docs recommend using Random Forest as the final classifier for /predict
    (Step 6) because the same model used for importance ranking in Step 5 is
    already a good classifier. This is efficient: one fit serves both purposes.

    class_weight='balanced': compensates for class imbalance (the dataset has
    more retained users than churned users). This tells scikit-learn to weight
    each class inversely proportional to its frequency, giving the minority
    class (churned) more influence during training.

    Parameters
    ----------
    X_train      : Training feature matrix
    y_train      : Training churn labels
    n_estimators : Number of trees (docs specify 100)

    Returns
    -------
    dict with keys:
      'importances' : pd.Series of averaged importance scores, sorted descending
      'top5'        : list of top 5 feature names
      'model'       : the fitted RandomForestClassifier object
    """
    print("─" * 60)
    print("  METHOD 4 — RANDOM FOREST IMPORTANCE")
    print("─" * 60)

    # n_estimators=100 and class_weight='balanced' as per documentation
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",   # handles the class imbalance in churn datasets
        random_state=42,           # reproducibility
        n_jobs=-1,                 # use all available CPU cores
    )
    rf.fit(X_train, y_train)

    # feature_importances_ is already averaged across all 100 trees
    importances = pd.Series(
        rf.feature_importances_,
        index=X_train.columns
    ).sort_values(ascending=False)

    top5 = importances.head(5).index.tolist()

    print(f"\n  Random Forest importance scores ({n_estimators} trees):")
    for i, (feat, imp) in enumerate(importances.items(), 1):
        marker = "✅" if feat in top5 else "  "
        print(f"      {marker} #{i}  {feat:<35} importance={imp:.4f}")

    print(f"\n  RF top-5: {top5}")

    return {
        "importances": importances,
        "top5": top5,
        "model": rf,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — COMPARISON TABLE (the required deliverable of Step 5)
# ═══════════════════════════════════════════════════════════════════════════════

def build_comparison_table(
    filter_results: dict,
    rfe_results: dict,
    dt_results: dict,
    rf_results: dict,
    all_features: list,
) -> pd.DataFrame:
    """
    Synthesize results from all four selection methods into the comparison
    table required by the documentation.

    The table is the central deliverable of Step 5. The docs say:
    "You should not just run each method in isolation — they should produce
    a table that shows, for every feature, how each method ranked it.
    Agreements across all four methods are strong signals to keep a feature.
    Disagreements are interesting findings to discuss."

    Table columns:
      Feature           : feature name
      Filter_ANOVA_Rank : rank by ANOVA F-score (1 = highest F-score)
      RFE_Selected      : ✅ if RFE kept this feature, ❌ if eliminated
      DT_Rank           : rank by Decision Tree importance (1 = most important)
      RF_Rank           : rank by Random Forest importance (1 = most important)
      Methods_Agreeing  : count of methods that consider this a top-5 feature
      Decision          : ✅ Keep / ⚠️ Optional / ❌ Drop

    Parameters
    ----------
    filter_results : output dict from filter_method()
    rfe_results    : output dict from rfe_method()
    dt_results     : output dict from decision_tree_method()
    rf_results     : output dict from random_forest_method()
    all_features   : list of all feature names (FEATURE_COLUMNS order)

    Returns
    -------
    pd.DataFrame — the comparison table, printed to stdout and returned
    """
    # Build rank series for each method (rank 1 = best)
    # ANOVA: anova_scores is already sorted descending, so rank = position + 1
    anova_ranks = {
        feat: i + 1
        for i, feat in enumerate(filter_results["anova_scores"].index)
    }

    # DT: importances sorted descending, so rank = position + 1
    dt_ranks = {
        feat: i + 1
        for i, feat in enumerate(dt_results["importances"].index)
    }

    # RF: same
    rf_ranks = {
        feat: i + 1
        for i, feat in enumerate(rf_results["importances"].index)
    }

    rows = []
    for feat in all_features:
        filter_rank  = anova_ranks.get(feat, 99)
        rfe_selected = "✅" if feat in rfe_results["selected"] else "❌"
        dt_rank      = dt_ranks.get(feat, 99)
        rf_rank      = rf_ranks.get(feat, 99)

        # Count how many methods consider this feature "top-5"
        in_filter_top5 = feat in filter_results["anova_top5"]
        in_rfe         = feat in rfe_results["selected"]
        in_dt_top5     = feat in dt_results["top5"]
        in_rf_top5     = feat in rf_results["top5"]
        agreements     = sum([in_filter_top5, in_rfe, in_dt_top5, in_rf_top5])

        # Decision rule:
        #   4/4 or 3/4 methods agree → ✅ Keep
        #   2/4 methods agree        → ⚠️ Optional (test both ways)
        #   0–1 methods agree        → ❌ Drop
        if agreements >= 3:
            decision = "✅ Keep"
        elif agreements == 2:
            decision = "⚠️  Optional"
        else:
            decision = "❌ Drop"

        rows.append({
            "Feature":          feat,
            "Filter_Rank":      filter_rank,
            "RFE_Selected":     rfe_selected,
            "DT_Rank":          dt_rank,
            "RF_Rank":          rf_rank,
            "Methods_Agreeing": f"{agreements}/4",
            "Decision":         decision,
        })

    table = pd.DataFrame(rows).set_index("Feature")

    print("\n" + "═" * 75)
    print("  STEP 5 — FEATURE SELECTION COMPARISON TABLE")
    print("═" * 75)
    print(table.to_string())
    print("═" * 75)
    print("\n  LEGEND:")
    print("    Filter_Rank : rank by ANOVA F-score (1 = strongest individual signal)")
    print("    RFE_Selected: ✅ kept by Recursive Feature Elimination, ❌ eliminated")
    print("    DT_Rank     : rank by single Decision Tree importance")
    print("    RF_Rank     : rank by Random Forest averaged importance (most trusted)")
    print("    Decision    : 3–4 methods agree → Keep | 2 → Optional | 0–1 → Drop")
    print()

    return table


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FULL SELECTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def select_features(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Run all four selection methods on the full feature matrix and return
    the comparison table plus all intermediate results.

    This is the function to call from the EDA notebook for Step 5 analysis.
    It handles the train/test split internally so the test set is NEVER
    seen during feature selection (avoiding selection bias).

    Parameters
    ----------
    X : Full feature matrix (all 10 features, all rows)
    y : Full churn label Series

    Returns
    -------
    dict with keys:
      'table'      : the comparison DataFrame
      'filter'     : filter_method() results
      'rfe'        : rfe_method() results
      'dt'         : decision_tree_method() results
      'rf'         : random_forest_method() results
      'X_train', 'X_test', 'y_train', 'y_test' : the splits used
    """
    print("=" * 60)
    print("  STEP 5 — FEATURE SELECTION (Four Methods)")
    print("=" * 60)
    print(f"  Input: {X.shape[0]} rows × {X.shape[1]} features\n")

    # ── Train/test split ──────────────────────────────────────────────────────
    # stratify=y preserves the churn ratio in both splits.
    # This is critical for imbalanced datasets — without stratification,
    # the test set might accidentally contain mostly one class.
    # test_size=0.2 gives an 80/20 split.
    # random_state=42 ensures reproducibility across runs.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,       # preserves churn ratio in both splits
        random_state=42,
    )

    print(f"  Train set: {X_train.shape[0]} rows  "
          f"(churn rate: {y_train.mean():.1%})")
    print(f"  Test set : {X_test.shape[0]} rows  "
          f"(churn rate: {y_test.mean():.1%})\n")

    # ── Run all four methods ──────────────────────────────────────────────────
    # Filter uses the full X (model-agnostic, no data leakage risk)
    filter_res = filter_method(X, y)

    # Wrapper and tree-based methods use only the training split
    rfe_res    = rfe_method(X_train, y_train, n_features=5)
    dt_res     = decision_tree_method(X_train, y_train, max_depth=5)
    rf_res     = random_forest_method(X_train, y_train, n_estimators=100)

    # ── Build comparison table ────────────────────────────────────────────────
    table = build_comparison_table(
        filter_results=filter_res,
        rfe_results=rfe_res,
        dt_results=dt_res,
        rf_results=rf_res,
        all_features=X.columns.tolist(),
    )

    return {
        "table":   table,
        "filter":  filter_res,
        "rfe":     rfe_res,
        "dt":      dt_res,
        "rf":      rf_res,
        "X_train": X_train,
        "X_test":  X_test,
        "y_train": y_train,
        "y_test":  y_test,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MODEL TRAINING & SERIALIZATION (bridge to Step 6)
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_save(
    X: pd.DataFrame,
    y: pd.Series,
    selected_features: list = None,
    model_path: str = MODEL_PATH,
) -> dict:
    """
    Train the final Random Forest on the selected features and save it
    to disk as model.pkl. This is the bridge between Step 5 and Step 6.

    WHY RANDOM FOREST as the final classifier:
    The docs explicitly recommend RF for the final model (Step 6) because:
    - It was already used in Step 5 for importance ranking
    - It naturally handles non-linear relationships
    - It is robust to class imbalance (with class_weight='balanced')
    - It provides predict_proba() for churn probability, not just binary output

    WHAT GETS SAVED IN model.pkl:
    joblib serializes the entire fitted RandomForestClassifier object,
    including all 100 trained trees and their split thresholds. When
    main.py calls joblib.load("model.pkl"), it gets back a fully
    functional model that can call .predict() and .predict_proba()
    immediately — no retraining needed.

    5-FOLD CROSS-VALIDATION:
    Before saving, we validate with 5-fold stratified cross-validation.
    This is the "Validate with Cross-Validation" requirement from the docs.
    5-fold means the data is split into 5 equal parts; the model is trained
    on 4 and tested on 1, five times, and the scores are averaged. This
    gives a more reliable performance estimate than a single train/test split.

    Parameters
    ----------
    X                 : Full feature matrix (all rows)
    y                 : Full churn label Series
    selected_features : Which features to use for the final model.
                        If None, uses FINAL_FEATURES constant from this file.
    model_path        : Where to save model.pkl

    Returns
    -------
    dict with performance metrics:
      'cv_accuracy', 'cv_precision', 'cv_recall', 'cv_f1',
      'test_accuracy', 'test_precision', 'test_recall', 'test_f1',
      'classification_report', 'model', 'selected_features'
    """
    if selected_features is None:
        selected_features = FINAL_FEATURES

    # Validate that all selected features actually exist in X
    missing = [f for f in selected_features if f not in X.columns]
    if missing:
        raise ValueError(
            f"Selected features not found in X: {missing}\n"
            f"Available: {X.columns.tolist()}"
        )

    # Subset to only the final selected features
    X_final = X[selected_features]

    print("=" * 60)
    print("  TRAINING FINAL MODEL")
    print("=" * 60)
    print(f"  Selected features ({len(selected_features)}): {selected_features}")

    # ── Train/test split (same seed as select_features for consistency) ───────
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # ── Final Random Forest classifier ────────────────────────────────────────
    final_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # ── 5-fold stratified cross-validation ───────────────────────────────────
    # StratifiedKFold preserves churn ratio in each fold — essential for
    # imbalanced datasets (if a fold happens to have no churned users,
    # precision/recall metrics would be undefined).
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n  Running 5-fold stratified cross-validation...")

    # Score on multiple metrics as required by the documentation
    cv_accuracy  = cross_val_score(final_rf, X_final, y, cv=cv, scoring="accuracy").mean()
    cv_precision = cross_val_score(final_rf, X_final, y, cv=cv, scoring="precision").mean()
    cv_recall    = cross_val_score(final_rf, X_final, y, cv=cv, scoring="recall").mean()
    cv_f1        = cross_val_score(final_rf, X_final, y, cv=cv, scoring="f1").mean()

    print(f"\n  Cross-Validation Results (5-fold, stratified):")
    print(f"    Accuracy  : {cv_accuracy:.3f}")
    print(f"    Precision : {cv_precision:.3f}")
    print(f"    Recall    : {cv_recall:.3f}")
    print(f"    F1 Score  : {cv_f1:.3f}")
    print()
    print("  NOTE: F1 is the primary metric for imbalanced churn datasets.")
    print("  A high F1 on held-out folds confirms the feature set is genuinely")
    print("  informative and not just memorizing the training data.\n")

    # ── Train final model on full training split ──────────────────────────────
    # Re-train on the full training set (not cross-val folds) for deployment
    final_rf.fit(X_train, y_train)

    # ── Evaluate on held-out test set ────────────────────────────────────────
    y_pred = final_rf.predict(X_test)

    test_accuracy  = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall    = recall_score(y_test, y_pred, zero_division=0)
    test_f1        = f1_score(y_test, y_pred, zero_division=0)
    report         = classification_report(y_test, y_pred,
                                           target_names=["Retained", "Churned"],
                                           zero_division=0)

    print("  Test Set Results (held-out 20%):")
    print(f"    Accuracy  : {test_accuracy:.3f}")
    print(f"    Precision : {test_precision:.3f}")
    print(f"    Recall    : {test_recall:.3f}")
    print(f"    F1 Score  : {test_f1:.3f}")
    print()
    print("  Classification Report:")
    print(report)

    # ── Save model to disk ────────────────────────────────────────────────────
    # joblib is preferred over pickle for scikit-learn objects because it
    # handles large numpy arrays (the 100 trees) more efficiently.
    os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else ".", exist_ok=True)
    joblib.dump(final_rf, model_path)
    print(f"  Model saved to: {model_path}")
    print(f"  Load in main.py with: model = joblib.load('{model_path}')\n")

    return {
        "cv_accuracy":           cv_accuracy,
        "cv_precision":          cv_precision,
        "cv_recall":             cv_recall,
        "cv_f1":                 cv_f1,
        "test_accuracy":         test_accuracy,
        "test_precision":        test_precision,
        "test_recall":           test_recall,
        "test_f1":               test_f1,
        "classification_report": report,
        "model":                 final_rf,
        "selected_features":     selected_features,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Run this script directly to execute the full Step 5 pipeline:

        python model.py

    What happens in order:
      1. Load feature matrix from data/processed/github_features.csv
         (or regenerate from data/raw/github_users.csv if not found)
      2. Run all four selection methods and print results
      3. Print the comparison table
      4. Train the final Random Forest on FINAL_FEATURES
      5. Run 5-fold cross-validation and print metrics
      6. Save model.pkl to disk for main.py to load

    WORKFLOW FOR FIRST RUN:
      1. Run this script once → read the comparison table output
      2. Update FINAL_FEATURES at the top of this file to match the
         features marked ✅ Keep in your comparison table
      3. Run this script again → the final model uses your empirical selection
      4. Proceed to main.py (Step 6) with your saved model.pkl
    """

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y = load_feature_matrix()

    # ── Run all four selection methods ────────────────────────────────────────
    selection_results = select_features(X, y)

    # ── Remind the user to update FINAL_FEATURES ─────────────────────────────
    # Identify which features were marked ✅ Keep in the comparison table
    table = selection_results["table"]
    keep_features = table[table["Decision"] == "✅ Keep"].index.tolist()

    print("─" * 60)
    print("  NEXT STEP: Update FINAL_FEATURES")
    print("─" * 60)
    print(f"  Features marked ✅ Keep in your table: {keep_features}")
    print()
    print("  Update the FINAL_FEATURES constant at the top of this file")
    print("  to match the list above, then re-run: python model.py")
    print()

    # ── Train and save the final model ────────────────────────────────────────
    # Uses whatever FINAL_FEATURES is set to at the top of this file.
    # On first run this uses the default placeholder; update and re-run.
    training_results = train_and_save(X, y)

    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  model.pkl saved → ready for main.py (Step 6)")
    print(f"  Final features used: {training_results['selected_features']}")
    print(f"  CV F1 Score: {training_results['cv_f1']:.3f}")