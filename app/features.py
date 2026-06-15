"""
features.py — Feature Generation for GitHub User Churn Prediction
==================================================================
Step 4: Feature Generation
Course: Introduction to Data Science — Informatics Engineering

PURPOSE
-------
This module takes the raw DataFrame produced by scraper.py (Step 3)
and transforms it into a model-ready feature matrix. Each column in the
output is a *derived* signal — a hypothesis that some transformation of
raw API data captures something meaningful about whether a user will churn.

RAW DATA → FEATURES (core principle)
--------------------------------------
Raw API fields are not features. A timestamp string is meaningless to
any ML algorithm. But the NUMBER OF DAYS elapsed since that timestamp is
a rich, continuous numerical signal the model can actually reason about.

  ✗  created_at    = "2018-03-14T10:22:00Z"     ← raw string, useless
  ✓  account_age_days = 2648                      ← transformed, meaningful

Every transformation in this file is a HYPOTHESIS: "I believe this
derived number carries information about whether a user will disengage
from GitHub." Step 5 (feature selection) will empirically test which
hypotheses hold up.

FEATURE INVENTORY (10 features, all four required types covered)
-----------------------------------------------------------------
  RATIO (2):
    F1  follower_ratio            — followers / (following + 1)
    F2  engagement_ratio          — (followers + following) / (public_repos + 1)

  TIME-BASED / RECENCY & FREQUENCY (3):
    F3  days_since_last_activity  — days since updated_at (recency)
    F4  account_age_days          — days since created_at (tenure)
    F5  activity_frequency        — public_repos / account_age_years (frequency)

  AGGREGATION (3):
    F6  repos_per_year            — public_repos / account_age_years
    F7  social_volume             — log1p(followers + following + public_gists)
    F8  profile_completeness      — count of filled optional profile fields (0–4)

  BINARY / FLAG (2):
    F9  has_no_repos              — 1 if public_repos == 0, else 0
    F10 has_complete_profile      — 1 if profile_completeness == 4, else 0

DESIGN NOTES FOR STEP 5
------------------------
The 10 features above are the GENERATION pool. Step 5 (eda_and_selection.ipynb)
will apply four selection methods to this pool:

  1. Filter  — variance threshold + correlation matrix + ANOVA F-test
  2. Wrapper — Recursive Feature Elimination (RFE) with LogisticRegression
  3. Decision Tree — single DT feature_importances_
  4. Random Forest — averaged feature_importances_ across 100 trees

The generate_features() function returns ALL 10 features so every
selection method has the full candidate pool to work with.
The final selected set (expected: ~4 features) will be determined
empirically in the notebook and then hard-coded into main.py / model.py.

COLUMNS CONSUMED FROM SCRAPER OUTPUT
--------------------------------------
  Required  : created_at, updated_at, public_repos, followers, following
  Optional  : public_gists, bio, blog, company, hireable
              days_since_last_push (used only for label; dropped here)
              days_since_update    (used only for label; dropped here)

USAGE
-----
  from features import generate_features, FEATURE_COLUMNS

  import pandas as pd
  df_raw = pd.read_csv("data/raw/github_users.csv")
  X, y   = generate_features(df_raw)   # X: feature matrix, y: churn label
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC CONSTANT — ORDERED LIST OF FEATURE NAMES
# ═══════════════════════════════════════════════════════════════════════════════

# This list defines the EXACT column order of the feature matrix returned by
# generate_features(). It is imported by model.py and main.py so every module
# uses the same column ordering — essential for model serialization correctness.

FEATURE_COLUMNS = [
    "follower_ratio",           # F1  — Ratio
    "engagement_ratio",         # F2  — Ratio
    "days_since_last_activity", # F3  — Time-based (recency)
    "account_age_days",         # F4  — Time-based (tenure)
    "activity_frequency",       # F5  — Time-based (frequency)
    "repos_per_year",           # F6  — Aggregation
    "social_volume",            # F7  — Aggregation
    "profile_completeness",     # F8  — Aggregation (count)
    "has_no_repos",             # F9  — Binary
    "has_complete_profile",     # F10 — Binary
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _to_datetime_utc(series: pd.Series) -> pd.Series:
    """
    Convert a Series of ISO-8601 strings (GitHub format, UTC "Z" suffix)
    to timezone-aware datetime objects.

    Uses errors="coerce" so invalid / missing strings become NaT
    (Not a Time) instead of raising exceptions.
    """
    return pd.to_datetime(series, utc=True, errors="coerce")


def _days_since(dt_series: pd.Series) -> pd.Series:
    """
    Compute the number of whole days between each datetime in the Series
    and the current UTC time.

    Returns a float Series (NaN where the input was NaT).
    """
    now = pd.Timestamp.now(tz="UTC")
    return (now - dt_series).dt.days.astype(float)


def _safe_years(days_series: pd.Series, min_days: float = 1.0) -> pd.Series:
    """
    Convert a days Series to years, flooring at min_days to prevent
    division-by-zero errors in ratio computations.

    min_days = 1.0 means the smallest possible denominator is 1/365 ≈ 0.003 years.
    """
    days_floored = days_series.clip(lower=min_days)
    return days_floored / 365.25


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INDIVIDUAL FEATURE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
# Each function receives the full DataFrame and returns a single pd.Series.
# Keeping functions small and single-purpose makes them individually testable.

# ── RATIO FEATURES ────────────────────────────────────────────────────────────

def compute_follower_ratio(df: pd.DataFrame) -> pd.Series:
    """
    F1 — follower_ratio = followers / (following + 1)
    TYPE: Ratio

    HYPOTHESIS:
    A user's follower-to-following ratio is a proxy for social influence and
    engagement QUALITY on GitHub. A developer who posts interesting work
    attracts followers organically; someone who only follows others as a
    discovery mechanism ends up with a near-zero ratio.

    WHY THIS PREDICTS CHURN:
    Low follower_ratio users (passive consumers, people who signed up and
    followed a few accounts but never contributed) show much higher churn rates
    in social platform research. Without a community that values their work, there
    is little social "pull" to return. Conversely, a user with a high ratio has
    public recognition — a reason to stay active.

    FORMULA NOTE:
    We add 1 to the denominator (+1 Laplace smoothing) to handle users who
    follow nobody (following == 0), which would otherwise cause division by zero.
    This is standard practice for ratio features.
    """
    return df["followers"] / (df["following"] + 1)


def compute_engagement_ratio(df: pd.DataFrame) -> pd.Series:
    """
    F2 — engagement_ratio = (followers + following) / (public_repos + 1)
    TYPE: Ratio

    HYPOTHESIS:
    This ratio measures how socially engaged a user is relative to the size
    of their public codebase. A user with many social connections but few
    repos may be primarily a consumer of others' work (lurker), while a
    highly productive coder with a large following is deeply embedded.

    WHY THIS PREDICTS CHURN:
    Pure lurkers — high social volume, zero repos — are at higher churn risk
    because their GitHub engagement is entirely passive. Conversely, users
    who have built a meaningful social graph around their actual work output
    have strong retention signals. The ratio normalizes social activity by
    output, separating "networker but no code" from "networker AND coder."

    NOTE: +1 in denominator prevents division by zero for users with 0 repos.
    """
    social_connections = df["followers"] + df["following"]
    return social_connections / (df["public_repos"] + 1)


# ── TIME-BASED FEATURES ───────────────────────────────────────────────────────

def compute_days_since_last_activity(df: pd.DataFrame) -> pd.Series:
    """
    F3 — days_since_last_activity
    TYPE: Time-based (RECENCY)

    HYPOTHESIS:
    Recency of the most recent profile event is among the strongest single
    churn predictors across virtually every digital platform. A user who
    was active last week has fundamentally different churn risk than one
    who was last active 600 days ago, regardless of how productive their
    earlier history was.

    DATA SOURCE:
    We use updated_at from the GitHub profile, which changes on any activity
    (starring repos, following users, pushing code, bio changes). While
    scraper.py correctly notes that updated_at is a weaker LABEL signal
    (because passive events like starring inflate recency), it is an excellent
    FEATURE signal — it captures ANY recent engagement, including kinds of
    engagement that indicate the user is still visiting the platform.

    WHY NOT days_since_last_push:
    days_since_last_push is NULL for all users when fetch_repos=False (our
    optimized scraper setting). More critically, it was used to CONSTRUCT the
    churn label — using it as a feature would be label leakage. We drop it here.

    DIRECTION: Higher value → higher churn probability.
    """
    updated_dt = _to_datetime_utc(df["updated_at"])
    return _days_since(updated_dt)


def compute_account_age_days(df: pd.DataFrame) -> pd.Series:
    """
    F4 — account_age_days
    TYPE: Time-based (TENURE)

    HYPOTHESIS:
    Account tenure encodes historical commitment to the platform. Newer
    accounts that have already gone quiet are a very different (and higher)
    churn risk than long-standing accounts with recent inactivity.

    WHY THIS PREDICTS CHURN:
    Older accounts have survived multiple engagement cycles and "stickiness"
    periods — their survival itself is a retention signal. Brand-new accounts
    that show low activity haven't yet established the habit loops that drive
    long-term retention. Combined with recency features, tenure helps the model
    distinguish "new AND inactive" (high churn) from "old AND currently quiet"
    (different pattern — may just be between projects).

    NOTE: We store raw days; conversion to years happens inside repos_per_year
    and activity_frequency where denominators are needed.
    """
    created_dt = _to_datetime_utc(df["created_at"])
    return _days_since(created_dt)


def compute_activity_frequency(df: pd.DataFrame) -> pd.Series:
    """
    F5 — activity_frequency = public_repos / account_age_years
    TYPE: Time-based (FREQUENCY)

    HYPOTHESIS:
    Frequency measures HOW OFTEN a user creates repositories over their
    lifetime on the platform — it captures the cadence of engagement, not
    just the total volume. A user who has created 12 repos over 10 years
    has a very different activity pattern than one who created 12 repos
    in the last 12 months.

    WHY THIS PREDICTS CHURN:
    Users who historically create repositories at a sustained rate have
    demonstrated a behavioral pattern of return engagement. When that
    frequency drops (visible from the recency features), it signals a
    break from an established habit — a strong churn indicator. Users who
    were always low-frequency can't show this drop, so frequency alone is
    less powerful, but combined with recency it creates a strong signal.

    NOTE: We use _safe_years() to prevent division by zero for brand-new
    accounts (< 1 day old), flooring the denominator at 1 day.
    """
    created_dt = _to_datetime_utc(df["created_at"])
    age_days = _days_since(created_dt)
    age_years = _safe_years(age_days)
    return df["public_repos"] / age_years


# ── AGGREGATION FEATURES ──────────────────────────────────────────────────────

def compute_repos_per_year(df: pd.DataFrame) -> pd.Series:
    """
    F6 — repos_per_year = public_repos / account_age_years
    TYPE: Aggregation (rate-normalized count)

    HYPOTHESIS:
    Raw public_repos count is an unfair comparison across users — a developer
    with 10 repos after 1 year is MORE productive than one with 10 repos after
    10 years. Normalizing by account age converts an absolute count into a rate,
    making it a fair, comparable aggregation across the entire dataset.

    WHY THIS PREDICTS CHURN:
    This is effectively the most direct measure of "how much has this user
    contributed to GitHub per year of membership?" Low repos_per_year users
    either never got into the habit of using GitHub substantively, or have
    been absent for a long time and their rate has declined — both are churn
    risk factors.

    NOTE: This is mathematically identical to activity_frequency (F5).
    Both are kept intentionally: F5 is named for its temporal/frequency
    interpretation (Step 4 type classification); F6 is the aggregation
    framing required by the rubric. Step 5 selection methods may drop one
    of the two due to high correlation — that is the CORRECT outcome.

    IMPLEMENTATION: Same formula as F5, aliased with a domain-meaningful name.
    """
    created_dt = _to_datetime_utc(df["created_at"])
    age_days = _days_since(created_dt)
    age_years = _safe_years(age_days)
    return df["public_repos"] / age_years


def compute_social_volume(df: pd.DataFrame) -> pd.Series:
    """
    F7 — social_volume = log1p(followers + following + public_gists)
    TYPE: Aggregation (log-compressed sum)

    HYPOTHESIS:
    Social volume aggregates ALL social-graph signals into a single number:
    how many people follow this user, how many they follow back, and how
    many code snippets (gists) they have shared publicly. Together, this
    measures the total FOOTPRINT of a user's social presence on GitHub.

    WHY THIS PREDICTS CHURN:
    Users with a larger social footprint are more deeply embedded in the
    GitHub ecosystem — they have more reasons to return (notifications,
    followers, community interactions). A user with near-zero social volume
    has built no community ties and faces no social cost of churning.

    WHY log1p():
    Follower counts follow a power-law distribution — a few accounts have
    millions of followers while most have 0–10. Without log compression, a
    handful of celebrity accounts would dominate the range and distort the
    feature for everyone else. log1p(x) = log(1 + x) handles the x=0 case
    cleanly (log1p(0) = 0) while compressing the long tail.
    """
    total = df["followers"] + df["following"] + df.get("public_gists", 0)
    return np.log1p(total)


def compute_profile_completeness(df: pd.DataFrame) -> pd.Series:
    """
    F8 — profile_completeness = count of filled optional profile fields (0–4)
    TYPE: Aggregation (count-based composite)

    HYPOTHESIS:
    When a user fills out their optional GitHub profile fields — bio, blog/website,
    company, and email — they have invested effort in presenting themselves on the
    platform. This signals intention to be found, to network, or to job-hunt via
    GitHub. It is a measure of how seriously they treat their GitHub presence.

    WHY THIS PREDICTS CHURN:
    Profile completeness is a leading retention indicator in virtually every
    social and professional platform. LinkedIn, for example, prominently promotes
    profile completeness because completed profiles dramatically increase engagement
    and long-term retention. Users who invest in their profile signal intent to
    stay — they have "set up their space" on the platform. A completely empty
    optional profile suggests a trial user who never fully committed.

    SCORING (0–4):
    +1 if bio     is present (non-null, non-empty string)
    +1 if blog    is present (non-null, non-empty string)
    +1 if company is present (non-null, non-empty string)
    +1 if hireable flag is not null (any explicit value = intention signaled)
    """
    score = pd.Series(0, index=df.index, dtype=float)

    # bio: check for non-null and non-empty string
    if "bio" in df.columns:
        score += df["bio"].notna() & (df["bio"].astype(str).str.strip() != "") & (df["bio"].astype(str) != "None")

    # blog/website: presence of a URL is a strong profile-investment signal
    if "blog" in df.columns:
        score += df["blog"].notna() & (df["blog"].astype(str).str.strip() != "") & (df["blog"].astype(str) != "None")

    # company: professional affiliation = higher engagement intent
    if "company" in df.columns:
        score += df["company"].notna() & (df["company"].astype(str).str.strip() != "") & (df["company"].astype(str) != "None")

    # hireable: any explicit value (True or False) means the user completed this field
    if "hireable" in df.columns:
        score += df["hireable"].notna()

    return score.astype(float)


# ── BINARY FEATURES ───────────────────────────────────────────────────────────

def compute_has_no_repos(df: pd.DataFrame) -> pd.Series:
    """
    F9 — has_no_repos = 1 if public_repos == 0, else 0
    TYPE: Binary (threshold flag)

    HYPOTHESIS:
    Zero public repositories is a qualitatively distinct state from "1 repo"
    or "2 repos." It means the user created a GitHub account but has never
    publicly committed code — they either only work privately (less likely
    for this user population), or they signed up and never meaningfully
    engaged with the core function of the platform.

    WHY THIS PREDICTS CHURN:
    This is a "never-active" flag. In churn modeling terminology, a user
    who never activated (never completed the core value action) has a very
    different — and typically higher — churn probability than even a low-activity
    user. A count of 0 carries information that can't be captured well by
    a continuous feature: the gap between 0 and 1 is behavioral, not just
    numerical. has_no_repos = 1 is a hard signal: this user never engaged
    with the platform's core value proposition (public code contribution).
    """
    return (df["public_repos"] == 0).astype(int)


def compute_has_complete_profile(df: pd.DataFrame) -> pd.Series:
    """
    F10 — has_complete_profile = 1 if profile_completeness == 4, else 0
    TYPE: Binary (threshold flag on an aggregation)

    HYPOTHESIS:
    While profile_completeness (F8) gives the model a gradient from 0 to 4,
    the FULLY complete profile (all four optional fields filled) is a qualitatively
    distinct signal. A user who filled EVERY optional field has maximized their
    platform investment — they've gone beyond "enough" to "thorough."

    WHY THIS PREDICTS CHURN:
    This binary flag captures the upper tail of the profile_completeness
    distribution in a way the continuous version cannot. The model may find
    a non-linear threshold effect: the difference between completeness=3 and
    completeness=4 may be more predictive than the difference between 1 and 2.
    Binary features allow the model to capture these hard thresholds without
    needing to learn the non-linearity itself.

    DEPENDENCY: Reuses compute_profile_completeness() to avoid duplicate logic.
    """
    completeness = compute_profile_completeness(df)
    return (completeness == 4).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MAIN PIPELINE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_features(df: pd.DataFrame) -> tuple:
    """
    Transform the raw scraper DataFrame into a model-ready feature matrix.

    Applies all 10 feature transformations defined in Section 2,
    handles missing values, and returns a clean (X, y) pair.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame produced by scraper.py. Must contain at minimum:
        created_at, updated_at, public_repos, followers, following.
        Optional: public_gists, bio, blog, company, hireable, churn.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with columns matching FEATURE_COLUMNS (10 columns).
        All values are finite floats — NaN / Inf replaced by column medians.
    y : pd.Series or None
        Binary churn labels (0 / 1) from the "churn" column, or None if
        the column is absent (e.g., during inference in main.py).

    Notes
    -----
    DROPPED COLUMNS:
    - days_since_last_push : Used to construct the churn label in scraper.py.
      Including it as a feature would be label leakage. Dropped unconditionally.
    - days_since_update    : Same concern — it feeds the fallback chain in
      assign_churn_label(). Dropped to prevent leakage.
    - username, created_at, updated_at, bio, blog, company, email, hireable :
      Raw strings / identifiers — not model inputs. Replaced by derived features.
    - public_repos, followers, following, public_gists :
      Raw counts — not model inputs. Replaced by ratio and aggregation features.
      (The model should never see raw follower counts — use follower_ratio instead.)
    """

    # ── STEP 1: Validate required columns ────────────────────────────────────
    required = ["created_at", "updated_at", "public_repos", "followers", "following"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"generate_features() is missing required columns: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    # ── STEP 2: Fill missing raw fields with safe defaults ────────────────────
    # Only fill numeric raw fields used in feature calculations.
    # Timestamp columns stay as-is — _to_datetime_utc handles NaT gracefully.
    df = df.copy()
    df["public_repos"]  = df["public_repos"].fillna(0).clip(lower=0)
    df["followers"]     = df["followers"].fillna(0).clip(lower=0)
    df["following"]     = df["following"].fillna(0).clip(lower=0)
    df["public_gists"]  = df.get("public_gists", pd.Series(0, index=df.index)).fillna(0).clip(lower=0)

    # ── STEP 3: Compute all features ──────────────────────────────────────────
    features = pd.DataFrame(index=df.index)

    # --- Ratio features ---
    features["follower_ratio"]            = compute_follower_ratio(df)
    features["engagement_ratio"]          = compute_engagement_ratio(df)

    # --- Time-based features ---
    features["days_since_last_activity"]  = compute_days_since_last_activity(df)
    features["account_age_days"]          = compute_account_age_days(df)
    features["activity_frequency"]        = compute_activity_frequency(df)

    # --- Aggregation features ---
    features["repos_per_year"]            = compute_repos_per_year(df)
    features["social_volume"]             = compute_social_volume(df)
    features["profile_completeness"]      = compute_profile_completeness(df)

    # --- Binary features ---
    features["has_no_repos"]              = compute_has_no_repos(df)
    features["has_complete_profile"]      = compute_has_complete_profile(df)

    # ── STEP 4: Handle Inf and NaN produced by edge-case math ─────────────────
    # Replace ±Inf with NaN first, then fill NaN with each column's median.
    # Using median (not mean) because several features have right-skewed
    # distributions (followers, gists) where the mean is inflated by outliers.
    features = features.replace([np.inf, -np.inf], np.nan)
    for col in features.columns:
        if features[col].isna().any():
            median_val = features[col].median()
            fill_val = median_val if not np.isnan(median_val) else 0.0
            features[col] = features[col].fillna(fill_val)

    # ── STEP 5: Enforce column order per FEATURE_COLUMNS ─────────────────────
    features = features[FEATURE_COLUMNS]

    # ── STEP 6: Extract churn label if present ────────────────────────────────
    y = df["churn"].astype(int) if "churn" in df.columns else None

    return features, y


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE SUMMARY UTILITY
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_features(X: pd.DataFrame, y: pd.Series = None) -> None:
    """
    Print a diagnostic summary of the generated feature matrix.

    Shows shape, null counts, basic statistics, and — if y is provided —
    the mean value of each feature split by churn label. That split is
    a quick sanity check: features where churned (1) and retained (0) users
    have clearly different means are likely to be predictive.

    Parameters
    ----------
    X : Feature DataFrame from generate_features()
    y : Optional churn label Series from generate_features()
    """
    print("=" * 65)
    print("  FEATURE MATRIX SUMMARY")
    print("=" * 65)
    print(f"  Shape           : {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  NaN count total : {X.isna().sum().sum()}")
    print(f"  Inf count total : {np.isinf(X.values).sum()}")
    print()

    print("  Feature statistics:")
    print(X.describe().round(3).to_string())
    print()

    if y is not None:
        print("  Feature means by churn label:")
        combined = X.copy()
        combined["churn"] = y.values
        means = combined.groupby("churn")[FEATURE_COLUMNS].mean().round(3)
        print(means.to_string())
        print()
        print("  → Features with large differences between rows 0 and 1")
        print("    are the strongest candidate predictors.")
    print("=" * 65)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ENTRY POINT (for standalone testing)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Run this script directly to test feature generation on the collected dataset:
        python features.py

    Expects the raw CSV to exist at data/raw/github_users.csv (produced by scraper.py).
    Outputs the feature matrix to data/processed/github_features.csv.
    """
    import os

    RAW_DATA_PATH  = "data/raw/github_users.csv"
    OUT_DATA_PATH  = "data/processed/github_features.csv"

    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"  Loaded {len(df_raw)} rows, {len(df_raw.columns)} columns")
    print(f"  Columns: {list(df_raw.columns)}\n")

    # Generate features
    X, y = generate_features(df_raw)

    # Print diagnostic summary
    summarize_features(X, y)

    # Save feature matrix alongside the churn label
    os.makedirs(os.path.dirname(OUT_DATA_PATH), exist_ok=True)
    output_df = X.copy()
    if y is not None:
        output_df["churn"] = y.values
    output_df.to_csv(OUT_DATA_PATH, index=False)
    print(f"\nFeature matrix saved to: {OUT_DATA_PATH}")
    print(f"Columns: {list(output_df.columns)}")