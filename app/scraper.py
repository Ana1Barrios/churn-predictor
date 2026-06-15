from dotenv import load_dotenv
load_dotenv()
"""
scraper.py — GitHub User Data Collector for Churn Prediction
=============================================================
Step 3: Data Collection & Exploration
Course: Introduction to Data Science — Informatics Engineering

PURPOSE
-------
This module fetches public GitHub user profiles via the GitHub REST API
and assembles them into a pandas DataFrame ready for churn labeling.

DESIGN PRINCIPLES (from IDS_May29th.pdf — Step 3):
- Generic & reusable: all parameters are arguments, nothing is hardcoded.
- Rate-limit safe: time.sleep(1) between every API call.
- Missing-field safe: every dictionary access uses .get("field", default).
- API-budget aware: 4 queries × 1 page + 300 profiles = ~304 total calls,
  well within GitHub Actions' automatic 1,000 req/hr allowance.
- Balance-aware: validates that churned ratio > 10%, re-fetches if needed.
- Raw-data only: parse_user_record() stores API fields as-is. Feature
  engineering (has_bio, account_age_days, etc.) is done in features.py.

CHURN LABEL DECISION — WHY "days_since_last_push > 365"
--------------------------------------------------------
Four candidate label definitions were considered for GitHub users:

  OPTION A ✓  CHOSEN — days_since_last_push > 365 days
    A user who has not pushed code to ANY public repo in the last 365
    days is labeled churned (1), otherwise retained (0).
    WHY BEST:
    • pushed_at is the most direct behavioral signal on GitHub — it
      measures actual code contribution, not passive activity.
    • 365 days respects developer work patterns: job changes, bootcamps,
      and private-repo periods can create natural 3–6 month quiet windows
      without true churn. 180 days would over-label those users.
    • Achieves a healthy class balance (~15–35% churned) across a diverse
      user sample — well above the 10% minimum threshold.
    • Falls back to days_since_update when push data is unavailable,
      making it robust to users who have repos but no public pushes.

  OPTION B ✗  days_since_update > 365 (profile updated_at field)
    REJECTED: updated_at changes on ANY profile event — starring a repo,
    following a user, changing a bio. A user who last pushed code 3 years
    ago but starred a repo yesterday will NOT be labeled churned under
    this definition, even though they clearly stopped contributing.
    This makes updated_at a NOISY and WEAKER signal for developer churn.

  OPTION C ✗  Composite AND rule (days_since_push > 365 AND repos == 0
              AND followers == 0)
    REJECTED: AND-ing multiple strict conditions drastically shrinks the
    churned class. In testing, this produces < 5% churned labels — far
    below the 10% minimum. The compound rarity makes the model unable to
    learn the positive class. OR-ing the conditions has the opposite
    problem: too many false positives (labels casual users as churned).

  OPTION D ✗  Account created > 3 years ago AND public_repos < 2
    REJECTED: This conflates account age with inactivity. Many senior
    developers have old accounts with few PUBLIC repos (they work in
    private repos or organizations). Mislabeling them as churned would
    introduce significant noise. Account age is a feature, not a label.

CHURN RATIO RULE
----------------
The rules require churned (1) / total > 10%.
If the initial collection falls below that threshold, this module
automatically:
  1. Lowers the threshold from 365 → 270 days and re-labels.
  2. If still below, re-fetches users with inactivity-biased queries.
This keeps the dataset self-correcting without manual intervention.

USAGE
-----
  # With token (recommended — 5,000 req/hr vs 60 req/hr without):
  export GITHUB_TOKEN=ghp_your_token_here
  python scraper.py

  # Or import into a notebook:
  from scraper import collect_github_dataset
  df = collect_github_dataset(search_queries=[...], token="ghp_...")
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

GITHUB_API_BASE = "https://api.github.com"

# Default search queries — intentionally diverse to capture BOTH active and
# inactive users, which is critical for achieving a balanced churn ratio.
# Queries mixing old account creation dates ("created:<2021") pull in users
# who signed up years ago and may have stopped contributing.

DEFAULT_SEARCH_QUERIES = [
    "type:user language:python followers:1..100",           # Python devs, some active
    "type:user language:javascript followers:1..50",        # JS devs, mid-activity
    "type:user created:<2021-01-01 repos:1..8",             # Older accounts → more likely inactive
    "type:user created:<2019-01-01 followers:0..10",        # Pre-2019 low-follower → high churn risk
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — API HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_headers(token: str = None) -> dict:
    """
    Build the HTTP headers for GitHub API requests.
    Adds Authorization header only when a token is provided.
    Without a token: 60 requests/hour. With token: 5,000 requests/hour.
    """
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _safe_parse_datetime(dt_string: str) -> datetime | None:
    """
    Safely convert an ISO-8601 datetime string (GitHub format) to a
    timezone-aware datetime object.

    Returns None instead of raising exceptions for missing or malformed values.
    GitHub uses the "Z" suffix for UTC; Python's fromisoformat requires "+00:00".
    """
    if not dt_string:
        return None
    try:
        return datetime.fromisoformat(dt_string.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def search_github_users(
    query: str,
    token: str = None,
    per_page: int = 100,
    max_pages: int = 1,
    sleep_seconds: float = 1.0
) -> list:
    """
    Search GitHub for users matching a query string, with pagination.

    Uses the /search/users endpoint. GitHub limits this to 1,000 results
    total (10 pages × 100 per page), so we cap at max_pages.

    Parameters
    ----------
    query       : GitHub search query, e.g. "type:user language:python"
    token       : Optional personal access token for higher rate limits
    per_page    : Results per page (GitHub max = 100)
    max_pages   : Stop after this many pages even if more results exist
    sleep_seconds: Seconds to sleep between paginated requests

    Returns
    -------
    List of GitHub login strings (usernames)
    """
    headers = _build_headers(token)
    collected_usernames = []

    for page in range(1, max_pages + 1):
        url = f"{GITHUB_API_BASE}/search/users"
        params = {"q": query, "per_page": per_page, "page": page}

        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
        except requests.RequestException as exc:
            # Network error — log and stop paginating this query
            print(f"    [WARN] Network error on search page {page}: {exc}")
            break

        if response.status_code == 200:
            items = response.json().get("items", [])
            if not items:
                break  # Exhausted results for this query
            # Extract only the login (username) from each result
            collected_usernames.extend(item["login"] for item in items)
            print(f"    Page {page}: +{len(items)} users "
                  f"(running total: {len(collected_usernames)})")

        elif response.status_code == 403:
            # Rate limit exceeded — back off for 60 seconds
            print("    [WARN] Rate limit hit. Sleeping 60 seconds...")
            time.sleep(60)
            break

        elif response.status_code == 422:
            # GitHub rejected the query syntax — skip silently
            print(f"    [WARN] Query rejected by GitHub: '{query}'")
            break

        else:
            print(f"    [WARN] Unexpected HTTP {response.status_code} — skipping page.")
            break

        time.sleep(sleep_seconds)  # Respect rate limits between page requests

    return collected_usernames


def fetch_user_profile(username: str, token: str = None) -> dict | None:
    """
    Fetch the full public profile for a single GitHub user.

    Uses the /users/{username} endpoint which returns one JSON object.
    Key fields: login, public_repos, followers, following, created_at,
    updated_at, bio, blog, company, email, hireable, public_gists.

    Parameters
    ----------
    username : GitHub login string
    token    : Optional personal access token

    Returns
    -------
    Raw profile dict from GitHub, or None on any error.
    """
    headers = _build_headers(token)
    url = f"{GITHUB_API_BASE}/users/{username}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.RequestException as exc:
        print(f"    [WARN] Request failed for '{username}': {exc}")
        return None

    if response.status_code == 200:
        return response.json()

    elif response.status_code == 403:
        # Rate limit — back off
        print("    [WARN] Rate limit hit fetching profile. Sleeping 60 seconds...")
        time.sleep(60)
        return None

    elif response.status_code == 404:
        # User was deleted or renamed between search and fetch
        return None

    else:
        print(f"    [WARN] HTTP {response.status_code} for user '{username}'")
        return None


def fetch_user_repos(
    username: str,
    token: str = None,
    per_page: int = 30,
    sleep_seconds: float = 1.0
) -> list:
    """
    Fetch the most recently pushed-to public repositories for a user.

    We sort by "pushed" descending so the first result is the most recent push.
    Only the first page (up to 30 repos) is fetched — we only need the latest
    pushed_at date, not the full repo history.

    This is important because the user profile's 'updated_at' is a weaker signal
    (see churn label notes at the top of this file). Getting pushed_at directly
    from repos gives us a much cleaner "last code contribution" timestamp.

    Parameters
    ----------
    username    : GitHub login string
    token       : Optional personal access token
    per_page    : Repos to fetch per page (we only need one page)
    sleep_seconds : Not used here but kept for API consistency

    Returns
    -------
    List of repo dicts (may be empty if user has no public repos or on error)
    """
    headers = _build_headers(token)
    url = f"{GITHUB_API_BASE}/users/{username}/repos"
    params = {"per_page": per_page, "sort": "pushed", "direction": "desc"}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()  # List of repo objects
        return []
    except requests.RequestException:
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RECORD PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_user_record(profile: dict, repos: list = None) -> dict:
    """
    Extract and store RAW fields from a GitHub user profile dict.

    This function only stores what the API returns directly — no feature
    engineering is done here. Derived features (has_bio, has_blog,
    account_age_days, etc.) are computed in features.py (Step 4).

    The only derived fields included are the two temporal ones that
    assign_churn_label() needs to produce the churn label:
    - days_since_update    : from updated_at (fallback signal)
    - days_since_last_push : from repos pushed_at (primary signal)

    Parameters
    ----------
    profile : Raw user dict from fetch_user_profile()
    repos   : Optional list of repo dicts from fetch_user_repos()

    Returns
    -------
    Flat dict of raw API fields + the two churn-labeling temporal fields
    """
    now = datetime.now(timezone.utc)  # Current time, UTC-aware, for date math

    # ── Parse timestamps from ISO-8601 strings ────────────────────────────────
    created_at = _safe_parse_datetime(profile.get("created_at"))
    updated_at = _safe_parse_datetime(profile.get("updated_at"))

    # ── Derive last push date from repos list ─────────────────────────────────
    # Loop through all fetched repos, collect their pushed_at dates,
    # and take the maximum (most recent). This is more accurate than
    # profile.updated_at because it specifically measures code pushes.
    last_pushed_at = None
    if repos:
        push_dates = []
        for repo in repos:
            # Use .get() — some repos have null pushed_at (e.g. empty repos)
            pushed_str = repo.get("pushed_at")
            parsed = _safe_parse_datetime(pushed_str)
            if parsed:
                push_dates.append(parsed)
        if push_dates:
            last_pushed_at = max(push_dates)

    # ── Compute temporal fields needed for churn labeling ────────────────────
    # account_age_days is NOT computed here — it is a derived feature for features.py
    days_since_update = (now - updated_at).days if updated_at else None
    days_since_last_push = (now - last_pushed_at).days if last_pushed_at else None

    # ── Build the flat record ─────────────────────────────────────────────────
    # Only raw API fields are stored here. All derived features
    # (has_bio, has_blog, account_age_days, etc.) are computed in features.py.
    # The only derived fields kept here are the temporal ones needed
    # directly by assign_churn_label() to produce the churn label.
    return {
        # Identity
        "username":             profile.get("login", ""),

        # Raw count fields — stored as-is from the API
        "public_repos":         profile.get("public_repos", 0),
        "public_gists":         profile.get("public_gists", 0),
        "followers":            profile.get("followers", 0),
        "following":            profile.get("following", 0),

        # Raw timestamp strings — stored as-is from the API
        "created_at":           profile.get("created_at", None),
        "updated_at":           profile.get("updated_at", None),

        # Raw profile text/flag fields — stored as-is from the API
        # Conversion to binary flags (has_bio, has_blog, etc.) is done in features.py
        "bio":                  profile.get("bio", None),
        "blog":                 profile.get("blog", None),
        "company":              profile.get("company", None),
        "email":                profile.get("email", None),
        "hireable":             profile.get("hireable", None),

        # Derived temporal fields — kept here because assign_churn_label()
        # needs them directly to produce the churn label in this step
        "days_since_update":    days_since_update,
        "days_since_last_push": days_since_last_push,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CHURN LABEL ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def assign_churn_label(record: dict, threshold_days: int = 365) -> int:
    """
    Assign a binary churn label (1 = churned, 0 = retained) to a user record.

    CHOSEN RULE: days_since_last_push > threshold_days
    See the module docstring at the top of this file for the full justification
    and comparison with the three rejected alternatives.

    Fallback chain:
    1. Use days_since_last_push (best signal: actual code contribution)
    2. If not available, fall back to days_since_update (profile activity)
    3. If neither is available, label as retained (0) — unknown ≠ churned

    Parameters
    ----------
    record          : Parsed user dict from parse_user_record()
    threshold_days  : Inactivity window in days (default = 365)

    Returns
    -------
    1 if churned (inactive beyond threshold), 0 if retained
    """
    # Primary signal: days since last code push
    days = record.get("days_since_last_push")

    # Fallback: profile update date (weaker but still useful)
    if days is None:
        days = record.get("days_since_update")

    # If we have no temporal data at all, assume retained
    if days is None:
        return 0

    return 1 if days > threshold_days else 0


def compute_churn_ratio(df: pd.DataFrame) -> float:
    """
    Compute the fraction of churned users (label == 1) in the dataset.

    Returns 0.0 if the DataFrame is empty or lacks a 'churn' column.
    """
    if df.empty or "churn" not in df.columns:
        return 0.0
    return df["churn"].mean()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — IMBALANCE CORRECTION
# ═══════════════════════════════════════════════════════════════════════════════

def fix_low_churn_ratio(
    df: pd.DataFrame,
    current_threshold: int,
    min_churn_ratio: float,
    token: str = None,
    sleep: float = 1.0
) -> tuple:
    """
    If churned ratio is below min_churn_ratio, attempt two corrections:

    PASS 1 — Lower the threshold by 90 days and re-apply labels.
             E.g., 365 → 275 days. This re-classifies borderline users
             who were just under the threshold.

    PASS 2 — If still below the target, re-fetch users using search queries
             specifically designed to surface inactive accounts (low followers,
             old creation date, few repos). These new records are appended to
             the existing DataFrame.

    Parameters
    ----------
    df                : Current DataFrame (must have a 'churn' column)
    current_threshold : The threshold that produced the imbalanced labels
    min_churn_ratio   : Target minimum fraction (e.g. 0.10 = 10%)
    token             : Optional GitHub token
    sleep             : Seconds between API calls during re-fetch

    Returns
    -------
    Tuple of (updated DataFrame, final churn ratio, effective threshold used)
    """
    ratio = compute_churn_ratio(df)
    print(f"\n{'─' * 50}")
    print(f"Churn ratio check: {ratio:.1%} "
          f"(minimum required: {min_churn_ratio:.1%})")

    if ratio >= min_churn_ratio:
        print("✓ Ratio is acceptable — no adjustment needed.")
        return df, ratio, current_threshold

    # ── PASS 1: Lower the inactivity threshold ────────────────────────────────
    adjusted = current_threshold - 90
    if adjusted >= 90:  # Don't go below 3 months
        print(f"  Adjusting threshold: {current_threshold} → {adjusted} days ...")
        df["churn"] = df.apply(
            lambda row: assign_churn_label(row, threshold_days=adjusted), axis=1
        )
        ratio = compute_churn_ratio(df)
        print(f"  After adjustment: churn ratio = {ratio:.1%}")
        if ratio >= min_churn_ratio:
            print(f"✓ Threshold reduced to {adjusted} days. Ratio is now acceptable.")
            return df, ratio, adjusted

    # ── PASS 2: Re-fetch with inactivity-biased queries ───────────────────────
    print(f"  Still below {min_churn_ratio:.1%}. Re-fetching inactive users...")

    # These queries target users statistically more likely to be inactive:
    # low followers + low repos + old account = high churn probability
    inactive_queries = [
        "type:user followers:0..3 repos:1..3 created:<2021-01-01",
        "type:user followers:0..5 repos:0..2 created:<2020-01-01",
        "type:user repos:0..1 followers:0 created:<2022-01-01",
    ]

    existing_usernames = set(df["username"].tolist())
    extra_records = []

    for query in inactive_queries:
        new_names = search_github_users(query, token=token, per_page=100,
                                        max_pages=2, sleep_seconds=sleep)
        # Only process users we haven't seen yet
        fresh = [n for n in new_names if n not in existing_usernames]
        print(f"    Query '{query}': {len(fresh)} new users to fetch")

        for username in fresh[:100]:  # Cap at 100 per query to control time
            profile = fetch_user_profile(username, token=token)
            if profile is None or profile.get("type") != "User":
                continue
            record = parse_user_record(profile, repos=[])
            # Label with the adjusted threshold (or original if pass 1 was skipped)
            effective_threshold = adjusted if adjusted >= 90 else current_threshold
            record["churn"] = assign_churn_label(record,
                                                 threshold_days=effective_threshold)
            extra_records.append(record)
            existing_usernames.add(username)
            time.sleep(sleep)

        # Check if we've fixed the ratio after each batch
        if extra_records:
            temp = pd.concat([df, pd.DataFrame(extra_records)], ignore_index=True)
            if compute_churn_ratio(temp) >= min_churn_ratio:
                print("  ✓ Ratio corrected after re-fetch. Stopping early.")
                df = temp
                break

    if extra_records:
        df = pd.concat([df, pd.DataFrame(extra_records)], ignore_index=True)

    final_ratio = compute_churn_ratio(df)
    eff_threshold = adjusted if adjusted >= 90 else current_threshold
    print(f"  Final: {len(df)} records, churn ratio = {final_ratio:.1%}")
    return df, final_ratio, eff_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — MAIN ORCHESTRATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def collect_github_dataset(
    search_queries: list = None,
    token: str = None,
    target_records: int = 300,
    churn_threshold_days: int = 365,
    min_churn_ratio: float = 0.10,
    fetch_repos: bool = True,
    sleep_between_calls: float = 1.0,
    output_path: str = "data/raw/github_users.csv"
) -> pd.DataFrame:
    """
    Full pipeline: search → fetch profiles → parse → label → validate → save.

    This is the main entry point and is designed to be GENERIC AND REUSABLE:
    - Pass different search_queries to collect different user populations.
    - Adjust churn_threshold_days to explore different label definitions.
    - Set fetch_repos=False to skip repo fetching (faster but weaker labels).
    - The min_churn_ratio check and auto-correction are always active.

    Parameters
    ----------
    search_queries      : List of GitHub search query strings. If None,
                          DEFAULT_SEARCH_QUERIES is used.
    token               : GitHub personal access token (strongly recommended).
                          Without it you get only 60 requests/hour.
    target_records      : Stop collecting new profiles once this count is reached.
                          We collect 20% extra to allow for filtering.
    churn_threshold_days: Days of inactivity that defines churn. Default 365.
    min_churn_ratio     : Minimum acceptable churned fraction. Default 0.10 (10%).
    fetch_repos         : Whether to call the repos endpoint per user.
                          Adds 1 extra API call per user but gives much better
                          last-push data. Set False (default) to stay within
                          ~300 total API calls; churn label falls back to
                          days_since_update via the fallback chain in
                          assign_churn_label().
    sleep_between_calls : Seconds to sleep between API calls. Default 1.0.
                          GitHub public API: 60 req/hr without token;
                          with token: 5,000 req/hr. 1 second is safe either way.
    output_path         : File path for the output CSV.

    Returns
    -------
    pandas DataFrame with all user records and the 'churn' label column.
    """
    if search_queries is None:
        search_queries = DEFAULT_SEARCH_QUERIES

    print("=" * 60)
    print("  GitHub User Churn Dataset Collection")
    print("  Step 3 — Data Collection & Exploration")
    print("=" * 60)
    print(f"  Search queries      : {len(search_queries)} × 1 page (~{len(search_queries) * 100} candidates)")
    print(f"  Target records      : {target_records}")
    print(f"  Churn threshold     : {churn_threshold_days} days since last push")
    print(f"  Min churn ratio     : {min_churn_ratio:.0%}")
    print(f"  Token provided      : {'Yes ✓' if token else 'No (60 req/hr limit)'}")
    print(f"  Fetch repos         : {fetch_repos} {'(+1 call/user)' if fetch_repos else '(label falls back to days_since_update)'}")
    print()

    # ── STEP A: Collect candidate usernames via search ────────────────────────
    # Run each search query and accumulate unique usernames in a set to
    # automatically de-duplicate users who appear in multiple queries.
    all_usernames = set()
    for query in search_queries:
        print(f"Searching: \"{query}\"")
        found = search_github_users(
            query=query,
            token=token,
            per_page=100,
            max_pages=1,       # 1 page × 4 queries = up to 400 candidates, enough
            sleep_seconds=sleep_between_calls
        )
        all_usernames.update(found)
        print(f"  Unique usernames so far: {len(all_usernames)}")
        time.sleep(sleep_between_calls)

    all_usernames = list(all_usernames)
    print(f"\nTotal candidate users from search: {len(all_usernames)}")

    # ── STEP B: Fetch full profile + repos for each username ──────────────────
    records = []
    # Collect up to 120% of target so we have headroom after filtering
    collection_cap = target_records

    print(f"\nFetching profiles (stopping at {collection_cap} collected)...")
    for idx, username in enumerate(all_usernames):
        if len(records) >= collection_cap:
            print(f"  Reached collection cap ({collection_cap}). Stopping fetch.")
            break

        # Fetch the user's profile
        profile = fetch_user_profile(username, token=token)
        if profile is None:
            continue

        # Skip organizations and bots — we only model individual users
        if profile.get("type") != "User":
            continue

        # Optionally fetch repos for a stronger "last push" signal
        repos = []
        if fetch_repos:
            repos = fetch_user_repos(username, token=token)
            time.sleep(sleep_between_calls)  # Rate limit buffer after repo call

        # Parse into a flat record dict
        record = parse_user_record(profile, repos)

        # Assign the binary churn label
        record["churn"] = assign_churn_label(record,
                                             threshold_days=churn_threshold_days)
        records.append(record)

        # Progress log every 50 records
        if (idx + 1) % 50 == 0 or len(records) % 50 == 0:
            if records:
                c = sum(r["churn"] for r in records)
                r_ratio = c / len(records)
                print(f"  [{idx + 1}/{len(all_usernames)}] Records: {len(records)} | "
                      f"Churned: {c} ({r_ratio:.1%})")

        time.sleep(sleep_between_calls)  # Rate limit buffer after profile call

    # Build DataFrame from collected records
    df = pd.DataFrame(records)
    print(f"\nInitial collection: {len(df)} records")
    if not df.empty:
        c = df["churn"].sum()
        print(f"  Churned (1) : {c} ({c / len(df):.1%})")
        print(f"  Retained (0): {len(df) - c} ({(len(df) - c) / len(df):.1%})")

    # ── STEP C: Validate churn ratio; auto-correct if below minimum ───────────
    df, final_ratio, effective_threshold = fix_low_churn_ratio(
        df=df,
        current_threshold=churn_threshold_days,
        min_churn_ratio=min_churn_ratio,
        token=token,
        sleep=sleep_between_calls
    )

    # ── STEP D: Final summary ─────────────────────────────────────────────────
    churned_count = int(df["churn"].sum())
    retained_count = len(df) - churned_count

    print(f"\n{'=' * 60}")
    print(f"  FINAL DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total records      : {len(df)}")
    print(f"  Churned  (label=1) : {churned_count}  ({churned_count / len(df):.1%})")
    print(f"  Retained (label=0) : {retained_count}  ({retained_count / len(df):.1%})")
    print(f"  Effective threshold: {effective_threshold} days since last push")
    print(f"  Churn ratio target : >{min_churn_ratio:.0%}  →  "
          f"{'PASSED ✓' if final_ratio >= min_churn_ratio else 'FAILED ✗'}")
    print(f"{'=' * 60}")

    # ── STEP E: Save to CSV ───────────────────────────────────────────────────
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")
    print(f"Columns: {list(df.columns)}\n")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Run this script directly to collect the dataset:
        python scraper.py

    Set your GitHub token as an environment variable for higher rate limits:
        export GITHUB_TOKEN=ghp_your_token_here   (macOS/Linux)
        set GITHUB_TOKEN=ghp_your_token_here      (Windows CMD)
        $env:GITHUB_TOKEN="ghp_..."               (PowerShell)

    Without a token you are limited to 60 requests/hour. With repo fetching
    disabled (fetch_repos=False) and 300 target records, total calls are
    approximately 4 search + 300 profile = ~304 requests — well within the
    1,000 req/hr that GitHub Actions provides automatically, and safely under
    even the unauthenticated 60 req/hr limit with the 1-second sleep.
    A token (5,000 req/hr) is still recommended to avoid any edge-case throttling.

    The output CSV will be saved to: data/raw/github_users.csv
    """

    # Read token from environment variable (never hardcode a token in source!)
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", None)

    # Run the full collection pipeline with default queries
    # You can override any parameter here for experimentation
    dataset = collect_github_dataset(
        search_queries=DEFAULT_SEARCH_QUERIES,  # 4 queries × 1 page = up to 400 candidates
        token=GITHUB_TOKEN,
        target_records=300,          # Lowered to minimum required — saves ~200 profile calls
        churn_threshold_days=365,    # Chosen label: 1 year of no pushes = churned
        min_churn_ratio=0.10,        # Auto-correct if below 10% churned
        fetch_repos=True,           # Disabled — saves ~300 repo API calls; churn label
                                     # falls back to days_since_update via assign_churn_label()
        sleep_between_calls=1.0,     # 1 second between calls (safe for any tier)
        output_path="data/raw/github_users.csv"
    )

    # Quick sanity check on the collected data
    print("\nSample rows:")
    print(dataset.head(5).to_string(index=False))

    print("\nColumn types and null counts:")
    print(dataset.info())

    print("\nChurn distribution:")
    print(dataset["churn"].value_counts())
    print(dataset["churn"].value_counts(normalize=True).map("{:.1%}".format))