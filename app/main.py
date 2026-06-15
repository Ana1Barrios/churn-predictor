"""
main.py — FastAPI Churn Prediction & Recommendation Endpoint
=============================================================
Step 6 + Step 10: Build the FastAPI Prediction & Recommendation Endpoints
Course: Introduction to Data Science — Informatics Engineering

PURPOSE
-------
This is the production-facing layer of the entire project. It loads the
trained model (model.pkl produced by model.py in Step 5) and the SVD
decomposition (svd_data.pkl produced by recommend.py in Step 9) once at
startup and exposes four HTTP endpoints:

  POST /predict    -- accepts user feature values, returns churn prediction
  POST /recommend  -- returns top-N re-engagement actions for at-risk users
  GET  /health     -- returns {"status": "ok"} for Docker health checks
  GET  /features   -- returns the list of expected input fields (documentation)

ARCHITECTURE POSITION
---------------------
  scraper.py    --> raw data                          (Step 3)
  features.py   --> feature matrix                    (Step 4)
  model.py      --> trained model.pkl                 (Step 5)
  recommend.py  --> svd_data.pkl                      (Step 9)
  main.py       --> API that uses both pkl files      (Step 6 + Step 10)
"""

import os
import numpy as np
import joblib

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Import FINAL_FEATURES from model.py so the API is always in sync
# with what the model was trained on. No manual synchronization needed.
from model import FINAL_FEATURES


# ===============================================================================
# SECTION 1 -- APPLICATION SETUP
# ===============================================================================

app = FastAPI(
    title="GitHub User Churn Predictor",
    description=(
        "Predicts whether a GitHub user will churn (stop contributing) "
        "based on behavioral features derived from their public profile, "
        "and recommends re-engagement actions for at-risk users. "
        "Built for Introduction to Data Science -- Informatics Engineering."
    ),
    version="2.0.0",
)


# ===============================================================================
# SECTION 2 -- MODEL AND SVD DATA LOADING (once at startup)
# ===============================================================================

# --- Load Random Forest model (Step 5) ----------------------------------------
# Loaded once at module level. Every /predict request reuses this object.
# Fails immediately at startup if model.pkl is missing -- fail fast and clearly.
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"model.pkl not found at '{MODEL_PATH}'.\n"
        f"Run model.py first to train and save the model:\n"
        f"    python model.py\n"
        f"Then ensure model.pkl is in the app/ directory before building Docker."
    )

# --- Load SVD recommendation data (Step 9) ------------------------------------
# svd_data.pkl is produced by recommend.py. It contains the predicted score
# matrix from SVD decomposition of the user-feature matrix, used by /recommend
# to find re-engagement opportunities for at-risk users.
# If the file is missing, /recommend returns 503 gracefully rather than
# crashing the whole API -- /predict still works without it.
SVD_PATH = os.getenv("SVD_PATH", "svd_data.pkl")

try:
    svd_data         = joblib.load(SVD_PATH)
    predicted_scores = svd_data["predicted_scores"]  # shape: (n_users, n_features)
    user_item_matrix = svd_data["user_item_matrix"]  # shape: (n_users, n_features), scaled
    usernames        = svd_data["usernames"]          # list of GitHub usernames or row indices
    feature_cols     = svd_data["feature_cols"]       # ordered list of all 10 feature names
except FileNotFoundError:
    # Degrade gracefully: /predict works, /recommend returns 503
    svd_data         = None
    predicted_scores = None
    user_item_matrix = None
    usernames        = []
    feature_cols     = []


# ===============================================================================
# SECTION 3 -- PYDANTIC INPUT MODELS
# ===============================================================================

class UserFeatures(BaseModel):
    """
    Input schema for POST /predict.

    All fields are DERIVED features from features.py -- NOT raw GitHub API
    fields. Values should come from running generate_features() on a user's
    raw profile data before calling this endpoint.

    These three fields match the empirical FINAL_FEATURES selected in Step 5:
      - days_since_last_activity  (4/4 methods agreed -- primary recency signal)
      - activity_frequency        (3/4 methods agreed -- cadence signal)
      - repos_per_year            (4/4 methods agreed -- rate-normalized productivity)
    """

    days_since_last_activity: float = Field(
        ...,
        description=(
            "Days since the user's GitHub profile was last updated (updated_at). "
            "Primary recency signal -- higher values indicate longer inactivity "
            "and higher churn risk."
        ),
        example=200.0,
        ge=0,
    )

    activity_frequency: float = Field(
        ...,
        description=(
            "public_repos / account_age_years. Cadence of repository creation -- "
            "how often the user creates repos per year of membership. "
            "Low values indicate the user has slowed or stopped creating repos."
        ),
        example=2.5,
        ge=0,
    )

    repos_per_year: float = Field(
        ...,
        description=(
            "public_repos / account_age_years. Rate-normalized productivity -- "
            "identical formula to activity_frequency, kept as the aggregation-type "
            "representative in the final feature set."
        ),
        example=2.5,
        ge=0,
    )


class RecommendRequest(BaseModel):
    """
    Input schema for POST /recommend.

    user_id is the 0-based row index of the user in the dataset (same order
    as data/processed/github_features.csv). This identifies which user's
    SVD scores to look up.
    """

    user_id: int = Field(
        ...,
        description=(
            "0-based row index of the user in the processed dataset "
            "(github_features.csv). Use values 0 to N-1 where N is the "
            "number of users collected by scraper.py."
        ),
        example=42,
        ge=0,
    )

    top_n: int = Field(
        default=5,
        description="Number of re-engagement recommendations to return. Default is 5.",
        example=5,
        ge=1,
        le=10,
    )


# ===============================================================================
# SECTION 4 -- PYDANTIC RESPONSE MODELS
# ===============================================================================

class PredictionResponse(BaseModel):
    """Output schema for POST /predict."""

    churned: bool = Field(
        ...,
        description=(
            "True if the model predicts this user will churn, "
            "False if predicted to be retained."
        ),
        example=True,
    )

    churn_probability: float = Field(
        ...,
        description=(
            "Probability of churn between 0.0 and 1.0, from Random Forest "
            "predict_proba(). Values above 0.5 correspond to churned=True."
        ),
        example=0.874,
    )

    risk_level: str = Field(
        ...,
        description=(
            "Human-readable risk tier: "
            "LOW (< 0.3) | MEDIUM (0.3-0.6) | HIGH (0.6-0.8) | CRITICAL (> 0.8)"
        ),
        example="HIGH",
    )


class RecommendationItem(BaseModel):
    """A single re-engagement recommendation for one feature gap."""

    feature: str = Field(
        ...,
        description="The feature where this user scores below similar retained users.",
        example="activity_frequency",
    )

    gap_score: float = Field(
        ...,
        description=(
            "Difference between SVD predicted score (similar retained users) "
            "and this user's actual score. Larger = bigger re-engagement opportunity."
        ),
        example=0.312,
    )

    action: str = Field(
        ...,
        description="Concrete re-engagement action tied to this feature gap.",
        example="Try creating a new repository this month.",
    )


class RecommendResponse(BaseModel):
    """Output schema for POST /recommend."""

    user_id: int = Field(..., example=42)
    username: str = Field(..., example="ana_barrios")
    churn_probability: float = Field(..., example=0.78)
    at_risk: bool = Field(
        ...,
        description="True if churn_probability >= 0.5. Recommendations only returned when True.",
        example=True,
    )
    recommendations: list[RecommendationItem] = Field(
        ...,
        description="Ordered list of re-engagement actions, highest-gap feature first.",
    )
    message: str = Field(..., example="User is at risk (78.0%). Top 5 re-engagement actions returned.")


class HealthResponse(BaseModel):
    """Output schema for GET /health."""
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
    svd_loaded: bool = Field(..., example=True)
    features_expected: int = Field(..., example=3)


class FeaturesResponse(BaseModel):
    """Output schema for GET /features."""
    features: list[str] = Field(
        ...,
        description="Ordered list of feature names POST /predict expects as input.",
        example=["days_since_last_activity", "activity_frequency", "repos_per_year"],
    )
    count: int = Field(..., example=3)


# ===============================================================================
# SECTION 5 -- INTERNAL HELPERS
# ===============================================================================

def _build_feature_array(user: UserFeatures) -> np.ndarray:
    """
    Convert a validated UserFeatures Pydantic object into a 2D numpy array
    in the EXACT column order the Random Forest was trained on (FINAL_FEATURES).

    Column order matters: if the model learned "split on feature[0] > 400",
    feature[0] must be days_since_last_activity, not repos_per_year.

    Returns shape (1, n_features) -- one row (one user), n_features columns.
    """
    values = [getattr(user, feat) for feat in FINAL_FEATURES]
    return np.array([values])


def _get_risk_level(probability: float) -> str:
    """
    Map a churn probability to a human-readable risk tier.

    Thresholds are calibrated for GitHub developer churn:
      LOW      (< 0.30) : strong engagement signals, no intervention needed
      MEDIUM   (0.30-0.60) : some inactivity, worth monitoring
      HIGH     (0.60-0.80) : significant risk, intervention warranted
      CRITICAL (> 0.80) : almost certainly disengaged, urgent action needed
    """
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    elif probability < 0.80:
        return "HIGH"
    else:
        return "CRITICAL"


# Maps feature names to concrete GitHub re-engagement actions.
# Used by /recommend to translate SVD gap scores into actionable advice.
# Each action is tied to the specific behavioral signal the feature captures.
_FEATURE_ACTIONS = {
    "days_since_last_activity": (
        "Push a commit or update your profile -- even small activity signals "
        "you are still engaged with the platform."
    ),
    "activity_frequency": (
        "Try creating a new repository this month. Users similar to you who "
        "stayed active create repos regularly."
    ),
    "repos_per_year": (
        "Start a new project. Your repo creation rate is below similar "
        "active users -- consistency is the strongest retention signal."
    ),
    "social_volume": (
        "Follow 5-10 developers in your area of interest to rebuild "
        "your community ties on GitHub."
    ),
    "follower_ratio": (
        "Share a project or write a detailed README to attract followers. "
        "Social recognition is a strong retention driver."
    ),
    "engagement_ratio": (
        "Engage with others' repos by starring, forking, or contributing -- "
        "community engagement correlates strongly with long-term retention."
    ),
    "profile_completeness": (
        "Complete your GitHub profile (bio, website, company) -- "
        "complete profiles have significantly lower churn rates."
    ),
    "account_age_days": (
        "Your account has history -- leverage it by pinning your best "
        "repositories to make your profile more discoverable."
    ),
    "has_no_repos": (
        "Create your first public repository. Users with no public repos "
        "churn at significantly higher rates than those with even one."
    ),
    "has_complete_profile": (
        "Fill in all optional profile fields (bio, blog, company, hireable). "
        "A fully complete profile signals long-term platform commitment."
    ),
}


# ===============================================================================
# SECTION 6 -- ENDPOINTS
# ===============================================================================

# -- POST /predict --------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict GitHub user churn",
    tags=["Prediction"],
)
def predict_churn(user: UserFeatures) -> PredictionResponse:
    """
    Accept a user's derived feature values and return a churn prediction.

    FastAPI validates the JSON body against UserFeatures automatically.
    Returns 422 if any field is missing or has the wrong type.

    Example request:
      {"days_since_last_activity": 600, "activity_frequency": 0.8, "repos_per_year": 0.8}

    Example response:
      {"churned": true, "churn_probability": 0.82, "risk_level": "CRITICAL"}
    """
    try:
        feature_array = _build_feature_array(user)

        # predict() returns array of shape (1,) -- 0 = retained, 1 = churned
        prediction = model.predict(feature_array)[0]

        # predict_proba() returns shape (1, 2):
        #   [:, 0] = P(retained), [:, 1] = P(churned)
        churn_prob = float(model.predict_proba(feature_array)[0][1])

        return PredictionResponse(
            churned=bool(prediction),
            churn_probability=round(churn_prob, 3),
            risk_level=_get_risk_level(churn_prob),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )


# -- POST /recommend ------------------------------------------------------------

@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Recommend re-engagement actions for at-risk users",
    tags=["Recommendation"],
)
def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Return top-N personalized re-engagement recommendations for a GitHub user.

    FLOW (as specified in the Unit 9 docs):
      1. Look up the user by user_id (row index in github_features.csv)
      2. Reconstruct their feature vector and get churn probability from the RF model
      3. If prob < 0.5 --> return empty recommendations (user not at risk)
      4. If prob >= 0.5 --> use SVD predicted scores to find which features this
         user scores lowest on compared to similar users in latent space.
         Those gaps are the re-engagement opportunities.

    WHY SVD FOR RECOMMENDATIONS:
    SVD decomposes the user-feature matrix into latent user vectors. Users
    close together in this latent space behave similarly on GitHub. For an
    at-risk user, we compare their actual feature scores against the SVD
    reconstruction (which reflects what similar users look like). The features
    with the largest positive gap -- where similar users score high but this
    user scores low -- are the behavioral levers most likely to re-engage them.
    """
    # Guard: SVD data must be loaded
    if svd_data is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "SVD data not available. Run recommend.py to generate svd_data.pkl "
                "and ensure it is in the app/ directory."
            ),
        )

    # Guard: user_id must be within dataset bounds
    n_users = len(usernames)
    if request.user_id >= n_users:
        raise HTTPException(
            status_code=404,
            detail=(
                f"user_id {request.user_id} is out of range. "
                f"Dataset has {n_users} users (valid range: 0 to {n_users - 1})."
            ),
        )

    user_idx = request.user_id
    username = str(usernames[user_idx])

    # -- Step 1: Get churn probability for this user ---------------------------
    # The RF model was trained on FINAL_FEATURES (3 features), but the SVD
    # matrix has all 10 features. We slice to the final 3 for prediction.
    # MinMaxScaler was applied during SVD so scores are already in [0,1].
    final_indices = [
        feature_cols.index(f)
        for f in FINAL_FEATURES
        if f in feature_cols
    ]
    user_final_features = user_item_matrix[user_idx][final_indices].reshape(1, -1)
    churn_prob = float(model.predict_proba(user_final_features)[0][1])

    # -- Step 2: Gate on churn probability (docs requirement: prob >= 0.5) -----
    # This is the core requirement from the Unit 9 documentation:
    # "Must check churn probability first and only return results for
    # at-risk users (prob >= 0.5)."
    if churn_prob < 0.5:
        return RecommendResponse(
            user_id=user_idx,
            username=username,
            churn_probability=round(churn_prob, 3),
            at_risk=False,
            recommendations=[],
            message="User is not at risk. No intervention needed.",
        )

    # -- Step 3: Compute feature gaps from SVD ---------------------------------
    # predicted_scores[user_idx] = what SVD reconstructs this user's scores
    # as, based on the latent structure of ALL users. This reflects the
    # "average" behavior of users similar to this one in latent space.
    #
    # user_item_matrix[user_idx] = this user's ACTUAL feature scores.
    #
    # gap = predicted - actual:
    #   positive gap = similar users score high here, this user scores low
    #                = re-engagement opportunity
    #   negative gap = this user already exceeds similar users here
    #                = not an intervention target
    svd_predicted = predicted_scores[user_idx]        # shape: (n_features,)
    actual_scores  = user_item_matrix[user_idx]        # shape: (n_features,)
    gaps           = svd_predicted - actual_scores     # shape: (n_features,)

    # Sort by gap descending -- biggest opportunity first
    sorted_indices = np.argsort(gaps)[::-1][:request.top_n]

    # Build RecommendationItem objects for the response
    recommendations = []
    for idx in sorted_indices:
        feat_name = feature_cols[idx]
        gap_value = float(gaps[idx])
        action    = _FEATURE_ACTIONS.get(
            feat_name,
            f"Improve your {feat_name} score to match similar retained users."
        )
        recommendations.append(
            RecommendationItem(
                feature=feat_name,
                gap_score=round(gap_value, 4),
                action=action,
            )
        )

    return RecommendResponse(
        user_id=user_idx,
        username=username,
        churn_probability=round(churn_prob, 3),
        at_risk=True,
        recommendations=recommendations,
        message=(
            f"User is at risk ({churn_prob:.1%}). "
            f"Top {request.top_n} re-engagement actions returned."
        ),
    )


# -- GET /health ---------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["Operations"],
)
def health() -> HealthResponse:
    """
    Health check endpoint -- returns status ok when the API is running.

    Used by Docker and cloud platforms to verify the container is alive.
    Also confirms both model.pkl and svd_data.pkl loaded successfully.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        svd_loaded=svd_data is not None,
        features_expected=len(FINAL_FEATURES),
    )


# -- GET /features -------------------------------------------------------------

@app.get(
    "/features",
    response_model=FeaturesResponse,
    summary="List expected input features for /predict",
    tags=["Operations"],
)
def get_features() -> FeaturesResponse:
    """
    Returns the ordered list of feature names POST /predict expects.

    Always synchronized with FINAL_FEATURES in model.py -- the single
    source of truth for what the Random Forest was trained on.
    """
    return FeaturesResponse(
        features=FINAL_FEATURES,
        count=len(FINAL_FEATURES),
    )


# -- GET / (root) --------------------------------------------------------------

@app.get(
    "/",
    summary="API root",
    tags=["Operations"],
    include_in_schema=False,
)
def root() -> dict:
    """Welcome message with links to all endpoints."""
    return {
        "message": "GitHub User Churn Predictor API",
        "docs": "http://localhost:8000/docs",
        "endpoints": {
            "POST /predict":    "Predict churn for a GitHub user",
            "POST /recommend":  "Get re-engagement recommendations for at-risk users",
            "GET  /health":     "Health check (confirms model + SVD loaded)",
            "GET  /features":   "List expected input features for /predict",
        },
    }


# ===============================================================================
# SECTION 7 -- LOCAL DEVELOPMENT ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    """
    Run locally without Docker:
        python main.py

    Or with uvicorn directly (recommended for development):
        uvicorn main:app --host 0.0.0.0 --port 8000 --reload

    Docker uses:
        CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
    """
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )