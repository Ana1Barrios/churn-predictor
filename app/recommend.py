import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import pandas as pd
import joblib

# Load your feature matrix (users × features)
df = pd.read_csv("../data/processed/github_features.csv")
feature_cols = ["follower_ratio", "engagement_ratio", "days_since_last_activity",
                "account_age_days", "activity_frequency", "repos_per_year",
                "social_volume", "profile_completeness", "has_no_repos",
                "has_complete_profile"]

X = df[feature_cols].values
y = df["churn"].values

# Normalize to [0,1] so all features are on the same scale for SVD
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert to sparse matrix — svds requires this
user_item_matrix = csr_matrix(X_scaled)

# k=5 latent factors (keep small for 300 users × 10 features)
# k must be < min(rows, cols) - 1
U, sigma, Vt = svds(user_item_matrix, k=5)
sigma_diag = np.diag(sigma)

# Reconstruct predicted scores for every user × feature combination
predicted_scores = np.dot(np.dot(U, sigma_diag), Vt)

# Save everything main.py needs
joblib.dump({
    "predicted_scores": predicted_scores,
    "user_item_matrix": X_scaled,
    "usernames": df["username"].tolist() if "username" in df.columns else list(range(len(df))),
    "feature_cols": feature_cols,
    "churn_probs": None,   # filled at runtime by the RF model
}, "svd_data.pkl")

print("SVD decomposition complete. svd_data.pkl saved.")
print(f"Matrix shape: {X_scaled.shape}")
print(f"U shape: {U.shape}, sigma: {sigma}, Vt shape: {Vt.shape}")