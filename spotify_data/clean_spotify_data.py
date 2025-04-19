import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# === Step 1: Load data ===
df = pd.read_csv("tracks_features.csv")

# === Step 2: Select relevant numerical features ===
features = [
    'danceability', 'energy', 'key', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'duration_ms'
]

df_features = df[features]

# === Step 3: Drop rows with missing or infinite values ===
df_clean = df.dropna(subset=features).copy()
df_clean = df_clean[~df_clean[features].isin([float("inf"), float("-inf")]).any(axis=1)]

# === Step 4: Normalize the features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# === Step 5: PCA to 2D ===
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# === Step 6: Add back track IDs and save ===
df_output = pd.DataFrame({
    'x': X_pca[:, 0],
    'y': X_pca[:, 1],
    'z': X_pca[:, 2]
})

# Optional: save original clusterable features too, if needed later
# for f in features:
#     df_output[f] = df_clean[f].values

df_output.to_csv("spotify_pca_cleaned.csv", index=False)
print("✅ Saved cleaned PCA dataset as 'spotify_pca_cleaned.csv'")
