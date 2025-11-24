import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


def parse_features(feat_str):
    return np.fromstring(feat_str.strip("[]"), sep=",")


def solve_reid_weighted(train_path, test_path, output_path):
    print("Loading and processing data...")
    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)

    # Convert timestamps to datetime
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

    # 1. Parse Embeddings
    train_df["feat_vec"] = train_df["features"].apply(parse_features)
    test_df["feat_vec"] = test_df["features"].apply(parse_features)

    # Stack features for matrix operations
    # Normalize vectors so dot product = cosine similarity
    train_feats = np.stack(train_df["feat_vec"].values)
    train_feats = train_feats / np.linalg.norm(train_feats, axis=1, keepdims=True)

    test_feats = np.stack(test_df["feat_vec"].values)
    test_feats = test_feats / np.linalg.norm(test_feats, axis=1, keepdims=True)

    results = []

    # --- TUNABLE WEIGHTS ---
    # If your model matches wrong cars, INCREASE alpha (trust physics more)
    # If your model misses matches due to time gaps, DECREASE alpha
    ALPHA_VISUAL = 0.7  # Importance of looking the same (0.0 to 1.0)
    ALPHA_TIME = 0.3  # Importance of being close in time

    # Threshold to say "This is a NEW vehicle"
    # If composite score is below this, we assume it's a new truck.
    MATCH_THRESHOLD = 0.65

    print("Computing matches...")

    for idx, row in test_df.iterrows():
        test_vec = row["feat_vec"].reshape(1, -1)
        test_time = row["timestamp"]
        test_dir = row["direction"]
        test_utm = row["position_utm"]  # Assuming this is a scalar or [x,y]

        # --- 1. HARD CONSTRAINTS (The Logic Filter) ---
        # Only look at vehicles moving the same way
        # Only look at vehicles from the PAST (timestamp < current)
        candidates = train_df[
            (train_df["direction"] == test_dir)
            & (train_df["timestamp"] < test_time)
            # Add drone logic here if you want strictly 1->2->3
        ].copy()

        if candidates.empty:
            results.append({"id": row["id"], "reid": "new_vehicle"})
            continue

        # --- 2. CALCULATE SCORES ---

        # A. Visual Score (Cosine Similarity: 0 to 1)
        cand_feats = np.stack(candidates["feat_vec"].values)
        cand_feats = cand_feats / np.linalg.norm(cand_feats, axis=1, keepdims=True)

        # Dot product of normalized vectors is Cosine Similarity
        visual_scores = np.dot(cand_feats, test_vec.T).flatten()

        # B. Temporal Score (Normalized Time Proximity)
        # We want a score of 1.0 if times are identical, 0.0 if very far apart
        # Convert timedelta to seconds
        time_diffs = np.abs((candidates["timestamp"] - test_time).dt.total_seconds())

        # Normalize time diffs to 0-1 range relative to this batch
        # Use a soft decay: exp(-decay * time_diff)
        # Adjust '0.01' depending on your unit of time (seconds? ms?)
        # If timestamp is seconds, 0.01 means score drops by half every ~70 seconds
        time_scores = np.exp(-0.01 * time_diffs).values

        # --- 3. COMBINE SCORES ---
        # Final Score = Weighted Average of Visual and Time
        final_scores = (ALPHA_VISUAL * visual_scores) + (ALPHA_TIME * time_scores)

        # --- 4. PICK BEST ---
        best_idx = np.argmax(final_scores)
        best_score = final_scores[best_idx]

        if best_score > MATCH_THRESHOLD:
            matched_reid = candidates.iloc[best_idx]["reid"]
        else:
            matched_reid = "new_vehicle"

        results.append({"id": row["id"], "reid": matched_reid})

    # Save
    submission = pd.DataFrame(results)
    submission.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")


solve_reid_weighted(
    "TruckReidDataset/train.pkl", "TruckReidDataset/test.pkl", "submission_weighted.csv"
)
