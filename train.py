import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def parse_feature_string(feat_str):
    # Converts the string representation of list to actual numpy array
    # Adjust this based on your actual string format (e.g., removing brackets)
    return np.fromstring(feat_str.strip("[]"), sep=",")


def solve_reid(train_path, test_path, output_path):
    # 1. Load Data
    print("Loading data...")
    train_df = pd.read_pickle(train_path)
    test_df = pd.read_pickle(test_path)

    # 2. Parse Features (String -> Numpy Array)
    print("Parsing features...")
    train_df["feat_vec"] = train_df["features"].apply(parse_feature_string)
    test_df["feat_vec"] = test_df["features"].apply(parse_feature_string)

    # Stack features for fast matrix math
    train_feats = np.stack(train_df["feat_vec"].values)

    # Create a place to store results
    results = []

    # 3. Iterate through Test Samples
    for idx, row in test_df.iterrows():
        test_id = row["id"]
        test_feat = row["feat_vec"].reshape(1, -1)
        test_drone = row["drone"]
        test_dir = row["direction"]
        test_time = row["timestamp"]

        # --- CONSTRAINTS FILTERING ---

        # Filter 1: Direction must match
        # (Vehicles don't usually pull U-turns on highways between cameras)
        candidates = train_df[train_df["direction"] == test_dir].copy()

        # Filter 2: Drone Topology (South -> North: 1 -> 2 -> 3)
        # We need to handle both directions.
        # Northbound: 1 -> 2 -> 3 (Past is lower IDs)
        # Southbound: 3 -> 2 -> 1 (Past is higher IDs)

        if test_dir == "Northbound":
            if test_drone == 2:
                candidates = candidates[candidates["drone"] == 1]
            elif test_drone == 3:
                candidates = candidates[candidates["drone"].isin([1, 2])]
            elif test_drone == 1:
                candidates = candidates[candidates["drone"] == -1]  # Start
        elif test_dir == "Southbound":
            if test_drone == 2:
                candidates = candidates[candidates["drone"] == 3]
            elif test_drone == 1:
                candidates = candidates[candidates["drone"].isin([2, 3])]
            elif test_drone == 3:
                candidates = candidates[candidates["drone"] == -1]  # Start

        # Filter 3: Time Constraints
        # The candidate (past) must have appeared BEFORE the test (current)
        candidates = candidates[candidates["timestamp"] < test_time]

        # If no candidates remain after filtering, skip
        if candidates.empty:
            results.append({"id": test_id, "reid": "new_vehicle"})
            continue

        # --- VISUAL MATCHING ---

        # Calculate similarity between Test Image and ALL Valid Candidates
        candidate_feats = np.stack(candidates["feat_vec"].values)
        sim_scores = cosine_similarity(test_feat, candidate_feats)[0]

        # Find best match
        best_idx = np.argmax(sim_scores)
        best_score = sim_scores[best_idx]

        # Threshold: Only accept if visual similarity is high enough
        # (You will need to tune this, usually 0.6 to 0.8 for ReID)
        if best_score > 0.6:
            matched_reid = candidates.iloc[best_idx]["reid"]
        else:
            matched_reid = "new_vehicle"  # Or leave blank

        results.append({"id": test_id, "reid": matched_reid})

    # 4. Save Results
    submission = pd.DataFrame(results)
    submission.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


# Run it
solve_reid("TruckReidDataset/train.pkl", "TruckReidDataset/test.pkl", "submission.csv")
