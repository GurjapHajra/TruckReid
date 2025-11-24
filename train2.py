import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# --- HELPER FUNCTIONS ---
def parse_features(feat_str):
    return np.fromstring(feat_str.strip('[]'), sep=',')

def create_pairs(df, sample_size=10000):
    """
    Creates 'Positive' (Same ID) and 'Negative' (Diff ID) pairs for training.
    """
    pairs = []
    labels = []
    
    # Group by ID to find positives
    grouped = df.groupby('reid')
    ids = list(grouped.groups.keys())
    
    print("Generating training pairs...")
    
    # 1. POSITIVE PAIRS (Same vehicle, different times)
    for vid in ids:
        group = grouped.get_group(vid)
        if len(group) < 2: continue
        
        # Create pairs of detections for this single vehicle
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                row1 = group.iloc[i]
                row2 = group.iloc[j]
                pairs.append((row1, row2))
                labels.append(1) # Label 1 = Same Vehicle

    # 2. NEGATIVE PAIRS (Different vehicles)
    # We randomly sample negatives to keep dataset balanced
    num_positives = len(pairs)
    for _ in range(num_positives): 
        id1, id2 = np.random.choice(ids, 2, replace=False)
        row1 = grouped.get_group(id1).sample(1).iloc[0]
        row2 = grouped.get_group(id2).sample(1).iloc[0]
        pairs.append((row1, row2))
        labels.append(0) # Label 0 = Different Vehicle
        
    return pairs, labels

def extract_pair_features(row1, row2):
    """
    Calculates the physics and visual differences between two rows.
    """
    # 1. Visual Similarity (Cosine)
    vec1 = row1['feat_vec']
    vec2 = row2['feat_vec']
    vis_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # 2. Temporal Distance
    time_diff = abs(row1['timestamp'] - row2['timestamp']).total_seconds()
    
    # 3. Spatial Distance (Euclidean on UTM)
    # Assuming position_utm is a tuple or list in the dataframe
    pos1 = np.array(row1['position_utm'])
    pos2 = np.array(row2['position_utm'])
    dist = np.linalg.norm(pos1 - pos2)
    
    # 4. Speed (Distance / Time)
    speed = dist / (time_diff + 1e-5) # Avoid division by zero
    
    # 5. Direction Match (1 if same, 0 if diff)
    dir_match = 1 if row1['direction'] == row2['direction'] else 0
    
    return [vis_sim, time_diff, dist, speed, dir_match]

# --- MAIN EXECUTION ---

# 1. Load Data
train_df = pd.read_pickle('TruckReidDataset/train.pkl')
test_df = pd.read_pickle('TruckReidDataset/test.pkl')

# Convert timestamp to datetime objects
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

# Parse string features to numpy
train_df['feat_vec'] = train_df['features'].apply(parse_features)
test_df['feat_vec'] = test_df['features'].apply(parse_features)

# 2. Create Training Data (Pairs)
pairs, labels = create_pairs(train_df)
X = [extract_pair_features(p[0], p[1]) for p in pairs]
y = labels

# 3. Train XGBoost Model
print("Training Classifier...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Evaluate quickly
acc = model.score(X_val, y_val)
print(f"Model Validation Accuracy: {acc:.2f}")

# 4. Apply to Test Set
# Logic: For each Test truck, compare against Train trucks and find highest prob match
results = []
train_candidates = train_df.sample(frac=1.0) # Use full train set in reality

print("Matching test vehicles...")
for idx, test_row in test_df.iterrows():
    
    # OPTIONAL: Apply simple filters to reduce candidates (Direction/Drone)
    # to speed up the loop before ML prediction
    candidates = train_candidates[
        (train_candidates['direction'] == test_row['direction']) &
        (train_candidates['timestamp'] < test_row['timestamp'])
    ].copy()
    
    if candidates.empty:
        results.append({'id': test_row['id'], 'reid': 'new_vehicle'})
        continue

    # Prepare batch features for this test item against all candidates
    batch_features = []
    candidate_ids = []
    
    for _, cand_row in candidates.iterrows():
        feats = extract_pair_features(test_row, cand_row)
        batch_features.append(feats)
        candidate_ids.append(cand_row['reid'])
    
    # Predict Probabilities
    probs = model.predict_proba(batch_features)[:, 1] # Probability of "Same Class"
    
    best_idx = np.argmax(probs)
    best_prob = probs[best_idx]
    
    # Threshold (Learned by model, but we apply a safety cutoff)
    if best_prob > 0.5: 
        results.append({'id': test_row['id'], 'reid': candidate_ids[best_idx]})
    else:
        results.append({'id': test_row['id'], 'reid': 'new_vehicle'})

# Save
pd.DataFrame(results).to_csv('submission_ml.csv', index=False)
print("Done!")