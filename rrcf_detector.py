import os
import argparse
import numpy as np
import pandas as pd
import time
import rrcf
import json
from datetime import datetime, timedelta
from opensearch_client import get_opensearch_client, discover_indices
from scipy.stats import poisson

def generate_synthetic_data(num_points=1440, lambda_val=5, spike_index=720, spike_mult=150):
    """
    Generate synthetic Poisson-distributed data with a large spike.
    """
    print(f"Generating {num_points} synthetic data points (lambda={lambda_val})...")
    data = poisson.rvs(lambda_val, size=num_points)
    data[spike_index] = lambda_val * spike_mult
    
    # Create timestamps starting from 24 hours ago
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=num_points-1)
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_points)]
    
    return pd.Series(data, index=pd.to_datetime(timestamps), name="event_density_y")

def fetch_real_data(client, indices, start_date, end_date):
    """
    Fetch real alert data from OpenSearch indices.
    """
    query = {
        "query": {
            "range": {
                "@timestamp": {
                    "gte": start_date,
                    "lte": end_date
                }
            }
        },
        "size": 10000,
        "_source": ["@timestamp"]
    }
    
    all_hits = []
    # Only use indices that match the wazuh-alerts-4.x-* pattern
    target_indices = [idx for idx in indices if idx.startswith("wazuh-alerts-4.x-")]
    
    if not target_indices:
        print("No Wazuh alert indices found.")
        return None
    
    print(f"Querying indices: {target_indices}")
    try:
        resp = client.search(index=target_indices, body=query)
        hits = [h["_source"] for h in resp["hits"]["hits"]]
        if not hits:
            print("No data found in the specified time range.")
            return None
        
        df = pd.DataFrame(hits)
        df["@timestamp"] = pd.to_datetime(df["@timestamp"])
        df = df.set_index("@timestamp").sort_index()
        
        # Resample to 1-minute buckets
        density = df.resample("1min").size().rename("event_density_y")
        return density
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def create_anomalies_index(client, index_name="rrcf-anomalies"):
    """
    Create the anomalies index with explicit mapping.
    """
    mapping = {
        "mappings": {
            "properties": {
                "timestamp": {"type": "date"},
                "anomaly_grade": {"type": "float"},
                "is_anomaly": {"type": "boolean"},
                "detector": {"type": "keyword"},
                "feature": {"type": "keyword"},
                "value": {"type": "float"}
            }
        }
    }
    
    if not client.indices.exists(index=index_name):
        print(f"Creating index {index_name} with explicit mapping...")
        client.indices.create(index=index_name, body=mapping)
    else:
        print(f"Index {index_name} already exists.")

def fetch_analyst_feedback(client, index_name="rrcf-feedback"):
    """
    Placeholder for fetching analyst feedback.
    Analyzes FP rates and returns a threshold adjustment.
    """
    try:
        if not client.indices.exists(index=index_name):
            return 0.0
        
        # Example logic: count FPs in last 24h
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"feedback_type": "FP"}},
                        {"range": {"timestamp": {"gte": "now-24h"}}}
                    ]
                }
            }
        }
        resp = client.count(index=index_name, body=query)
        fp_count = resp["count"]
        
        # If FP rate is high, suggest raising threshold
        if fp_count > 10: # arbitrary threshold for demo
            print(f"High FP rate detected ({fp_count} in 24h). Adjusting threshold +0.05")
            return 0.05
        return 0.0
    except Exception as e:
        print(f"Error fetching feedback: {e}")
        return 0.0

def run_detection_cycle(client, indices, args, forest_state=None, drift_history=None, start_index=0):
    """
    Executes a single detection cycle (fetch, score, detect, index).
    Returns updated forest_state, drift_history, and start_index.
    """
    # 1. Setup Forest
    tree_count = 40
    window_size = 256
    shingle_size = 4
    
    if forest_state is None:
        print("Initializing new RRCF forest...")
        forest_state = [rrcf.RCTree() for _ in range(tree_count)]
    
    if drift_history is None:
        drift_history = [] # list of anomaly counts

    # 2. Fetch Data
    density = None
    if args.start and args.end:
        density = fetch_real_data(client, indices, args.start, args.end)
    elif args.mode == "scheduled":
        # Fetch last 120 minutes of data to ensure overlap and shingling
        end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        start_time = (datetime.now() - timedelta(minutes=120)).strftime("%Y-%m-%dT%H:%M:%S")
        density = fetch_real_data(client, indices, start_time, end_time)
        
    if density is None:
        print("No real data found. Using synthetic data for cycle...")
        density = generate_synthetic_data()

    # 3. Score
    scores_df, next_index = run_rrcf(density, tree_count, window_size, shingle_size, forest_state, start_index)
    
    # 4. Thresholding
    # Normalize to Anomaly Grade (0-1)
    min_s = scores_df["anomaly_score"].min()
    max_s = scores_df["anomaly_score"].max()
    if max_s > min_s:
        scores_df["anomaly_grade"] = (scores_df["anomaly_score"] - min_s) / (max_s - min_s)
    else:
        scores_df["anomaly_grade"] = 0.0
        
    current_threshold = args.threshold
    if args.auto_threshold:
        computed = scores_df["anomaly_grade"].mean() + 2 * scores_df["anomaly_grade"].std()
        print(f"Auto-threshold computed: {computed:.4f} (Default was {args.threshold})")
        current_threshold = computed
    
    # Feedback adjustment
    current_threshold += fetch_analyst_feedback(client)
    
    scores_df["is_anomaly"] = scores_df["anomaly_grade"] >= current_threshold
    
    # 5. Cooldown logic
    cooldown = 0
    is_anomaly_suppressed = []
    for _, row in scores_df.iterrows():
        if cooldown > 0:
            is_anomaly_suppressed.append(False)
            cooldown -= 1
            continue
        if row["is_anomaly"]:
            is_anomaly_suppressed.append(True)
            cooldown = args.cooldown
        else:
            is_anomaly_suppressed.append(False)
    scores_df["is_anomaly"] = is_anomaly_suppressed
    
    anomalies = scores_df[scores_df["is_anomaly"]]
    num_anomalies = len(anomalies)
    print(f"Cycle completed. Detected {num_anomalies} anomalies.")

    # 6. Drift Detection & Forest Reset
    drift_history.append(num_anomalies)
    if len(drift_history) > 3: # Keep last 3 drift windows (hours)
        drift_history.pop(0)
    
    # Rule: If anomalies > 5 in each of the last 3 hours -> Reset
    if len(drift_history) == 3 and all(count > 5 for count in drift_history):
        print("!!! DRIFT DETECTED !!! Anomalies high for 3 consecutive cycles. Resetting forest.")
        forest_state = [rrcf.RCTree() for _ in range(tree_count)]
        drift_history = []
        next_index = 0 # Reset index too
    
    # 7. Index Results
    if not args.dry_run and not anomalies.empty:
        create_anomalies_index(client)
        for _, row in anomalies.iterrows():
            doc = {
                "timestamp": row["timestamp"].isoformat(),
                "anomaly_grade": float(row["anomaly_grade"]),
                "is_anomaly": bool(row["is_anomaly"]),
                "detector": "rrcf-custom",
                "feature": "event_density_y",
                "value": float(row["value"])
            }
            client.index(index="rrcf-anomalies", body=doc)
    elif args.dry_run and not anomalies.empty:
        print("\nDry-run: Anomalies (top 5):")
        print(anomalies.head(5))

    return forest_state, drift_history, next_index

def run_rrcf(density, tree_count, window_size, shingle_size, forest, start_index=0):
    """
    Run RRCF on density series using provided forest state.
    """
    print(f"Scoring {len(density)} points with start_index={start_index}...")
    shingled_data = list(rrcf.shingle(density.values, shingle_size))
    scores = []
    relevant_timestamps = density.index[shingle_size - 1:]
    relevant_values = density.values[shingle_size - 1:]
    
    for i, point in enumerate(shingled_data):
        avg_score = 0
        idx = start_index + i
        for tree in forest:
            if len(tree.leaves) > window_size:
                tree.forget_point(min(tree.leaves.keys()))
            tree.insert_point(point, index=idx)
            avg_score += tree.codisp(idx)
        avg_score /= tree_count
        scores.append((relevant_timestamps[i], avg_score, relevant_values[i]))
            
    return pd.DataFrame(scores, columns=["timestamp", "anomaly_score", "value"]), start_index + len(shingled_data)

def main():
    parser = argparse.ArgumentParser(description="RRCF Anomaly Detection for Wazuh Alerts")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Print anomalies without writing to OpenSearch")
    parser.add_argument("--threshold", type=float, default=0.85, help="Anomaly grade threshold (0-1)")
    parser.add_argument("--cooldown", type=int, default=15, help="Cooldown period in points")
    parser.add_argument("--mode", type=str, choices=["batch", "scheduled"], default="batch", help="Execution mode")
    parser.add_argument("--drift-window", type=int, default=60, help="Drift window in minutes")
    parser.add_argument("--auto-threshold", action="store_true", help="Compute threshold dynamically (mean + 2*std)")
    parser.add_argument("--test-mode", action="store_true", help="Run with short sleep for verification")
    args = parser.parse_args()
    
    client = get_opensearch_client()
    indices = discover_indices()
    
    forest_state = None
    drift_history = None
    curr_index = 0
    
    if args.mode == "batch":
        run_detection_cycle(client, indices, args, forest_state, drift_history, curr_index)
    else:
        print(f"Entering scheduled mode. Cycle interval: {args.drift_window} minutes.")
        while True:
            forest_state, drift_history, curr_index = run_detection_cycle(client, indices, args, forest_state, drift_history, curr_index)
            sleep_time = 1 if args.test_mode else args.drift_window * 60
            print(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)

if __name__ == "__main__":
    main()
