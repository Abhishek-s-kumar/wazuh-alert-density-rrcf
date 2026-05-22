import os
import argparse
import numpy as np
import pandas as pd
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

def run_rrcf(density, tree_count=40, window_size=256, shingle_size=4):
    """
    Run Robust Random Cut Forest on the density series.
    """
    print(f"Running RRCF (trees={tree_count}, window={window_size}, shingle={shingle_size})...")
    forest = [rrcf.RCTree() for _ in range(tree_count)]
    
    # rrcf.shingle is a generator that yields windows of size shingle_size
    shingled_data = list(rrcf.shingle(density.values, shingle_size))
    
    scores = []
    # The first shingle corresponds to density.index[shingle_size - 1]
    # because shingle generator starts yielding after 'size' points
    relevant_timestamps = density.index[shingle_size - 1:]
    relevant_values = density.values[shingle_size - 1:]
    
    for i, point in enumerate(shingled_data):
        avg_score = 0
        for tree in forest:
            if len(tree.leaves) > window_size:
                tree.forget_point(min(tree.leaves.keys()))
            tree.insert_point(point, index=i)
            avg_score += tree.codisp(i)
        avg_score /= tree_count
        scores.append((relevant_timestamps[i], avg_score, relevant_values[i]))
            
    return pd.DataFrame(scores, columns=["timestamp", "anomaly_score", "value"])

def main():
    parser = argparse.ArgumentParser(description="RRCF Anomaly Detection for Wazuh Alerts")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Print anomalies without writing to OpenSearch")
    parser.add_argument("--threshold", type=float, default=0.65, help="Anomaly grade threshold (0-1)")
    args = parser.parse_args()
    
    client = get_opensearch_client()
    indices = discover_indices()
    
    density = None
    if args.start and args.end:
        density = fetch_real_data(client, indices, args.start, args.end)
        
    if density is None:
        print("Falling back to synthetic data...")
        density = generate_synthetic_data()
        
    scores_df = run_rrcf(density)
    
    # Normalize to Anomaly Grade (0-1)
    min_s = scores_df["anomaly_score"].min()
    max_s = scores_df["anomaly_score"].max()
    if max_s > min_s:
        scores_df["anomaly_grade"] = (scores_df["anomaly_score"] - min_s) / (max_s - min_s)
    else:
        scores_df["anomaly_grade"] = 0.0
        
    scores_df["is_anomaly"] = scores_df["anomaly_grade"] >= args.threshold
    
    # Cooldown logic: suppress "echo" detections within shingle_size points of an anomaly
    shingle_size = 4 # matches default in run_rrcf
    cooldown = 0
    is_anomaly_suppressed = []
    
    for _, row in scores_df.iterrows():
        if cooldown > 0:
            is_anomaly_suppressed.append(False)
            cooldown -= 1
            continue
        
        if row["is_anomaly"]:
            is_anomaly_suppressed.append(True)
            cooldown = shingle_size
        else:
            is_anomaly_suppressed.append(False)
            
    scores_df["is_anomaly"] = is_anomaly_suppressed
    
    anomalies = scores_df[scores_df["is_anomaly"]]
    print(f"\nDetected {len(anomalies)} anomalies.")
    
    if not args.dry_run:
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
            print(f"Indexed anomaly: {doc['timestamp']} - Grade: {doc['anomaly_grade']:.2f}")
    else:
        print("\nDry-run mode: Anomalies detected (first 10):")
        print(anomalies.head(10))

if __name__ == "__main__":
    main()
