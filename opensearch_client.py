import os
import time
from opensearchpy import OpenSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_opensearch_client():
    host = os.getenv("OPENSEARCH_HOST", "127.0.0.1")
    port = int(os.getenv("OPENSEARCH_PORT", 9200))
    user = os.getenv("OPENSEARCH_USER", "admin")
    password = os.getenv("OPENSEARCH_PASSWORD", "admin")
    use_ssl = os.getenv("OPENSEARCH_USE_SSL", "true").lower() == "true"
    verify_certs = os.getenv("OPENSEARCH_VERIFY_CERTS", "false").lower() == "true"
    auth_type = os.getenv("OPENSEARCH_AUTH_TYPE", "basic")

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_auth=(user, password),
        use_ssl=use_ssl,
        verify_certs=verify_certs,
        ssl_assert_hostname=False,
        ssl_show_warn=False
    )
    return client

def test_connection(max_retries=3, delay=5):
    client = get_opensearch_client()
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Attempting to connect to OpenSearch (Attempt {attempt}/{max_retries})...")
            if client.ping():
                print("Successfully connected to OpenSearch!")
                return True
            else:
                print("Could not ping OpenSearch.")
        except Exception as e:
            print(f"Connection attempt {attempt} failed: {e}")
        
        if attempt < max_retries:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
    
    return False

def discover_indices():
    client = get_opensearch_client()
    try:
        indices = client.cat.indices(format="json")
        index_names = [idx['index'] for idx in indices]
        print(f"Discovered {len(index_names)} indices.")
        return index_names
    except Exception as e:
        print(f"Failed to discover indices: {e}")
        return []

if __name__ == "__main__":
    if test_connection():
        indices = discover_indices()
        print("Indices found:")
        for idx in indices:
            if not idx.startswith('.'):
                print(f" - {idx}")
    else:
        print("Final connection test failed. Check your OpenSearch instance and credentials.")
