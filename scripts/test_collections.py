import argparse
import time
import pandas as pd
import requests

DEFAULT_USER = "user1"
DEFAULT_PASSWORD = "ZaYVK1fsbw1ZfbX3OX"


def get_token(base_url: str, username: str, password: str) -> str:
    """Authenticate and return JWT token."""
    resp = requests.post(
        f"{base_url}/token",
        data={"username": username, "password": password},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def wait_for_api(base_url: str, timeout: int = 60) -> None:
    """Wait until API is responsive on /health."""
    start = time.time()
    while True:
        try:
            resp = requests.get(f"{base_url}/health")
            if resp.ok:
                print("API is ready!")
                return
        except Exception as exc:
            print(f"Waiting for API: {exc}")
        if time.time() - start > timeout:
            raise RuntimeError("API did not become available in time")
        time.sleep(1)


def read_items(path: str):
    """Read dataset file and convert to list of request payloads."""
    df = pd.read_csv(path)
    required = {"id", "subject", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns {missing} not found in {path}")

    items = []
    for _, row in df.iterrows():
        item = {
            "id": str(row["id"]),
            "subject": str(row["subject"]),
            "description": str(row["description"]),
        }
        if "class_name" in df.columns and not pd.isna(row["class_name"]):
            item["class_name"] = str(row["class_name"])
        if "task" in df.columns and not pd.isna(row["task"]):
            item["task"] = str(row["task"])
        items.append(item)
    return items


def upload_dataset(items, token, base_url, collection=None):
    payload = {"items": items}
    if collection:
        payload["collection"] = collection
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(f"{base_url}/upload", json=payload, headers=headers)
    resp.raise_for_status()
    print(f"Uploaded {len(items)} items to {collection or 'default'} collection")


def run_search(token, base_url, collection=None):
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "subject": "test",
        "description": "sample",
        "limit": 1,
    }
    if collection:
        payload["collection"] = collection
    resp = requests.post(f"{base_url}/search", json=payload, headers=headers)
    resp.raise_for_status()
    print(f"Search in {collection or 'default'} collection returned: {resp.json()}")


def health_check(base_url: str):
    resp = requests.get(f"{base_url}/health")
    resp.raise_for_status()
    print(f"Health check returned: {resp.json()}")


def get_current_user(token: str, base_url: str):
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{base_url}/users/me", headers=headers)
    resp.raise_for_status()
    print(f"Current user: {resp.json()}")


def run_predict(item, token, base_url, collection=None):
    payload = dict(item)
    if collection:
        payload["collection"] = collection
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(f"{base_url}/predict", json=payload, headers=headers)
    resp.raise_for_status()
    print(f"Predict in {collection or 'default'} collection returned: {resp.json()}")


def clear_collection(token, base_url, collection=None):
    payload = {}
    if collection:
        payload["collection"] = collection
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(f"{base_url}/clear_index", json=payload or None, headers=headers)
    resp.raise_for_status()
    print(f"Cleared {collection or 'default'} collection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload two datasets to separate Qdrant collections and test search"
    )
    parser.add_argument("default_dataset", help="Dataset for the default collection")
    parser.add_argument("custom_dataset", help="Dataset for the custom collection")
    parser.add_argument(
        "--collection", default="requests_new", help="Name for custom collection"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Base URL of the API"
    )
    parser.add_argument("--user", default=DEFAULT_USER, help="API username")
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="API password")

    args = parser.parse_args()

    # Wait for API to become ready and basic endpoint tests
    wait_for_api(args.base_url)
    health_check(args.base_url)

    token = get_token(args.base_url, args.user, args.password)
    get_current_user(token, args.base_url)

    # Prepare collections
    clear_collection(token, args.base_url)
    clear_collection(token, args.base_url, args.collection)

    default_items = read_items(args.default_dataset)
    upload_dataset(default_items, token, args.base_url)

    custom_items = read_items(args.custom_dataset)
    upload_dataset(custom_items, token, args.base_url, args.collection)

    run_search(token, args.base_url)
    run_search(token, args.base_url, args.collection)

    # Predict on first item from each dataset
    run_predict(default_items[0], token, args.base_url)
    run_predict(custom_items[0], token, args.base_url, args.collection)

    # Cleanup
    clear_collection(token, args.base_url)
    clear_collection(token, args.base_url, args.collection)

    print("Testing completed successfully")
