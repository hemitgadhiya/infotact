import os
import urllib.request

def download_dataset(url, dest_path):
    """
    Download a file from url to dest_path, creating directories if needed.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        print(f"File already exists at {dest_path}, skipping download.")
        return
    print(f"Downloading from {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/jmatth11/King-County-House-Data-Set/master/kc_house_data.csv"
    dest = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "kc_house_data.csv"))
    download_dataset(url, dest)
