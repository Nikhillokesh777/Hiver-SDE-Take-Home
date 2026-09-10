"""
Download script for Kaggle Customer Support on Twitter (TWCS) dataset.
Source: HuggingFace CDN mirror of twcs.csv (516 MB).
Ensures data is stored strictly in data/raw/ on D: drive.
"""

import os
import sys
import urllib.request
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
DATA_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"
OUTPUT_FILE = os.path.join(DATA_DIR, "twcs.csv")
TEMP_FILE = os.path.join(DATA_DIR, "twcs.csv.tmp")
EXPECTED_MIN_SIZE_BYTES = 500_000_000  # ~500 MB


def download_twcs():
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE)
        if file_size >= EXPECTED_MIN_SIZE_BYTES:
            print(f"[OK] Dataset already present at {OUTPUT_FILE} ({file_size / (1024*1024):.2f} MB). Skipping download.")
            return

    print(f"Downloading TWCS dataset from {DATA_URL}...")
    print(f"Destination: {OUTPUT_FILE}")

    req = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Hiver-Evaluation-Agent/1.0"}
    )

    start_time = time.time()
    last_report_time = start_time

    with urllib.request.urlopen(req) as response, open(TEMP_FILE, "wb") as out_file:
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)

            now = time.time()
            if now - last_report_time >= 2.0 or downloaded == total_size:
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                pct = (downloaded / total_size * 100) if total_size else 0
                speed = mb_downloaded / (now - start_time) if (now - start_time) > 0 else 0
                print(f"Progress: {mb_downloaded:.1f}/{mb_total:.1f} MB ({pct:.1f}%) - Speed: {speed:.2f} MB/s", flush=True)
                last_report_time = now

    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    os.rename(TEMP_FILE, OUTPUT_FILE)
    total_time = time.time() - start_time
    print(f"[SUCCESS] Download completed in {total_time:.1f}s. Saved to {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)/(1024*1024):.2f} MB).")


if __name__ == "__main__":
    download_twcs()
