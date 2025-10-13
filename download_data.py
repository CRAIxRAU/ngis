#!/usr/bin/env python3
"""
Download EEG data from OpenNeuro dataset ds003766
"""

import requests
from pathlib import Path
from tqdm import tqdm

def download_file(url: str, output_path: Path):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"Downloaded: {output_path}")


def main():
    """Download sub-01 resting state data."""

    # OpenNeuro dataset base URL
    base_url = "https://s3.amazonaws.com/openneuro.org/ds003766"

    # Files to download
    subject = "sub-01"
    task = "resting"

    files = [
        f"sub-01/eeg/sub-01_task-{task}_eeg.set",
        f"sub-01/eeg/sub-01_task-{task}_eeg.fdt",
        f"sub-01/eeg/sub-01_task-{task}_eeg.json",
        f"sub-01/eeg/sub-01_task-{task}_channels.tsv",
        f"sub-01/eeg/sub-01_task-{task}_events.tsv",
    ]

    # Output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {subject} task-{task} data from OpenNeuro...")
    print(f"Output directory: {output_dir.absolute()}\n")

    for file_path in files:
        url = f"{base_url}/{file_path}"
        output_path = output_dir / Path(file_path).name

        if output_path.exists():
            print(f"Skipping (already exists): {output_path.name}")
            continue

        try:
            download_file(url, output_path)
        except Exception as e:
            print(f"Failed to download {file_path}: {e}")

    print("\n✅ Download complete!")
    print(f"Files are in: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
