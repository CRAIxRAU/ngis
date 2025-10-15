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
    """Download EEG data from ds003766."""

    # OpenNeuro dataset base URL
    base_url = "https://s3.amazonaws.com/openneuro.org/ds003766"

    # Configuration
    subjects = [f"sub-{i:02d}" for i in range(1, 32)]  # sub-01 to sub-31
    tasks = ["resting"]  # Start with resting, can add: food-choice, word-choice, image-choice

    # Output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(subjects)} subjects × {len(tasks)} tasks from OpenNeuro ds003766")
    print(f"Output directory: {output_dir.absolute()}\n")

    total_files = 0
    skipped_files = 0
    failed_files = 0

    for subject in subjects:
        for task in tasks:
            print(f"\n📥 {subject} task-{task}:")

            files = [
                f"{subject}/eeg/{subject}_task-{task}_eeg.set",
                f"{subject}/eeg/{subject}_task-{task}_eeg.fdt",
                f"{subject}/eeg/{subject}_task-{task}_eeg.json",
                f"{subject}/eeg/{subject}_task-{task}_channels.tsv",
                f"{subject}/eeg/{subject}_task-{task}_events.tsv",
            ]

            for file_path in files:
                url = f"{base_url}/{file_path}"
                # Filename already has subject ID: sub-01_task-resting_eeg.fdt
                output_path = output_dir / Path(file_path).name

                if output_path.exists():
                    print(f"  ⏭️  Skipping: {output_path.name}")
                    skipped_files += 1
                    continue

                try:
                    download_file(url, output_path)
                    total_files += 1
                except Exception as e:
                    print(f"  ❌ Failed {file_path}: {e}")
                    failed_files += 1

    print(f"\n✅ Download complete!")
    print(f"  Downloaded: {total_files} files")
    print(f"  Skipped: {skipped_files} files")
    print(f"  Failed: {failed_files} files")
    print(f"  Location: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
