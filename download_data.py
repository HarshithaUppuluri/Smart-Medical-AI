"""Downloads all four Kaggle datasets used by the project into data/raw/.

Requires kaggle.json (Kaggle API token) to be present in the project root.
Safe to re-run: skips datasets that are already downloaded.
"""
import os
import zipfile

from config import ROOT, RAW_DIR, DATASETS

os.environ.setdefault("KAGGLE_CONFIG_DIR", str(ROOT))


def download(ref: str, dest_folder: str):
    dest = RAW_DIR / dest_folder
    if dest.exists() and any(dest.iterdir()):
        print(f"[skip] {ref} already downloaded -> {dest}")
        return

    dest.mkdir(parents=True, exist_ok=True)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    print(f"[download] {ref} -> {dest}")
    api.dataset_download_files(ref, path=str(dest), unzip=False)

    for zip_path in dest.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        zip_path.unlink()


def main():
    download(DATASETS["classification"], "classification")
    download(DATASETS["nlp"], "nlp")
    download(DATASETS["sentiment"], "sentiment")
    download(DATASETS["images"], "images")
    print("\nAll datasets downloaded to", RAW_DIR)


if __name__ == "__main__":
    main()
