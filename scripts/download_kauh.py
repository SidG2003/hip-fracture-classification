"""Download KAUH-HipFx via individual public API files when ZIP creation fails."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import urlopen


REPOSITORY = "mohammadaminalqasem/kauh-hipfx"
API = "https://www.kaggle.com/api/v1/datasets"
VERSION = 1


def open_url(url: str):
    for attempt in range(4):
        try:
            return urlopen(url, timeout=90)
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code not in {429, 500, 502, 503, 504}:
                raise
            if attempt == 3:
                raise
            print(f"Retry {attempt + 1}/3 after {type(error).__name__}: {url}", flush=True)
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("Retry loop exited unexpectedly.")


def list_files() -> list[dict]:
    files = []
    token = None
    seen_tokens = set()
    while True:
        query = {"datasetVersionNumber": VERSION, "pageSize": 200}
        if token:
            query["pageToken"] = token
        with open_url(f"{API}/list/{REPOSITORY}?{urlencode(query)}") as response:
            page = json.load(response)
        files.extend(page["datasetFiles"])
        token = page.get("nextPageToken") or page.get("nextPageTokenNullable")
        if not token:
            return files
        if token in seen_tokens:
            raise RuntimeError("Kaggle returned a repeated pagination token.")
        seen_tokens.add(token)


def target_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or not relative.parts:
        raise ValueError(f"Unsafe dataset filename: {name}")
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Dataset path escapes destination: {name}")
    return target


def download_file(root: Path, entry: dict) -> dict:
    target = target_path(root, entry["name"])
    expected = entry["totalBytes"]
    if not target.is_file() or target.stat().st_size != expected:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        url = f"{API}/download/{REPOSITORY}/{quote(entry['name'], safe='')}?datasetVersionNumber={VERSION}"
        with open_url(url) as response, partial.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if partial.stat().st_size != expected:
            raise RuntimeError(f"Incomplete download: {entry['name']}; rerun to retry.")
        partial.replace(target)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"name": entry["name"], "bytes": expected, "sha256": digest}


def download(output: Path) -> None:
    files = list_files()
    if not files or len({entry["name"] for entry in files}) != len(files):
        raise ValueError("Kaggle file listing is empty or contains repeated filenames.")
    for entry in files:
        target_path(output, entry["name"])
    print(f"Downloading {len(files)} files ({sum(f['totalBytes'] for f in files):,} bytes) to {output}", flush=True)
    with open_url(f"{API}/view/{REPOSITORY}") as response:
        metadata = json.load(response)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "kaggle-metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for index, record in enumerate(pool.map(lambda entry: download_file(output, entry), files), start=1):
            records.append(record)
            if index % 50 == 0 or index == len(files):
                print(f"Verified file sizes: {index}/{len(files)}", flush=True)
    with (output / "download-manifest.json").open("w", encoding="utf-8") as stream:
        json.dump({"repository": REPOSITORY, "version": VERSION, "files": records}, stream, indent=2)
        stream.write("\n")
    print("Download complete. Local SHA-256 fingerprints recorded; Kaggle listing sizes verified.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/kauh-hipfx"))
    download(parser.parse_args().output)
