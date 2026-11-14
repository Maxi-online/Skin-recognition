"""Acquire the original FEI research dataset from the owner's website.

The dataset must not be redistributed. This script preserves downloaded
archives, records SHA-256 hashes, and never deletes existing files.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "fei"
SOURCE = "https://fei.edu.br/~cet/facedatabase.html"
BASE = "https://fei.edu.br/~cet/"


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def acquire(part: int) -> dict:
    name = f"originalimages_part{part}.zip"
    url = BASE + name
    archive = DATA / "archives" / name
    if not archive.exists():
        temporary = archive.with_suffix(".zip.download")
        with urllib.request.urlopen(url, timeout=120) as response:
            with temporary.open("wb") as target:
                shutil.copyfileobj(response, target, 1024 * 1024)
        with zipfile.ZipFile(temporary) as bundle:
            failure = bundle.testzip()
            if failure:
                raise ValueError(f"Archive CRC failure: {name}: {failure}")
        temporary.rename(archive)
    files = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = PurePosixPath(member.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
            if member.is_dir():
                continue
            destination = (DATA / "originals" / Path(*relative.parts)).resolve()
            if not destination.is_relative_to((DATA / "originals").resolve()):
                raise ValueError(f"Unsafe destination: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.stat().st_size != member.file_size:
                    raise ValueError(f"Existing file differs: {destination}")
            else:
                with bundle.open(member) as source, destination.open("xb") as target:
                    shutil.copyfileobj(source, target)
            files.append(str(destination.relative_to(DATA)))
    record = {"url": url, "archive": str(archive.relative_to(DATA)),
              "sha256": digest(archive), "bytes": archive.stat().st_size,
              "file_count": len(files), "files": files}
    print(json.dumps({k: v for k, v in record.items() if k != "files"}), flush=True)
    return record


def main() -> None:
    (DATA / "archives").mkdir(parents=True, exist_ok=True)
    (DATA / "originals").mkdir(parents=True, exist_ok=True)
    provenance = ROOT / "research" / "data"
    provenance.mkdir(parents=True, exist_ok=True)
    source_snapshot = provenance / "fei_source_2026-11-14.html"
    if not source_snapshot.exists():
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            source_snapshot.write_bytes(response.read())
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(acquire, range(1, 5)))
    manifest = {
        "dataset": "FEI Face Database, original images",
        "checked_on": "2026-11-14",
        "source_url": SOURCE,
        "source_snapshot_sha256": digest(source_snapshot),
        "usage": "Research only; no redistribution or reproduction of the database.",
        "session_metadata": "Not stated on the source page; do not infer sessions from image number.",
        "archives": records,
    }
    (provenance / "fei_acquisition.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {provenance / 'fei_acquisition.json'}", flush=True)


if __name__ == "__main__":
    main()
