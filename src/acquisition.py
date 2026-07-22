"""Download the Wisconsin Diagnostic Breast Cancer (WDBC) dataset from UCI.

Acquisition is code, not a manual download: anyone cloning this repo can
reproduce the exact input by running this module. It records provenance
(source URL, retrieval timestamp, SHA-256 checksum) so every number in the
analysis can be traced back to a specific file.

Why UCI and not ``sklearn.datasets.load_breast_cancer``
-------------------------------------------------------
scikit-learn ships a copy of this dataset, and importing it would be one line.
Two reasons not to:

1. The bundled copy has no provenance record and no ``id`` column. The IDs
   matter here — they are the only way to check whether the 569 rows are 569
   distinct patients (see ``profiling.py``), and a dataset that cannot be
   deduplicated cannot be trusted.
2. ``wdbc.names`` carries the measurement protocol and the authors' own
   reported accuracy. Reading it is how the modelling decisions downstream get
   made; a bare feature matrix hides the thing worth knowing.

Source
------
Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993).
Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5DW2B
Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

The data is NOT committed to this repository. CC BY 4.0 would permit
redistribution with attribution, but downloading at runtime keeps the repo
small and guarantees the provenance record matches the file actually used.

Usage
-----
    python src/acquisition.py
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

SOURCE_URL = (
    "https://archive.ics.uci.edu/static/public/17/"
    "breast+cancer+wisconsin+diagnostic.zip"
)
SOURCE_CITATION = (
    "Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). "
    "Breast Cancer Wisconsin (Diagnostic) [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B"
)
SOURCE_LICENCE = "CC BY 4.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DIR / "wdbc.zip"
PROVENANCE_PATH = RAW_DIR / "provenance.json"

# wdbc.data is headerless. The ten base measurements below are each recorded
# three times per image — mean, standard error, and "worst" (the mean of the
# three largest values) — giving 30 features in that fixed block order.
BASE_FEATURES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]
STATISTICS = ["mean", "se", "worst"]
FEATURE_COLUMNS = [f"{feat}_{stat}" for stat in STATISTICS for feat in BASE_FEATURES]
COLUMNS = ["id", "diagnosis"] + FEATURE_COLUMNS

CHUNK_BYTES = 1 << 16


def _sha256(path: Path) -> str:
    """Checksum a file without reading it entirely into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(force: bool = False) -> Path:
    """Fetch the source archive, skipping the network call if already present."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists() and not force:
        print(f"Archive already present, skipping download: {ZIP_PATH.name}")
        return ZIP_PATH

    print(f"Downloading from {SOURCE_URL} ...")
    response = requests.get(SOURCE_URL, stream=True, timeout=120)
    response.raise_for_status()

    with ZIP_PATH.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            handle.write(chunk)

    print(f"Saved {ZIP_PATH.stat().st_size / 1e3:.1f} KB to {ZIP_PATH}")
    return ZIP_PATH


def extract(zip_path: Path) -> dict[str, Path]:
    """Unpack the archive members we use: the data file and the codebook."""
    wanted = {"wdbc.data": "data", "wdbc.names": "names"}
    extracted: dict[str, Path] = {}

    with zipfile.ZipFile(zip_path) as archive:
        present = set(archive.namelist())
        missing = set(wanted) - present
        if missing:
            raise RuntimeError(
                f"Archive is missing expected members {sorted(missing)}; "
                f"found {sorted(present)}"
            )
        for member, key in wanted.items():
            target = RAW_DIR / member
            if not target.exists():
                archive.extract(member, RAW_DIR)
                print(f"Extracted {member}")
            else:
                print(f"Already extracted: {member}")
            extracted[key] = target

    return extracted


def load_raw(data_path: Path) -> pd.DataFrame:
    """Read the headerless CSV and attach the column names from wdbc.names.

    The header is asserted, not inferred, so a change in the upstream file
    layout fails loudly here rather than silently mislabelling 30 columns.
    """
    frame = pd.read_csv(data_path, header=None, names=COLUMNS)

    if frame.shape[1] != len(COLUMNS):
        raise RuntimeError(
            f"Expected {len(COLUMNS)} columns, got {frame.shape[1]}. "
            "The upstream file layout has changed."
        )
    unexpected = set(frame["diagnosis"].unique()) - {"M", "B"}
    if unexpected:
        raise RuntimeError(f"Unexpected diagnosis codes: {sorted(unexpected)}")

    return frame


def write_provenance(zip_path: Path, files: dict[str, Path], frame: pd.DataFrame) -> dict:
    """Record where the data came from and exactly which bytes we got."""
    record = {
        "source_url": SOURCE_URL,
        "citation": SOURCE_CITATION,
        "licence": SOURCE_LICENCE,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive_filename": zip_path.name,
        "archive_sha256": _sha256(zip_path),
        "archive_bytes": zip_path.stat().st_size,
        "members": {
            key: {
                "filename": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in files.items()
        },
        "rows": int(frame.shape[0]),
        "columns": int(frame.shape[1]),
        "class_counts": frame["diagnosis"].value_counts().to_dict(),
    }
    PROVENANCE_PATH.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Provenance written to {PROVENANCE_PATH.name}")
    return record


def main() -> pd.DataFrame:
    zip_path = download()
    files = extract(zip_path)
    frame = load_raw(files["data"])
    record = write_provenance(zip_path, files, frame)

    out_path = RAW_DIR / "wdbc_raw.parquet"
    frame.to_parquet(out_path, index=False)

    print(f"\n{record['rows']} rows x {record['columns']} columns -> {out_path.name}")
    print(f"Class balance: {record['class_counts']}")
    print(f"SHA-256 (wdbc.data): {record['members']['data']['sha256']}")
    return frame


if __name__ == "__main__":
    main()
