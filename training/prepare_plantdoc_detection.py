"""Download PlantDoc object-detection data and convert it to YOLOv8 format."""

from __future__ import annotations

import csv
import shutil
import time
import urllib.error
import urllib.request
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote


REPO_RAW = "https://raw.githubusercontent.com/pratikkayal/PlantDoc-Object-Detection-Dataset/master"
ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
RAW_DIR = DATASETS / "plantdoc_raw"
YOLO_DIR = DATASETS / "plantdoc_yolo"
TRAIN_CSV = DATASETS / "plantdoc_train_labels.csv"
TEST_CSV = DATASETS / "plantdoc_test_labels.csv"
INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


def download_file(url: str, destination: Path, retries: int = 1) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return

    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Codex"})
            with urllib.request.urlopen(request, timeout=20) as response:
                destination.write_bytes(response.read())
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(1.5 * attempt)


def safe_filename(filename: str) -> str:
    """Return a Windows-safe image filename while keeping it readable."""
    cleaned = INVALID_FILENAME_CHARS.sub("_", filename).strip().strip(".")
    return cleaned or "image"


def ensure_label_csvs() -> None:
    files = {
        TRAIN_CSV: f"{REPO_RAW}/train_labels.csv",
        TEST_CSV: f"{REPO_RAW}/test_labels.csv",
    }
    for path, url in files.items():
        download_file(url, path)


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def download_images(rows: list[dict[str, str]], split: str, workers: int = 4) -> None:
    filenames = sorted({row["filename"] for row in rows})
    print(f"Downloading {len(filenames)} {split} images...")

    def fetch(filename: str) -> str:
        url = f"{REPO_RAW}/{split.upper()}/{quote(filename)}"
        download_file(url, RAW_DIR / split.upper() / safe_filename(filename))
        return filename

    completed = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, filename) for filename in filenames]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failed.append(str(exc))
            completed += 1
            if completed % 100 == 0 or completed == len(filenames):
                print(f"  {split}: {completed}/{len(filenames)}")

    if failed:
        print(f"Retrying {len(failed)} failed {split} downloads one at a time...")
        remaining = [
            filename
            for filename in filenames
            if not (RAW_DIR / split.upper() / safe_filename(filename)).exists()
        ]
        still_missing: list[str] = []
        for index, filename in enumerate(remaining, start=1):
            try:
                fetch(filename)
            except Exception:
                still_missing.append(filename)
            if index % 25 == 0 or index == len(remaining):
                print(f"  {split} retries: {index}/{len(remaining)}")
        if still_missing:
            missing_path = RAW_DIR / f"missing_{split}.txt"
            missing_path.write_text("\n".join(still_missing) + "\n", encoding="utf-8")
            print(f"  skipped {len(still_missing)} missing {split} images; list saved to {missing_path}")


def build_class_map(rows: list[dict[str, str]]) -> dict[str, int]:
    names = sorted({row["class"].strip() for row in rows})
    return {name: index for index, name in enumerate(names)}


def yolo_line(row: dict[str, str], class_id: int) -> str | None:
    width = float(row["width"])
    height = float(row["height"])
    xmin = float(row["xmin"])
    ymin = float(row["ymin"])
    xmax = float(row["xmax"])
    ymax = float(row["ymax"])

    if width <= 0 or height <= 0 or xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height

    values = [class_id, x_center, y_center, box_width, box_height]
    return f"{values[0]} " + " ".join(f"{value:.6f}" for value in values[1:])


def reset_yolo_output() -> None:
    for child in [YOLO_DIR / "images", YOLO_DIR / "labels"]:
        if child.exists():
            shutil.rmtree(child)


def convert_split(rows: list[dict[str, str]], split: str, class_map: dict[str, int], raw_split: str | None = None) -> None:
    raw_split = raw_split or split
    image_dir = YOLO_DIR / "images" / split
    label_dir = YOLO_DIR / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["filename"], []).append(row)

    for filename, image_rows in grouped.items():
        safe_name = safe_filename(filename)
        source = RAW_DIR / raw_split.upper() / safe_name
        if not source.exists():
            continue

        destination = image_dir / safe_name
        if not destination.exists():
            shutil.copy2(source, destination)

        label_path = label_dir / f"{Path(safe_name).stem}.txt"
        lines = []
        for row in image_rows:
            line = yolo_line(row, class_map[row["class"].strip()])
            if line is not None:
                lines.append(line)
        if not lines:
            continue
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_available_train_rows(train_rows: list[dict[str, str]], val_fraction: float = 0.15) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    available = {
        path.name
        for path in (RAW_DIR / "TRAIN").glob("*")
        if path.is_file() and path.stat().st_size > 0
    }
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in train_rows:
        if safe_filename(row["filename"]) in available:
            grouped.setdefault(row["filename"], []).append(row)

    filenames = sorted(grouped)
    val_count = max(1, int(len(filenames) * val_fraction))
    val_names = set(filenames[:: max(1, len(filenames) // val_count)][:val_count])

    new_train: list[dict[str, str]] = []
    new_val: list[dict[str, str]] = []
    for filename, rows in grouped.items():
        if filename in val_names:
            new_val.extend(rows)
        else:
            new_train.extend(rows)
    return new_train, new_val


def write_data_yaml(class_map: dict[str, int]) -> None:
    names = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    yaml_lines = [
        f"path: {YOLO_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/val",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (YOLO_DIR / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_label_csvs()
    train_rows = read_rows(TRAIN_CSV)
    test_rows = read_rows(TEST_CSV)
    all_rows = train_rows + test_rows

    download_images(train_rows, "train")
    try:
        download_images(test_rows, "test")
    except Exception as exc:
        print(f"Could not download PlantDoc TEST images: {exc}")

    class_map = build_class_map(all_rows)
    reset_yolo_output()
    if any((RAW_DIR / "TEST").glob("*")):
        convert_split(train_rows, "train", class_map)
        convert_split(test_rows, "val", class_map, raw_split="test")
    else:
        print("Using a validation split from downloaded TRAIN images.")
        train_split, val_split = split_available_train_rows(train_rows)
        convert_split(train_split, "train", class_map, raw_split="train")
        convert_split(val_split, "val", class_map, raw_split="train")
    write_data_yaml(class_map)

    print(f"Prepared YOLO dataset at {YOLO_DIR}")
    print(f"Classes: {len(class_map)}")
    print("Train with: python training/train_detection.py")


if __name__ == "__main__":
    main()
