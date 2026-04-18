import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm

ADE20K_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
ARCHIVE_NAME = 'ADEChallengeData2016.zip'


class DownloadProgressBar(tqdm):
    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: Optional[int] = None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _has_ade20k_dataset(data_root: Path) -> bool:
    required_paths = [
        data_root / 'images' / 'training',
        data_root / 'images' / 'validation',
        data_root / 'annotations' / 'training',
        data_root / 'annotations' / 'validation',
    ]
    return all(path.exists() for path in required_paths)


def _download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(
        unit='B',
        unit_scale=True,
        miniters=1,
        desc=f'Downloading {target_path.name}',
    ) as progress:
        def report_hook(blocks, block_size, total_size=None):
            progress.update_to(blocks, block_size, total_size)

        urllib.request.urlretrieve(url, filename=str(target_path), reporthook=report_hook)


def _extract_zip(zip_path: Path, extract_to: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"Archive not found: {zip_path}")

    with zipfile.ZipFile(str(zip_path), 'r') as archive:
        archive.extractall(path=str(extract_to))


def ensure_ade20k_dataset(data_root: str, download: bool = False) -> None:
    data_root_path = Path(data_root)
    if _has_ade20k_dataset(data_root_path):
        return

    if not download:
        raise FileNotFoundError(
            f"ADE20K dataset not found at {data_root}.\n"
            "Run with --download-data to download and extract the dataset automatically, "
            "or manually download from http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip "
            "and extract to the expected path."
        )

    archive_dir = data_root_path.parent
    archive_path = archive_dir / ARCHIVE_NAME

    print(f"Dataset missing at {data_root_path}. Downloading ADE20K to {archive_path}...")
    _download_file(ADE20K_URL, archive_path)

    print(f"Extracting {archive_path} to {archive_dir}...")
    _extract_zip(archive_path, archive_dir)

    extracted_path = archive_dir / 'ADEChallengeData2016'
    if extracted_path.exists() and extracted_path != data_root_path:
        if data_root_path.exists():
            shutil.rmtree(str(data_root_path))
        shutil.move(str(extracted_path), str(data_root_path))

    if not _has_ade20k_dataset(data_root_path):
        raise RuntimeError(
            f"Failed to prepare ADE20K dataset at {data_root_path}. "
            "Please verify the extracted archive contains the expected ADE20K folder structure."
        )

    print(f"ADE20K dataset is ready at {data_root_path}")
