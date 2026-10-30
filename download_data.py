"""Try to download the public Kaggle archive; manual download is also supported."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
TARGET = DATA / "salesdaily.csv"
URL = "https://www.kaggle.com/api/v1/datasets/download/milanzdravkovic/pharma-sales-data?datasetVersionNumber=6"
MIRROR_URL = "https://raw.githubusercontent.com/mcallara/pharma-sales-data/main/salesdaily.csv"

if TARGET.exists():
    print(f"Dataset already exists: {TARGET}")
else:
    try:
        print("Downloading public Kaggle dataset...")
        with urlopen(URL, timeout=30) as response:
            archive = ZipFile(BytesIO(response.read()))
        name = next(n for n in archive.namelist() if Path(n).name.lower() == "salesdaily.csv")
        TARGET.write_bytes(archive.read(name))
        print(f"Saved {TARGET}")
    except Exception as kaggle_error:
        print(f"Kaggle download unavailable ({kaggle_error}); trying public GitHub copy...")
        try:
            with urlopen(MIRROR_URL, timeout=30) as response:
                payload = response.read()
            if not payload.startswith(b"datum,"):
                raise ValueError("Downloaded file did not look like salesdaily.csv")
            TARGET.write_bytes(payload)
            print(f"Saved {TARGET} ({len(payload):,} bytes)")
            print("Source mirror: github.com/mcallara/pharma-sales-data; original dataset by Milan Zdravkovic.")
        except Exception as mirror_error:
            print(f"Automatic download failed: {mirror_error}")
            print("Download salesdaily.csv from https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data")
            print(f"Then place it here: {TARGET}")
