"""Fetch an authorized CSV sales feed and retrain if its data changed."""
from pathlib import Path
import argparse
import json
import logging
import subprocess
import sys
from urllib.request import Request, urlopen
import pandas as pd

ROOT = Path(__file__).parent
CONFIG = ROOT / "config.json"
DATA = ROOT / "data" / "salesdaily.csv"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(LOGS / "daily_refresh.log", encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger("daily_refresh")
MEDICINES = {"M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"}

def get_url():
    if not CONFIG.exists():
        raise RuntimeError("No data source configured. Save a public CSV URL in the dashboard first.")
    url = json.loads(CONFIG.read_text(encoding="utf-8")).get("data_url", "").strip()
    if not url.startswith("https://"):
        raise RuntimeError("Data source must be an HTTPS CSV URL.")
    return url

def parse_csv(content):
    import io
    raw = pd.read_csv(io.BytesIO(content))
    date_col = next((c for c in raw.columns if c.lower() == "datum"), None)
    if date_col is None:
        raise ValueError("CSV needs a 'datum' date column.")
    found = [c for c in raw.columns if c.upper() in MEDICINES]
    if len(found) != len(MEDICINES) or {c.upper() for c in found} != MEDICINES:
        raise ValueError("CSV needs all 8 medicine columns: M01AB, M01AE, N02BA, N02BE, N05B, N05C, R03, R06.")
    raw[date_col] = pd.to_datetime(raw[date_col], dayfirst=False, errors="coerce")
    raw = raw.dropna(subset=[date_col]).rename(columns={date_col: "datum"})
    raw = raw.rename(columns={c: c.upper() for c in found})
    for col in MEDICINES:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if raw.empty or raw[list(MEDICINES)].isna().any().any() or (raw[list(MEDICINES)] < 0).any().any():
        raise ValueError("CSV contains empty, non-numeric, or negative daily sales.")
    return raw[["datum"] + sorted(MEDICINES)]

def refresh():
    url = get_url()
    log.info("Fetching the configured pharmacy CSV feed.")
    req = Request(url, headers={"User-Agent": "MedicineDemandPrototype/1.0"})
    with urlopen(req, timeout=60) as response:
        content = response.read(20_000_001)
    if len(content) > 20_000_000:
        raise ValueError("CSV exceeds the 20 MB safety limit.")
    incoming = parse_csv(content)
    old = pd.read_csv(DATA)
    old["datum"] = pd.to_datetime(old["datum"], dayfirst=False, errors="coerce")
    old = old.rename(columns={c: c.upper() for c in old.columns if c.upper() in MEDICINES})
    if not MEDICINES.issubset(old.columns):
        raise ValueError("Local sales history is missing one or more medicine groups.")
    old = old[["datum"] + sorted(MEDICINES)]
    merged = pd.concat([old, incoming], ignore_index=True, sort=False)
    merged = merged.dropna(subset=["datum"]).sort_values("datum").drop_duplicates("datum", keep="last")
    merged = merged.sort_values("datum")
    if len(merged) < 700:
        raise ValueError("Combined data needs at least 700 daily rows before training.")
    # Only update the source CSV if its content actually changed.
    if len(merged) == len(old) and merged.iloc[-1]["datum"] == old.iloc[-1]["datum"]:
        try:
            old2 = old.sort_values("datum").reset_index(drop=True)
            new2 = merged.reset_index(drop=True)
            pd.testing.assert_frame_equal(old2, new2, check_dtype=False)
            log.info("Feed contains no changes. Existing data and model left untouched.")
            return False
        except AssertionError:
            pass
    DATA.parent.mkdir(exist_ok=True)
    previous = DATA.read_bytes()
    temp = DATA.with_suffix(".csv.pending")
    merged.to_csv(temp, index=False, date_format="%m/%d/%Y")
    temp.replace(DATA)
    try:
        log.info("Merged %d days; latest date is %s. Starting model training.", len(merged), merged["datum"].max().date())
        subprocess.run([sys.executable, str(ROOT / "train_model.py")], cwd=ROOT, check=True)
    except Exception:
        DATA.write_bytes(previous)
        log.exception("Training failed; restored previous CSV and kept the existing model.")
        raise
    log.info("Refresh and model training completed.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one refresh now (used by the dashboard).")
    parser.parse_args()
    try:
        refresh()
    except Exception as exc:
        log.error("Refresh stopped: %s", exc)
        raise SystemExit(1)
