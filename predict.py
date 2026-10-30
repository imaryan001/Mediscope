"""Make a next-7/30-day forecast and optional stock warning."""
from pathlib import Path
import argparse
from forecasting import load_bundle, load_history, forecast_one

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medicine", help="Medicine group code, e.g. N02BE")
    parser.add_argument("--stock", type=float, help="Current stock units, if known")
    parser.add_argument("--lead-days", type=int, default=7, choices=(7, 30), help="Days until replenishment can arrive")
    parser.add_argument("--list", action="store_true", help="List medicine groups")
    args = parser.parse_args()
    bundle = load_bundle()
    if args.list:
        print("Medicine groups:", ", ".join(bundle["columns"])); return
    med = args.medicine
    if not med:
        parser.error("provide --medicine CODE or use --list")
    if med not in bundle["columns"]:
        raise SystemExit(f"Unknown group {med}. Use --list to see valid codes.")
    history = load_history()
    print(f"Forecast for {med}:")
    for horizon in (7, 30):
        result = forecast_one(bundle, history, med, horizon)
        if horizon == 7:
            print(f"  Starting {result['date'].date()}")
        print(f"  Next {horizon} days: about {result['quantity']} units (historical error band: {result['low']}-{result['high']}) [{result['model']}]")
        if args.stock is not None and horizon == args.lead_days:
            print(f"  Stock check: {'RESTOCK RISK' if args.stock < result['quantity'] else 'stock covers forecast'} (stock={args.stock:g}, predicted demand={result['quantity']})")

if __name__ == "__main__": main()
