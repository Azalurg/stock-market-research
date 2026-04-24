import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

CATEGORY_MAP = {
    "Information Technology": "Information Technology",
    "Technology": "Information Technology",
    "Informatyka": "Information Technology",
    "Nowe Technologie": "Information Technology",

    "Financials": "Financials",
    "Banki": "Financials",
    "Rynek Kapitałowy": "Financials",

    "Industrials": "Industrials",
    "Przemysł Elektromaszynowy": "Industrials",
    "Budownictwo": "Industrials",
    "Handel Hurtowy": "Industrials",
    "Recykling": "Industrials",

    "Consumer Discretionary": "Consumer Discretionary",
    "Motoryzacja": "Consumer Discretionary",
    "Rekreacja I Wypoczynek": "Consumer Discretionary",
    "Odzież I Kosmetyki": "Consumer Discretionary",
    "Gry": "Consumer Discretionary",
    "Sieci Handlowe": "Consumer Discretionary",

    "Health Care": "Health Care",
    "Biotechnologia": "Health Care",
    "Sprzęt I Materiały Medyczne": "Health Care",
    "Dystrubucja Leków": "Health Care",
    "Ochrona zdrowia - pozostałe": "Health Care",

    "Consumer Staples": "Consumer Staples",

    "Communication Services": "Communication Services",
    "Communication": "Communication Services",
    "Telecommunications": "Communication Services",
    "Telekomunikacja": "Communication Services",
    "Media": "Communication Services",

    "Energy": "Energy",
    "Energia": "Energy",

    "Materials": "Materials",
    "Basic Materials": "Materials",
    "Chemia": "Materials",
    "Górnictwo": "Materials",

    "Utilities": "Utilities",

    "Real Estate": "Real Estate",
    "Nieruchomości": "Real Estate",

    "Commodities": "Commodities",
    "Cash and/or Derivatives": "Cash/Derivatives",
    "Futures": "Cash/Derivatives",
    "UNKNOWN": "Unknown",
    "": "Empty",
}

CATEGORY_COLUMN = "Sector"
FALLBACK_VALUE = "Other"


def normalize_category(value: str) -> str:
    value = (value or "").strip()
    return CATEGORY_MAP.get(value, FALLBACK_VALUE)


def process_folder(folder: str) -> None:
    folder_path = Path(folder)
    csv_files = list(folder_path.glob("*.csv"))

    if not csv_files:
        print(f"[WARN] No CSV files found in '{folder_path.resolve()}'")
        return

    unmapped: defaultdict[str, set] = defaultdict(set)

    for csv_path in csv_files:
        df = pd.read_csv(csv_path, dtype=str)

        if CATEGORY_COLUMN not in df.columns:
            print(f"[SKIP] '{csv_path.name}' — column '{CATEGORY_COLUMN}' not found")
            continue

        original = df[CATEGORY_COLUMN].fillna("")
        df[CATEGORY_COLUMN] = original.apply(normalize_category)

        # collect values that fell through to FALLBACK_VALUE
        for raw_val in original[df[CATEGORY_COLUMN] == FALLBACK_VALUE].unique():
            if raw_val not in CATEGORY_MAP:
                unmapped[csv_path.name].add(raw_val)

        df.to_csv(csv_path, index=False)
        print(f"[OK]   '{csv_path.name}' — processed")

    if unmapped:
        print("\n[WARN] The following values were not found in CATEGORY_MAP "
              f"and were mapped to '{FALLBACK_VALUE}':")
        for fname, values in unmapped.items():
            for v in sorted(values):
                print(f"       {fname}: '{v}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Normalize the 'Category' column in all CSV files inside a folder."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing CSV files"
    )
    args = parser.parse_args()
    
    print(f"Sectors: {set(CATEGORY_MAP.values())}")
    
    # process_folder(args.folder)