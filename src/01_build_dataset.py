"""
01_build_dataset.py
--------------------
Builds a labeled URL dataset for the quishing-detector-baseline project.

Sources:
  - Phishing URLs : JPCERT/CC official phishing URL dataset
                    https://github.com/JPCERTCC/phishurl-list
                    (years 2022-2026 used for recency/relevance)
  - Legitimate URLs: Cisco Umbrella Top Domains (via cjbarker/top-domains mirror)
                    https://github.com/cjbarker/top-domains

Output:
  data/processed/urls_labeled.csv  with columns: url, label (1 = phishing, 0 = legitimate)
"""

import pandas as pd
import glob
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
JPCERT_DIR = os.path.join(RAW_DIR, "jpcert_raw")
TOPDOMAINS_FILE = os.path.join(RAW_DIR, "top_domains_raw", "top-recs", "top-sites-100000.csv")
OUT_DIR = os.path.join(RAW_DIR, "processed")
OUT_FILE = os.path.join(OUT_DIR, "urls_labeled.csv")

# How many of each class to keep (balanced dataset, manageable size for baseline)
N_PER_CLASS = 20000

# Which years of JPCERT data to use (recent years = more relevant to current quishing patterns)
YEARS_TO_USE = ["2022", "2023", "2024", "2025", "2026"]


def load_phishing_urls():
    frames = []
    for year in YEARS_TO_USE:
        pattern = os.path.join(JPCERT_DIR, year, "*.csv")
        for f in glob.glob(pattern):
            df = pd.read_csv(f, usecols=["URL", "description"])
            frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.rename(columns={"URL": "url", "description": "target_brand"})
    combined["label"] = 1
    combined = combined.drop_duplicates(subset="url")
    return combined[["url", "target_brand", "label"]]


def load_legitimate_urls():
    df = pd.read_csv(TOPDOMAINS_FILE, header=None, names=["rank", "domain"])
    df["url"] = "https://" + df["domain"].astype(str)
    df["target_brand"] = None
    df["label"] = 0
    df = df.drop_duplicates(subset="url")
    return df[["url", "target_brand", "label"]]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    phishing = load_phishing_urls()
    legit = load_legitimate_urls()

    print(f"Loaded {len(phishing)} unique phishing URLs (2022-2026)")
    print(f"Loaded {len(legit)} unique legitimate URLs")

    # Balance classes by sampling
    phishing_sample = phishing.sample(n=min(N_PER_CLASS, len(phishing)), random_state=42)
    legit_sample = legit.sample(n=min(N_PER_CLASS, len(legit)), random_state=42)

    dataset = pd.concat([phishing_sample, legit_sample], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    dataset.to_csv(OUT_FILE, index=False)

    print(f"\nFinal dataset: {len(dataset)} rows")
    print(dataset["label"].value_counts())
    print(f"Saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
