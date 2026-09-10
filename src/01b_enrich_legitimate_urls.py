"""
01b_enrich_legitimate_urls.py
-------------------------------
FIX for a dataset artifact discovered during feature sanity-checking:

  The legitimate URLs (from Cisco Umbrella top-domains list) were bare
  domains (e.g. "https://google.com"), while phishing URLs (from JPCERT/CC)
  are naturally full page URLs with paths (e.g. "https://x.cn/login/verify").

  This meant path_length, path_depth, has_query, count_question, count_equal
  were IDENTICALLY 0.0 for every legitimate example -> the model would learn
  "has a path => phishing", which is a trivial dataset artifact, not a real
  signal (amazon.com/gp/product/... is a completely normal legitimate URL).

FIX APPLIED:
  1. Sample realistic path structures from a small real crawled legitimate
     URL corpus (ISCX-2016-derived, via chamanthmvs/Phishing-Website-Detection)
     and apply them to our larger domain pool.
  2. Add a modest proportion of realistic query strings (search/tracking
     params), since the source crawl predates modern query-string-heavy
     browsing patterns and would otherwise still make has_query=0 an artifact.

LIMITATION (documented, not hidden):
  Paths/queries are *resampled* onto domains rather than being genuinely
  crawled per-domain. This is a reasonable baseline compromise given time
  and data-access constraints, but a stronger follow-up would crawl real
  page URLs per legitimate domain directly. Flagged as future work in README.
"""

import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(__file__)
DOMAINS_FILE = os.path.join(BASE, "..", "data", "top_domains_raw", "top-recs", "top-sites-100000.csv")
REAL_PATHS_FILE = os.path.join(BASE, "..", "data", "chaman_legit_paths.csv")  # copied in separately
OUT_FILE = os.path.join(BASE, "..", "data", "processed", "urls_labeled.csv")

N_LEGIT = 20000
N_PHISH = 20000

# Realistic synthetic query patterns representative of common legitimate
# traffic (search, tracking, pagination, product/session IDs). Documented
# as synthesized since the real crawled corpus pre-dates common query-string
# usage.
QUERY_TEMPLATES = [
    "?q={term}", "?search={term}", "?id={num}", "?page={num}",
    "?ref=homepage", "?utm_source=newsletter", "?utm_campaign=summer",
    "?sort=popular", "?category={term}", "?sessionid={num}",
]
TERMS = ["shoes", "laptop", "flights", "news", "recipe", "weather", "jobs", "python"]


def load_real_paths():
    df = pd.read_csv(REAL_PATHS_FILE)
    paths = df["Path"].dropna().tolist()
    paths = [p for p in paths if isinstance(p, str) and p.strip() != ""]
    return paths


def build_legitimate_urls(rng):
    domains_df = pd.read_csv(DOMAINS_FILE, header=None, names=["rank", "domain"])
    domains_df = domains_df.drop_duplicates(subset="domain")
    sampled_domains = domains_df.sample(n=min(N_LEGIT, len(domains_df)), random_state=42)["domain"].tolist()

    real_paths = load_real_paths()

    urls = []
    for domain in sampled_domains:
        # 25% stay as bare root, 75% get a realistic sampled path (matches
        # a plausible real-world mix of homepage vs deep-page traffic)
        if rng.random() < 0.25 or not real_paths:
            path = ""
        else:
            path = rng.choice(real_paths)

        url = f"https://{domain}{path}"

        # ~15% of URLs get a query string appended (search/tracking/etc.)
        if rng.random() < 0.15:
            template = rng.choice(QUERY_TEMPLATES)
            query = template.format(term=rng.choice(TERMS), num=rng.integers(1, 99999))
            url += query

        urls.append(url)

    return pd.DataFrame({"url": urls, "target_brand": None, "label": 0})


def load_phishing_urls_from_existing():
    # Reuse the already-built dataset's phishing rows (already deduplicated
    # and sampled in 01_build_dataset.py)
    existing = pd.read_csv(OUT_FILE)
    return existing[existing.label == 1][["url", "target_brand", "label"]]


def main():
    rng = np.random.default_rng(42)

    phishing = load_phishing_urls_from_existing()
    legit = build_legitimate_urls(rng)

    print(f"Phishing URLs (unchanged): {len(phishing)}")
    print(f"Legitimate URLs (re-built with realistic paths): {len(legit)}")
    print(f"  - bare root URLs: {(legit['url'].str.count('/') == 2).sum()}")
    print(f"  - URLs with a path: {(legit['url'].str.count('/') > 2).sum()}")

    dataset = pd.concat([phishing, legit], ignore_index=True)
    dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

    dataset.to_csv(OUT_FILE, index=False)
    print(f"\nSaved corrected dataset to {OUT_FILE}")
    print(dataset["label"].value_counts())


if __name__ == "__main__":
    main()
