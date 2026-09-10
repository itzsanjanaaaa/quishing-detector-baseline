"""
03_feature_extraction.py
--------------------------
Extracts lexical and structural features from raw URLs for the quishing
detection baseline. These are classic, well-established phishing-URL
features (see e.g. Hannousse & Yahiouche 2021; Sahoo, Liu & Hoi 2017 survey)
chosen for interpretability over raw-string deep learning, since the goal
of this baseline is an explainable centralized model to contrast against
the privacy-preserving federated model in the dissertation.

Usage:
    python 03_feature_extraction.py
Reads:  data/processed/urls_labeled.csv
Writes: data/processed/features.csv
"""

import pandas as pd
import numpy as np
import re
import math
from urllib.parse import urlparse
import os

IN_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "urls_labeled.csv")
OUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")

# Common brand names targeted by phishing/quishing (extend as needed)
BRAND_KEYWORDS = [
    "paypal", "amazon", "apple", "google", "microsoft", "netflix", "facebook",
    "instagram", "bank", "visa", "mastercard", "jcb", "amex", "wallet",
    "irs", "gov", "login", "signin", "account", "secure", "verify", "update",
    "confirm", "billing", "invoice",
]

# TLDs frequently abused in phishing campaigns (cheap/anonymous registration)
SUSPICIOUS_TLDS = {
    "top", "xyz", "cn", "tk", "ml", "ga", "cf", "gq", "click", "link",
    "loan", "work", "men", "win", "review", "cyou", "icu", "vip",
}

IP_PATTERN = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")


def shannon_entropy(s: str) -> float:
    """Higher entropy = more random-looking string (typical of generated phishing domains)."""
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def extract_features(url: str) -> dict:
    # Ensure a scheme so urlparse behaves consistently
    parse_url = url if "://" in url else "http://" + url
    parsed = urlparse(parse_url)

    hostname = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""

    tld = hostname.split(".")[-1].lower() if "." in hostname else ""
    subdomain_count = max(hostname.count(".") - 1, 0) if hostname else 0

    full_url_lower = url.lower()

    features = {
        # --- length-based ---
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),

        # --- character composition ---
        "count_dots": url.count("."),
        "count_hyphens": url.count("-"),
        "count_at": url.count("@"),
        "count_question": url.count("?"),
        "count_equal": url.count("="),
        "count_percent": url.count("%"),
        "count_digits": sum(c.isdigit() for c in url),
        "digit_ratio": sum(c.isdigit() for c in url) / len(url) if url else 0,

        # --- entropy ---
        "hostname_entropy": shannon_entropy(hostname),

        # --- structural ---
        "subdomain_count": subdomain_count,
        "path_depth": path.count("/"),
        "has_ip_address": int(bool(IP_PATTERN.match(hostname))),
        "has_port": int(parsed.port is not None),
        "has_query": int(len(query) > 0),

        # --- security signals ---
        "is_https": int(parsed.scheme == "https"),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "suspicious_keyword_count": sum(kw in full_url_lower for kw in BRAND_KEYWORDS),

        # --- brand mimicry: brand keyword present but NOT as the actual registered domain ---
        "brand_in_path_not_domain": int(
            any(kw in (path + query).lower() for kw in BRAND_KEYWORDS)
            and not any(kw in hostname.lower() for kw in BRAND_KEYWORDS)
        ),
    }
    return features


def main():
    df = pd.read_csv(IN_FILE)
    print(f"Loaded {len(df)} URLs")

    feature_dicts = df["url"].apply(extract_features)
    features_df = pd.DataFrame(list(feature_dicts))

    result = pd.concat([df[["url", "label"]], features_df], axis=1)
    result.to_csv(OUT_FILE, index=False)

    print(f"Extracted {features_df.shape[1]} features")
    print(f"Saved to {OUT_FILE}")
    print("\nFeature summary (mean by class):")
    print(result.groupby("label")[features_df.columns].mean().T)


if __name__ == "__main__":
    main()
