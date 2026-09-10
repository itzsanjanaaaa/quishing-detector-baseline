#!/bin/bash
# 00_download_data.sh
# ---------------------
# Downloads the raw data sources used by this project. Run this once before
# 01_build_dataset.py. Raw data is NOT committed to the repo (see .gitignore)
# to keep it lightweight and reproducible instead.

set -e
cd "$(dirname "$0")/.."   # move to project root

mkdir -p data

echo "==> Downloading JPCERT/CC phishing URL list..."
curl -sL -o data/phishurl-list.tar.gz "https://codeload.github.com/JPCERTCC/phishurl-list/tar.gz/refs/heads/main"
mkdir -p data/jpcert_raw
tar -xzf data/phishurl-list.tar.gz -C data/jpcert_raw --strip-components=1
rm data/phishurl-list.tar.gz

echo "==> Downloading Cisco Umbrella top-domains list (via cjbarker/top-domains)..."
curl -sL -o data/top-domains.tar.gz "https://codeload.github.com/cjbarker/top-domains/tar.gz/refs/heads/master"
mkdir -p data/top_domains_raw
tar -xzf data/top-domains.tar.gz -C data/top_domains_raw --strip-components=1
rm data/top-domains.tar.gz

echo "==> Downloading real crawled legitimate-URL paths (for realistic path sampling)..."
curl -sL -o /tmp/chaman.tar.gz "https://codeload.github.com/chamanthmvs/Phishing-Website-Detection/tar.gz/refs/heads/master"
mkdir -p /tmp/chaman_extract
tar -xzf /tmp/chaman.tar.gz -C /tmp/chaman_extract --strip-components=1
cp /tmp/chaman_extract/extracted_csv_files/legitimate-urls.csv data/chaman_legit_paths.csv
rm -rf /tmp/chaman.tar.gz /tmp/chaman_extract

echo "==> Done. Raw data is in data/jpcert_raw, data/top_domains_raw, data/chaman_legit_paths.csv"
echo "==> Next: python3 src/01_build_dataset.py"
