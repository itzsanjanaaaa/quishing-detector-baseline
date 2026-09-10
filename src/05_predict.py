"""
05_predict.py
--------------
End-to-end inference: takes a QR code image, decodes the embedded URL,
extracts features, and returns a phishing/legitimate verdict with a
confidence score using the trained baseline model.

This is the piece that ties the whole Phase 1 pipeline together:
  QR image -> URL (02_qr_decoder) -> features (03_feature_extraction)
  -> verdict (trained model from 04_train_model)

Usage (CLI):
    python src/05_predict.py --image path/to/qr_code.png
    python src/05_predict.py --url "https://example.com/some/path"   # skip QR decoding, test a raw URL directly
"""

import argparse
import sys
import os
import joblib
import pandas as pd
import importlib

sys.path.insert(0, os.path.dirname(__file__))
qr_decoder = importlib.import_module("02_qr_decoder")
feature_extraction = importlib.import_module("03_feature_extraction")

BASE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE, "..", "models", "best_model.joblib")

_model = None


def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run src/04_train_model.py first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_url(url: str) -> dict:
    """Run the feature extractor + trained model on a single URL."""
    model = load_model()

    features = feature_extraction.extract_features(url)
    feature_order = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(features.keys())
    X = pd.DataFrame([features])[feature_order]

    pred = model.predict(X)[0]
    prob_phishing = model.predict_proba(X)[0][1]

    return {
        "url": url,
        "verdict": "PHISHING (quishing risk)" if pred == 1 else "LEGITIMATE",
        "phishing_probability": round(float(prob_phishing), 4),
        "features": features,
    }


def predict_from_qr_image(image_path: str) -> dict:
    """Decode a QR image, then classify the extracted URL."""
    decoded = qr_decoder.decode_qr_from_file(image_path)

    if not decoded:
        return {"error": f"No QR code detected in {image_path}"}

    # If multiple QR codes are present, classify all of them
    results = [predict_url(url) for url in decoded]
    return {"image": image_path, "decoded_count": len(decoded), "results": results}


def _print_result(result: dict):
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return

    if "results" in result:  # from a QR image
        print(f"Image: {result['image']}  ({result['decoded_count']} QR code(s) found)\n")
        for r in result["results"]:
            _print_single(r)
    else:  # direct URL check
        _print_single(result)


def _print_single(r: dict):
    print(f"URL:        {r['url']}")
    print(f"Verdict:    {r['verdict']}")
    print(f"Confidence: {r['phishing_probability']*100:.1f}% phishing probability")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Quishing detector - QR image or URL classifier")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a QR code image file")
    group.add_argument("--url", help="Classify a raw URL directly (skip QR decoding)")
    args = parser.parse_args()

    if args.image:
        result = predict_from_qr_image(args.image)
    else:
        result = predict_url(args.url)

    _print_result(result)


if __name__ == "__main__":
    main()
