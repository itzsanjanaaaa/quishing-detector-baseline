"""
02_qr_decoder.py
-----------------
Core "quishing" component: decodes QR code images to extract embedded URLs,
which are then passed to the URL classifier (03_feature_extraction.py /
04_train_model.py).

This module is what distinguishes this project from a generic phishing-URL
detector: the threat model is "malicious QR code" rather than "malicious link
in an email", so the pipeline must start from an image, not a raw string.

BACKEND CHOICE:
  Uses OpenCV's built-in QRCodeDetector as the primary decoder. OpenCV is
  already a project dependency and has NO external native-library dependency
  beyond the pip package itself.

  pyzbar (a wrapper around the native ZBar library) is used as a secondary
  fallback ONLY if it happens to be importable and finds something OpenCV
  missed - ZBar is sometimes more robust on damaged/rotated real-world photos.
  It is never required: if pyzbar or its underlying libzbar DLL is missing
  or fails to load (a common issue on Windows - see README "Troubleshooting"),
  this module still works correctly using OpenCV alone.

Usage:
    from qr_decoder import decode_qr_from_file, generate_qr_for_testing

    url = decode_qr_from_file("path/to/qr.png")
"""

from PIL import Image
import numpy as np
import cv2
import io

_pyzbar_available = None  # lazily determined on first use


def decode_qr_from_file(image_path: str):
    """
    Decode all QR codes found in an image file.
    Returns a list of decoded string payloads (usually URLs).
    Returns an empty list if no QR code is found.
    """
    img = Image.open(image_path)
    return _decode_image(img)


def decode_qr_from_bytes(image_bytes: bytes):
    """Decode QR codes from raw image bytes (e.g. from an upload/API)."""
    img = Image.open(io.BytesIO(image_bytes))
    return _decode_image(img)


def _decode_with_opencv(img: Image.Image):
    """Primary backend - pure Python/OpenCV, no external native library needed."""
    rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    detector = cv2.QRCodeDetector()
    try:
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(bgr)
        if retval:
            return [s for s in decoded_info if s]
    except cv2.error:
        pass

    # Fallback for older OpenCV builds without detectAndDecodeMulti
    data, points, _ = detector.detectAndDecode(bgr)
    return [data] if data else []


def _decode_with_pyzbar(img: Image.Image):
    """Optional secondary backend - only used if pyzbar/ZBar is available."""
    global _pyzbar_available
    if _pyzbar_available is False:
        return []
    try:
        from pyzbar.pyzbar import decode as zbar_decode
        results = zbar_decode(img)
        _pyzbar_available = True
        return [r.data.decode("utf-8", errors="replace") for r in results]
    except Exception:
        # Covers ImportError (not installed) AND the Windows DLL-load
        # FileNotFoundError ("Could not find module 'libzbar-64.dll'").
        # Either way, fall back to OpenCV silently rather than crashing.
        _pyzbar_available = False
        return []


def _decode_image(img: Image.Image):
    payloads = _decode_with_opencv(img)
    if payloads:
        return payloads
    # Only try pyzbar if OpenCV found nothing (e.g. a harder real-world photo)
    return _decode_with_pyzbar(img)


def generate_qr_for_testing(url: str, out_path: str):
    """
    Utility to generate a QR code PNG for a given URL, for testing the
    decode pipeline end-to-end. Requires the `qrcode` package.
    """
    import qrcode
    img = qrcode.make(url)
    img.save(out_path)
    return out_path


if __name__ == "__main__":
    # Quick self-test: generate a QR code, then decode it back
    test_url = "https://example.com/test-quishing-pipeline"
    test_path = "/tmp/test_qr.png"

    generate_qr_for_testing(test_url, test_path)
    decoded = decode_qr_from_file(test_path)

    print(f"Encoded URL : {test_url}")
    print(f"Decoded URL : {decoded}")
    assert decoded == [test_url], "Round-trip decode failed!"
    print("Self-test passed: QR encode/decode round-trip successful (OpenCV backend).")
