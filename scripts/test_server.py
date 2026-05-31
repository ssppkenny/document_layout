#!/usr/bin/env python3
"""
test_server.py — smoke test for the /page endpoint.

Usage (server must be running):
    pixi run python test_server.py [IMAGE_PATH] [SERVER_URL]

Defaults:
    IMAGE_PATH  = images/dvurog_p019.png  (or first .png found in images/)
    SERVER_URL  = http://localhost:8000
"""

import json
import sys
import urllib.request
import urllib.parse
import os
from pathlib import Path


def find_test_image() -> Path:
    candidates = [
        "images/dvurog_p019.png",
        "images/hlw_p040.png",
        "images/gang_p023.png",
    ]
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    # Fall back to any PNG in images/
    images_dir = Path("images")
    if images_dir.exists():
        pngs = list(images_dir.glob("*.png"))
        if pngs:
            return pngs[0]
    raise FileNotFoundError("No test image found. Pass an image path as argument.")


def post_multipart(url: str, image_path: Path) -> dict:
    """POST image as multipart/form-data using only stdlib."""
    boundary = "----TestBoundary7a3f9b2c"
    image_bytes = image_path.read_bytes()
    filename = image_path.name

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n"
        f"\r\n"
    ).encode() + image_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_test_image()
    server_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"

    print(f"Image : {image_path}")
    print(f"Server: {server_url}")

    # --- Health check ---
    print("\n[1] Health check…")
    with urllib.request.urlopen(f"{server_url}/health", timeout=5) as r:
        health = json.loads(r.read())
    print(f"    {health}")
    assert health.get("status") == "ok", "Health check failed"

    # --- POST /page ---
    params = urllib.parse.urlencode({
        "page_width": 2000,
        "zoom_factor": 2.5,
        "toc_algorithm": "original",  # faster than layoutlm for smoke test
    })
    endpoint = f"{server_url}/page?{params}"
    print(f"\n[2] POST {endpoint}")
    print("    (this may take 30-60 s on first call while models load…)")

    response = post_multipart(endpoint, image_path)

    # --- Validate response structure ---
    print("\n[3] Validating response…")
    required_top = ["image_width", "image_height", "background_color",
                    "skew_angle", "is_toc", "zoom_factor", "page_width",
                    "left_margin", "right_margin", "blocks"]
    for key in required_top:
        assert key in response, f"Missing top-level key: {key}"

    print(f"    image size  : {response['image_width']} × {response['image_height']}")
    print(f"    background  : {response['background_color']}")
    print(f"    skew_angle  : {response['skew_angle']:.3f}°")
    print(f"    is_toc      : {response['is_toc']}")
    print(f"    left_margin : {response['left_margin']}")
    print(f"    right_margin: {response['right_margin']}")
    print(f"    blocks      : {len(response['blocks'])}")

    total_words = 0
    for block in response["blocks"]:
        assert "index" in block
        assert "block_type" in block
        assert "bbox" in block and len(block["bbox"]) == 4
        assert "is_narrow" in block
        assert "gap_before_px" in block
        assert "lines" in block
        for line in block["lines"]:
            assert "words" in line
            for word in line["words"]:
                for field in ("xmin", "ymin", "xmax", "ymax", "bl", "above"):
                    assert field in word, f"Missing word field: {field}"
                total_words += 1

    print(f"    total words : {total_words}")

    # --- Print block summary ---
    print("\n[4] Block summary:")
    for block in response["blocks"]:
        n_words = sum(len(l["words"]) for l in block["lines"])
        bbox = block["bbox"]
        print(f"    [{block['index']:2d}] {block['block_type']:<30s}  "
              f"bbox=({bbox[0]},{bbox[1]})→({bbox[2]},{bbox[3]})  "
              f"narrow={block['is_narrow']}  "
              f"lines={len(block['lines'])}  words={n_words}")

    # --- Save full response for inspection ---
    out_path = Path("test_server_response.json")
    out_path.write_text(json.dumps(response, indent=2))
    print(f"\n[5] Full response saved to {out_path}")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
