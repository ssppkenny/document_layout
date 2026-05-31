"""Benchmark /ocr_page endpoint: sequential (batched=false) vs batched (batched=true).

Usage:
    python bench_server_ocr.py archimedes.djvu --pages 20 21 22 23 24 --server http://localhost:8000
"""

import argparse
import io
import os
import time

import requests


def render_page_to_jpeg(djvu_path: str, page: int) -> bytes:
    """Render a DjVu page to JPEG bytes using document_loader."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from ocr_reflow.document_loader import load_page
    from pathlib import Path
    import cv2
    img_bgr = load_page(Path(djvu_path), page)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"Failed to encode page {page} as JPEG")
    return buf.tobytes()
    with open(tmp, "rb") as f:
        return f.read()


def ocr_request(server: str, jpeg_bytes: bytes, batched: bool) -> float:
    """POST to /ocr_page, return elapsed seconds."""
    url = f"{server}/ocr_page?batched={'true' if batched else 'false'}"
    t = time.perf_counter()
    resp = requests.post(url, files={"image": ("page.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}, timeout=300)
    elapsed = time.perf_counter() - t
    resp.raise_for_status()
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("djvu")
    parser.add_argument("--pages", nargs="+", type=int, default=[20, 21, 22, 23, 24])
    parser.add_argument("--server", default="http://localhost:8000")
    args = parser.parse_args()

    print(f"Rendering {len(args.pages)} pages from {args.djvu}...", flush=True)
    page_images = {}
    for p in args.pages:
        page_images[p] = render_page_to_jpeg(args.djvu, p)
        print(f"  page {p}: {len(page_images[p])//1024}KB", flush=True)

    results = {}

    for mode, batched in [("sequential", False), ("batched", True)]:
        print(f"\n--- Mode: {mode} (batched={batched}) ---", flush=True)
        times = {}
        for p in args.pages:
            elapsed = ocr_request(args.server, page_images[p], batched)
            times[p] = elapsed
            print(f"  page {p}: {elapsed:.2f}s", flush=True)
        results[mode] = times

    # Summary table
    print("\n" + "="*60)
    print(f"{'Page':<8} {'Sequential':>12} {'Batched':>12} {'Speedup':>10}")
    print("-"*60)
    speedups = []
    for p in args.pages:
        seq = results["sequential"][p]
        bat = results["batched"][p]
        sp = seq / bat
        speedups.append(sp)
        flag = " ⚠" if sp < 1.0 else ""
        print(f"{p:<8} {seq:>11.2f}s {bat:>11.2f}s {sp:>9.2f}x{flag}")
    print("-"*60)
    seq_avg = sum(results["sequential"][p] for p in args.pages) / len(args.pages)
    bat_avg = sum(results["batched"][p] for p in args.pages) / len(args.pages)
    avg_sp = seq_avg / bat_avg
    print(f"{'Average':<8} {seq_avg:>11.2f}s {bat_avg:>11.2f}s {avg_sp:>9.2f}x")
    print("="*60)


if __name__ == "__main__":
    main()
