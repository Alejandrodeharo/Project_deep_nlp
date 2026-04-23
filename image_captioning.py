from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import easyocr


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class OCRHit:
    text: str
    confidence: float
    box: np.ndarray  # shape (4, 2), float or int
    region_name: str


@dataclass
class ScoreCandidate:
    home: str
    away: str
    confidence: float
    region_name: str
    center_distance_penalty: float
    box_area: float
    raw_text: str


# ----------------------------
# Utility functions
# ----------------------------

def normalize_text(text: str) -> str:
    """
    Normalize OCR text so score-like patterns are easier to match.
    """
    t = text.strip()

    replacements = {
        "—": "-",
        "–": "-",
        "−": "-",
        "_": "-",
        "~": "-",
        ":": "-",
        ";": "-",
        "|": "1",   # common OCR confusion
        "I": "1",
        "l": "1",
        "O": "0",
        "o": "0",
        "S": "5",
    }

    for k, v in replacements.items():
        t = t.replace(k, v)

    # Keep only digits, hyphens, and spaces for score matching.
    t = re.sub(r"[^0-9\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def crop_region(img_bgr: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """
    Crop using fractional coordinates in [0,1].
    """
    h, w = img_bgr.shape[:2]
    xa = clamp(int(x0 * w), 0, w)
    ya = clamp(int(y0 * h), 0, h)
    xb = clamp(int(x1 * w), 0, w)
    yb = clamp(int(y1 * h), 0, h)
    return img_bgr[ya:yb, xa:xb].copy()


def preprocess_variants(img_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Generate multiple image variants for OCR robustness.
    """
    variants = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    # Upscaled grayscale
    up2 = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(up2)

    # Bilateral + Otsu threshold
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # Inverted threshold
    variants.append(255 - otsu)

    # Adaptive threshold
    adap = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    variants.append(adap)
    variants.append(255 - adap)

    # Contrast enhanced
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(clahe)

    return variants


def polygon_center(box: np.ndarray) -> Tuple[float, float]:
    pts = np.asarray(box, dtype=np.float32)
    return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))


def polygon_area(box: np.ndarray) -> float:
    pts = np.asarray(box, dtype=np.float32)
    return float(cv2.contourArea(pts))


def run_easyocr(reader: easyocr.Reader, img: np.ndarray, region_name: str) -> List[OCRHit]:
    """
    Run OCR and return hits.
    """
    results = reader.readtext(
        img,
        detail=1,
        paragraph=False,
        batch_size=1,
        width_ths=0.7,
        height_ths=0.7,
        decoder="greedy",
        text_threshold=0.5,
        low_text=0.3,
        link_threshold=0.4,
    )

    hits: List[OCRHit] = []
    for item in results:
        box, text, conf = item
        box_np = np.array(box, dtype=np.float32)
        hits.append(OCRHit(text=str(text), confidence=float(conf), box=box_np, region_name=region_name))
    return hits


# ----------------------------
# Score extraction logic
# ----------------------------

SCORE_REGEXES = [
    re.compile(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})\s+(\d{1,2})\b"),  # fallback when separator is missed
]


def extract_score_candidates(hits: Sequence[OCRHit], region_shape: Tuple[int, int]) -> List[ScoreCandidate]:
    """
    Convert OCR hits into score candidates using regex + heuristics.
    """
    h, w = region_shape[:2]
    cx_target, cy_target = w / 2.0, h / 2.0
    candidates: List[ScoreCandidate] = []

    # 1) Direct single-hit matches such as "2-3", "2 - 3", or "2 3"
    for hit in hits:
        norm = normalize_text(hit.text)
        for rx in SCORE_REGEXES:
            m = rx.search(norm)
            if m:
                home, away = m.group(1), m.group(2)
                cx, cy = polygon_center(hit.box)
                dist = ((cx - cx_target) ** 2 + (cy - cy_target) ** 2) ** 0.5
                area = polygon_area(hit.box)
                candidates.append(
                    ScoreCandidate(
                        home=home,
                        away=away,
                        confidence=hit.confidence,
                        region_name=hit.region_name,
                        center_distance_penalty=dist,
                        box_area=area,
                        raw_text=hit.text,
                    )
                )

    # 2) Combine nearby digit-only hits across the same line (e.g., "2", "-", "3" or just "2" + "3")
    digit_hits = []
    for hit in hits:
        norm = normalize_text(hit.text)
        if re.fullmatch(r"\d{1,2}", norm):
            cx, cy = polygon_center(hit.box)
            digit_hits.append((hit, norm, cx, cy))

    digit_hits.sort(key=lambda x: (x[3], x[2]))  # by y then x

    for i in range(len(digit_hits)):
        hit1, d1, x1, y1 = digit_hits[i]
        for j in range(i + 1, len(digit_hits)):
            hit2, d2, x2, y2 = digit_hits[j]

            # Same line-ish and not too far apart
            if abs(y1 - y2) > 0.08 * h:
                continue
            if not (0 < (x2 - x1) < 0.45 * w):
                continue

            mid_x = (x1 + x2) / 2.0
            mid_y = (y1 + y2) / 2.0
            dist = ((mid_x - cx_target) ** 2 + (mid_y - cy_target) ** 2) ** 0.5
            area = polygon_area(hit1.box) + polygon_area(hit2.box)
            conf = (hit1.confidence + hit2.confidence) / 2.0

            candidates.append(
                ScoreCandidate(
                    home=d1,
                    away=d2,
                    confidence=conf * 0.85,  # slight penalty vs direct match
                    region_name=hit1.region_name,
                    center_distance_penalty=dist,
                    box_area=area,
                    raw_text=f"{hit1.text} | {hit2.text}",
                )
            )

    return candidates


def choose_best_candidate(candidates: Sequence[ScoreCandidate]) -> Optional[ScoreCandidate]:
    """
    Rank candidates with score-aware heuristics.
    """
    if not candidates:
        return None

    def rank_key(c: ScoreCandidate):
        # Prefer:
        # - higher OCR confidence
        # - regions closer to image center
        # - larger text boxes
        # - scoreboard-focused regions
        region_bonus = 0.0
        if "center_lower" in c.region_name:
            region_bonus = 35.0
        elif "center_mid" in c.region_name:
            region_bonus = 25.0
        elif "whole" in c.region_name:
            region_bonus = 5.0

        # Main ranking score
        score = (
            c.confidence * 100.0
            + min(c.box_area / 500.0, 40.0)
            + region_bonus
            - min(c.center_distance_penalty / 20.0, 60.0)
        )
        return score

    return max(candidates, key=rank_key)


def format_score(home: str, away: str) -> str:
    """
    Normalize to exact required format.
    """
    home = str(int(home))
    away = str(int(away))
    return f"{home} - {away}"


# ----------------------------
# Main pipeline
# ----------------------------

def read_score(
    image_path: str,
    verbose: bool = False,
    reader: Optional[easyocr.Reader] = None,
) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img_pil = Image.open(image_path).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    use_gpu = torch.cuda.is_available()
    if verbose:
        device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
        print(f"[INFO] Using device: {'CUDA' if use_gpu else 'CPU'} ({device_name})", file=sys.stderr)

    ocr_reader = reader or easyocr.Reader(["en"], gpu=use_gpu, verbose=verbose)

    # Search regions:
    # - whole image
    # - center-lower area (common scoreboard placement)
    # - center-mid
    # - lower band
    # - center strip
    regions = {
        "whole": img_bgr,
        "center_lower": crop_region(img_bgr, 0.10, 0.55, 0.90, 0.95),
        "center_mid": crop_region(img_bgr, 0.18, 0.42, 0.82, 0.78),
        "lower_band": crop_region(img_bgr, 0.05, 0.60, 0.95, 0.98),
        "center_strip": crop_region(img_bgr, 0.25, 0.45, 0.75, 0.85),
    }

    all_candidates: List[ScoreCandidate] = []

    for region_name, region_img in regions.items():
        for variant in preprocess_variants(region_img):
            hits = run_easyocr(ocr_reader, variant, region_name=region_name)
            cands = extract_score_candidates(hits, region_shape=variant.shape)
            all_candidates.extend(cands)

            if verbose:
                for hit in hits:
                    print(
                        f"[OCR] region={region_name} conf={hit.confidence:.3f} text={hit.text!r}",
                        file=sys.stderr,
                    )

    best = choose_best_candidate(all_candidates)
    if best is None:
        raise RuntimeError("Could not detect a valid score in the image.")

    if verbose:
        print(
            f"[BEST] region={best.region_name} conf={best.confidence:.3f} raw={best.raw_text!r} "
            f"-> {best.home} - {best.away}",
            file=sys.stderr,
        )

    return format_score(best.home, best.away)


def get_data_images(data_dir: Path) -> List[Path]:
    """
    Return image files found directly inside the data directory.
    """
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    images = sorted(
        path for path in data_dir.iterdir() if path.is_file() and path.suffix.lower() in allowed_suffixes
    )

    if not images:
        raise RuntimeError(f"No image files found in: {data_dir}")

    return images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read the visible football match scores from all images in the data directory."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print OCR/debug information to stderr",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(__file__).resolve().parent / "data"
    use_gpu = torch.cuda.is_available()
    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=args.verbose)

    for image_path in get_data_images(data_dir):
        score = read_score(str(image_path), verbose=args.verbose, reader=reader)
        print(f"{image_path.name}: {score}")


if __name__ == "__main__":
    main()
