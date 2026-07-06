# ============================================================
# UNIVERSAL GDOT CROSS-SECTION GRID REMOVAL PIPELINE
# Updated: detects CROSS-SECTIONS / CROSS SECTIONS
# ============================================================

# !pip install pymupdf pillow opencv-python-headless numpy -q

import fitz
import os
import re
import shutil
import zipfile
import cv2
import numpy as np
from PIL import Image
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    files = None
    IN_COLAB = False

# =========================
# INPUT
# =========================
PDF_PATH = "F:/Github/Infra_DMT_POC/files/FULL PLAN-Gulledge at Creekview Realignment.pdf"   # change input PDF here
OUTPUT_DIR = "universal_grid_removed_output"
ZIP_PATH = "universal_grid_removed_output.zip"
ZOOM = 4.0

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# PAGE DETECTION
# ============================================================

def normalize_text(text):
    text = text.upper()
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_title_block_text(page):
    rect = page.rect
    title_rect = fitz.Rect(
        rect.width * 0.50,
        rect.height * 0.65,
        rect.width,
        rect.height
    )
    return page.get_text("text", clip=title_rect)


def get_drawing_no_from_page(page):
    text = normalize_text(page.get_text("text"))

    match = re.search(r"\b(19|23)\s*(\d{4})\b", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    match2 = re.search(r"\b(19|23)-\d{4}\b", page.get_text("text").upper())
    if match2:
        return match2.group(0)

    return None


def has_scale(page):
    text = normalize_text(page.get_text("text"))

    patterns = [
        "VERTICAL",
        "HORIZONTAL",
        "VERT",
        "HORIZ",
        '1" = 10',
        '1"=10',
        '1" = 20',
        '1"=20'
    ]

    return any(p in text for p in patterns)


def get_station_numbers(page):
    text = page.get_text("text").upper()
    return re.findall(r"\b\d{1,3}\+\d{2}(?:\.\d+)?\b", text)


def has_station_pattern(page):
    stations = get_station_numbers(page)
    return len(stations) >= 2


def is_cross_section_page(page):
    text_raw = page.get_text("text")
    title_raw = get_title_block_text(page)

    text = normalize_text(text_raw)
    title = normalize_text(title_raw)

    drawing_no = get_drawing_no_from_page(page)

    exclude_keywords = [
        "CONSTRUCTION PLAN",
        "CONSTRUCTION STAGING PLAN",
        "CONSTRUCTION STAGING PROFILE",
        "MAINLINE PROFILE",
        "DRIVEWAY PROFILE",
        "CROSSROAD PROFILE",
        "TYPICAL SECTIONS",
        "SUMMARY QUANTITIES",
        "GENERAL NOTES",
        "DRAINAGE PROFILE",
        "DRAINAGE PROFILES",
        "DRAINAGE AREA MAP",
        "SPECIAL GRADING"
    ]

    # Strong accepts for real cross-section pages
    if "CONSTRUCTION STAGING CROSS SECTIONS" in text:
        return True

    if "EARTHWORK CROSS SECTIONS" in text:
        return True

    if "CROSS SECTIONS" in title and drawing_no:
        return True

    # Backup station-based detection
    if drawing_no and drawing_no.startswith(("19-", "23-")):
        if has_station_pattern(page) and has_scale(page):
            for bad in exclude_keywords:
                if bad in text:
                    return False
            return True

    return False


# ============================================================
# VECTOR GRID REMOVAL
# ============================================================

def drawing_bbox(drawing):
    rect = drawing.get("rect")
    if rect:
        return rect

    xs, ys = [], []

    for it in drawing.get("items", []):
        for obj in it[1:]:
            if hasattr(obj, "x") and hasattr(obj, "y"):
                xs.append(obj.x)
                ys.append(obj.y)

    if not xs or not ys:
        return None

    return fitz.Rect(min(xs), min(ys), max(xs), max(ys))


def is_axis_aligned_line_item(it):
    if it[0] != "l":
        return False, None, None, None, None

    a, b = it[1], it[2]
    dx = abs(b.x - a.x)
    dy = abs(b.y - a.y)

    horizontal = dy < 0.8
    vertical = dx < 0.8

    return horizontal or vertical, horizontal, vertical, dx, dy


def is_grid_vector_object(drawing, page_w, page_h):
    width = drawing.get("width") or 0

    if width > 1.4:
        return False

    items = drawing.get("items", [])
    if not items:
        return False

    rect = drawing_bbox(drawing)
    if rect is None:
        return False

    bbox_w = rect.width
    bbox_h = rect.height

    if bbox_w < page_w * 0.002 and bbox_h < page_h * 0.002:
        return False

    line_items = []
    horizontal_count = 0
    vertical_count = 0
    diagonal_count = 0

    for it in items:
        op = it[0]

        if op == "c":
            return False

        if op in ("re", "qu"):
            if width <= 1.2:
                if bbox_w > page_w * 0.20 or bbox_h > page_h * 0.05:
                    return True
            return False

        if op != "l":
            continue

        ok, horizontal, vertical, dx, dy = is_axis_aligned_line_item(it)

        if not ok:
            diagonal_count += 1
            continue

        line_items.append(it)

        if horizontal:
            horizontal_count += 1

        if vertical:
            vertical_count += 1

    if not line_items:
        return False

    if diagonal_count > 0:
        return False

    total = len(line_items)

    if horizontal_count == total:
        max_dx = max(abs(it[2].x - it[1].x) for it in line_items)
        if max_dx > page_w * 0.25:
            return True

    if vertical_count == total:
        max_dy = max(abs(it[2].y - it[1].y) for it in line_items)
        if max_dy > page_h * 0.025:
            return True

    if vertical_count >= 3 and vertical_count == total:
        avg_h = np.mean([abs(it[2].y - it[1].y) for it in line_items])
        if avg_h > page_h * 0.015:
            return True

    return False


def remove_grid_vector(page, zoom=4.0):
    pw, ph = page.rect.width, page.rect.height

    out_doc = fitz.open()
    new_page = out_doc.new_page(width=pw, height=ph)
    new_page.draw_rect(new_page.rect, color=None, fill=(1, 1, 1))

    shape = new_page.new_shape()

    removed = 0
    kept = 0

    for dr in page.get_drawings():

        if is_grid_vector_object(dr, pw, ph):
            removed += 1
            continue

        for it in dr.get("items", []):
            op = it[0]

            try:
                if op == "l":
                    shape.draw_line(it[1], it[2])
                elif op == "c":
                    shape.draw_bezier(it[1], it[2], it[3], it[4])
                elif op == "re":
                    shape.draw_rect(it[1])
                elif op == "qu":
                    shape.draw_quad(it[1])
            except Exception:
                pass

        dashes = dr.get("dashes")
        if dashes in ("", "[] 0", None):
            dashes = None

        try:
            shape.finish(
                color=dr.get("color") or (0, 0, 0),
                fill=dr.get("fill"),
                width=dr.get("width") or 1,
                dashes=dashes,
                closePath=dr.get("closePath", False)
            )
            kept += 1
        except Exception:
            pass

    shape.commit()

    pix = new_page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    out_doc.close()

    return np.array(img), removed, kept


# ============================================================
# RASTER FALLBACK
# ============================================================

def render_page_to_image(page, zoom=4.0):
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)


def remove_grid_raster(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)

    h, w = bw.shape

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(40, w // 35), 1)
    )
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(25, h // 45))
    )
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)

    grid_mask = cv2.bitwise_or(horizontal, vertical)

    grid_mask = cv2.dilate(
        grid_mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        iterations=1
    )

    cleaned = img_bgr.copy()
    cleaned[grid_mask > 0] = (255, 255, 255)

    return cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)


# ============================================================
# STRIP EXTRACTION
# ============================================================

def find_station_labels(page):
    rect = page.rect
    labels = []

    words = page.get_text("words")

    for w in words:
        x0, y0, x1, y1, word = w[:5]

        if re.match(r"^\d{1,3}\+\d{2}(?:\.\d+)?$", word):
            labels.append(fitz.Rect(x0, y0, x1, y1))

    labels = sorted(labels, key=lambda r: r.y0)

    # remove close duplicates
    cleaned = []
    for lab in labels:
        if not cleaned or abs(lab.y0 - cleaned[-1].y0) > 15:
            cleaned.append(lab)

    return cleaned


def extract_cross_section_strips(page):
    rect = page.rect
    labels = find_station_labels(page)

    crops = []

    if len(labels) >= 1:
        last_bottom = rect.height * 0.04

        for idx, lab in enumerate(labels):
            y_top = max(0, last_bottom)

            if idx + 1 < len(labels):
                y_bottom = min(lab.y1 + 50, labels[idx + 1].y0 - 15)
            else:
                y_bottom = min(lab.y1 + 65, rect.height * 0.88)

            if y_bottom <= y_top:
                continue

            crop = fitz.Rect(
                rect.width * 0.02,
                y_top,
                rect.width * 0.98,
                y_bottom
            )

            crops.append((idx + 1, crop))
            last_bottom = y_bottom

    return crops


def crop_numpy_image(img_rgb, page_rect, crop_rect):
    h, w = img_rgb.shape[:2]

    x0 = int((crop_rect.x0 / page_rect.width) * w)
    x1 = int((crop_rect.x1 / page_rect.width) * w)
    y0 = int((crop_rect.y0 / page_rect.height) * h)
    y1 = int((crop_rect.y1 / page_rect.height) * h)

    return img_rgb[y0:y1, x0:x1]


# ============================================================
# MAIN PROCESS
# ============================================================

def process_pdf(pdf_path):
    doc = fitz.open(pdf_path)

    print("PDF:", pdf_path)
    print("Total pages:", len(doc))
    print("Output:", OUTPUT_DIR)
    print("-" * 70)

    total_saved = 0
    skipped = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_no = page_index + 1

        if not is_cross_section_page(page):
            skipped += 1
            continue

        drawing_no = get_drawing_no_from_page(page)
        stations = get_station_numbers(page)

        print(f"Processing page {page_no}")
        print("  Drawing:", drawing_no)
        print("  Stations:", len(stations))

        try:
            processed_img, removed, kept = remove_grid_vector(page, zoom=ZOOM)
            print(f"  Vector used | removed={removed}, kept={kept}")
        except Exception as e:
            print("  Vector failed:", e)
            processed_img = render_page_to_image(page, zoom=ZOOM)
            processed_img = remove_grid_raster(processed_img)
            print("  Raster fallback used")

        crops = extract_cross_section_strips(page)

        for strip_no, crop_rect in crops:
            crop_img = crop_numpy_image(processed_img, page.rect, crop_rect)

            if crop_img.size == 0:
                continue

            filename = f"page_{page_no:03d}_section_{strip_no:02d}_no_grid.png"
            out_path = os.path.join(OUTPUT_DIR, filename)

            Image.fromarray(crop_img).save(out_path)
            total_saved += 1

        print("  Saved strips:", len(crops))

    doc.close()

    print("-" * 70)
    print("Done")
    print("Saved images:", total_saved)
    print("Skipped pages:", skipped)


def zip_output():
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in os.listdir(OUTPUT_DIR):
            if f.lower().endswith(".png"):
                zipf.write(os.path.join(OUTPUT_DIR, f), f)

    print("ZIP created:", ZIP_PATH)
    if files is not None:
        files.download(ZIP_PATH)


# ============================================================
# RUN
# ============================================================

process_pdf(PDF_PATH)
zip_output()