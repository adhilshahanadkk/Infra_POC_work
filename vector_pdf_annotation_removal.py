!pip install pymupdf opencv-python-headless pandas -q

import fitz
import os
import re
import math
import cv2
import numpy as np
import pandas as pd
import shutil
import zipfile
from IPython.display import Image, display

# =====================================================
# INPUT / OUTPUT
# =====================================================
INPUT_PDF = "/content/highway_planning1 (1).pdf"
OUTPUT_DIR = "/content/all_19series_annotation_removed_outputs"

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

ORIGINAL_DIR = os.path.join(OUTPUT_DIR, "01_original")
TEXTREMOVED_DIR = os.path.join(OUTPUT_DIR, "02_text_removed")
PROFILE_DIR = os.path.join(OUTPUT_DIR, "03_profile_only")
RESTORED_DIR = os.path.join(OUTPUT_DIR, "04_text_removed_profile_restored")
CSV_DIR = os.path.join(OUTPUT_DIR, "05_profile_csv")

for d in [ORIGINAL_DIR, TEXTREMOVED_DIR, PROFILE_DIR, RESTORED_DIR, CSV_DIR]:
    os.makedirs(d, exist_ok=True)


# =====================================================
# FIND ALL 19-SERIES CROSS SECTION PAGES
# =====================================================
def find_all_19_pages(pdf_path):

    doc = fitz.open(pdf_path)
    pages = []

    for i in range(len(doc)):

        page = doc[i]
        rect = page.rect

        br_rect = fitz.Rect(
            rect.width * 0.70,
            rect.height * 0.80,
            rect.width,
            rect.height
        )

        corner_text = page.get_text("text", clip=br_rect).strip()

        is_drawing_19 = False

        for line in corner_text.split("\n"):

            clean_line = re.sub(
                r'(?i)DRAWING|NO\.?|DRG|[:\s]',
                '',
                line
            ).strip()

            if (
                (clean_line.startswith("19-") or clean_line.startswith("19"))
                and "+" not in clean_line
            ):
                is_drawing_19 = True
                break

        if not is_drawing_19:
            continue

        if re.search(r'(?i)cross[- \s]*sections?', corner_text):
            pages.append(i)

    doc.close()
    return pages


# =====================================================
# ANNOTATION TEXT FILTER
# =====================================================
def is_annotation_text(text):

    text = text.strip()

    if text == "":
        return False

    patterns = [
        r"^[+-]?\d+(\.\d+)?$",          # 652.73, -50.20
        r"^[+-]?\d+(\.\d+)?%$",         # 8.0%, 10.6%
        r"^\d+:\d+$",                   # 2:1, 4:1
        r"^\d+\.\d+:\d+$",              # 5.5:1
        r"^\d+\+\d+$",                  # 12+50
        r"^[+-]?\d+$",                  # 650, -140
        r"^P\d+$",                      # P1, P2
        r"^LT$",                         # LT
        r"^RT$",                         # RT
        r"^CL$",                         # CL
        r"^TEMPORARY$",                  # Temporary
        r"^PIPE$",                       # Pipe
        r"^SHORING$",                    # Shoring
        r"^STAGE$",                      # Stage
    ]

    return any(re.match(p, text, re.I) for p in patterns)


# =====================================================
# EXTRACT PROFILE VECTOR LINES
# =====================================================
def extract_profile_vectors(pdf_path, page_index):

    MIN_PROFILE_LENGTH = 25       # safer for all pages
    MAX_PROFILE_ANGLE = 55
    MAX_LINE_LENGTH = 300

    doc = fitz.open(pdf_path)
    page = doc[page_index]
    page_h = page.rect.height

    drawings = page.get_drawings()
    rows = []

    for d_idx, drawing in enumerate(drawings):

        for item in drawing.get("items", []):

            if item[0] != "l":
                continue

            p1 = item[1]
            p2 = item[2]

            x1, y1 = p1.x, p1.y
            x2, y2 = p2.x, p2.y

            dx = x2 - x1
            dy = y2 - y1

            length = math.sqrt(dx * dx + dy * dy)

            if length < MIN_PROFILE_LENGTH:
                continue

            angle = abs(math.degrees(math.atan2(dy, dx)))

            rows.append({
                "drawing_id": d_idx,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "length": length,
                "angle": angle,
                "stroke_width": drawing.get("width"),
                "color": drawing.get("color")
            })

    doc.close()

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Remove vertical / arrow / scale marks
    df = df[
        ~((df["angle"] > 70) & (df["angle"] < 110))
    ].copy()

    # Remove steep leader/annotation lines
    df = df[
        df["angle"] <= MAX_PROFILE_ANGLE
    ].copy()

    # Mark near-horizontal
    df["near_horizontal"] = df["angle"].apply(
        lambda a: a < 2 or abs(a - 180) < 2
    )

    # Remove very long horizontal grid/axis lines
    df = df[
        (~df["near_horizontal"]) |
        ((df["near_horizontal"]) & (df["length"] < 120))
    ].copy()

    # Remove very large border/title lines
    df = df[
        df["length"] < MAX_LINE_LENGTH
    ].copy()

    # Remove top title and bottom title block area
    df = df[
        (df["y1"] > 120) &
        (df["y2"] > 120) &
        (df["y1"] < page_h * 0.88) &
        (df["y2"] < page_h * 0.88)
    ].copy()

    return df


# =====================================================
# PROCESS ONE PAGE
# =====================================================
def process_page(pdf_path, page_index, out_index):

    zoom = 3

    original_path = os.path.join(
        ORIGINAL_DIR,
        f"page_{out_index:03d}_pdfpage_{page_index + 1}.png"
    )

    text_removed_path = os.path.join(
        TEXTREMOVED_DIR,
        f"page_{out_index:03d}_pdfpage_{page_index + 1}.png"
    )

    profile_only_path = os.path.join(
        PROFILE_DIR,
        f"page_{out_index:03d}_pdfpage_{page_index + 1}.png"
    )

    restored_path = os.path.join(
        RESTORED_DIR,
        f"page_{out_index:03d}_pdfpage_{page_index + 1}.png"
    )

    csv_path = os.path.join(
        CSV_DIR,
        f"page_{out_index:03d}_pdfpage_{page_index + 1}_profile_vectors.csv"
    )

    profile_df = extract_profile_vectors(pdf_path, page_index)
    profile_df.to_csv(csv_path, index=False)

    doc = fitz.open(pdf_path)
    page = doc[page_index]

    pix = page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False
    )
    pix.save(original_path)

    one_doc = fitz.open()
    one_doc.insert_pdf(
        doc,
        from_page=page_index,
        to_page=page_index
    )

    doc.close()

    clean_page = one_doc[0]

    words = clean_page.get_text("words")
    hidden_count = 0

    for word in words:

        x0, y0, x1, y1, text = word[:5]

        if not is_annotation_text(text):
            continue

        shrink = 0.80

        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2

        new_w = (x1 - x0) * shrink
        new_h = (y1 - y0) * shrink

        rect = fitz.Rect(
            cx - new_w / 2,
            cy - new_h / 2,
            cx + new_w / 2,
            cy + new_h / 2
        )

        clean_page.draw_rect(
            rect,
            color=(1, 1, 1),
            fill=(1, 1, 1),
            overlay=True
        )

        hidden_count += 1

    pix = clean_page.get_pixmap(
        matrix=fitz.Matrix(zoom, zoom),
        alpha=False
    )
    pix.save(text_removed_path)

    one_doc.close()

    img = cv2.imread(text_removed_path)

    if img is None:
        raise ValueError(f"Could not read image: {text_removed_path}")

    profile_only = np.full_like(img, 255)

    for _, row in profile_df.iterrows():

        x1 = int(row["x1"] * zoom)
        y1 = int(row["y1"] * zoom)
        x2 = int(row["x2"] * zoom)
        y2 = int(row["y2"] * zoom)

        # Restore selected profile vectors in black
        cv2.line(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 0),
            2
        )

        # Profile-only debug image in blue
        cv2.line(
            profile_only,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    cv2.imwrite(restored_path, img)
    cv2.imwrite(profile_only_path, profile_only)

    return {
        "pdf_page": page_index + 1,
        "hidden_words": hidden_count,
        "profile_segments": len(profile_df),
        "original": original_path,
        "text_removed": text_removed_path,
        "profile_only": profile_only_path,
        "restored": restored_path,
        "csv": csv_path
    }


# =====================================================
# MAIN: PROCESS ALL 19-SERIES PAGES
# =====================================================
pages = find_all_19_pages(INPUT_PDF)

print("Total 19-series pages found:", len(pages))
print("PDF pages:", [p + 1 for p in pages])

results = []

for idx, page_index in enumerate(pages, start=1):

    print(f"\nProcessing {idx}/{len(pages)} | PDF page {page_index + 1}")

    try:
        result = process_page(
            INPUT_PDF,
            page_index,
            idx
        )

        results.append(result)

        print(
            "Hidden words:",
            result["hidden_words"],
            "| Profile segments:",
            result["profile_segments"]
        )

    except Exception as e:
        print("ERROR on page:", page_index + 1, e)


# =====================================================
# SAVE SUMMARY CSV
# =====================================================
summary_df = pd.DataFrame(results)

summary_path = os.path.join(
    OUTPUT_DIR,
    "processing_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nSummary saved:", summary_path)


# =====================================================
# SHOW FIRST 5 RESTORED OUTPUTS
# =====================================================
print("\nDisplaying first 5 restored outputs:")

for r in results[:5]:

    print(
        f"Restored | PDF page {r['pdf_page']} | "
        f"Hidden words: {r['hidden_words']} | "
        f"Profile segments: {r['profile_segments']}"
    )

    display(Image(filename=r["restored"]))


# =====================================================
# ZIP ALL OUTPUTS
# =====================================================
zip_path = "/content/all_19series_annotation_removed_outputs.zip"

with zipfile.ZipFile(
    zip_path,
    "w",
    zipfile.ZIP_DEFLATED
) as z:

    for root, dirs, files in os.walk(OUTPUT_DIR):

        for file in files:

            fp = os.path.join(root, file)

            z.write(
                fp,
                os.path.relpath(fp, OUTPUT_DIR)
            )

print("\nDONE")
print("Output folder:", OUTPUT_DIR)
print("Output zip:", zip_path)