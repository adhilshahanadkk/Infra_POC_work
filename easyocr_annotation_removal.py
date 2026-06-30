!pip install easyocr pymupdf opencv-python-headless -q

import cv2
import os
import shutil
import numpy as np
import fitz
import easyocr
from google.colab.patches import cv2_imshow

# ============================================
# PATHS
# ============================================
PDF_PATH = "/content/cross_sections (1) (1).pdf"

OUTPUT_DIR = "easyocr_pdf_text_removed"

# ============================================
# CLEAN OUTPUT
# ============================================
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR)

# ============================================
# LOAD OCR
# ============================================
reader = easyocr.Reader(['en'], gpu=False)

# ============================================
# REMOVE TEXT FUNCTION
# ============================================
def remove_text_easyocr(img):

    out = img.copy()

    h, w = out.shape[:2]

    results = reader.readtext(
        out,
        detail=1,
        paragraph=False,
        text_threshold=0.25,
        low_text=0.15,
        link_threshold=0.20
    )

    for box, text, conf in results:

        text = str(text).strip()

        # remove only numeric/engineering text
        if not any(ch.isdigit() for ch in text):
            continue

        pts = np.array(box).astype(np.int32)

        x, y, bw, bh = cv2.boundingRect(pts)

        # avoid removing huge regions
        if bw > 350 or bh > 250:
            continue

        pad = 28

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)

        cv2.rectangle(
            out,
            (x1, y1),
            (x2, y2),
            (255,255,255),
            -1
        )

    # remove bottom scales
    out[int(h * 0.86):h, :] = 255

    # remove left/right scales
    out[:, 0:int(w * 0.10)] = 255
    out[:, int(w * 0.90):w] = 255

    return out

# ============================================
# OPEN PDF
# ============================================
doc = fitz.open(PDF_PATH)

print("Total Pages:", len(doc))

# ============================================
# PROCESS EACH PAGE
# ============================================
for i in range(len(doc)):

    page = doc[i]

    # render high resolution page
    pix = page.get_pixmap(
        matrix=fitz.Matrix(3,3),
        alpha=False
    )

    img = np.frombuffer(
        pix.samples,
        dtype=np.uint8
    ).reshape(
        pix.height,
        pix.width,
        pix.n
    )

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cleaned = remove_text_easyocr(img)

    out_path = os.path.join(
        OUTPUT_DIR,
        f"page_{i+1}.png"
    )

    cv2.imwrite(out_path, cleaned)

    print("Saved:", out_path)

# ============================================
# SHOW SAMPLE
# ============================================
sample = cv2.imread(
    os.path.join(OUTPUT_DIR, os.listdir(OUTPUT_DIR)[0])
)

cv2_imshow(sample)

print("DONE")