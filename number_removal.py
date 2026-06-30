!apt-get install -y tesseract-ocr > /dev/null
!pip install pymupdf opencv-python-headless pytesseract -q

import fitz, cv2, os, shutil, zipfile, pytesseract
import numpy as np
from google.colab.patches import cv2_imshow

PDF_PATH = "/content/cross_sections (1) (1).pdf"
OUTPUT_DIR = "numbers_removed_keep_grid"

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

def build_line_restore_mask(gray):
    _, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    h, w = bw.shape

    # grid horizontal + vertical lines
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1))
    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 120))

    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_k)
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_k)

    line_mask = cv2.bitwise_or(horizontal, vertical)

    # cross-section/profile line restore using Hough
    edges = cv2.Canny(gray, 80, 180)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=60,
        minLineLength=45,
        maxLineGap=8
    )

    hough_mask = np.zeros_like(gray)

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            length = ((x2-x1)**2 + (y2-y1)**2) ** 0.5

            # keep longer line segments only, avoid text strokes
            if length > 45:
                cv2.line(hough_mask, (x1,y1), (x2,y2), 255, 3)

    line_mask = cv2.bitwise_or(line_mask, hough_mask)

    # thicken restore mask slightly
    line_mask = cv2.dilate(line_mask, np.ones((3,3), np.uint8), iterations=1)

    return line_mask


def remove_text_components(img, line_mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # remove line pixels from component detection
    text_candidate = cv2.bitwise_and(bw, cv2.bitwise_not(line_mask))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(text_candidate, 8)

    remove_mask = np.zeros_like(gray)

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # number/text-like parts
        if 5 <= area <= 1800 and ww <= 90 and hh <= 120:
            cv2.rectangle(remove_mask, (x-4,y-4), (x+ww+4,y+hh+4), 255, -1)

    img[remove_mask > 0] = 255
    return img


def remove_ocr_numbers(img):
    out = img.copy()
    h, w = out.shape[:2]

    for angle in [0, 90, 270]:
        if angle == 0:
            rot = out.copy()
        elif angle == 90:
            rot = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
        else:
            rot = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)

        gray = cv2.cvtColor(rot, cv2.COLOR_BGR2GRAY)

        data = pytesseract.image_to_data(
            gray,
            config="--psm 11 -c tessedit_char_whitelist=0123456789.+/%:-",
            output_type=pytesseract.Output.DICT
        )

        rh, rw = rot.shape[:2]

        for i, txt in enumerate(data["text"]):
            txt = txt.strip()

            if txt == "":
                continue

            if any(c.isdigit() for c in txt):
                x = data["left"][i]
                y = data["top"][i]
                ww = data["width"][i]
                hh = data["height"][i]

                pad = 12
                cv2.rectangle(
                    rot,
                    (max(0, x-pad), max(0, y-pad)),
                    (min(rw, x+ww+pad), min(rh, y+hh+pad)),
                    (255,255,255),
                    -1
                )

        if angle == 0:
            out = rot
        elif angle == 90:
            out = cv2.rotate(rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            out = cv2.rotate(rot, cv2.ROTATE_90_CLOCKWISE)

    return out


def final_clean_page(img):
    original = img.copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    line_mask = build_line_restore_mask(gray)

    cleaned = remove_ocr_numbers(original.copy())
    cleaned = remove_text_components(cleaned, line_mask)

    # restore grid + cross-section line pixels
    cleaned[line_mask > 0] = original[line_mask > 0]

    return cleaned


doc = fitz.open(PDF_PATH)

for i in range(len(doc)):
    page = doc[i]
    pix = page.get_pixmap(matrix=fitz.Matrix(4,4), alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cleaned = final_clean_page(img)

    out_path = f"{OUTPUT_DIR}/page_{i+1}.png"
    cv2.imwrite(out_path, cleaned)
    print("Saved:", out_path)

zip_name = "numbers_removed_keep_grid.zip"

with zipfile.ZipFile(zip_name, "w") as zipf:
    for f in os.listdir(OUTPUT_DIR):
        zipf.write(os.path.join(OUTPUT_DIR, f), f)

print("DONE:", zip_name)

sample = cv2.imread(f"{OUTPUT_DIR}/page_1.png")
cv2_imshow(sample)