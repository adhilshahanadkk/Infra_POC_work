!pip install opencv-python-headless -q

import cv2, os, zipfile, shutil
import numpy as np
from google.colab.patches import cv2_imshow

ZIP_PATH = "/content/grid_removing (1).zip"
TEMP_DIR = "temp_input"
OUTPUT_DIR = "aggressive_numbers_removed"

for folder in [TEMP_DIR, OUTPUT_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    z.extractall(TEMP_DIR)

def remove_text_keep_profile(img):
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    h, w = bw.shape

    # 1. Detect profile/dashed/solid line segments
    edges = cv2.Canny(gray, 40, 160)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=18,
        minLineLength=25,
        maxLineGap=25
    )

    line_mask = np.zeros_like(bw)

    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]

            length = np.hypot(x2-x1, y2-y1)
            angle = abs(np.degrees(np.arctan2(y2-y1, x2-x1)))

            # keep horizontal/sloped engineering profile lines
            # ignore near-vertical text strokes
            if length >= 25 and angle <= 75:
                cv2.line(line_mask, (x1, y1), (x2, y2), 255, 3)

    # make profile protection stronger
    line_mask = cv2.dilate(line_mask, np.ones((3,3), np.uint8), iterations=1)

    # 2. Create search area around profile only
    near_zone = cv2.dilate(line_mask, np.ones((150,150), np.uint8), iterations=1)

    # ignore bottom scale numbers
    near_zone[int(h * 0.78):h, :] = 0

    # ignore side scale numbers
    near_zone[:, 0:int(w * 0.08)] = 0
    near_zone[:, int(w * 0.92):w] = 0

    # 3. Remove all non-profile black pixels near profile
    remove_mask = cv2.bitwise_and(bw, near_zone)
    remove_mask = cv2.bitwise_and(remove_mask, cv2.bitwise_not(line_mask))

    # expand removal to cover full number thickness
    remove_mask = cv2.dilate(remove_mask, np.ones((12,12), np.uint8), iterations=1)

    cleaned = original.copy()

    # remove text/numbers
    cleaned[remove_mask > 0] = 255

    # restore profile/dashed/solid lines
    cleaned[line_mask > 0] = original[line_mask > 0]

    return cleaned

image_files = []
for root, dirs, files in os.walk(TEMP_DIR):
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            image_files.append(os.path.join(root, f))

image_files = sorted(image_files)
print("Total images:", len(image_files))

for idx, path in enumerate(image_files, start=1):
    img = cv2.imread(path)
    if img is None:
        continue

    cleaned = remove_text_keep_profile(img)

    out_path = os.path.join(OUTPUT_DIR, f"page_{idx}.png")
    cv2.imwrite(out_path, cleaned)
    print("Saved:", out_path)

zip_name = "aggressive_numbers_removed.zip"

with zipfile.ZipFile(zip_name, "w") as zipf:
    for f in os.listdir(OUTPUT_DIR):
        zipf.write(os.path.join(OUTPUT_DIR, f), f)

print("DONE:", zip_name)

sample = cv2.imread(os.path.join(OUTPUT_DIR, os.listdir(OUTPUT_DIR)[0]))
cv2_imshow(sample)