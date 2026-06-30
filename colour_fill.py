import cv2
import numpy as np
import os, zipfile, shutil

FILE_PATH = "/content/grid_removing (1).zip"
TMP_EXTRACT_DIR = "/content/tmp_stable_extraction"
OUTPUT_DIR = "/content/output_no_gaps_final"

for folder in [OUTPUT_DIR, TMP_EXTRACT_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def strong_fill_color_gaps(color_output):
    fixed = color_output.copy()
    b, g, r = cv2.split(fixed)

    green_mask = ((g > 100) & (r < 170) & (b < 170)).astype(np.uint8) * 255
    red_mask   = ((r > 100) & (g < 170) & (b < 170)).astype(np.uint8) * 255

    # Very strong gap closing for vertical number/text gaps
    green_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (180, 45))
    red_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (120, 35))

    green_closed = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, green_kernel)
    red_closed   = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, red_kernel)

    # Fill full closed green/red areas
    fixed[green_closed > 0] = [0, 210, 0]
    fixed[red_closed > 0] = [0, 0, 255]

    return fixed


def process_precision_logic(img_gray, filename):
    h_img, w_img = img_gray.shape

    _, bw = cv2.threshold(img_gray, 235, 255, cv2.THRESH_BINARY_INV)

    ver_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 150))
    hor_k = cv2.getStructuringElement(cv2.MORPH_RECT, (150, 1))

    grid_mask = cv2.add(
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_k),
        cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_k)
    )

    diagram_only = cv2.subtract(bw, grid_mask)
    bridge_shield = cv2.dilate(diagram_only, np.ones((3, 3), np.uint8), iterations=1)
    eraser = cv2.subtract(grid_mask, bridge_shield)
    clean_bw = cv2.subtract(bw, eraser)

    color_output = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    color_output[eraser > 0] = [255, 255, 255]

    nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(clean_bw, 8, cv2.CV_32S)

    solid_mask = np.zeros_like(clean_bw)
    dotted_mask = np.zeros_like(clean_bw)

    scale_line_y_threshold = int(h_img * 0.88)

    for i in range(1, nlabels):
        x, y, w, h, area = stats[i]

        if x < 3 or y < 3 or (x + w) > (w_img - 3) or (y + h) > (h_img - 3):
            continue

        if y > scale_line_y_threshold:
            continue

        if h >= 16 and w <= 45:
            continue

        diag_len = np.sqrt(w ** 2 + h ** 2)
        aspect_ratio = w / h if h > 0 else 0

        if 3 <= w <= 45 and 2 <= h <= 12 and aspect_ratio > 1.0:
            dotted_mask[labels == i] = 255

        elif diag_len > 55 or w > 50 or h > 50:
            solid_mask[labels == i] = 255

    heal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    solid_boundary = cv2.morphologyEx(solid_mask, cv2.MORPH_CLOSE, heal_kernel)
    dotted_boundary = cv2.morphologyEx(dotted_mask, cv2.MORPH_CLOSE, heal_kernel)

    red_overlay = np.zeros_like(clean_bw)
    green_overlay = np.zeros_like(clean_bw)

    for c in range(w_img):
        solid_rows = np.where(solid_boundary[:, c] == 255)[0]

        if len(solid_rows) == 0:
            continue

        st = np.min(solid_rows)
        sb = np.max(solid_rows)

        dotted_above = np.where(dotted_boundary[:st, c] == 255)[0]
        dotted_below = np.where(dotted_boundary[sb:, c] == 255)[0]

        if len(dotted_above) > 0:
            boundary_y = np.max(dotted_above)
            red_overlay[boundary_y:st, c] = 255

        elif len(dotted_below) > 0:
            boundary_y = np.min(dotted_below) + sb
            green_overlay[sb:boundary_y, c] = 255

    # Strong overlay smoothing
    hor_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (120, 1))
    ver_close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 12))
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    green_overlay = cv2.morphologyEx(green_overlay, cv2.MORPH_CLOSE, hor_close_kernel)
    green_overlay = cv2.morphologyEx(green_overlay, cv2.MORPH_CLOSE, ver_close_kernel)
    green_overlay = cv2.dilate(green_overlay, smooth_kernel, iterations=1)

    red_overlay = cv2.morphologyEx(red_overlay, cv2.MORPH_CLOSE, hor_close_kernel)
    red_overlay = cv2.morphologyEx(red_overlay, cv2.MORPH_CLOSE, ver_close_kernel)
    red_overlay = cv2.dilate(red_overlay, smooth_kernel, iterations=1)

    color_output[green_overlay == 255] = [0, 210, 0]
    color_output[red_overlay == 255] = [0, 0, 255]

    # Do NOT restore text/numbers.
    # Do NOT recolor solid lines blue.
    # This allows fill to cover numbers fully.

    color_output = strong_fill_color_gaps(color_output)

    return color_output


if os.path.exists(FILE_PATH):
    print("Unzipping input zip...")

    with zipfile.ZipFile(FILE_PATH, "r") as z:
        z.extractall(TMP_EXTRACT_DIR)

    valid_file_paths = []

    for root, _, files in os.walk(TMP_EXTRACT_DIR):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")) and "__macosx" not in root.lower():
                valid_file_paths.append(os.path.join(root, file))

    print("Images found:", len(valid_file_paths))

    for idx, img_path in enumerate(valid_file_paths, start=1):
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img_gray is None:
            continue

        out_name = "precision_" + os.path.basename(img_path)
        result = process_precision_logic(img_gray, out_name)

        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), result)
        print(f"Processed {idx}/{len(valid_file_paths)}:", out_name)

    print("Done")
    print("Output folder:", OUTPUT_DIR)

else:
    print("ERROR: Input zip not found:", FILE_PATH)