#Economic_Marks.py
#STEP_4


import cv2
import numpy as np
import csv
import pathlib
CROP_DIR             = pathlib.Path(r"CROP_OUTPUT")
GRID_LINES_DIR       = CROP_DIR / "INTERP_OUTPUT" 
OUT_CSV              = CROP_DIR / "COORD_DETECTOR.csv" 			  
IMAGES_CLOSE_BLOBS   = CROP_DIR / "FLAGGED_IMAGES.csv" 	 
# —————————— PARAMETERS
MIN_DIM_PCT       = 0.008    		# minimum width and length (0.8% of the crop)
HEIGHT_MAX_PCT    = 0.025   		# maximum height (2.5% of the crop)
GRID_BUFFER_PCT   = 0.02   		# buffer (2% of the crop) both on the right and left side of the grid lines
WORD_DIST_PCT_X   = 0.10    		# distance by X for filter "close to external words" (10% of the crop)
WORD_DIST_PCT_Y   = 0.10    		# distance by Y for filter "close to external words" (10% of the crop)
WORD_BUFFER_PCT   = 0.07    		# maximum distance from table border (7% of the crop)
BOTTOM_NOISE_PCT  = 0.011    		# 1.1% from top to bottom (for end of the page noise)
CLOSE_DIST_PCT    = 0.01     		# 1% of crop length for too close blobs
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
# —————————— DETECTOR
def extract_buffered_mask(mask_full, buf_px):
    k = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * buf_px + 1, 1)
    )
    mask_buffered = cv2.dilate(mask_full, k)
    return mask_buffered
with open(OUT_CSV, "w", newline="") as wf:
    writer = csv.writer(wf)
    writer.writerow(["file", "x", "y", "w", "h", "reason"])
    images_flagged = []
    for crop_path in sorted(CROP_DIR.glob("*.jpg")):
        fname = crop_path.name
        crop_color = cv2.imread(str(crop_path))
        if crop_color is None:
            continue
        crop_gray = cv2.cvtColor(crop_color, cv2.COLOR_BGR2GRAY)
        h_c, w_c = crop_gray.shape
        stem = crop_path.stem
        grid_lines_path = GRID_LINES_DIR / f"{stem}_lines_medie.jpg"
        mask_full = cv2.imread(str(grid_lines_path), cv2.IMREAD_GRAYSCALE)
        if mask_full is None:
            print(f"[Warning] Line mask not found for {fname}.")
            continue
        if mask_full.shape != crop_gray.shape:
            print(f"[Warning] Different dimensions: mask {mask_full.shape} vs crop {crop_gray.shape} for {fname}")
            continue
        xs = np.where(mask_full > 0)[1]
        if xs.size == 0:
            print(f"[Warning] No lines found in mask_full for {fname}, ignore.")
            continue
        splits = np.where(np.diff(xs) > 1)[0] + 1
        groups = np.split(xs, splits)
        left_block  = groups[0]
        right_block = groups[-1]
        left_x  = int(round((int(left_block.min())  + int(left_block.max()))  / 2.0))
        right_x = int(round((int(right_block.min()) + int(right_block.max())) / 2.0))
        buf_px          = int(round(w_c * GRID_BUFFER_PCT))
        word_buf_px     = int(round(w_c * WORD_BUFFER_PCT))
        word_dist_px_x  = int(round(w_c * WORD_DIST_PCT_X))
        word_dist_px_y  = int(round(h_c * WORD_DIST_PCT_Y))
        bottom_noise_px = int(round(h_c * BOTTOM_NOISE_PCT))
        close_dist_px   = int(round(w_c * CLOSE_DIST_PCT))
        mask_buffered = extract_buffered_mask(mask_full, buf_px)
        bin0 = cv2.adaptiveThreshold(
            crop_gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            15, 15
        )
        bin1 = bin0.copy()
        bin1[mask_full > 0] = 0
        num, labels, stats, cents = cv2.connectedComponentsWithStats(bin1, connectivity=8)
        min_w     = max(1, int(round(w_c * MIN_DIM_PCT)))
        min_h     = max(1, int(round(h_c * MIN_DIM_PCT)))
        height_max = h_c * HEIGHT_MAX_PCT
        kept_blobs = []
        primary_pass = []
        for i in range(1, num):
            x, y, w_, h_, area = stats[i]
            cx, cy = cents[i]
            cx, cy = int(cx), int(cy)
            if w_ < min_w or h_ < min_h:
                continue
            if h_ > height_max:
                continue
            submask = mask_buffered[y : y + h_, x : x + w_]
            if np.all(submask > 0):
                continue
            if cy >= (h_c - bottom_noise_px):
                continue
            primary_pass.append((i, x, y, w_, h_, cx, cy))
        discarded_A = []
        for (i, x, y, w_, h_, cx, cy) in primary_pass:
            if cx < left_x or cx > right_x:
                discarded_A.append((cx, cy))
        for i in range(1, num):
            x, y, w_, h_, area = stats[i]
            cx, cy = cents[i]
            cx, cy = int(cx), int(cy)
            reason = "kept"
            if w_ < min_w or h_ < min_h:
                reason = "too_small"
            elif h_ > height_max:
                reason = "height_exceed"
            else:
                submask = mask_buffered[y : y + h_, x : x + w_]
                if np.all(submask > 0):
                    reason = "grid_prox"
                else:
                    if cy >= (h_c - bottom_noise_px):
                        reason = "bottom_noise"
                    else:
                        if cx < left_x or cx > right_x:
                            reason = "out_of_bounds"
                        else:
                            if left_x <= cx <= (left_x + word_buf_px):
                                for (ax, ay) in discarded_A:
                                    if abs(cx - ax) <= word_dist_px_x and abs(cy - ay) <= word_dist_px_y:
                                        reason = "out_of_words"
                                        break
                            if reason == "kept" and (right_x - word_buf_px) <= cx <= right_x:
                                for (ax, ay) in discarded_A:
                                    if abs(cx - ax) <= word_dist_px_x and abs(cy - ay) <= word_dist_px_y:
                                        reason = "out_of_words"
                                        break
            if reason == "kept":
                kept_blobs.append((x, y, w_, h_))
            writer.writerow([fname, x, y, w_, h_, reason])
        flagged = False
        n_kept = len(kept_blobs)
        for idx1 in range(n_kept):
            x1, y1, w1, h1 = kept_blobs[idx1]
            y1_bot = y1 + h1 - 1
            for idx2 in range(idx1 + 1, n_kept):
                x2, y2, w2, h2 = kept_blobs[idx2]
                if not (x1 + w1 - 1 < x2 or x2 + w2 - 1 < x1):
                    if y2 > y1:
                        gap = y2 - (y1 + h1)
                    else:
                        gap = y1 - (y2 + h2)
                    if gap < 0:
                        gap = 0   
                    if gap < close_dist_px:
                        images_flagged.append(fname)
                        flagged = True
                        break
            if flagged:
                break
    with open(IMAGES_CLOSE_BLOBS, "w", newline="") as wf2:
        w2 = csv.writer(wf2)
        w2.writerow(["file"])
        seen = set()
        for name in images_flagged:
            if name not in seen:
                w2.writerow([name])
                seen.add(name)
    print("Detector completated →", OUT_CSV)
    print("Images flagged →", IMAGES_CLOSE_BLOBS)
