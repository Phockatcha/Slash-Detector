#Gridline_Interp.py
#STEP_3

import cv2
import numpy as np
import pathlib
CROP_DIR   = pathlib.Path(r"CROP_OUTPUT") 		
MASK_DIR   = CROP_DIR / "MASK_OUTPUT" 				
OUTPUT_DIR = CROP_DIR / "INTERP_OUTPUT" 			
OUTPUT_DIR.mkdir(exist_ok=True)
DP           = 1.0 				#same Hough resolution as that of image
THETA        = np.pi / 180 			#maximum angle rotation = 1 degree
THRESH       = 100 				#minimum count = 100 px per line
MIN_LINE_LEN = 50 				#minimum line lenght = 50 px
MAX_LINE_GAP = 10 				#maximum distance bw px of the same line = 10 px
ANGLE_TOL      = 1.0  				#angle tolerance = 1 degree
X_CLUSTER_PCT  = 0.05 				#tolered distance between line sets = 5% of the image   
for mask_path in sorted(MASK_DIR.glob("*_mask.png")):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    h, w = mask.shape
    x_tol = w * X_CLUSTER_PCT
    lines = cv2.HoughLinesP(mask, DP, THETA, THRESH,
                            minLineLength=MIN_LINE_LEN,
                            maxLineGap=MAX_LINE_GAP)
    if lines is None:
        continue
    segs = []    
    angles = []   
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if 80 < angle < 100:
            if dx == 0:
                slope = np.inf
            else:
                slope = dy / dx
            segs.append((x1, y1, x2, y2, slope))
            angles.append(angle)
    if not segs:
        continue
    mean_angle = np.median(angles)
    filtered = []
    for x1, y1, x2, y2, _ in segs:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if abs(angle - mean_angle) > ANGLE_TOL:
            continue
        filtered.append((x1, y1, x2, y2))
    if not filtered:
        continue
    centers = [((x1 + x2) / 2, (x1, y1, x2, y2)) for (x1, y1, x2, y2) in filtered]
    centers.sort(key=lambda c: c[0])
    groups = []
    for xm, seg in centers:
        placed = False
        for grp in groups:
            if abs(xm - grp[0]) <= x_tol:
                grp[1].append(seg)
                grp[0] = np.mean([((s[0] + s[2]) / 2) for s in grp[1]])
                placed = True
                break
        if not placed:
            groups.append([xm, [seg]])
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for _, segs_in_group in groups:
        tops = []
        bots = []
        for x1, y1, x2, y2 in segs_in_group:
            if x1 == x2:
                xt = x1
                xb = x2
            else:
                m = (y2 - y1) / float(x2 - x1)
                q = y1 - m * x1
                xt = int(round((0 - q) / m))
                xb = int(round(((h - 1) - q) / m))
            tops.append(xt)
            bots.append(xb)
        x_top_avg = int(np.mean(tops))
        x_bot_avg = int(np.mean(bots))
        cv2.line(out, (x_top_avg, 0), (x_bot_avg, h - 1), (0, 255, 255), 1)
    out_name = mask_path.stem.replace("_mask", "_lines_avg") + ".jpg"
    cv2.imwrite(str(OUTPUT_DIR / out_name), out)
print("Final lines avg saved in", OUTPUT_DIR)
