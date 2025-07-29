#Grid_Masks.py
#STEP_2

import cv2
import numpy as np
import pathlib
CROP_DIR = pathlib.Path(r"CROP_OUTPUT")	
MASK_DIR = CROP_DIR / "MASK_OUTPUT" 				
MASK_DIR.mkdir(exist_ok=True)
VK = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 35))
HK = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
DIL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
for img_path in sorted(CROP_DIR.glob("*.jpg")):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    bin0 = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15, 15
    )
    vert = cv2.morphologyEx(bin0, cv2.MORPH_OPEN, VK)
    hori = cv2.morphologyEx(bin0, cv2.MORPH_OPEN, HK)
    grid = cv2.dilate(vert | hori, DIL)
    out_path = MASK_DIR / f"{img_path.stem}_mask.png"
    cv2.imwrite(str(out_path), grid)
print("Grid masks saved in", MASK_DIR)
