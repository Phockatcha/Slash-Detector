#Visualizer1.py
#STEP_5.1

import cv2
import csv
import pathlib
from collections import defaultdict
CROP_DIR   = pathlib.Path(r"CROP_OUTPUT")
CSV_PATH   = CROP_DIR / "COORD_DETECTOR.csv" 		 
VIS_DIR    = CROP_DIR / "VIS_1" 				 
VIS_DIR.mkdir(exist_ok=True)
COLOR_KEPT = (0, 255, 0)  
records = defaultdict(list)
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["reason"] == "kept": 
            records[row["file"]].append(row)
for fname, items in records.items():
    img_path = CROP_DIR / fname
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[Warning] Imagine not found: {fname}")
        continue
    for r in items:
        x = int(r["x"])
        y = int(r["y"])
        w = int(r["w"])
        h = int(r["h"])
        cv2.rectangle(img, (x, y), (x + w, y + h), COLOR_KEPT, 2)
    out_path = VIS_DIR / fname
    cv2.imwrite(str(out_path), img)
print("Visualization completed (only kept) →", VIS_DIR)
