#Visualizer2.py
#STEP_5.2


import cv2
import csv
import pathlib
from collections import defaultdict
CROP_DIR   = pathlib.Path(r"CROP_OUTPUT")
CSV_PATH   = CROP_DIR / "COORD_DETECTOR.csv" 		 
VIS_DIR    = CROP_DIR / "VIS_2" 				 
VIS_DIR.mkdir(exist_ok=True)
COLORS = {
    "kept":          (0, 255,  0),    		# green
    "too_small":     (0, 255, 255),  		# yellow
    "height_exceed": (0, 165, 255),   		# orange
    "grid_prox":     (0, 0, 255),   		# red
    "out_of_bounds": (255,   0,   0), 		# blue
    "out_of_words":  (255,   0, 255), 		# magenta
    "bottom_noise":  (255, 255,   0) 		# cyan
}
records = defaultdict(list)
with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
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
        reason = r["reason"]
        color = COLORS.get(reason, (255, 255, 255))   
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    out_path = VIS_DIR / fname
    cv2.imwrite(str(out_path), img)
print("Visualization complete →", VIS_DIR)
